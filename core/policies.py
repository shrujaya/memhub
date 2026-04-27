"""
core/policies.py  — MemHub Memory Lifecycle Policies
======================================================

This module implements the three core lifecycle operations that govern how
memories flow through MemHub's two-tier storage architecture:

┌─────────────────────────────────────────────────────────────────┐
│  Tier 1  ─  SQLite  (Working Memory / Short-Term)               │
│     ↕  eviction / promotion / demotion                          │
│  Tier 2  ─  ChromaDB  (Long-Term / Vector Memory)              │
└─────────────────────────────────────────────────────────────────┘

Key Policies
────────────
  • EvictionPolicy  – Decides *which* Tier-1 rows to remove when the
                      working-memory budget is full.
  • PromotionPolicy – Detects frequently retrieved Tier-2 chunks and
                      promotes them back to Tier 1 for faster access.
  • DemotionPolicy  – Orchestrates the full compress-and-move pipeline:
                      summarise → store summary in Tier 1 → embed raw
                      chunks in Tier 2 → delete originals from Tier 1.

Coordinating with summarisation
────────────────────────────────
  DemotionPolicy calls `summarize_working_memory()` from
  `core.summarization` to compress the oldest 70 % of working-memory
  rows before writing them into ChromaDB, so the LLM context window
  always receives a compact, high-signal representation of history.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb  # type: ignore

from core.summarization import count_tokens, summarize_working_memory

# ── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Token budget for an agent's Tier-1 working memory before eviction fires.
WORKING_MEMORY_TOKEN_BUDGET: int = 2_000

# Number of retrievals that promote a Tier-2 chunk back to Tier 1.
PROMOTION_HIT_THRESHOLD: int = 3

# Minimum age (seconds) before a Tier-1 row is eligible for demotion.
MIN_AGE_FOR_DEMOTION_SECS: int = 60

# ChromaDB collection used for long-term vector memory.
CHROMA_COLLECTION_NAME: str = "memhub_longterm"

# Maximum number of Tier-1 rows returned to a policy in one pass.
POLICY_BATCH_SIZE: int = 500


# ── Data models ──────────────────────────────────────────────────────────────

class EvictionStrategy(str, Enum):
    """Available eviction algorithms for Tier-1 working memory."""
    LRU = "lru"      # Least-Recently Used  (access time)
    FIFO = "fifo"    # First-In First-Out   (creation time)
    LFU = "lfu"      # Least-Frequently Used (hit count)


@dataclass
class MemoryRecord:
    """
    A single working-memory row surfaced from SQLite.

    Attributes:
        id:           Primary key / UUID of this memory entry.
        agent_id:     Owning agent identifier.
        content:      The raw text of the memory.
        created_at:   Unix timestamp of insertion.
        last_accessed: Unix timestamp of the last retrieval.
        access_count: Cumulative access count (for LFU / promotion).
        namespace:    'private' | 'shared' — used for ACL enforcement.
        metadata:     Optional extra key/value pairs (source, tags, …).
    """
    id: str
    agent_id: str
    content: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    namespace: str = "private"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    """
    The structured outcome returned by every policy run.

    Attributes:
        evicted_ids:       Tier-1 row IDs removed outright (no demotion).
        demoted_ids:       Tier-1 row IDs moved to Tier 2.
        promoted_ids:      Tier-2 chunk IDs moved back to Tier 1.
        new_summary:       Compressed text inserted as a new Tier-1 row
                           (only set after demotion + summarisation).
        retained_memories: Content strings left untouched in Tier 1.
        tokens_before:     Token count before the policy ran.
        tokens_after:      Estimated token count after the policy ran.
    """
    evicted_ids: List[str] = field(default_factory=list)
    demoted_ids: List[str] = field(default_factory=list)
    promoted_ids: List[str] = field(default_factory=list)
    new_summary: Optional[str] = None
    retained_memories: List[str] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0


# ── Abstract base ─────────────────────────────────────────────────────────────

class BasePolicy(ABC):
    """
    Common interface all MemHub policies must implement.

    Concrete policies receive a SQLite connection and a ChromaDB client
    at construction time to avoid re-opening connections on every call.
    """

    def __init__(self, db: sqlite3.Connection, chroma: chromadb.Client) -> None:
        self._db = db
        self._chroma = chroma
        self._collection: chromadb.Collection = chroma.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _fetch_working_memory(
        self,
        agent_id: str,
        order_by: str = "created_at ASC",
        limit: int = POLICY_BATCH_SIZE,
    ) -> List[MemoryRecord]:
        """
        Load Tier-1 memory rows for *agent_id* from SQLite.

        Args:
            agent_id:  The target agent.
            order_by:  SQL ORDER BY clause (column + direction).
            limit:     Maximum rows to retrieve.

        Returns:
            A list of :class:`MemoryRecord` instances.
        """
        cursor = self._db.execute(
            f"""
            SELECT id, agent_id, content, created_at, last_accessed,
                   access_count, namespace
              FROM working_memory
             WHERE agent_id = ?
             ORDER BY {order_by}
             LIMIT ?
            """,
            (agent_id, limit),
        )
        rows = cursor.fetchall()
        return [
            MemoryRecord(
                id=r[0],
                agent_id=r[1],
                content=r[2],
                created_at=r[3],
                last_accessed=r[4],
                access_count=r[5],
                namespace=r[6],
            )
            for r in rows
        ]

    def _delete_from_tier1(self, ids: Sequence[str]) -> None:
        """Hard-delete a batch of rows from the working-memory table."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._db.execute(
            f"DELETE FROM working_memory WHERE id IN ({placeholders})", list(ids)
        )
        self._db.commit()
        logger.debug("Deleted %d Tier-1 row(s): %s", len(ids), ids)

    def _insert_summary_to_tier1(
        self, agent_id: str, summary: str, namespace: str = "private"
    ) -> str:
        """
        Insert a compressed summary back into Tier-1 (SQLite) so that it
        immediately becomes available for the agent's next context window.

        Returns the new row's generated ID.
        """
        import uuid

        new_id = str(uuid.uuid4())
        now = time.time()
        self._db.execute(
            """
            INSERT INTO working_memory
                    (id, agent_id, content, created_at, last_accessed,
                     access_count, namespace)
            VALUES  (?,  ?,        ?,       ?,          ?,            0,
                     ?)
            """,
            (new_id, agent_id, summary, now, now, namespace),
        )
        self._db.commit()
        logger.info(
            "Inserted summarised memory for agent '%s' (id=%s, %d tokens).",
            agent_id,
            new_id,
            count_tokens(summary),
        )
        return new_id

    def _embed_to_tier2(self, records: Sequence[MemoryRecord]) -> None:
        """
        Persist a batch of memory records into ChromaDB (Tier 2) for
        long-term vector retrieval.

        NOTE: Embeddings are computed by ChromaDB's default embedding
        function. Swap for a custom one by setting
        ``collection.embedding_function`` at collection creation time.
        """
        if not records:
            return
        self._collection.add(
            ids=[r.id for r in records],
            documents=[r.content for r in records],
            metadatas=[
                {
                    "agent_id": r.agent_id,
                    "created_at": r.created_at,
                    "access_count": r.access_count,
                    "namespace": r.namespace,
                    **r.metadata,
                }
                for r in records
            ],
        )
        logger.info(
            "Embedded %d record(s) into Tier-2 ChromaDB collection '%s'.",
            len(records),
            CHROMA_COLLECTION_NAME,
        )

    @abstractmethod
    async def run(self, agent_id: str, **kwargs: Any) -> PolicyResult:
        """Execute the policy for *agent_id* and return a :class:`PolicyResult`."""


# ── EvictionPolicy ────────────────────────────────────────────────────────────

class EvictionPolicy(BasePolicy):
    """
    Controls *when* and *which* Tier-1 memories are discarded.

    Supported strategies
    ────────────────────
    LRU  – Remove the rows with the oldest ``last_accessed`` timestamp.
           Best for conversational agents that revisit recent topics.

    FIFO – Remove the oldest inserted rows regardless of access pattern.
           Simplest strategy; good baseline for benchmarking.

    LFU  – Remove the rows with the fewest total accesses.
           Preserves frequently-retrieved knowledge; suits research agents.

    The policy only fires when the agent's working-memory token budget
    (``WORKING_MEMORY_TOKEN_BUDGET``) is exceeded.  Eviction is
    *destructive* — evicted rows are deleted without being moved to Tier 2.
    If you need historical preservation, use :class:`DemotionPolicy` instead.
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        chroma: chromadb.Client,
        strategy: EvictionStrategy = EvictionStrategy.LRU,
        token_budget: int = WORKING_MEMORY_TOKEN_BUDGET,
    ) -> None:
        super().__init__(db, chroma)
        self.strategy = strategy
        self.token_budget = token_budget

    # ── strategy → ORDER BY mapping ──────────────────────────────────────────

    _STRATEGY_ORDER: Dict[EvictionStrategy, str] = {
        EvictionStrategy.LRU:  "last_accessed ASC",
        EvictionStrategy.FIFO: "created_at ASC",
        EvictionStrategy.LFU:  "access_count ASC, created_at ASC",
    }

    async def run(self, agent_id: str, **kwargs: Any) -> PolicyResult:
        """
        Evict Tier-1 rows for *agent_id* until the token budget is satisfied.

        Steps
        ─────
        1. Load all Tier-1 rows, sorted by the chosen strategy.
        2. Walk oldest-first, accumulate tokens until budget is satisfied.
        3. Hard-delete the rows that exceed the budget.

        Returns:
            :class:`PolicyResult` with ``evicted_ids`` populated.
        """
        order_by = self._STRATEGY_ORDER[self.strategy]
        records = self._fetch_working_memory(agent_id, order_by=order_by)

        if not records:
            logger.debug("No Tier-1 records found for agent '%s'.", agent_id)
            return PolicyResult()

        # Measure current total
        tokens_before = sum(count_tokens(r.content) for r in records)
        result = PolicyResult(tokens_before=tokens_before)

        if tokens_before <= self.token_budget:
            logger.debug(
                "Agent '%s' within budget (%d/%d tokens); skipping eviction.",
                agent_id,
                tokens_before,
                self.token_budget,
            )
            result.tokens_after = tokens_before
            result.retained_memories = [r.content for r in records]
            return result

        logger.info(
            "Agent '%s' over budget (%d/%d). Running %s eviction.",
            agent_id,
            tokens_before,
            self.token_budget,
            self.strategy.value.upper(),
        )

        evict_ids: List[str] = []
        running_tokens = tokens_before

        for record in records:
            if running_tokens <= self.token_budget:
                break
            evict_ids.append(record.id)
            running_tokens -= count_tokens(record.content)

        self._delete_from_tier1(evict_ids)

        result.evicted_ids = evict_ids
        result.tokens_after = running_tokens
        result.retained_memories = [
            r.content for r in records if r.id not in set(evict_ids)
        ]

        logger.info(
            "Eviction complete for agent '%s': %d row(s) removed, %d tokens remaining.",
            agent_id,
            len(evict_ids),
            running_tokens,
        )
        return result


# ── PromotionPolicy ───────────────────────────────────────────────────────────

class PromotionPolicy(BasePolicy):
    """
    Promotes frequently-accessed Tier-2 (ChromaDB) memories back to
    Tier-1 (SQLite working memory), giving the agent fast, exact-match
    access to knowledge it keeps coming back to.

    Trigger
    ───────
    A Tier-2 chunk is eligible for promotion when its ``access_count``
    metadata field reaches ``hit_threshold``.  The retrieval layer
    (``core/retrieval.py``) is responsible for incrementing that counter
    each time the chunk is returned to an agent.

    Conflict Avoidance
    ──────────────────
    Before promoting, the policy checks whether an identical ``id``
    already exists in SQLite to prevent duplicates.
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        chroma: chromadb.Client,
        hit_threshold: int = PROMOTION_HIT_THRESHOLD,
    ) -> None:
        super().__init__(db, chroma)
        self.hit_threshold = hit_threshold

    def _fetch_promotion_candidates(
        self, agent_id: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Query ChromaDB for Tier-2 chunks belonging to *agent_id* whose
        access_count has reached the promotion threshold.

        Returns a list of ``(id, document, metadata)`` tuples.
        """
        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"agent_id": {"$eq": agent_id}},
                        {"access_count": {"$gte": self.hit_threshold}},
                    ]
                },
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed during promotion check: %s", exc)
            return []

        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        return list(zip(ids, docs, metas))

    def _already_in_tier1(self, chunk_id: str) -> bool:
        """Return True if *chunk_id* already exists in working memory."""
        row = self._db.execute(
            "SELECT 1 FROM working_memory WHERE id = ?", (chunk_id,)
        ).fetchone()
        return row is not None

    async def run(self, agent_id: str, **kwargs: Any) -> PolicyResult:
        """
        Promote eligible Tier-2 chunks to Tier 1 for *agent_id*.

        Steps
        ─────
        1. Fetch Tier-2 chunks with access_count ≥ hit_threshold.
        2. Filter out chunks already present in Tier 1.
        3. Re-insert them into SQLite as fresh working-memory rows.
        4. Remove them from ChromaDB to avoid duplication.

        Returns:
            :class:`PolicyResult` with ``promoted_ids`` populated.
        """
        candidates = self._fetch_promotion_candidates(agent_id)
        if not candidates:
            logger.debug("No promotion candidates for agent '%s'.", agent_id)
            return PolicyResult()

        promoted_ids: List[str] = []
        now = time.time()

        for chunk_id, document, meta in candidates:
            if self._already_in_tier1(chunk_id):
                logger.debug("Chunk '%s' already in Tier 1; skipping.", chunk_id)
                continue

            namespace = meta.get("namespace", "private")
            original_created = meta.get("created_at", now)

            self._db.execute(
                """
                INSERT INTO working_memory
                        (id, agent_id, content, created_at, last_accessed,
                         access_count, namespace)
                VALUES  (?,  ?,        ?,       ?,          ?,            ?,
                         ?)
                """,
                (
                    chunk_id,
                    agent_id,
                    document,
                    original_created,
                    now,
                    meta.get("access_count", self.hit_threshold),
                    namespace,
                ),
            )
            promoted_ids.append(chunk_id)

        if promoted_ids:
            self._db.commit()
            # Remove promoted chunks from Tier 2 to avoid duplication
            self._collection.delete(ids=promoted_ids)
            logger.info(
                "Promoted %d Tier-2 chunk(s) to Tier 1 for agent '%s': %s",
                len(promoted_ids),
                agent_id,
                promoted_ids,
            )

        return PolicyResult(promoted_ids=promoted_ids)


# ── DemotionPolicy ────────────────────────────────────────────────────────────

class DemotionPolicy(BasePolicy):
    """
    Orchestrates the full **compress → archive → clean-up** pipeline.

    This is the central policy that ties summarisation and storage together:

    ┌──────────────────────────────────────────────────────────────┐
    │  Tier-1 (SQLite)                                             │
    │    oldest 70 % of rows  ──►  summarize_working_memory()      │
    │                                    │                         │
    │                         ┌──────────┴──────────┐             │
    │                         ▼                     ▼             │
    │                   embed raw chunks          insert           │
    │                   into Tier-2 (Chroma)    compressed         │
    │                                          summary into         │
    │                                          Tier-1 (SQLite)     │
    │    newest 30 % of rows  ──►  retained untouched in Tier 1   │
    └──────────────────────────────────────────────────────────────┘

    Trigger conditions
    ──────────────────
    • Token budget exceeded  AND
    • At least one row is older than ``MIN_AGE_FOR_DEMOTION_SECS``

    This dual guard prevents the policy from compressing brand-new,
    high-value memories that simply happened to be verbose.
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        chroma: chromadb.Client,
        token_budget: int = WORKING_MEMORY_TOKEN_BUDGET,
        min_age_secs: int = MIN_AGE_FOR_DEMOTION_SECS,
    ) -> None:
        super().__init__(db, chroma)
        self.token_budget = token_budget
        self.min_age_secs = min_age_secs

    async def run(self, agent_id: str, **kwargs: Any) -> PolicyResult:
        """
        Execute the full demotion pipeline for *agent_id*.

        Steps
        ─────
        1. Load Tier-1 rows ordered oldest-first.
        2. Check token budget; bail early if under budget.
        3. Filter out rows too young for demotion.
        4. Call ``summarize_working_memory()`` → get demoted + retained split.
        5. Embed the raw demoted records in ChromaDB (Tier 2).
        6. Insert the LLM-generated summary as a new Tier-1 row.
        7. Delete the original demoted rows from SQLite.

        Keyword Args (passed via **kwargs):
            namespace (str): ACL namespace to apply on the summary row
                             (default: ``"private"``).

        Returns:
            :class:`PolicyResult` with all fields populated.
        """
        namespace: str = kwargs.get("namespace", "private")
        records = self._fetch_working_memory(agent_id, order_by="created_at ASC")

        if not records:
            return PolicyResult()

        # ── Step 1: Check token budget ────────────────────────────────────────
        all_contents = [r.content for r in records]
        tokens_before = sum(count_tokens(c) for c in all_contents)
        result = PolicyResult(tokens_before=tokens_before)

        if tokens_before <= self.token_budget:
            logger.debug(
                "Agent '%s' within budget (%d/%d tokens); skipping demotion.",
                agent_id,
                tokens_before,
                self.token_budget,
            )
            result.tokens_after = tokens_before
            result.retained_memories = all_contents
            return result

        # ── Step 2: Filter by minimum age ────────────────────────────────────
        now = time.time()
        eligible = [r for r in records if (now - r.created_at) >= self.min_age_secs]

        if not eligible:
            logger.info(
                "Agent '%s' over budget but no rows old enough for demotion "
                "(min_age=%ds). Consider running EvictionPolicy instead.",
                agent_id,
                self.min_age_secs,
            )
            result.tokens_after = tokens_before
            result.retained_memories = all_contents
            return result

        # ── Step 3: Summarise via core.summarization ──────────────────────────
        logger.info(
            "Running demotion pipeline for agent '%s' (%d eligible rows, %d tokens).",
            agent_id,
            len(eligible),
            tokens_before,
        )

        eligible_contents = [r.content for r in eligible]
        summary_result = await summarize_working_memory(
            agent_id=agent_id,
            memories=eligible_contents,
            threshold=self.token_budget,
        )

        if not summary_result["needs_update"]:
            # Summariser decided no compression needed after all (edge case)
            result.tokens_after = tokens_before
            result.retained_memories = all_contents
            return result

        demoted_contents: List[str] = summary_result["demoted_memories"]
        retained_contents: List[str] = summary_result["retained_memories"]
        new_summary: str = summary_result["new_summary"]

        # Map demoted content back to their MemoryRecord objects (preserves IDs)
        demoted_contents_set = set(demoted_contents)
        demoted_records = [r for r in eligible if r.content in demoted_contents_set]
        demoted_ids = [r.id for r in demoted_records]

        # ── Step 4: Embed raw demoted records → Tier 2 (ChromaDB) ────────────
        self._embed_to_tier2(demoted_records)

        # ── Step 5: Insert compressed summary → Tier 1 (SQLite) ──────────────
        summary_id = self._insert_summary_to_tier1(agent_id, new_summary, namespace)
        logger.info(
            "Inserted demotion summary for agent '%s' (row id=%s).",
            agent_id,
            summary_id,
        )

        # ── Step 6: Delete original demoted rows from Tier 1 ─────────────────
        self._delete_from_tier1(demoted_ids)

        # Compute approximate post-demotion token count
        tokens_after = sum(count_tokens(c) for c in retained_contents) + count_tokens(
            new_summary
        )

        result.demoted_ids = demoted_ids
        result.new_summary = new_summary
        result.retained_memories = retained_contents
        result.tokens_after = tokens_after

        logger.info(
            "Demotion complete for agent '%s': %d row(s) demoted, "
            "%d → %d tokens (%.1f%% reduction).",
            agent_id,
            len(demoted_ids),
            tokens_before,
            tokens_after,
            100 * (1 - tokens_after / max(tokens_before, 1)),
        )
        return result


# ── PolicyEngine — unified entry point ───────────────────────────────────────

class PolicyEngine:
    """
    Top-level coordinator called by the API layer (``api/routes.py``) or
    background scheduler to run the full policy suite for an agent in the
    correct order:

        PromotionPolicy → DemotionPolicy → EvictionPolicy

    Running promotion first guarantees that hot Tier-2 content is
    available in Tier 1 before eviction has a chance to discard it.

    Usage
    ─────
    .. code-block:: python

        engine = PolicyEngine(db_conn, chroma_client)
        result = await engine.run_all(agent_id="agent-42")
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        chroma: chromadb.Client,
        eviction_strategy: EvictionStrategy = EvictionStrategy.LRU,
        token_budget: int = WORKING_MEMORY_TOKEN_BUDGET,
    ) -> None:
        self._promotion = PromotionPolicy(db, chroma)
        self._demotion = DemotionPolicy(db, chroma, token_budget=token_budget)
        self._eviction = EvictionPolicy(
            db, chroma, strategy=eviction_strategy, token_budget=token_budget
        )

    async def run_all(
        self, agent_id: str, namespace: str = "private"
    ) -> Dict[str, PolicyResult]:
        """
        Execute all three policies sequentially for *agent_id*.

        Args:
            agent_id:  The target agent.
            namespace: ACL namespace forwarded to DemotionPolicy.

        Returns:
            A dict keyed by policy name containing each :class:`PolicyResult`.
        """
        logger.info("PolicyEngine: starting full policy sweep for agent '%s'.", agent_id)

        promotion_result = await self._promotion.run(agent_id)
        demotion_result = await self._demotion.run(agent_id, namespace=namespace)
        eviction_result = await self._eviction.run(agent_id)

        results = {
            "promotion": promotion_result,
            "demotion": demotion_result,
            "eviction": eviction_result,
        }

        logger.info(
            "PolicyEngine sweep complete for agent '%s'. "
            "Promoted=%d, Demoted=%d, Evicted=%d.",
            agent_id,
            len(promotion_result.promoted_ids),
            len(demotion_result.demoted_ids),
            len(eviction_result.evicted_ids),
        )
        return results
