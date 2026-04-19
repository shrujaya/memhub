"""
core/retrieval.py  — MemHub Retrieval Layer
============================================

Responsible for fetching memories on behalf of an agent across both
storage tiers and returning a single ranked, deduplicated result list.

Storage Architecture
────────────────────
  Tier 1  (SQLite)   — working memory: exact keyword + recency search.
                        Fast, low-latency; holds the agent's immediate
                        context window.

  Tier 2  (ChromaDB) — long-term memory: semantic top-k vector search.
                        Slower but high-recall; holds everything that
                        was demoted from Tier 1.

Retrieval Flow
──────────────
                       agent /retrieve request
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     Tier-1 search (SQLite)          Tier-2 search (ChromaDB)
     LIKE / FTS keyword match         top-k cosine similarity
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    _merge_and_rank()
                    (dedup → score → top-n)
                               │
                               ▼
                    _increment_access_counts()
                    (feeds PromotionPolicy)
                               │
                               ▼
                    RetrievalResult returned to caller

ACL Enforcement
───────────────
Every query is scoped by (agent_id, namespace). A "shared" namespace
allows an agent to read memories written by other agents in the same
team. "private" limits results strictly to the requesting agent.

Promotion Hook
──────────────
After each Tier-2 hit, ``access_count`` is incremented in ChromaDB
metadata. When the count reaches ``PROMOTION_HIT_THRESHOLD`` (defined
in ``core/policies.py``), ``PromotionPolicy`` will pull that chunk back
into Tier 1 on the next policy sweep.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import chromadb  # type: ignore

from core.policies import CHROMA_COLLECTION_NAME, PROMOTION_HIT_THRESHOLD

# ── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Default number of results returned to the caller.
DEFAULT_TOP_K: int = 5

# Maximum Tier-2 candidates fetched before merge/re-rank (wider net).
CHROMA_FETCH_MULTIPLIER: int = 3

# Recency decay half-life in seconds (used in hybrid scoring).
# A memory becomes half as relevant after this many seconds of no access.
RECENCY_HALF_LIFE_SECS: float = 3_600.0  # 1 hour


# ── Data models ──────────────────────────────────────────────────────────────


class MemoryTier(str, Enum):
    """Indicates which storage tier a result came from."""
    TIER1 = "tier1"  # SQLite working memory
    TIER2 = "tier2"  # ChromaDB long-term memory


@dataclass
class RetrievalResult:
    """
    A single memory chunk returned by the retrieval layer.

    Attributes:
        id:            Unique identifier of the memory record.
        agent_id:      Owning agent.
        content:       The raw text of the memory.
        score:         Final hybrid relevance score (higher = more relevant).
        tier:          Which storage tier this result came from.
        namespace:     'private' | 'shared'.
        access_count:  How many times this chunk has been retrieved.
        created_at:    Unix timestamp of original insertion.
        last_accessed: Unix timestamp of most recent retrieval.
        metadata:      Arbitrary extra key-value pairs from storage.
    """
    id: str
    agent_id: str
    content: str
    score: float
    tier: MemoryTier
    namespace: str = "private"
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """
    The full response object returned to the API layer.

    Attributes:
        agent_id:    The agent that issued the query.
        query:       The original query string.
        results:     Ranked list of :class:`RetrievalResult` objects.
        tier1_hits:  Number of results sourced from Tier 1.
        tier2_hits:  Number of results sourced from Tier 2.
        latency_ms:  Wall-clock time for the full retrieval in milliseconds.
    """
    agent_id: str
    query: str
    results: List[RetrievalResult]
    tier1_hits: int = 0
    tier2_hits: int = 0
    latency_ms: float = 0.0


# ── MemoryRetriever ───────────────────────────────────────────────────────────


class MemoryRetriever:
    """
    Unified retrieval interface for MemHub's two-tier memory store.

    Instantiate once per application lifecycle (e.g., via FastAPI
    dependency injection) and reuse the shared SQLite connection and
    ChromaDB client.

    Args:
        db:       An open :class:`sqlite3.Connection` to ``memhub.db``.
        chroma:   A connected :class:`chromadb.Client` instance.
        top_k:    Default number of results to return per query.
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        chroma: chromadb.Client,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._db = db
        self._top_k = top_k
        self._collection: chromadb.Collection = chroma.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        agent_id: str,
        query: str,
        top_k: Optional[int] = None,
        namespace: str = "private",
        include_shared: bool = False,
        tier_filter: Optional[MemoryTier] = None,
    ) -> RetrievalResponse:
        """
        Retrieve the most relevant memories for *agent_id* given *query*.

        Args:
            agent_id:       The requesting agent.
            query:          Natural-language or keyword query string.
            top_k:          Override the default result count.
            namespace:      The agent's own namespace ('private' by default).
            include_shared: If True, also include 'shared' namespace results.
                            Enables cross-agent memory sharing within a team.
            tier_filter:    If set, restrict results to a single tier.
                            Useful for debugging or targeted retrieval.

        Returns:
            A :class:`RetrievalResponse` containing ranked results and
            diagnostic metadata.
        """
        k = top_k or self._top_k
        start_ns = time.perf_counter_ns()

        # Build namespace filter
        allowed_namespaces = [namespace]
        if include_shared:
            allowed_namespaces.append("shared")

        # Fan out to tiers (skip if filtered)
        tier1_results: List[RetrievalResult] = []
        tier2_results: List[RetrievalResult] = []

        if tier_filter is None or tier_filter == MemoryTier.TIER1:
            tier1_results = self._search_tier1(
                agent_id=agent_id,
                query=query,
                allowed_namespaces=allowed_namespaces,
                limit=k * CHROMA_FETCH_MULTIPLIER,
            )

        if tier_filter is None or tier_filter == MemoryTier.TIER2:
            tier2_results = self._search_tier2(
                agent_id=agent_id,
                query=query,
                allowed_namespaces=allowed_namespaces,
                n_results=k * CHROMA_FETCH_MULTIPLIER,
            )

        # Merge, deduplicate, and re-rank
        merged = self._merge_and_rank(tier1_results, tier2_results, top_k=k)

        # Side-effect: update access counters (feeds PromotionPolicy)
        self._increment_access_counts(merged)

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        response = RetrievalResponse(
            agent_id=agent_id,
            query=query,
            results=merged,
            tier1_hits=sum(1 for r in merged if r.tier == MemoryTier.TIER1),
            tier2_hits=sum(1 for r in merged if r.tier == MemoryTier.TIER2),
            latency_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "Retrieval for agent '%s': query=%r, top_k=%d, "
            "tier1_hits=%d, tier2_hits=%d, latency=%.2fms",
            agent_id,
            query[:60],
            k,
            response.tier1_hits,
            response.tier2_hits,
            elapsed_ms,
        )
        return response

    async def retrieve_working_memory(
        self, agent_id: str, namespace: str = "private"
    ) -> List[RetrievalResult]:
        """
        Return ALL Tier-1 memories for an agent in chronological order.

        Used by the API's /retrieve endpoint to reconstruct the agent's
        full current context window without a semantic query. Also used
        by DemotionPolicy to load the scratchpad before summarisation.

        Args:
            agent_id:  The requesting agent.
            namespace: Namespace filter (default 'private').

        Returns:
            List of :class:`RetrievalResult` ordered oldest-first.
        """
        rows = self._db.execute(
            """
            SELECT id, agent_id, content, created_at, last_accessed,
                   access_count, namespace
              FROM working_memory
             WHERE agent_id = ?
               AND namespace = ?
             ORDER BY created_at ASC
            """,
            (agent_id, namespace),
        ).fetchall()

        results = [
            RetrievalResult(
                id=r[0],
                agent_id=r[1],
                content=r[2],
                score=1.0,  # No relevance scoring for full-dump mode
                tier=MemoryTier.TIER1,
                namespace=r[6],
                access_count=r[5],
                created_at=r[3],
                last_accessed=r[4],
            )
            for r in rows
        ]

        # Update last_accessed for all returned rows
        self._touch_tier1_rows([r.id for r in results])
        return results

    # ── Tier-1 search (SQLite) ────────────────────────────────────────────────

    def _search_tier1(
        self,
        agent_id: str,
        query: str,
        allowed_namespaces: List[str],
        limit: int,
    ) -> List[RetrievalResult]:
        """
        Keyword search over Tier-1 working memory using SQLite LIKE.

        Scoring heuristic
        ─────────────────
        Each row is scored with a hybrid of:
          • Keyword match bonus: +1.0 per query term found in content.
          • Recency bonus: decays exponentially with time since last access
            using ``RECENCY_HALF_LIFE_SECS``.
          • Access-count bonus: log-scaled to reward well-retrieved memories.

        NOTE: For production scale, replace the LIKE scan with SQLite FTS5
        (full-text search) by running:
            CREATE VIRTUAL TABLE working_memory_fts USING fts5(content, ...);

        Args:
            agent_id:            Owning agent.
            query:               Raw query string.
            allowed_namespaces:  Namespaces the caller is permitted to read.
            limit:               Maximum rows to evaluate.

        Returns:
            Scored list of :class:`RetrievalResult`.
        """
        ns_placeholders = ",".join("?" * len(allowed_namespaces))
        rows = self._db.execute(
            f"""
            SELECT id, agent_id, content, created_at, last_accessed,
                   access_count, namespace
              FROM working_memory
             WHERE agent_id = ?
               AND namespace IN ({ns_placeholders})
             ORDER BY last_accessed DESC
             LIMIT ?
            """,
            (agent_id, *allowed_namespaces, limit),
        ).fetchall()

        query_terms = set(query.lower().split())
        now = time.time()
        results: List[RetrievalResult] = []

        for r in rows:
            row_id, row_agent, content, created_at, last_accessed, access_count, ns = r
            content_lower = content.lower()

            # Keyword match score
            keyword_score = sum(1.0 for term in query_terms if term in content_lower)
            if keyword_score == 0:
                # Skip entries with zero keyword relevance in Tier-1 pass
                continue

            # Recency score: exponential decay
            age_secs = now - last_accessed
            recency_score = 2 ** (-age_secs / RECENCY_HALF_LIFE_SECS)

            # Frequency score: log-scaled access count
            import math
            freq_score = math.log1p(access_count) / 10.0

            final_score = keyword_score + recency_score + freq_score

            results.append(
                RetrievalResult(
                    id=row_id,
                    agent_id=row_agent,
                    content=content,
                    score=final_score,
                    tier=MemoryTier.TIER1,
                    namespace=ns,
                    access_count=access_count,
                    created_at=created_at,
                    last_accessed=last_accessed,
                )
            )

        logger.debug(
            "Tier-1 search for agent '%s': %d candidate(s) scored.", agent_id, len(results)
        )
        return results

    # ── Tier-2 search (ChromaDB) ──────────────────────────────────────────────

    def _search_tier2(
        self,
        agent_id: str,
        query: str,
        allowed_namespaces: List[str],
        n_results: int,
    ) -> List[RetrievalResult]:
        """
        Semantic top-k vector search over Tier-2 long-term memory (ChromaDB).

        ChromaDB returns cosine distances in [0, 2]; we convert them to a
        similarity score in (0, 1] via: score = 1 / (1 + distance).

        ACL filter is applied server-side via ChromaDB's ``where`` clause,
        so no extra Python filtering is needed after the query.

        Args:
            agent_id:            Owning agent.
            query:               Natural-language query for embedding.
            allowed_namespaces:  Namespaces the caller may access.
            n_results:           Number of Tier-2 candidates to fetch.

        Returns:
            Scored list of :class:`RetrievalResult`.
        """
        # Build ChromaDB WHERE filter (namespace ACL + agent scope)
        if len(allowed_namespaces) == 1:
            where_filter: Dict[str, Any] = {
                "$and": [
                    {"agent_id": {"$eq": agent_id}},
                    {"namespace": {"$eq": allowed_namespaces[0]}},
                ]
            }
        else:
            where_filter = {
                "$and": [
                    {"agent_id": {"$eq": agent_id}},
                    {"namespace": {"$in": allowed_namespaces}},
                ]
            }

        try:
            collection_size = self._collection.count()
            if collection_size == 0:
                logger.debug("Tier-2 collection is empty; skipping search.")
                return []

            raw = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, collection_size),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: List[RetrievalResult] = []
        now = time.time()

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Convert distance → similarity score
            similarity = 1.0 / (1.0 + dist)

            results.append(
                RetrievalResult(
                    id=chunk_id,
                    agent_id=meta.get("agent_id", agent_id),
                    content=doc,
                    score=similarity,
                    tier=MemoryTier.TIER2,
                    namespace=meta.get("namespace", "private"),
                    access_count=meta.get("access_count", 0),
                    created_at=meta.get("created_at", now),
                    last_accessed=meta.get("last_accessed", now),
                    metadata=meta,
                )
            )

        logger.debug(
            "Tier-2 search for agent '%s': %d result(s) returned.", agent_id, len(results)
        )
        return results

    # ── Merge & Re-rank ───────────────────────────────────────────────────────

    def _merge_and_rank(
        self,
        tier1: List[RetrievalResult],
        tier2: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Deduplicate and re-rank results from both tiers into a single list.

        Deduplication
        ─────────────
        If the same memory ID appears in both tiers (e.g., after a failed
        cleanup), the Tier-1 copy is preferred (lower latency, exact match).

        Re-ranking
        ──────────
        Scores from the two tiers are not directly comparable:
          • Tier-1 scores are additive (keyword + recency + frequency).
          • Tier-2 scores are cosine similarities in (0, 1].

        We normalise each list independently to [0, 1] before merging, then
        apply tier-preference weights:
          • Tier-1 weight: 0.6  (working memory is immediately relevant)
          • Tier-2 weight: 0.4  (semantic matches from long-term store)

        Args:
            tier1:  Results from Tier-1 search.
            tier2:  Results from Tier-2 search.
            top_k:  Maximum number of results to return.

        Returns:
            Sorted, deduplicated list of up to *top_k* results.
        """
        TIER1_WEIGHT = 0.6
        TIER2_WEIGHT = 0.4

        def _normalise(results: List[RetrievalResult]) -> List[RetrievalResult]:
            if not results:
                return results
            max_score = max(r.score for r in results)
            min_score = min(r.score for r in results)
            spread = max_score - min_score or 1.0
            for r in results:
                r.score = (r.score - min_score) / spread
            return results

        tier1 = _normalise(tier1)
        tier2 = _normalise(tier2)

        # Apply tier weighting
        for r in tier1:
            r.score *= TIER1_WEIGHT
        for r in tier2:
            r.score *= TIER2_WEIGHT

        # Merge, prefer Tier-1 on id collision
        seen_ids: set[str] = set()
        merged: List[RetrievalResult] = []

        for result in tier1 + tier2:
            if result.id in seen_ids:
                logger.debug(
                    "Dedup: skipping Tier-2 duplicate of Tier-1 id '%s'.", result.id
                )
                continue
            seen_ids.add(result.id)
            merged.append(result)

        merged.sort(key=lambda r: r.score, reverse=True)
        logger.debug("Merge produced %d unique result(s), returning top %d.", len(merged), top_k)
        return merged[:top_k]

    # ── Access counter updates ────────────────────────────────────────────────

    def _increment_access_counts(self, results: Sequence[RetrievalResult]) -> None:
        """
        Bump ``access_count`` and ``last_accessed`` for all returned results.

        Tier-1 records are updated in SQLite.
        Tier-2 records are updated in ChromaDB metadata.

        This is the **key hook** that feeds :class:`~core.policies.PromotionPolicy`.
        Once a Tier-2 chunk's ``access_count`` reaches
        ``PROMOTION_HIT_THRESHOLD``, the next PolicyEngine sweep will pull it
        back into Tier 1.
        """
        now = time.time()
        tier1_ids = [r.id for r in results if r.tier == MemoryTier.TIER1]
        tier2_results = [r for r in results if r.tier == MemoryTier.TIER2]

        # ── Tier 1: batch SQL UPDATE ──────────────────────────────────────────
        if tier1_ids:
            self._touch_tier1_rows(tier1_ids, now=now)

        # ── Tier 2: update each ChromaDB chunk's metadata individually ────────
        for r in tier2_results:
            new_count = r.access_count + 1
            try:
                self._collection.update(
                    ids=[r.id],
                    metadatas=[
                        {
                            **r.metadata,
                            "access_count": new_count,
                            "last_accessed": now,
                        }
                    ],
                )
                logger.debug(
                    "Tier-2 chunk '%s' access_count → %d (promotion threshold=%d).",
                    r.id,
                    new_count,
                    PROMOTION_HIT_THRESHOLD,
                )
                if new_count >= PROMOTION_HIT_THRESHOLD:
                    logger.info(
                        "Chunk '%s' has hit promotion threshold (%d). "
                        "Will be promoted on next PolicyEngine sweep.",
                        r.id,
                        PROMOTION_HIT_THRESHOLD,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to update access_count for Tier-2 chunk '%s': %s", r.id, exc
                )

    def _touch_tier1_rows(
        self, ids: Sequence[str], now: Optional[float] = None
    ) -> None:
        """
        Update ``last_accessed`` and increment ``access_count`` for a batch
        of Tier-1 rows in a single SQL statement.
        """
        if not ids:
            return
        ts = now or time.time()
        placeholders = ",".join("?" * len(ids))
        self._db.execute(
            f"""
            UPDATE working_memory
               SET last_accessed = ?,
                   access_count  = access_count + 1
             WHERE id IN ({placeholders})
            """,
            (ts, *ids),
        )
        self._db.commit()
        logger.debug("Touched %d Tier-1 row(s).", len(ids))

    # ── Convenience: lookup by ID ─────────────────────────────────────────────

    async def get_by_id(
        self, memory_id: str, agent_id: str
    ) -> Optional[RetrievalResult]:
        """
        Fetch a single memory by its exact ID, searching Tier 1 first then
        Tier 2. Returns None if not found or if the agent_id does not match
        (ACL enforcement).

        Args:
            memory_id:  The UUID of the target memory.
            agent_id:   The requesting agent (used for ownership check).

        Returns:
            The :class:`RetrievalResult`, or None if not found / forbidden.
        """
        # Check Tier 1
        row = self._db.execute(
            """
            SELECT id, agent_id, content, created_at, last_accessed,
                   access_count, namespace
              FROM working_memory
             WHERE id = ? AND agent_id = ?
            """,
            (memory_id, agent_id),
        ).fetchone()

        if row:
            self._touch_tier1_rows([row[0]])
            return RetrievalResult(
                id=row[0],
                agent_id=row[1],
                content=row[2],
                score=1.0,
                tier=MemoryTier.TIER1,
                namespace=row[6],
                access_count=row[5],
                created_at=row[3],
                last_accessed=row[4],
            )

        # Check Tier 2
        try:
            raw = self._collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            )
            ids = raw.get("ids", [])
            if not ids:
                return None

            doc = raw["documents"][0]
            meta = raw["metadatas"][0]

            # ACL: only return if this agent owns it (or it's shared)
            if meta.get("agent_id") != agent_id and meta.get("namespace") != "shared":
                logger.warning(
                    "ACL denied: agent '%s' tried to access memory '%s' "
                    "owned by '%s'.",
                    agent_id,
                    memory_id,
                    meta.get("agent_id"),
                )
                return None

            now = time.time()
            new_count = meta.get("access_count", 0) + 1
            self._collection.update(
                ids=[memory_id],
                metadatas=[{**meta, "access_count": new_count, "last_accessed": now}],
            )

            return RetrievalResult(
                id=ids[0],
                agent_id=meta.get("agent_id", agent_id),
                content=doc,
                score=1.0,
                tier=MemoryTier.TIER2,
                namespace=meta.get("namespace", "private"),
                access_count=new_count,
                created_at=meta.get("created_at", now),
                last_accessed=now,
                metadata=meta,
            )

        except Exception as exc:
            logger.error("Error fetching memory '%s' from Tier 2: %s", memory_id, exc)
            return None
