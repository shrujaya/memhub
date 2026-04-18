"""
api/routes.py  — MemHub REST API Endpoints
===========================================

Exposes four endpoint groups via a single APIRouter:

  POST /store           — write a memory to Tier-1 working memory
  POST /retrieve        — hybrid Tier-1 + Tier-2 semantic search
  POST /policies/run    — manually trigger a PolicyEngine sweep (admin)
  GET  /health          — liveness + readiness probe
  GET  /memory/{id}     — fetch a single memory by exact UUID

All endpoints are ACL-gated via the FastAPI dependency functions in
``api/auth.py``.  The ``X-Agent-ID`` header is required on every call.

Storage connections (SQLite + ChromaDB) are injected from
``app.state`` as set up in ``api/main.py``.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from typing import Optional

import chromadb  # type: ignore
from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.auth import (
    ensure_agent_registered,
    get_agent_id,
    require_read_access,
    require_shared_read,
    require_write_access,
)
from api.models import (
    EvictionStrategyEnum,
    HealthResponse,
    MemoryItem,
    MemoryTierEnum,
    Namespace,
    PolicyRunRequest,
    PolicyRunResponse,
    PolicySummary,
    RetrieveRequest,
    RetrieveResponse,
    StoreRequest,
    StoreResponse,
)
from core.policies import EvictionStrategy, PolicyEngine, WORKING_MEMORY_TOKEN_BUDGET
from core.retrieval import MemoryRetriever, MemoryTier
from core.summarization import count_tokens

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Dependency helpers ────────────────────────────────────────────────────────


def _get_db(request: Request) -> sqlite3.Connection:
    return request.app.state.db


def _get_chroma(request: Request) -> chromadb.Client:
    return request.app.state.chroma


def _get_retriever(request: Request) -> MemoryRetriever:
    return request.app.state.retriever


# ── POST /store ───────────────────────────────────────────────────────────────


@router.post(
    "/store",
    response_model=StoreResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store a memory",
    description=(
        "Write a new memory entry into Tier-1 (SQLite) working memory for the "
        "calling agent. If the agent's token budget is exceeded after the write "
        "and ``run_policies`` is True, the PolicyEngine runs automatically "
        "(Promote → Demote → Evict)."
    ),
    tags=["Memory"],
)
async def store_memory(
    payload: StoreRequest,
    request: Request,
    agent_id: str = Depends(get_agent_id),
) -> StoreResponse:
    """
    Steps
    ─────
    1. ACL: verify the agent may write to the requested namespace.
    2. Auto-register the agent if it's new.
    3. Insert the memory row into SQLite working_memory.
    4. Optionally trigger the PolicyEngine.
    5. Return the new memory ID and token count.
    """
    db: sqlite3.Connection = _get_db(request)
    chroma: chromadb.Client = _get_chroma(request)

    # ── ACL guard ─────────────────────────────────────────────────────────────
    # Validate that the header agent matches the payload agent_id
    if agent_id != payload.agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"X-Agent-ID header ('{agent_id}') does not match "
                f"payload agent_id ('{payload.agent_id}')."
            ),
        )

    ensure_agent_registered(db, agent_id, payload.namespace.value)

    # ── Insert into Tier 1 ────────────────────────────────────────────────────
    memory_id = str(uuid.uuid4())
    now = time.time()
    token_count = count_tokens(payload.content)

    db.execute(
        """
        INSERT INTO working_memory
                (id, agent_id, content, created_at, last_accessed,
                 access_count, namespace)
        VALUES  (?,  ?,        ?,       ?,          ?,            0,
                 ?)
        """,
        (memory_id, agent_id, payload.content, now, now, payload.namespace.value),
    )
    db.commit()

    logger.info(
        "Stored memory id=%s for agent='%s' (%d tokens, namespace='%s').",
        memory_id,
        agent_id,
        token_count,
        payload.namespace.value,
    )

    # ── Optionally run PolicyEngine ───────────────────────────────────────────
    policies_triggered = False
    if payload.run_policies:
        engine = PolicyEngine(db, chroma)
        await engine.run_all(agent_id=agent_id, namespace=payload.namespace.value)
        policies_triggered = True

    return StoreResponse(
        memory_id=memory_id,
        agent_id=agent_id,
        token_count=token_count,
        policies_triggered=policies_triggered,
    )


# ── POST /retrieve ────────────────────────────────────────────────────────────


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve memories",
    description=(
        "Perform a hybrid Tier-1 keyword + Tier-2 semantic search for the "
        "calling agent. Results are ranked, deduplicated, and capped at "
        "``top_k``.\n\n"
        "Pass ``include_shared=true`` to also search the shared team namespace "
        "(requires 'shared' read access in the ACL table).\n\n"
        "Pass ``memory_id`` to perform an exact ID lookup instead of a query."
    ),
    tags=["Memory"],
)
async def retrieve_memories(
    payload: RetrieveRequest,
    request: Request,
    agent_id: str = Depends(get_agent_id),
) -> RetrieveResponse:
    """
    Steps
    ─────
    1. ACL: verify agent identity and shared-namespace access if requested.
    2. Delegate to MemoryRetriever for hybrid ranked search.
    3. Map internal RetrievalResult objects to MemoryItem response schema.
    4. Return RetrieveResponse with tier hit counts and latency.
    """
    # ── ACL guard ─────────────────────────────────────────────────────────────
    if agent_id != payload.agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"X-Agent-ID header ('{agent_id}') does not match "
                f"payload agent_id ('{payload.agent_id}')."
            ),
        )

    db: sqlite3.Connection = _get_db(request)
    ensure_agent_registered(db, agent_id, payload.namespace.value)

    # Shared namespace permission check
    if payload.include_shared:
        from api.auth import _has_namespace_permission
        if not _has_namespace_permission(db, agent_id, "shared", require_write=False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Agent '{agent_id}' does not have access to the "
                    "'shared' namespace."
                ),
            )

    retriever: MemoryRetriever = _get_retriever(request)

    # ── Exact ID lookup path ──────────────────────────────────────────────────
    if payload.memory_id is not None:
        result = await retriever.get_by_id(
            memory_id=str(payload.memory_id),
            agent_id=agent_id,
        )
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory '{payload.memory_id}' not found or access denied.",
            )
        item = MemoryItem(
            id=result.id,
            agent_id=result.agent_id,
            content=result.content,
            score=result.score,
            tier=MemoryTierEnum(result.tier.value),
            namespace=Namespace(result.namespace),
            access_count=result.access_count,
            created_at=result.created_at,
            last_accessed=result.last_accessed,
            metadata=result.metadata,
        )
        return RetrieveResponse(
            agent_id=agent_id,
            query=str(payload.memory_id),
            results=[item],
            tier1_hits=1 if result.tier == MemoryTier.TIER1 else 0,
            tier2_hits=1 if result.tier == MemoryTier.TIER2 else 0,
            total_results=1,
            latency_ms=0.0,
        )

    # ── Hybrid search path ────────────────────────────────────────────────────
    tier_filter: Optional[MemoryTier] = None
    if payload.tier_filter is not None:
        tier_filter = MemoryTier(payload.tier_filter.value)

    response = await retriever.retrieve(
        agent_id=agent_id,
        query=payload.query,
        top_k=payload.top_k,
        namespace=payload.namespace.value,
        include_shared=payload.include_shared,
        tier_filter=tier_filter,
    )

    items = [
        MemoryItem(
            id=r.id,
            agent_id=r.agent_id,
            content=r.content,
            score=round(r.score, 4),
            tier=MemoryTierEnum(r.tier.value),
            namespace=Namespace(r.namespace),
            access_count=r.access_count,
            created_at=r.created_at,
            last_accessed=r.last_accessed,
            metadata=r.metadata,
        )
        for r in response.results
    ]

    return RetrieveResponse(
        agent_id=agent_id,
        query=payload.query,
        results=items,
        tier1_hits=response.tier1_hits,
        tier2_hits=response.tier2_hits,
        total_results=len(items),
        latency_ms=response.latency_ms,
    )


# ── POST /policies/run ────────────────────────────────────────────────────────


@router.post(
    "/policies/run",
    response_model=PolicyRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run memory lifecycle policies",
    description=(
        "Manually trigger a full PolicyEngine sweep for an agent: "
        "Promote → Demote (with LLM summarisation) → Evict.\n\n"
        "This endpoint is intended for admin tooling, background schedulers, "
        "or testing. Under normal operation policies run automatically after "
        "each /store call."
    ),
    tags=["Admin"],
)
async def run_policies(
    payload: PolicyRunRequest,
    request: Request,
    agent_id: str = Depends(get_agent_id),
) -> PolicyRunResponse:
    db: sqlite3.Connection = _get_db(request)
    chroma: chromadb.Client = _get_chroma(request)

    # Map API enum → core enum
    strategy_map = {
        EvictionStrategyEnum.LRU:  EvictionStrategy.LRU,
        EvictionStrategyEnum.FIFO: EvictionStrategy.FIFO,
        EvictionStrategyEnum.LFU:  EvictionStrategy.LFU,
    }

    engine = PolicyEngine(
        db=db,
        chroma=chroma,
        eviction_strategy=strategy_map[payload.strategy],
    )

    results = await engine.run_all(
        agent_id=payload.agent_id,
        namespace=payload.namespace.value,
    )

    def _to_summary(policy_key: str) -> PolicySummary:
        r = results[policy_key]
        compression = (
            round(100 * (1 - r.tokens_after / max(r.tokens_before, 1)), 1)
            if r.tokens_before > 0
            else 0.0
        )
        preview = (r.new_summary[:200] + "…") if r.new_summary else None
        return PolicySummary(
            evicted_count=len(r.evicted_ids),
            demoted_count=len(r.demoted_ids),
            promoted_count=len(r.promoted_ids),
            tokens_before=r.tokens_before,
            tokens_after=r.tokens_after,
            compression_pct=compression,
            new_summary_preview=preview,
        )

    return PolicyRunResponse(
        agent_id=payload.agent_id,
        promotion=_to_summary("promotion"),
        demotion=_to_summary("demotion"),
        eviction=_to_summary("eviction"),
    )


# ── GET /memory/{memory_id} ───────────────────────────────────────────────────


@router.get(
    "/memory/{memory_id}",
    response_model=MemoryItem,
    status_code=status.HTTP_200_OK,
    summary="Fetch memory by ID",
    description="Retrieve a single memory by its UUID, searching Tier 1 then Tier 2.",
    tags=["Memory"],
)
async def get_memory_by_id(
    memory_id: str,
    request: Request,
    agent_id: str = Depends(get_agent_id),
) -> MemoryItem:
    retriever: MemoryRetriever = _get_retriever(request)

    result = await retriever.get_by_id(memory_id=memory_id, agent_id=agent_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found or access denied.",
        )

    return MemoryItem(
        id=result.id,
        agent_id=result.agent_id,
        content=result.content,
        score=result.score,
        tier=MemoryTierEnum(result.tier.value),
        namespace=Namespace(result.namespace),
        access_count=result.access_count,
        created_at=result.created_at,
        last_accessed=result.last_accessed,
        metadata=result.metadata,
    )


# ── GET /health ───────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Liveness and readiness probe. Checks SQLite and ChromaDB connectivity.",
    tags=["Ops"],
)
async def health_check(request: Request) -> HealthResponse:
    """
    Performs lightweight connectivity probes:
      • Tier 1: executes ``SELECT 1`` on the SQLite connection.
      • Tier 2: calls ``collection.count()`` on the ChromaDB collection.
    """
    tier1_ok = False
    tier2_ok = False

    # Tier-1 probe
    try:
        db: sqlite3.Connection = _get_db(request)
        db.execute("SELECT 1")
        tier1_ok = True
    except Exception as exc:
        logger.error("Health check: Tier-1 (SQLite) probe failed: %s", exc)

    # Tier-2 probe
    try:
        chroma: chromadb.Client = _get_chroma(request)
        chroma.heartbeat()
        tier2_ok = True
    except Exception as exc:
        logger.error("Health check: Tier-2 (ChromaDB) probe failed: %s", exc)

    overall_status = "ok" if (tier1_ok and tier2_ok) else "degraded"
    return HealthResponse(
        status=overall_status,
        tier1_connected=tier1_ok,
        tier2_connected=tier2_ok,
    )
