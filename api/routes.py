"""
api/routes.py
=============
FastAPI route handlers for MemHub.

All three public endpoints are defined here and registered on an APIRouter so
that api/main.py can include them onto the root FastAPI app. This keeps the
app-factory logic separate from the business logic.

Endpoints
---------
POST /store      – Store a memory fragment (Tier 1 or Tier 2)
POST /retrieve   – Semantic retrieval with ACL filtering
GET  /health     – Liveness / readiness probe
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException

from api.models import MemoryResponse, RetrieveRequest, StoreRequest
from api.auth import _upsert_agent
from core.summarization import summarize_content, OLLAMA_AVAILABLE

logger = logging.getLogger("memhub.routes")

# These module-level references are injected by api/main.py after DB init so
# that routes never import DB globals at import time (avoids circular deps and
# makes unit-testing with mock collections straightforward).
_sqlite_db_path: str = ""
_long_term_collection = None  # chromadb.Collection
_working_memory_summary_threshold: int = 2_000
_long_term_promotion_threshold: int = 500

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: POST /store
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/store", response_model=Dict[str, str], summary="Store a memory fragment")
async def store_memory(request: StoreRequest = Body(...)):
    """
    Store a memory fragment for an agent.

    Routing heuristic:
      • Content ≤ 500 chars AND not shared  → Tier 1 (SQLite working memory)
      • Content > 500 chars OR is_shared     → Tier 2 (ChromaDB long-term memory)

    Summarization policy:
      After appending to working memory, if the cumulative scratchpad exceeds
      WORKING_MEMORY_SUMMARY_THRESHOLD (2,000 chars), the entire scratchpad is
      compressed by summarize_content() before being saved back. This keeps the
      agent's hot context small and injection-ready.
    """
    agent_id     = request.agent_id
    content      = request.text_content
    is_shared    = request.is_shared
    workspace_id = request.workspace_id

    summarization_triggered = False

    # ── Tier 1: SQLite ─────────────────────────────────────────────────────────
    try:
        conn   = sqlite3.connect(_sqlite_db_path)
        cursor = conn.cursor()
        conn.execute("PRAGMA journal_mode=WAL")   # safer concurrent writes

        _upsert_agent(cursor, agent_id, workspace_id)

        if len(content) <= _long_term_promotion_threshold and not is_shared:
            # Stay in working memory (Tier 1)
            cursor.execute(
                "SELECT working_memory_content FROM agent_memory WHERE agent_id = ?",
                (agent_id,),
            )
            current = cursor.fetchone()[0] or ""
            updated = (current + "\n" + content).strip()

            # ── Policy Engine: Auto-Summarize if scratchpad is too large ──────
            # WHY: A bloated working-memory string passed verbatim into an LLM
            # prompt wastes tokens. MemHub proactively summarizes so every
            # downstream call gets a clean, compact context window — something
            # LMCache cannot do because it doesn't understand content.
            if len(updated) > _working_memory_summary_threshold:
                updated = summarize_content(updated)
                summarization_triggered = True
                logger.info(
                    "Working memory for agent '%s' was summarized (threshold=%d chars)",
                    agent_id,
                    _working_memory_summary_threshold,
                )

            cursor.execute(
                "UPDATE agent_memory SET working_memory_content = ?, last_updated = ? WHERE agent_id = ?",
                (updated, datetime.now(), agent_id),
            )
            conn.commit()
            destination = "working_memory"

        else:
            # ── Tier 2: ChromaDB long-term memory ─────────────────────────────
            # Commit the SQLite agent record first, then write to ChromaDB.
            conn.commit()

            _long_term_collection.add(
                documents=[content],
                metadatas=[{
                    "agent_id":    agent_id,
                    "workspace_id": workspace_id,
                    "is_shared":   is_shared,   # stored as bool in metadata
                    "timestamp":   str(datetime.now()),
                }],
                ids=[str(uuid.uuid4())],
            )
            destination = "long_term_memory"

    except sqlite3.Error as exc:
        logger.error("SQLite error in /store: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    except Exception as exc:
        logger.error("Unexpected error in /store: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
    finally:
        conn.close()

    return {
        "status":                  "success",
        "stored_in":               destination,
        "agent_id":                agent_id,
        "workspace_id":            workspace_id,
        "summarization_triggered": str(summarization_triggered),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: POST /retrieve
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/retrieve", response_model=MemoryResponse, summary="Retrieve memories for an agent")
async def retrieve_memory(request: RetrieveRequest = Body(...)):
    """
    Retrieve memories for an agent — Tier 1 (SQLite) + Tier 2 (ChromaDB).

    Access Control — The Bouncer
    ─────────────────────────────
    The ChromaDB query uses a compound $or / $and filter so that an agent can
    see ONLY:
      (a) Memories it stored itself (agent_id == requestor), OR
      (b) Memories explicitly shared (is_shared == True) AND belonging to the
          same workspace as the agent (workspace_id matches).

    This is the core difference from a naive 'filter by agent_id' query:
    shared team memory is made accessible without leaking private notes from
    other agents or other workspaces.

    WHY NOT RELY ON APP-LAYER FILTERING?
    ChromaDB metadata filtering happens inside the HNSW index, before results
    are returned. This is more efficient than retrieving all candidates and
    filtering in Python, and more correct (no risk of accidentally exposing
    private results due to a Python bug).
    """
    agent_id = request.agent_id
    query    = request.query
    top_k    = request.top_k

    # ── 1. SQLite: Fetch working memory & workspace info ───────────────────────
    try:
        conn   = sqlite3.connect(_sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT working_memory_content, workspace_id, authorized_spaces FROM agent_memory WHERE agent_id = ?",
            (agent_id,),
        )
        row = cursor.fetchone()
    except sqlite3.Error as exc:
        logger.error("SQLite error in /retrieve: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Call /store first to register.",
        )

    working_content, workspace_id, authorized_spaces_json = row
    try:
        authorized_spaces: List[str] = json.loads(authorized_spaces_json or "[]")
    except json.JSONDecodeError:
        authorized_spaces = []

    # ── 2. ChromaDB: Semantic retrieval with compound ACL filter ──────────────
    #
    #  The "Bouncer" filter expressed in ChromaDB's $or / $and DSL:
    #
    #  WHERE
    #    (agent_id == <me>)                            -- my own memories
    #    OR
    #    (is_shared == True AND workspace_id == <ws>)  -- shared team memories
    #
    #  IMPORTANT: ChromaDB stores booleans as-is but requires the value to be
    #  a bool (not the string "True"). We set is_shared as bool in /store.
    #
    #  COMPARISON TO LMCache: LMCache has no concept of agent identity or
    #  permissions. Every caller sees the whole cache. MemHub's access control
    #  is enforced at query time, inside the vector index, making it safe for
    #  multi-tenant deployments.

    acl_filter: Dict[str, Any] = {
        "$or": [
            # Branch A — agent's own memories (private + shared ones they wrote)
            {"agent_id": {"$eq": agent_id}},
            # Branch B — shared memories from the same workspace
            {
                "$and": [
                    {"is_shared":    {"$eq": True}},
                    {"workspace_id": {"$eq": workspace_id}},
                ]
            },
        ]
    }

    long_term_results: List[Dict[str, Any]] = []
    try:
        # Count documents first to avoid ChromaDB "n_results > collection size"
        collection_count = _long_term_collection.count()
        effective_top_k  = min(top_k, max(1, collection_count))

        if collection_count > 0:
            results = _long_term_collection.query(
                query_texts=[query],
                n_results=effective_top_k,
                where=acl_filter,
                include=["documents", "metadatas", "distances"],
            )

            docs      = results.get("documents",  [[]])[0]
            metas     = results.get("metadatas",  [[]])[0]
            distances = results.get("distances",  [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                long_term_results.append({
                    "content":  doc,
                    "metadata": meta,
                    "distance": round(dist, 6),
                })

    except Exception as exc:
        # Non-fatal: log and return empty long-term results rather than 500-ing
        logger.error("ChromaDB query failed: %s", exc)

    return MemoryResponse(
        agent_id=agent_id,
        workspace_id=workspace_id,
        working_memory=working_content or None,
        long_term_memory=long_term_results,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: GET /health
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health", summary="Health check")
async def health():
    """Returns service status and the counts of records in both tiers."""
    sqlite_count  = 0
    chroma_count  = 0
    sqlite_status = "ok"
    chroma_status = "ok"

    try:
        conn   = sqlite3.connect(_sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_memory")
        sqlite_count = cursor.fetchone()[0]
        conn.close()
    except sqlite3.Error as exc:
        sqlite_status = f"error: {exc}"

    try:
        chroma_count = _long_term_collection.count()
    except Exception as exc:
        chroma_status = f"error: {exc}"

    return {
        "status":             "healthy",
        "tier1_sqlite":       {"status": sqlite_status, "agent_count": sqlite_count},
        "tier2_chromadb":     {"status": chroma_status, "document_count": chroma_count},
        "ollama_available":   OLLAMA_AVAILABLE,
        "summarization_threshold": _working_memory_summary_threshold,
    }
