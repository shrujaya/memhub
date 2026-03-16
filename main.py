#!/usr/bin/env python3
"""
MemHub — Centralized Semantic Memory Service for LLM Agents
============================================================

WHY MEMHUB IS DIFFERENT FROM LMCache
--------------------------------------
LMCache (and similar GPU-level caching systems) operate at the **infrastructure**
layer: they cache raw KV (key-value) attention tensors on GPU/CPU/NVMe to reduce
TTFT (Time-To-First-Token) by avoiding redundant prefill passes. They have no
concept of *which agent* owns a piece of context, *why* something should be
remembered, or *who* should be allowed to read it.

MemHub operates at the **application semantics** layer:
  1. SEMANTIC STATE  — We store *meaning* (vector embeddings via sentence-transformers)
     so agents can retrieve conceptually relevant past context, not just exact-match
     token sequences. ChromaDB is the index for this semantic long-term memory.
  2. LOGICAL PERMISSIONS — We enforce fine-grained access control: an agent can only
     read its OWN private memories OR explicitly shared workspace memories. This is
     expressed as structured metadata filters on ChromaDB, not GPU-level opaque blobs.
  3. POLICY-DRIVEN SUMMARIZATION — When working memory grows too large (>2,000 chars),
     MemHub automatically compresses it using an LLM (via Ollama), preserving
     *semantic content* while reducing token footprint. LMCache has no awareness of
     content—it just caches and evicts raw tensors.
  4. WORKSPACE-AWARE ROUTING — Memories are scoped to `workspace_id`, allowing multiple
     independent teams (workspaces) to share a MemHub instance without cross-contamination.

In short: LMCache saves GPU compute. MemHub saves agent cognition.
"""

import json
import logging
import sqlite3
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import chromadb

# ── Sentence-Transformers for local, free embeddings ───────────────────────────
# NOTE: We deliberately avoid OpenAI-style API calls here. sentence-transformers
# runs entirely on-device, keeping MemHub self-contained and cost-free.
# This also means embeddings are deterministic across runs.
from chromadb.utils import embedding_functions

# ── Ollama for local LLM summarization ─────────────────────────────────────────
# ollama must be running locally (`ollama serve`). We prefer llama3 but fall back
# to mistral if llama3 is not pulled.
from core.summarization import summarize_content, OLLAMA_AVAILABLE

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memhub")

# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MemHub",
    version="2.0.0",
    description=(
        "Centralized memory service for LLM agents — manages Semantic State and "
        "Logical Permissions, not raw GPU KV-caches."
    ),
)

# ── Constants & Configuration ──────────────────────────────────────────────────
SQLITE_DB_PATH = "memhub_working_memory.db"
CHROMA_DB_PATH = "./chroma_db"

# Policy thresholds
WORKING_MEMORY_SUMMARY_THRESHOLD = 2_000   # chars: trigger summarization above this
LONG_TERM_PROMOTION_THRESHOLD    = 500     # chars: route to ChromaDB above this

# Preferred Ollama models, tried in order (Moved to core/summarization.py)

# ── Embedding Function (local, sentence-transformers) ─────────────────────────
# sentence-transformers is a free, local alternative to OpenAI Embeddings.
# The model is downloaded once and cached by the library.
_sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def init_sqlite() -> None:
    """
    Bootstrap the SQLite schema (Tier 1 — Working Memory & Permissions).

    Two tables:
      • agent_memory  : per-agent fast-access scratchpad + workspace assignment.
      • workspace_perms : maps (agent_id, workspace_id) to an access role.

    Design decision: keeping permissions in SQLite (not ChromaDB) means ACL
    checks are O(1) indexed lookups, not vector-scan operations.
    """
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Tier 1 Working Memory — per-agent scratchpad
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                agent_id              TEXT PRIMARY KEY,
                namespace             TEXT NOT NULL,
                workspace_id          TEXT NOT NULL DEFAULT 'default',
                working_memory_content TEXT,
                authorized_spaces     TEXT,          -- JSON list of workspace_ids
                last_updated          TIMESTAMP
            )
        """)

        # Workspace Permissions — which agents belong to which workspace
        # Role can be 'member' or 'admin'; reserved for future RBAC expansion.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_perms (
                agent_id     TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                role         TEXT NOT NULL DEFAULT 'member',
                joined_at    TIMESTAMP,
                PRIMARY KEY (agent_id, workspace_id)
            )
        """)

        conn.commit()
        logger.info("SQLite schema initialized at %s", SQLITE_DB_PATH)
    except sqlite3.Error as exc:
        logger.error("SQLite initialization failed: %s", exc)
        raise RuntimeError(f"Cannot initialize SQLite: {exc}") from exc
    finally:
        conn.close()


init_sqlite()

# ── ChromaDB (Tier 2 — Long-Term Semantic Memory) ─────────────────────────────
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    long_term_collection = chroma_client.get_or_create_collection(
        name="long_term_memory",
        embedding_function=_sentence_ef,  # local embeddings, no API key needed
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB initialized at %s", CHROMA_DB_PATH)
except Exception as exc:
    logger.error("ChromaDB initialization failed: %s", exc)
    raise RuntimeError(f"Cannot initialize ChromaDB: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# POLICY ENGINE — AUTOMATIC SUMMARIZATION (Moved to core/summarization.py)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class StoreRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    text_content: str = Field(..., description="The content to store")
    is_shared: bool = Field(
        False,
        description="If True, this memory is visible to all agents in the same workspace",
    )
    workspace_id: str = Field(
        "default",
        description="The workspace/team this memory belongs to. Controls shared visibility.",
    )


class RetrieveRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the querying agent")
    query: str = Field(..., description="Natural-language query for semantic retrieval")
    top_k: int = Field(5, ge=1, le=20, description="Number of long-term results to return")


class MemoryResponse(BaseModel):
    agent_id: str
    workspace_id: Optional[str]
    working_memory: Optional[str]
    long_term_memory: List[Dict[str, Any]]
    summarization_triggered: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — SQLite Agent Upsert
# ══════════════════════════════════════════════════════════════════════════════

def _upsert_agent(
    cursor: sqlite3.Cursor,
    agent_id: str,
    workspace_id: str,
) -> None:
    """
    Ensure the agent row exists in agent_memory and that a workspace_perms
    record is also created for the given workspace.

    Shared Workspace Support
    ─────────────────────────
    workspace_id is the team-level scope for shared memories. By recording it
    in both agent_memory (the agent's primary workspace) AND workspace_perms
    (the ACL join table), we support scenarios where one agent belongs to
    multiple workspaces over time without schema migrations.
    """
    cursor.execute(
        "SELECT agent_id, workspace_id FROM agent_memory WHERE agent_id = ?",
        (agent_id,),
    )
    row = cursor.fetchone()

    if not row:
        cursor.execute(
            """
            INSERT INTO agent_memory
                (agent_id, namespace, workspace_id, working_memory_content, authorized_spaces, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (agent_id, f"ns_{agent_id}", workspace_id, "", json.dumps([workspace_id]), datetime.now()),
        )
    else:
        # Update the authorised_spaces list to include any new workspace_id
        existing_ws = row[1]
        cursor.execute(
            "SELECT authorized_spaces FROM agent_memory WHERE agent_id = ?", (agent_id,)
        )
        spaces_str = cursor.fetchone()[0] or "[]"
        spaces: List[str] = json.loads(spaces_str)
        if workspace_id not in spaces:
            spaces.append(workspace_id)
            cursor.execute(
                "UPDATE agent_memory SET authorized_spaces = ?, last_updated = ? WHERE agent_id = ?",
                (json.dumps(spaces), datetime.now(), agent_id),
            )

    # Maintain workspace_perms join table
    cursor.execute(
        """
        INSERT OR IGNORE INTO workspace_perms (agent_id, workspace_id, role, joined_at)
        VALUES (?, ?, 'member', ?)
        """,
        (agent_id, workspace_id, datetime.now()),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/store", response_model=Dict[str, str], summary="Store a memory fragment")
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
        conn   = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        conn.execute("PRAGMA journal_mode=WAL")   # safer concurrent writes

        _upsert_agent(cursor, agent_id, workspace_id)

        if len(content) <= LONG_TERM_PROMOTION_THRESHOLD and not is_shared:
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
            if len(updated) > WORKING_MEMORY_SUMMARY_THRESHOLD:
                updated = summarize_content(updated)
                summarization_triggered = True
                logger.info(
                    "Working memory for agent '%s' was summarized (threshold=%d chars)",
                    agent_id,
                    WORKING_MEMORY_SUMMARY_THRESHOLD,
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

            long_term_collection.add(
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

    result = {
        "status":                  "success",
        "stored_in":               destination,
        "agent_id":                agent_id,
        "workspace_id":            workspace_id,
        "summarization_triggered": str(summarization_triggered),
    }
    return result


@app.post("/retrieve", response_model=MemoryResponse, summary="Retrieve memories for an agent")
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
        conn   = sqlite3.connect(SQLITE_DB_PATH)
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
        collection_count = long_term_collection.count()
        effective_top_k  = min(top_k, max(1, collection_count))

        if collection_count > 0:
            results = long_term_collection.query(
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


@app.get("/health", summary="Health check")
async def health():
    """Returns service status and the counts of records in both tiers."""
    sqlite_count  = 0
    chroma_count  = 0
    sqlite_status = "ok"
    chroma_status = "ok"

    try:
        conn   = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_memory")
        sqlite_count = cursor.fetchone()[0]
        conn.close()
    except sqlite3.Error as exc:
        sqlite_status = f"error: {exc}"

    try:
        chroma_count = long_term_collection.count()
    except Exception as exc:
        chroma_status = f"error: {exc}"

    return {
        "status":             "healthy",
        "tier1_sqlite":       {"status": sqlite_status, "agent_count": sqlite_count},
        "tier2_chromadb":     {"status": chroma_status, "document_count": chroma_count},
        "ollama_available":   OLLAMA_AVAILABLE,
        "summarization_threshold": WORKING_MEMORY_SUMMARY_THRESHOLD,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
