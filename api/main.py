"""
api/main.py
===========
MemHub — Application factory.

Responsibilities
----------------
1. Initialize Tier 1 (SQLite) schema.
2. Initialize Tier 2 (ChromaDB) client and collection.
3. Create the FastAPI app and wire the router from api/routes.py.
4. Inject DB references into api/routes so routes never import DB globals
   at import time (keeps the module testable in isolation).

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
     read its OWN private memories OR explicitly shared workspace memories.
  3. POLICY-DRIVEN SUMMARIZATION — When working memory grows too large (>2,000 chars),
     MemHub automatically compresses it using an LLM (via Ollama).
  4. WORKSPACE-AWARE ROUTING — Memories are scoped to `workspace_id`, allowing multiple
     independent teams to share a MemHub instance without cross-contamination.

In short: LMCache saves GPU compute. MemHub saves agent cognition.
"""

import logging
import sqlite3

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI

import api.routes as routes_module
from api.routes import router

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memhub")

# ── Constants & Configuration ──────────────────────────────────────────────────
SQLITE_DB_PATH = "memhub_working_memory.db"
CHROMA_DB_PATH = "./chroma_db"

# Policy thresholds
WORKING_MEMORY_SUMMARY_THRESHOLD = 2_000   # chars: trigger summarization above this
LONG_TERM_PROMOTION_THRESHOLD    = 500     # chars: route to ChromaDB above this

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
      • agent_memory    : per-agent fast-access scratchpad + workspace assignment.
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


# ── Run DB init ────────────────────────────────────────────────────────────────
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


# ── Inject DB references into routes ──────────────────────────────────────────
# routes.py declares module-level variables that are set here so the route
# handlers can access shared DB state without a circular import.
routes_module._sqlite_db_path                   = SQLITE_DB_PATH
routes_module._long_term_collection             = long_term_collection
routes_module._working_memory_summary_threshold = WORKING_MEMORY_SUMMARY_THRESHOLD
routes_module._long_term_promotion_threshold    = LONG_TERM_PROMOTION_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="MemHub",
    version="2.0.0",
    description=(
        "Centralized memory service for LLM agents — manages Semantic State and "
        "Logical Permissions, not raw GPU KV-caches."
    ),
)

app.include_router(router)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
