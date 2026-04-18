"""
api/main.py  — MemHub FastAPI Application Entry Point
======================================================

Creates, configures, and exports the FastAPI ``app`` instance.

Startup lifecycle
─────────────────
  1. Initialise SQLite (create tables if needed).
  2. Initialise ChromaDB (persistent local client).
  3. Instantiate MemoryRetriever and bind it to app.state.
  4. Mount all routes from api/routes.py under a versioned prefix.

Shutdown lifecycle
──────────────────
  1. Close the SQLite connection gracefully.

Running
───────
  From the repo root:
      uvicorn api.main:app --reload --port 8000

  Or via the helper script:
      bash scripts/run_server.sh
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import chromadb  # type: ignore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.retrieval import MemoryRetriever

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH: Path = Path(os.getenv("MEMHUB_DB_PATH", str(_REPO_ROOT / "db" / "memhub.db")))
CHROMA_PATH: Path = Path(
    os.getenv("MEMHUB_CHROMA_PATH", str(_REPO_ROOT / "db" / "chroma_db"))
)

# ── SQLite schema bootstrap ───────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS working_memory (
    id            TEXT    PRIMARY KEY,
    agent_id      TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    created_at    REAL    NOT NULL,
    last_accessed REAL    NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    namespace     TEXT    NOT NULL DEFAULT 'private'
);

CREATE INDEX IF NOT EXISTS idx_wm_agent_ns
    ON working_memory (agent_id, namespace);

CREATE INDEX IF NOT EXISTS idx_wm_last_accessed
    ON working_memory (last_accessed);

CREATE TABLE IF NOT EXISTS agent_acl (
    agent_id   TEXT    NOT NULL,
    namespace  TEXT    NOT NULL DEFAULT 'private',
    can_read   INTEGER NOT NULL DEFAULT 1,
    can_write  INTEGER NOT NULL DEFAULT 1,
    created_at REAL    NOT NULL,
    PRIMARY KEY (agent_id, namespace)
);
"""


def _init_sqlite(path: Path) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at *path* and apply the schema.

    WAL mode is enabled for better concurrent read performance when
    multiple agents query the /retrieve endpoint simultaneously.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    logger.info("SQLite (Tier 1) initialised at: %s", path)
    return conn


def _init_chroma(path: Path) -> chromadb.Client:
    """
    Create or connect to a persistent ChromaDB instance stored at *path*.
    """
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    logger.info("ChromaDB (Tier 2) initialised at: %s", path)
    return client


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Everything before ``yield`` runs at startup; everything after runs
    at shutdown.
    """
    logger.info("MemHub starting up…")

    db = _init_sqlite(DB_PATH)
    chroma = _init_chroma(CHROMA_PATH)
    retriever = MemoryRetriever(db=db, chroma=chroma)

    # Bind to app.state so routes can access them via request.app.state
    app.state.db = db
    app.state.chroma = chroma
    app.state.retriever = retriever

    logger.info("MemHub ready. Listening for requests.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("MemHub shutting down — closing SQLite connection.")
    db.close()


# ── Application factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Separated into a factory function so that tests can call
    ``create_app()`` directly without triggering the lifespan.
    """
    app = FastAPI(
        title="MemHub",
        summary="Centralized Memory-as-a-Service for Multi-Agent Systems",
        description=(
            "MemHub provides a two-tier (SQLite + ChromaDB) shared memory store "
            "for AutoGen / LangGraph agent teams, with automatic eviction, "
            "demotion, and LLM-based summarisation policies.\n\n"
            "## Authentication\n"
            "All endpoints require the ``X-Agent-ID`` header. "
            "Agents are auto-registered on first use.\n\n"
            "## Tiers\n"
            "| Tier | Store | Role |\n"
            "|------|-------|------|\n"
            "| 1 | SQLite | Working memory (fast, keyword search) |\n"
            "| 2 | ChromaDB | Long-term memory (semantic vector search) |"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Restrict in production to the specific agent orchestration hosts.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("MEMHUB_CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/v1")

    return app


# ── Module-level app instance (used by uvicorn) ───────────────────────────────

app: FastAPI = create_app()
