"""
api/auth.py  — MemHub Namespace & ACL Enforcement
==================================================

MemHub uses a lightweight, header-based authentication model with two
access-control concepts:

  1. Agent Identity — every request must identify itself via the
     ``X-Agent-ID`` header. This is validated against a SQLite ACL table
     that maps agent IDs to their permitted namespaces.

  2. Namespace Enforcement — an agent may only *write* to its own
     ``private`` namespace or to a ``shared`` namespace that it has been
     explicitly granted access to. Read rules mirror write rules, with the
     addition that ``include_shared`` on a retrieve request is checked
     against the ACL table before the retrieval layer fans out.

ACL table schema (in memhub.db)
────────────────────────────────
  CREATE TABLE IF NOT EXISTS agent_acl (
      agent_id       TEXT NOT NULL,
      namespace      TEXT NOT NULL DEFAULT 'private',
      can_read       INTEGER NOT NULL DEFAULT 1,
      can_write      INTEGER NOT NULL DEFAULT 1,
      created_at     REAL NOT NULL,
      PRIMARY KEY (agent_id, namespace)
  );

For development / single-agent use, ACL checks can be disabled by
setting the environment variable MEMHUB_DISABLE_AUTH=1.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Set MEMHUB_DISABLE_AUTH=1 to bypass ACL checks (dev / testing only).
_AUTH_DISABLED: bool = os.getenv("MEMHUB_DISABLE_AUTH", "0").strip() == "1"

if _AUTH_DISABLED:
    logger.warning(
        "MEMHUB_DISABLE_AUTH is set — ACL enforcement is DISABLED. "
        "Do NOT use this in production."
    )

# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_db(request: Request) -> sqlite3.Connection:
    """Extract the shared SQLite connection from the FastAPI app state."""
    return request.app.state.db


def _agent_exists(db: sqlite3.Connection, agent_id: str) -> bool:
    """Return True if agent_id has at least one ACL entry."""
    row = db.execute(
        "SELECT 1 FROM agent_acl WHERE agent_id = ? LIMIT 1", (agent_id,)
    ).fetchone()
    return row is not None


def _has_namespace_permission(
    db: sqlite3.Connection,
    agent_id: str,
    namespace: str,
    require_write: bool = False,
) -> bool:
    """
    Check whether *agent_id* has read (and optionally write) access to
    *namespace* in the ACL table.

    Args:
        db:             Open SQLite connection.
        agent_id:       The requesting agent.
        namespace:      The namespace being accessed.
        require_write:  If True, both can_read AND can_write must be 1.

    Returns:
        True if permission is granted, False otherwise.
    """
    if _AUTH_DISABLED:
        return True

    query = (
        "SELECT can_read, can_write FROM agent_acl "
        "WHERE agent_id = ? AND namespace = ?"
    )
    row = db.execute(query, (agent_id, namespace)).fetchone()
    if row is None:
        return False

    can_read, can_write = row
    if require_write:
        return bool(can_read) and bool(can_write)
    return bool(can_read)


def ensure_agent_registered(
    db: sqlite3.Connection,
    agent_id: str,
    namespace: str = "private",
) -> None:
    """
    Auto-register an agent with a default private namespace entry if it
    does not already exist. This gives frictionless onboarding for new
    agents without requiring a separate registration step.

    In production, replace or augment this with a proper registration
    endpoint that validates agent credentials before inserting the ACL row.
    """
    existing = db.execute(
        "SELECT 1 FROM agent_acl WHERE agent_id = ? AND namespace = ?",
        (agent_id, namespace),
    ).fetchone()

    if existing is None:
        db.execute(
            """
            INSERT INTO agent_acl (agent_id, namespace, can_read, can_write, created_at)
            VALUES (?, ?, 1, 1, ?)
            """,
            (agent_id, namespace, time.time()),
        )
        db.commit()
        logger.info(
            "Auto-registered agent '%s' with namespace '%s'.", agent_id, namespace
        )


# ── FastAPI dependency functions ──────────────────────────────────────────────


async def get_agent_id(
    x_agent_id: Optional[str] = Header(
        default=None,
        alias="X-Agent-ID",
        description="Unique identifier of the calling agent. Required for all endpoints.",
    )
) -> str:
    """
    FastAPI dependency: extract and validate the X-Agent-ID header.

    Raises:
        HTTPException 401 if the header is missing.
        HTTPException 400 if the header value is blank.
    """
    if x_agent_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required header: X-Agent-ID",
        )
    stripped = x_agent_id.strip()
    if not stripped:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Agent-ID header must not be blank.",
        )
    return stripped


def require_write_access(namespace: str):
    """
    Parametrised FastAPI dependency factory.

    Returns a dependency function that verifies the requesting agent has
    write access to *namespace* via the ACL table.

    Usage in route:
        @router.post("/store")
        async def store(
            payload: StoreRequest,
            agent_id: str = Depends(get_agent_id),
            _: None = Depends(require_write_access("private")),
        ): ...
    """

    async def _check(
        request: Request,
        agent_id: str = Depends(get_agent_id),
    ) -> None:
        if _AUTH_DISABLED:
            return
        db: sqlite3.Connection = _get_db(request)

        # Auto-register unknown agents (dev-friendly)
        ensure_agent_registered(db, agent_id, namespace)

        if not _has_namespace_permission(db, agent_id, namespace, require_write=True):
            logger.warning(
                "Write access denied: agent='%s', namespace='%s'.",
                agent_id,
                namespace,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Agent '{agent_id}' does not have write access to "
                    f"namespace '{namespace}'."
                ),
            )

    return _check


def require_read_access(namespace: str):
    """
    Parametrised FastAPI dependency factory for read access.
    Mirrors :func:`require_write_access` but only checks ``can_read``.
    """

    async def _check(
        request: Request,
        agent_id: str = Depends(get_agent_id),
    ) -> None:
        if _AUTH_DISABLED:
            return
        db: sqlite3.Connection = _get_db(request)
        ensure_agent_registered(db, agent_id, namespace)

        if not _has_namespace_permission(db, agent_id, namespace, require_write=False):
            logger.warning(
                "Read access denied: agent='%s', namespace='%s'.",
                agent_id,
                namespace,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Agent '{agent_id}' does not have read access to "
                    f"namespace '{namespace}'."
                ),
            )

    return _check


async def require_shared_read(
    request: Request,
    agent_id: str = Depends(get_agent_id),
) -> None:
    """
    FastAPI dependency that checks whether the requesting agent may read
    from the ``shared`` namespace.

    Use this in retrieve routes when ``include_shared=True`` is set.
    """
    if _AUTH_DISABLED:
        return
    db: sqlite3.Connection = _get_db(request)
    if not _has_namespace_permission(db, agent_id, "shared", require_write=False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Agent '{agent_id}' does not have access to the 'shared' namespace. "
                "Contact your team admin to grant shared-memory access."
            ),
        )
