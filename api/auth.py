"""
api/auth.py
===========
Access-control helpers for MemHub.

_upsert_agent
─────────────
Ensures the agent row exists in SQLite and that the agent is granted membership
in the given workspace. This is intentionally kept separate from the route
handlers so it can be unit-tested in isolation and reused by future endpoints
without importing the full route module.

Design note: ACL data lives in SQLite (not ChromaDB) so that all permission
checks are O(1) indexed lookups rather than vector-scan operations.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import List

logger = logging.getLogger("memhub.auth")


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
