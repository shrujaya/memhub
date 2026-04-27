"""
agents/tools.py  — MemHub Callable Tool Functions for AutoGen Agents
=====================================================================

This module defines the tool-callable functions that AutoGen agents use
to interact with the MemHub API.  Each function maps directly to a
MemHub REST endpoint and is designed to be registered as a tool in an
AutoGen ``ConversableAgent`` or ``AssistantAgent``.

Tool registration pattern
─────────────────────────
  from agents.tools import store_memory, query_team_memory

  assistant = AssistantAgent(
      name="researcher",
      llm_config={
          "tools": [store_memory.schema, query_team_memory.schema],
      },
  )
  register_function(store_memory, caller=assistant, executor=user_proxy, ...)

All tools are synchronous wrappers around the async MemHub REST API,
using ``requests`` so they can be called from the AutoGen tool-execution
thread without event-loop conflicts.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MEMHUB_BASE_URL: str = os.getenv("MEMHUB_BASE_URL", "http://localhost:8000/v1")
MEMHUB_TIMEOUT: int = int(os.getenv("MEMHUB_TIMEOUT", "15"))


def _headers(agent_id: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Agent-ID": agent_id,
    }


def _raise_for_status(response: requests.Response, context: str) -> None:
    """Raise a descriptive RuntimeError on non-2xx responses."""
    if not response.ok:
        raise RuntimeError(
            f"MemHub {context} failed [{response.status_code}]: {response.text}"
        )


# ── Core tool functions ───────────────────────────────────────────────────────


def store_memory(
    agent_id: str,
    content: str,
    namespace: str = "private",
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
    run_policies: bool = True,
) -> Dict[str, Any]:
    """
    Store a piece of information in the agent's working memory (Tier 1).

    This should be called by an agent whenever it produces an observation,
    decision, or fact it wants to retain across future turns or share
    with teammates.

    Args:
        agent_id:     The calling agent's unique ID.
        content:      The raw text to remember.
        namespace:    'private' (default) or 'shared' for team-visible memories.
        tags:         Optional list of string tags for categorisation.
        source:       Origin label (e.g. 'tool_output', 'user_message').
        run_policies: Trigger automatic eviction/demotion after write.

    Returns:
        dict with keys: memory_id, token_count, policies_triggered, message.

    Example (LLM-callable docstring used by AutoGen tool schema):
        Store the following finding so the team can use it later:
        store_memory(agent_id="agent-01", content="ACME Q2 revenue = $4.2B",
                     namespace="shared", tags=["finance"])
    """
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "content": content,
        "namespace": namespace,
        "run_policies": run_policies,
        "metadata": {
            "tags": tags or [],
            "source": source or "agent_tool",
        },
    }

    try:
        resp = requests.post(
            f"{MEMHUB_BASE_URL}/store",
            json=payload,
            headers=_headers(agent_id),
            timeout=MEMHUB_TIMEOUT,
        )
        _raise_for_status(resp, "store")
        result = resp.json()
        logger.info(
            "store_memory: agent='%s' id=%s tokens=%d",
            agent_id,
            result.get("memory_id"),
            result.get("token_count", 0),
        )
        return result
    except requests.RequestException as exc:
        logger.error("store_memory network error: %s", exc)
        raise RuntimeError(f"MemHub unreachable: {exc}") from exc


def query_team_memory(
    agent_id: str,
    query: str,
    top_k: int = 5,
    include_shared: bool = True,
    namespace: str = "private",
) -> List[Dict[str, Any]]:
    """
    Search the agent's working memory and the shared team memory for
    information relevant to *query*.

    This is the primary retrieval tool.  It performs a hybrid Tier-1
    keyword + Tier-2 semantic search and returns the top-k results, ranked
    by relevance score.

    Args:
        agent_id:       The calling agent's unique ID.
        query:          Natural-language question or keyword string.
        top_k:          Maximum number of results to return (1-50).
        include_shared: Also search the shared team namespace.
        namespace:      The agent's own namespace (default 'private').

    Returns:
        List of dicts, each with keys: id, content, score, tier,
        namespace, access_count, created_at.

    Example:
        results = query_team_memory(
            agent_id="agent-02",
            query="What did agent-01 find about ACME revenue?",
            include_shared=True,
        )
    """
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "query": query,
        "top_k": top_k,
        "namespace": namespace,
        "include_shared": include_shared,
    }

    try:
        resp = requests.post(
            f"{MEMHUB_BASE_URL}/retrieve",
            json=payload,
            headers=_headers(agent_id),
            timeout=MEMHUB_TIMEOUT,
        )
        _raise_for_status(resp, "retrieve")
        data = resp.json()
        results = data.get("results", [])
        logger.info(
            "query_team_memory: agent='%s' query=%r → %d result(s) "
            "(tier1=%d, tier2=%d, latency=%.1fms)",
            agent_id,
            query[:60],
            len(results),
            data.get("tier1_hits", 0),
            data.get("tier2_hits", 0),
            data.get("latency_ms", 0.0),
        )
        return results
    except requests.RequestException as exc:
        logger.error("query_team_memory network error: %s", exc)
        raise RuntimeError(f"MemHub unreachable: {exc}") from exc


def get_memory_by_id(agent_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a specific memory by its exact UUID.

    Useful when an agent stores a reference to a memory ID and wants to
    retrieve it directly later without a semantic query.

    Args:
        agent_id:   The calling agent's unique ID.
        memory_id:  UUID of the target memory.

    Returns:
        Memory dict, or None if not found / access denied.
    """
    try:
        resp = requests.get(
            f"{MEMHUB_BASE_URL}/memory/{memory_id}",
            headers=_headers(agent_id),
            timeout=MEMHUB_TIMEOUT,
        )
        if resp.status_code == 404:
            logger.warning("get_memory_by_id: '%s' not found.", memory_id)
            return None
        _raise_for_status(resp, "get_by_id")
        return resp.json()
    except requests.RequestException as exc:
        logger.error("get_memory_by_id network error: %s", exc)
        raise RuntimeError(f"MemHub unreachable: {exc}") from exc


def store_shared_finding(
    agent_id: str,
    content: str,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for ``store_memory`` that always writes to the
    ``shared`` namespace so all team agents can immediately read it.

    Intended for agents to broadcast key findings, decisions, or facts
    that other agents need to coordinate around.

    Args:
        agent_id: The broadcasting agent.
        content:  The fact / decision / finding to share.
        tags:     Optional categorisation tags.
        source:   Origin label.

    Returns:
        StoreResponse dict.
    """
    return store_memory(
        agent_id=agent_id,
        content=content,
        namespace="shared",
        tags=tags,
        source=source or "agent_finding",
    )


def trigger_policy_sweep(
    agent_id: str,
    namespace: str = "private",
    strategy: str = "lru",
) -> Dict[str, Any]:
    """
    Manually trigger a MemHub PolicyEngine sweep for this agent.

    Normally policies run automatically after /store calls, but an agent
    can invoke this to compact its memory mid-task (e.g., before a long
    reasoning chain that will generate many new memories).

    Args:
        agent_id:  The agent whose memory to sweep.
        namespace: Namespace for the demotion summary insertion.
        strategy:  Eviction strategy: 'lru' | 'fifo' | 'lfu'.

    Returns:
        PolicyRunResponse dict with per-policy statistics.
    """
    payload = {
        "agent_id": agent_id,
        "namespace": namespace,
        "strategy": strategy,
    }
    try:
        resp = requests.post(
            f"{MEMHUB_BASE_URL}/policies/run",
            json=payload,
            headers=_headers(agent_id),
            timeout=MEMHUB_TIMEOUT * 3,  # Policy sweeps may take longer
        )
        _raise_for_status(resp, "policies/run")
        return resp.json()
    except requests.RequestException as exc:
        logger.error("trigger_policy_sweep network error: %s", exc)
        raise RuntimeError(f"MemHub unreachable: {exc}") from exc


def check_memhub_health() -> Dict[str, Any]:
    """
    Check whether the MemHub service is healthy and both storage tiers
    are reachable.  Agents can call this at startup to fail fast if
    the memory service is unavailable.

    Returns:
        HealthResponse dict with 'status', 'tier1_connected', 'tier2_connected'.
    """
    try:
        resp = requests.get(
            f"{MEMHUB_BASE_URL}/health",
            timeout=5,
        )
        _raise_for_status(resp, "health")
        return resp.json()
    except requests.RequestException as exc:
        logger.error("MemHub health check failed: %s", exc)
        return {"status": "unreachable", "tier1_connected": False, "tier2_connected": False}
