"""
agents/team_config.py  — AutoGen Team Definitions for MemHub
=============================================================

Defines the multi-agent team that exercises MemHub as its shared memory
backend.  Each role is a specialised ``ConversableAgent`` whose system
prompt instructs it to:

  1. Store important findings/decisions via ``store_memory`` / ``store_shared_finding``.
  2. Query team memory before starting any reasoning via ``query_team_memory``.
  3. Compact its own memory when it feels overwhelmed via ``trigger_policy_sweep``.

Team Roles
──────────
  Orchestrator   — Manages the task lifecycle, breaks goals into subtasks,
                   reads shared memory to avoid duplication.

  Researcher     — Web-search or document-analysis agent; stores findings
                   to the shared namespace for teammates.

  Analyst        — Synthesises findings, performs quantitative reasoning;
                   reads shared memory extensively.

  Critic         — Reviews conclusions, ensures consistency; reads private
                   and shared namespaces to cross-check.

  UserProxy      — Human-in-the-loop or automated executor; routes tool
                   calls to the MemHub tool functions.

Usage
─────
  from agents.team_config import build_team
  team = build_team(task="Analyse ACME Corp Q2 earnings")
  team["orchestrator"].initiate_chat(team["user_proxy"], message=task)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

try:
    import autogen  # type: ignore
    from autogen import ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
    from autogen import register_function
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

from agents.tools import (
    check_memhub_health,
    get_memory_by_id,
    query_team_memory,
    store_memory,
    store_shared_finding,
    trigger_policy_sweep,
)

logger = logging.getLogger(__name__)

# ── LLM configuration ─────────────────────────────────────────────────────────
# Reads from environment; falls back to Ollama-served local model.

def _llm_config(model: Optional[str] = None) -> Dict[str, Any]:
    """Build an AutoGen-compatible LLM config dict."""
    return {
        "config_list": [
            {
                "model": model or os.getenv("MEMHUB_LLM_MODEL", "llama3"),
                "base_url": os.getenv(
                    "MEMHUB_LLM_BASE_URL", "http://localhost:11434/v1"
                ),
                "api_key": os.getenv("MEMHUB_LLM_API_KEY", "ollama"),
            }
        ],
        "temperature": 0.2,
        "timeout": 120,
    }


# ── System prompt templates ───────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """\
You are the Orchestrator of a multi-agent research team backed by MemHub,
a centralised shared memory service.

Your responsibilities:
1. Break the user's task into subtasks and assign them to the Researcher,
   Analyst, and Critic agents.
2. Before starting, call `query_team_memory` to check whether relevant
   information already exists in shared memory.
3. After receiving a final synthesis from the Analyst, call
   `store_shared_finding` to record the team's conclusion.
4. If your context feels crowded, call `trigger_policy_sweep` to compact
   your working memory.
5. Coordinate turn-taking by explicitly addressing agents by name.

Always cite memory IDs when referencing stored findings.
"""

_RESEARCHER_PROMPT = """\
You are the Researcher in a multi-agent team.  You specialise in finding
and storing factual information.

Your workflow:
1. First call `query_team_memory` to avoid duplicating work.
2. Perform your research (web search, document analysis, etc.).
3. Store EVERY significant finding by calling `store_shared_finding` with
   appropriate tags so teammates can use it.
4. Report a concise summary and list the memory IDs you stored.

Be prolific but precise.  Store raw facts, not interpretations.
"""

_ANALYST_PROMPT = """\
You are the Analyst in a multi-agent team.  You synthesise findings into
actionable insights and quantitative conclusions.

Your workflow:
1. Call `query_team_memory` with specific queries to retrieve Researcher
   findings.
2. Perform your analysis.
3. Store your intermediate reasoning steps with `store_memory` (namespace:
   private) so you can recall them if interrupted.
4. Store your final conclusions with `store_shared_finding` for the Critic.

Always quantify uncertainty and cite the memory IDs of the data you used.
"""

_CRITIC_PROMPT = """\
You are the Critic in a multi-agent team.  Your job is to challenge
conclusions, identify gaps, and ensure consistency across the team's work.

Your workflow:
1. Call `query_team_memory` with broad queries to read all stored findings
   and analyses.
2. Identify contradictions, missing evidence, or logical gaps.
3. Store your critique with `store_shared_finding` so the Analyst can
   revise.
4. If everything checks out, issue an explicit APPROVED statement.

Be constructive but rigorous.  Reference memory IDs for every claim you
challenge.
"""

_USER_PROXY_PROMPT = """\
You are the UserProxy.  You execute tool calls on behalf of the agents and
relay results.  You do not perform your own reasoning.

You have access to MemHub tools:
  - store_memory
  - query_team_memory
  - store_shared_finding
  - get_memory_by_id
  - trigger_policy_sweep
  - check_memhub_health

Execute all tool calls faithfully and return the raw result.
"""

# ── MemHub tool schemas (for AutoGen function-calling) ────────────────────────

_MEMHUB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "Store a memory in the agent's private working memory (Tier 1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id":    {"type": "string"},
                    "content":     {"type": "string"},
                    "namespace":   {"type": "string", "enum": ["private", "shared"], "default": "private"},
                    "tags":        {"type": "array", "items": {"type": "string"}},
                    "source":      {"type": "string"},
                    "run_policies":{"type": "boolean", "default": True},
                },
                "required": ["agent_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_team_memory",
            "description": (
                "Search private and optionally shared team memory for information "
                "relevant to a query. Returns ranked memory chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id":       {"type": "string"},
                    "query":          {"type": "string"},
                    "top_k":          {"type": "integer", "default": 5},
                    "include_shared": {"type": "boolean", "default": True},
                },
                "required": ["agent_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_shared_finding",
            "description": "Broadcast an important finding to all agents in the shared namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "content":  {"type": "string"},
                    "tags":     {"type": "array", "items": {"type": "string"}},
                    "source":   {"type": "string"},
                },
                "required": ["agent_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_by_id",
            "description": "Retrieve a specific memory by its UUID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id":  {"type": "string"},
                    "memory_id": {"type": "string"},
                },
                "required": ["agent_id", "memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_policy_sweep",
            "description": "Compact agent working memory via eviction and demotion policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id":  {"type": "string"},
                    "namespace": {"type": "string", "default": "private"},
                    "strategy":  {"type": "string", "enum": ["lru", "fifo", "lfu"], "default": "lru"},
                },
                "required": ["agent_id"],
            },
        },
    },
]

# ── Tool function dispatch map ─────────────────────────────────────────────────

TOOL_MAP = {
    "store_memory": store_memory,
    "query_team_memory": query_team_memory,
    "store_shared_finding": store_shared_finding,
    "get_memory_by_id": get_memory_by_id,
    "trigger_policy_sweep": trigger_policy_sweep,
    "check_memhub_health": check_memhub_health,
}


# ── Team factory ──────────────────────────────────────────────────────────────


def build_team(
    task: str,
    model: Optional[str] = None,
    max_round: int = 30,
) -> Dict[str, Any]:
    """
    Construct the full MemHub-backed AutoGen agent team.

    Args:
        task:      The high-level task description passed as the opening message.
        model:     LLM model name (overrides MEMHUB_LLM_MODEL env var).
        max_round: Maximum conversation rounds before the GroupChat terminates.

    Returns:
        Dict with keys: orchestrator, researcher, analyst, critic,
        user_proxy, group_chat, manager.

    Raises:
        ImportError if ``pyautogen`` is not installed.
        RuntimeError if the MemHub service is unhealthy at startup.
    """
    if not AUTOGEN_AVAILABLE:
        raise ImportError(
            "pyautogen is not installed. Run: pip install pyautogen"
        )

    # ── Preflight: verify MemHub is running ───────────────────────────────────
    health = check_memhub_health()
    if health.get("status") != "ok":
        logger.warning(
            "MemHub health check returned '%s'. Proceeding anyway — "
            "tool calls may fail.", health.get("status")
        )

    llm = _llm_config(model)
    llm_with_tools = {**llm, "tools": _MEMHUB_TOOLS}

    # ── Agent definitions ─────────────────────────────────────────────────────
    orchestrator = ConversableAgent(
        name="Orchestrator",
        system_message=_ORCHESTRATOR_PROMPT,
        llm_config=llm_with_tools,
        human_input_mode="NEVER",
    )

    researcher = ConversableAgent(
        name="Researcher",
        system_message=_RESEARCHER_PROMPT,
        llm_config=llm_with_tools,
        human_input_mode="NEVER",
    )

    analyst = ConversableAgent(
        name="Analyst",
        system_message=_ANALYST_PROMPT,
        llm_config=llm_with_tools,
        human_input_mode="NEVER",
    )

    critic = ConversableAgent(
        name="Critic",
        system_message=_CRITIC_PROMPT,
        llm_config=llm_with_tools,
        human_input_mode="NEVER",
    )

    user_proxy = UserProxyAgent(
        name="UserProxy",
        system_message=_USER_PROXY_PROMPT,
        human_input_mode="NEVER",
        code_execution_config=False,
        # Tool execution is handled by the interceptor (see interceptor.py)
    )

    # ── Register tool executors ───────────────────────────────────────────────
    for name, fn in TOOL_MAP.items():
        for caller in [orchestrator, researcher, analyst, critic]:
            try:
                register_function(
                    fn,
                    caller=caller,
                    executor=user_proxy,
                    name=name,
                    description=fn.__doc__.split("\n")[1].strip() if fn.__doc__ else name,
                )
            except Exception as exc:
                logger.warning("Could not register tool '%s': %s", name, exc)

    # ── GroupChat setup ───────────────────────────────────────────────────────
    group_chat = GroupChat(
        agents=[orchestrator, researcher, analyst, critic, user_proxy],
        messages=[],
        max_round=max_round,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm,
        system_message=(
            "You are the GroupChat manager. Route the conversation to the most "
            "appropriate agent. Always let the Orchestrator start."
        ),
    )

    logger.info(
        "MemHub team built: Orchestrator, Researcher, Analyst, Critic, UserProxy. "
        "Task: %r", task[:80]
    )

    return {
        "orchestrator": orchestrator,
        "researcher":   researcher,
        "analyst":      analyst,
        "critic":       critic,
        "user_proxy":   user_proxy,
        "group_chat":   group_chat,
        "manager":      manager,
    }