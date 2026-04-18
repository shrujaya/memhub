"""
agents/interceptor.py  — MemHub Context Interceptor for AutoGen
================================================================

The interceptor monkey-patches the AutoGen message pipeline so that:

  1. **Before each LLM call** — the agent's current MemHub working memory
     is prepended to the message list as a ``system`` turn, giving the
     LLM immediate access to the full memory context without the agent
     needing to explicitly call ``query_team_memory`` for every turn.

  2. **After each LLM response** — any text that the model annotates with
     the special tag ``[REMEMBER: <text>]`` is automatically extracted and
     stored back to MemHub, closing the read→reason→write loop without
     requiring the agent to call ``store_memory`` explicitly.

  3. **Token budget guard** — if the injected context would push the total
     prompt beyond a configurable token ceiling, the interceptor triggers
     a silent ``trigger_policy_sweep`` before injecting, ensuring the
     agent never receives a context that overflows its LLM window.

Integration
───────────
Apply the interceptor to a ConversableAgent with:

    from agents.interceptor import MemHubInterceptor
    interceptor = MemHubInterceptor(agent_id="researcher-01")
    interceptor.attach(researcher_agent)

Implementation note
───────────────────
AutoGen v0.2 does not expose a first-class hook for pre/post LLM
processing.  We achieve the same effect by wrapping the agent's
``generate_reply`` method with a thin decorator that injects context
and post-processes the reply.  This is compatible with AutoGen ≥ 0.2.0.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from core.summarization import count_tokens
from agents.tools import (
    query_team_memory,
    store_memory,
    trigger_policy_sweep,
)

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Max tokens of injected MemHub context before we trigger a policy sweep.
CONTEXT_TOKEN_CEILING: int = 1_500

# Regex to extract [REMEMBER: ...] annotations from LLM output.
_REMEMBER_RE = re.compile(r"\[REMEMBER:\s*(.+?)\]", re.DOTALL)

# How many MemHub memories to inject per turn.
CONTEXT_TOP_K: int = 5


class MemHubInterceptor:
    """
    Wraps an AutoGen ``ConversableAgent`` to transparently inject MemHub
    context before each LLM call and persist annotated findings after.

    Args:
        agent_id:           The MemHub agent ID for this agent.
        include_shared:     Also inject shared team memories into context.
        namespace:          The agent's namespace for reads and writes.
        context_top_k:      How many memory chunks to inject per turn.
        token_ceiling:      Max tokens of injected context (triggers sweep if exceeded).
        auto_store_tag:     If True, parse [REMEMBER: ...] tags in replies.
    """

    def __init__(
        self,
        agent_id: str,
        include_shared: bool = True,
        namespace: str = "private",
        context_top_k: int = CONTEXT_TOP_K,
        token_ceiling: int = CONTEXT_TOKEN_CEILING,
        auto_store_tag: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.include_shared = include_shared
        self.namespace = namespace
        self.context_top_k = context_top_k
        self.token_ceiling = token_ceiling
        self.auto_store_tag = auto_store_tag
        self._agent: Any = None  # Set by attach()

    # ── Attachment ─────────────────────────────────────────────────────────────

    def attach(self, agent: Any) -> None:
        """
        Wrap *agent*'s ``generate_reply`` method with the MemHub interceptor.

        Args:
            agent: An AutoGen ConversableAgent (or subclass) instance.
        """
        self._agent = agent
        original_generate_reply = agent.generate_reply

        interceptor = self  # capture self for the closure

        def _intercepted_generate_reply(
            messages: Optional[List[Dict[str, Any]]] = None,
            sender: Optional[Any] = None,
            **kwargs: Any,
        ) -> Union[str, Dict[str, Any], None]:
            # ── Pre-call: inject MemHub context ───────────────────────────────
            augmented_messages = interceptor._inject_context(messages or [])

            # ── LLM call ──────────────────────────────────────────────────────
            reply = original_generate_reply(
                messages=augmented_messages, sender=sender, **kwargs
            )

            # ── Post-call: parse and store [REMEMBER: ...] annotations ────────
            if interceptor.auto_store_tag and isinstance(reply, str):
                interceptor._extract_and_store(reply)

            return reply

        agent.generate_reply = _intercepted_generate_reply
        logger.info(
            "MemHubInterceptor attached to agent '%s' (agent_id='%s').",
            getattr(agent, "name", str(agent)),
            self.agent_id,
        )

    # ── Context injection ─────────────────────────────────────────────────────

    def _inject_context(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories and prepend them to the message list
        as a synthetic ``system`` turn.

        If the retrieved context exceeds ``token_ceiling``, a silent
        policy sweep is triggered first to compact working memory.

        Args:
            messages: The current conversation message list.

        Returns:
            Augmented message list with MemHub context prepended.
        """
        # Build query from the last user/agent turn
        query = self._build_query(messages)
        if not query:
            return messages

        try:
            memories = query_team_memory(
                agent_id=self.agent_id,
                query=query,
                top_k=self.context_top_k,
                include_shared=self.include_shared,
                namespace=self.namespace,
            )
        except Exception as exc:
            logger.warning("MemHub context injection failed: %s", exc)
            return messages  # Fail open — don't block the agent

        if not memories:
            return messages

        # Format the context block
        context_lines = [
            f"[{m.get('tier', '?').upper()} | score={m.get('score', 0):.3f}] "
            f"{m.get('content', '')}"
            for m in memories
        ]
        context_text = "\n".join(context_lines)
        context_tokens = count_tokens(context_text)

        # Token ceiling guard: compact before injecting
        if context_tokens > self.token_ceiling:
            logger.info(
                "Context injection: %d tokens > ceiling %d for agent '%s'. "
                "Running silent policy sweep.",
                context_tokens,
                self.token_ceiling,
                self.agent_id,
            )
            try:
                trigger_policy_sweep(agent_id=self.agent_id, namespace=self.namespace)
            except Exception as exc:
                logger.warning("Silent policy sweep failed: %s", exc)

        injection = {
            "role": "system",
            "content": (
                f"## MemHub Working Memory (top-{len(memories)} relevant chunks)\n\n"
                f"{context_text}\n\n"
                "Use the above memory context to inform your response. "
                "Cite memory IDs where relevant. "
                "Tag new findings with [REMEMBER: <concise fact>] to auto-store them."
            ),
        }

        logger.debug(
            "Injected %d memories (%d tokens) into context for agent '%s'.",
            len(memories),
            context_tokens,
            self.agent_id,
        )

        # Prepend after the existing system message (if any)
        if messages and messages[0].get("role") == "system":
            return [messages[0], injection, *messages[1:]]
        return [injection, *messages]

    # ── Post-processing ────────────────────────────────────────────────────────

    def _extract_and_store(self, reply: str) -> None:
        """
        Parse all ``[REMEMBER: <text>]`` annotations from *reply* and
        store each as a private memory entry in MemHub.

        This gives the LLM an ergonomic way to persist insights without
        making an explicit tool call.

        Args:
            reply: The raw string returned by the LLM.
        """
        matches = _REMEMBER_RE.findall(reply)
        if not matches:
            return

        for fact in matches:
            fact = fact.strip()
            if not fact:
                continue
            try:
                result = store_memory(
                    agent_id=self.agent_id,
                    content=fact,
                    namespace=self.namespace,
                    source="auto_remember_tag",
                    run_policies=False,  # Avoid repeated sweeps mid-turn
                )
                logger.info(
                    "Auto-stored [REMEMBER] tag for agent '%s': id=%s, content=%r",
                    self.agent_id,
                    result.get("memory_id"),
                    fact[:80],
                )
            except Exception as exc:
                logger.warning(
                    "Failed to auto-store [REMEMBER] tag '%s': %s", fact[:60], exc
                )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_query(messages: List[Dict[str, Any]]) -> str:
        """
        Extract the most recent non-system message content to use as the
        MemHub retrieval query.  Falls back to the last 3 messages joined
        if no clear single query is found.
        """
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                # Truncate to avoid excessively long queries
                return content.strip()[:512]

        # Fallback: concatenate last 3 messages
        recent = [
            m.get("content", "") for m in messages[-3:]
            if isinstance(m.get("content"), str)
        ]
        return " ".join(recent)[:512]


# ── Convenience: attach interceptor to all team agents ────────────────────────


def attach_interceptors_to_team(
    team: Dict[str, Any],
    agent_name_to_id: Dict[str, str],
    include_shared: bool = True,
) -> None:
    """
    Attach a :class:`MemHubInterceptor` to every speaking agent in *team*.

    Args:
        team:               Dict returned by ``build_team()`` in team_config.py.
        agent_name_to_id:   Mapping of AutoGen agent ``name`` → MemHub ``agent_id``.
                            Example: {"Researcher": "agent-researcher-01"}
        include_shared:     Whether each interceptor reads the shared namespace.
    """
    speaking_agents = ["orchestrator", "researcher", "analyst", "critic"]

    for role in speaking_agents:
        agent = team.get(role)
        if agent is None:
            continue

        name = getattr(agent, "name", role)
        agent_id = agent_name_to_id.get(name)
        if not agent_id:
            logger.warning(
                "No MemHub agent_id mapping found for agent '%s'; skipping interceptor.",
                name,
            )
            continue

        interceptor = MemHubInterceptor(
            agent_id=agent_id,
            include_shared=include_shared,
        )
        interceptor.attach(agent)

    logger.info(
        "MemHubInterceptors attached to team roles: %s", speaking_agents
    )
