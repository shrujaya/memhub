"""
api/models.py  — MemHub Pydantic Data Models
=============================================

Defines the request and response schemas for the MemHub REST API.
All models use Pydantic v2 for validation, serialisation, and
OpenAPI schema generation.

Shape summary
─────────────
  Requests
    StoreRequest    → POST /store
    RetrieveRequest → POST /retrieve

  Responses
    MemoryItem      → individual memory chunk surfaced in a response
    StoreResponse   → POST /store result
    RetrieveResponse→ POST /retrieve result
    PolicyRunResponse → POST /policies/run result (admin endpoint)
    HealthResponse  → GET /health
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class Namespace(str, Enum):
    """Access-control namespace for a memory entry."""
    PRIVATE = "private"   # Only accessible by the owning agent
    SHARED  = "shared"    # Accessible by all agents in the same team


class EvictionStrategyEnum(str, Enum):
    """Eviction strategy choices exposed via the admin API."""
    LRU  = "lru"
    FIFO = "fifo"
    LFU  = "lfu"


class MemoryTierEnum(str, Enum):
    """Which storage tier a returned chunk came from."""
    TIER1 = "tier1"
    TIER2 = "tier2"


# ── Shared sub-models ─────────────────────────────────────────────────────────


class MemoryMetadata(BaseModel):
    """
    Arbitrary key-value metadata attached to a memory entry.

    The ``source``, ``tags``, and ``priority`` fields are optional but
    indexed — the retrieval layer can filter on them via ChromaDB's
    ``where`` clause.
    """
    source: Optional[str] = Field(
        None,
        description="Origin of this memory (e.g. 'user_message', 'tool_output', 'observation').",
        examples=["tool_output"],
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Arbitrary string tags for categorisation.",
        examples=[["finance", "Q2-report"]],
    )
    priority: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Importance weight. Higher values bias retrieval scoring.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Catch-all bag for any additional structured metadata.",
    )

    model_config = {"extra": "allow"}


# ── Request Models ────────────────────────────────────────────────────────────


class StoreRequest(BaseModel):
    """
    Payload for POST /store.

    Stores one or more memory entries for an agent in Tier-1 (SQLite)
    working memory. If storage causes the token budget to be exceeded, the
    PolicyEngine runs automatically after the write.
    """

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier of the agent writing this memory.",
        examples=["agent-researcher-01"],
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="The raw text content of the memory to store.",
        examples=["The quarterly revenue for ACME Corp grew 12 % YoY to $4.2 B."],
    )
    namespace: Namespace = Field(
        default=Namespace.PRIVATE,
        description=(
            "'private' memories are only visible to the owning agent. "
            "'shared' memories are readable by all agents in the same team."
        ),
    )
    metadata: MemoryMetadata = Field(
        default_factory=MemoryMetadata,
        description="Optional structured metadata attached to this memory.",
    )
    run_policies: bool = Field(
        default=True,
        description=(
            "If True, triggers the PolicyEngine (eviction / demotion) "
            "after the write. Set to False for bulk imports."
        ),
    )

    @field_validator("agent_id")
    @classmethod
    def agent_id_no_whitespace(cls, v: str) -> str:
        if v != v.strip():
            raise ValueError("agent_id must not have leading or trailing whitespace.")
        return v

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank or whitespace-only.")
        return v


class RetrieveRequest(BaseModel):
    """
    Payload for POST /retrieve.

    Performs a hybrid Tier-1 keyword + Tier-2 semantic search and returns
    the top-k most relevant memories for the agent.
    """

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="The agent requesting memory retrieval.",
        examples=["agent-researcher-01"],
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=4_096,
        description="Natural-language or keyword query to search memories with.",
        examples=["What is ACME Corp's Q2 revenue?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of memory chunks to return.",
    )
    namespace: Namespace = Field(
        default=Namespace.PRIVATE,
        description="The agent's own namespace to search within.",
    )
    include_shared: bool = Field(
        default=False,
        description=(
            "Also search the 'shared' namespace, enabling cross-agent "
            "memory access for team collaboration."
        ),
    )
    tier_filter: Optional[MemoryTierEnum] = Field(
        default=None,
        description=(
            "Restrict search to a single tier. "
            "If None, both Tier-1 and Tier-2 are searched (recommended)."
        ),
    )
    memory_id: Optional[UUID] = Field(
        default=None,
        description=(
            "If provided, fetch a single memory by exact ID instead of "
            "running a query. All other search fields are ignored."
        ),
    )


class PolicyRunRequest(BaseModel):
    """Payload for POST /policies/run (admin endpoint)."""

    agent_id: str = Field(..., description="Agent to run policies for.")
    namespace: Namespace = Field(
        default=Namespace.PRIVATE,
        description="Namespace forwarded to DemotionPolicy for summary insertion.",
    )
    strategy: EvictionStrategyEnum = Field(
        default=EvictionStrategyEnum.LRU,
        description="Eviction strategy to use during this policy sweep.",
    )


# ── Response Models ───────────────────────────────────────────────────────────


class MemoryItem(BaseModel):
    """
    A single memory chunk in a retrieval response.
    Mirrors :class:`~core.retrieval.RetrievalResult` for API serialisation.
    """

    id: str = Field(..., description="Unique ID of the memory entry.")
    agent_id: str = Field(..., description="Owning agent identifier.")
    content: str = Field(..., description="Raw memory text.")
    score: float = Field(..., description="Relevance score (higher = more relevant).")
    tier: MemoryTierEnum = Field(..., description="Storage tier the result came from.")
    namespace: Namespace = Field(..., description="Access namespace.")
    access_count: int = Field(..., description="Total retrieval hit count.")
    created_at: float = Field(..., description="Unix timestamp of insertion.")
    last_accessed: float = Field(..., description="Unix timestamp of last retrieval.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StoreResponse(BaseModel):
    """Response returned by POST /store."""

    memory_id: str = Field(..., description="UUID assigned to the stored memory.")
    agent_id: str
    token_count: int = Field(
        ..., description="Token count of the stored content (tiktoken cl100k_base)."
    )
    policies_triggered: bool = Field(
        ...,
        description="Whether the PolicyEngine ran after this store operation.",
    )
    message: str = Field(
        default="Memory stored successfully.",
        description="Human-readable status message.",
    )


class RetrieveResponse(BaseModel):
    """Response returned by POST /retrieve."""

    agent_id: str
    query: str
    results: List[MemoryItem]
    tier1_hits: int = Field(..., description="Number of results from Tier-1 (SQLite).")
    tier2_hits: int = Field(..., description="Number of results from Tier-2 (ChromaDB).")
    total_results: int = Field(..., description="Total results returned.")
    latency_ms: float = Field(..., description="Server-side retrieval latency in ms.")


class PolicySummary(BaseModel):
    """Per-policy statistics within a PolicyRunResponse."""

    evicted_count: int = 0
    demoted_count: int = 0
    promoted_count: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    compression_pct: float = Field(
        0.0, description="Token reduction percentage achieved by demotion."
    )
    new_summary_preview: Optional[str] = Field(
        None, description="First 200 chars of the generated demotion summary."
    )


class PolicyRunResponse(BaseModel):
    """Response returned by POST /policies/run."""

    agent_id: str
    promotion: PolicySummary
    demotion: PolicySummary
    eviction: PolicySummary
    message: str = "Policy sweep complete."


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = "ok"
    tier1_connected: bool
    tier2_connected: bool
    version: str = "0.1.0"
