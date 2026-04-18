"""
api/models.py
=============
Pydantic request and response schemas for MemHub.

Keeping models in a dedicated module lets every other layer (routes, tests,
clients) import shared types without creating circular dependencies.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
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


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class MemoryResponse(BaseModel):
    agent_id: str
    workspace_id: Optional[str]
    working_memory: Optional[str]
    long_term_memory: List[Dict[str, Any]]
    summarization_triggered: bool = False
