"""Request/response contracts for the orchestrator HTTP API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


MessageRole = Literal["human", "ai", "system", "tool"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Stable id per conversation thread.")
    user_id: str = Field(default="default", description="User identifier for multi-tenant setups.")
    message: str = Field(..., min_length=1, description="User's message for this turn.")
    active_db_id: Optional[str] = Field(
        default=None,
        description="Optional database the user wants to target for this turn.",
    )


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    reply: str = Field(..., description="Assistant's natural-language reply for this turn.")
    messages_delta: list[ChatMessage] = Field(
        default_factory=list,
        description="Messages emitted in this turn (last one is the reply).",
    )
    active_db_id: Optional[str] = None
    last_sql: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class HistoryMessage(BaseModel):
    role: MessageRole
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
    active_db_id: Optional[str] = None
    last_sql: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
