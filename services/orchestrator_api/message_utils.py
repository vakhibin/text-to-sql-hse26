"""Shared helpers for turning LangChain messages into API DTOs."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from services.orchestrator_api.schemas import ChatMessage


def message_to_chat(msg: Any) -> ChatMessage | None:
    """Convert a LangChain message to the API ``ChatMessage`` DTO.

    Returns ``None`` for message types we do not surface (e.g. internal
    function messages). Keeping this logic in one place lets the chat and
    sessions routers stay identical in their message handling.
    """
    if isinstance(msg, HumanMessage):
        return ChatMessage(role="human", content=as_text(msg.content))
    if isinstance(msg, AIMessage):
        return ChatMessage(role="ai", content=as_text(msg.content))
    if isinstance(msg, SystemMessage):
        return ChatMessage(role="system", content=as_text(msg.content))
    if isinstance(msg, ToolMessage):
        return ChatMessage(
            role="tool",
            content=as_text(msg.content),
            name=getattr(msg, "name", None),
        )
    return None


def as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(content)
