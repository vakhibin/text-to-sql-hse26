"""FastAPI application for the conversational orchestrator.

Phase 3: session memory + LLM agent node. Tool calls are wired in Phase 4.

Endpoints:

- ``GET  /health``                     — liveness
- ``POST /chat``                       — one conversational turn
- ``GET  /sessions/{session_id}``      — inspect current state of a session
- ``DELETE /sessions/{session_id}``    — forget a session (best-effort)

A single compiled LangGraph is created at startup and reused for all
requests. Session isolation is provided by LangGraph's ``thread_id``
mechanism (thread_id == session_id), backed by the checkpointer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from services.orchestrator_api.graph import build_orchestrator_graph
from services.orchestrator_api.memory import CheckpointerHandle, create_checkpointer
from services.orchestrator_api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
    SessionDeleteResponse,
)

SERVICE_NAME = "orchestrator_api"
SERVICE_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    handle = await create_checkpointer()
    graph = build_orchestrator_graph(handle.saver)
    app.state.checkpointer = handle
    app.state.graph = graph
    try:
        yield
    finally:
        await handle.aclose()


app = FastAPI(title="Orchestrator API", version=SERVICE_VERSION, lifespan=lifespan)


def _message_to_chat(msg: Any) -> ChatMessage | None:
    if isinstance(msg, HumanMessage):
        return ChatMessage(role="human", content=_as_text(msg.content))
    if isinstance(msg, AIMessage):
        return ChatMessage(role="ai", content=_as_text(msg.content))
    if isinstance(msg, SystemMessage):
        return ChatMessage(role="system", content=_as_text(msg.content))
    if isinstance(msg, ToolMessage):
        return ChatMessage(
            role="tool",
            content=_as_text(msg.content),
            name=getattr(msg, "name", None),
        )
    return None


def _as_text(content: Any) -> str:
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    graph = app.state.graph
    config = {"configurable": {"thread_id": req.session_id}}

    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=req.message)],
        "session_id": req.session_id,
        "user_id": req.user_id,
    }
    if req.active_db_id is not None:
        input_state["active_db_id"] = req.active_db_id

    try:
        result = await graph.ainvoke(input_state, config=config)
    except Exception as exc:  # pragma: no cover - safety net for LLM errors
        raise HTTPException(status_code=502, detail=f"agent invocation failed: {exc}") from exc

    messages = cast(list[Any], result.get("messages") or [])
    ai_message = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    reply = _as_text(ai_message.content) if ai_message is not None else ""

    last_human_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    turn_raw = messages[last_human_idx:] if last_human_idx is not None else messages
    turn_messages = [
        chat_msg
        for chat_msg in (_message_to_chat(m) for m in turn_raw)
        if chat_msg is not None
    ]

    return ChatResponse(
        session_id=req.session_id,
        user_id=req.user_id,
        reply=reply,
        messages_delta=turn_messages,
        active_db_id=result.get("active_db_id"),
        last_sql=result.get("last_sql"),
        warnings=[],
    )


@app.get("/sessions/{session_id}", response_model=HistoryResponse)
async def get_session(session_id: str) -> HistoryResponse:
    graph = app.state.graph
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values) if snapshot and snapshot.values else {}
    if not values:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")
    raw_messages = cast(list[Any], values.get("messages") or [])
    history = [
        HistoryMessage(role=m.role, content=m.content)
        for m in (_message_to_chat(msg) for msg in raw_messages)
        if m is not None
    ]
    return HistoryResponse(
        session_id=session_id,
        messages=history,
        active_db_id=values.get("active_db_id"),
        last_sql=values.get("last_sql"),
        extra={
            "user_id": values.get("user_id"),
            "last_rows_columns": values.get("last_rows_columns"),
            "pending_confirmation": values.get("pending_confirmation"),
        },
    )


@app.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str) -> SessionDeleteResponse:
    """Best-effort session deletion via the checkpointer's ``delete_thread`` API.

    Not all checkpointers expose thread deletion; we report ``deleted=False``
    when unsupported so the client can decide whether to retry or ignore.
    """
    handle: CheckpointerHandle = app.state.checkpointer
    saver = handle.saver
    delete_method = getattr(saver, "adelete_thread", None)
    if delete_method is None:
        return SessionDeleteResponse(session_id=session_id, deleted=False)
    try:
        await delete_method(session_id)
    except Exception:  # pragma: no cover - backend-specific
        return SessionDeleteResponse(session_id=session_id, deleted=False)
    return SessionDeleteResponse(session_id=session_id, deleted=True)
