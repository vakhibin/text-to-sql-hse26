"""Session management endpoints: inspect and forget sessions."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request

from orchestrator_agent.memory import CheckpointerHandle
from services.orchestrator_api.message_utils import message_to_chat
from services.orchestrator_api.schemas import (
    HistoryMessage,
    HistoryResponse,
    SessionDeleteResponse,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}", response_model=HistoryResponse)
async def get_session(session_id: str, request: Request) -> HistoryResponse:
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    values = dict(snapshot.values) if snapshot and snapshot.values else {}
    if not values:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")
    raw_messages = cast(list[Any], values.get("messages") or [])
    history = [
        HistoryMessage(role=m.role, content=m.content)
        for m in (message_to_chat(msg) for msg in raw_messages)
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
            "last_row_count": values.get("last_row_count"),
            "last_result_export": values.get("last_result_export"),
            "sql_history": values.get("sql_history"),
            "pending_confirmation": values.get("pending_confirmation"),
        },
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
    """Best-effort session deletion via the checkpointer's ``delete_thread`` API.

    Not all checkpointers expose thread deletion; we report ``deleted=False``
    when unsupported so the client can decide whether to retry or ignore.
    """
    handle: CheckpointerHandle = request.app.state.checkpointer
    saver = handle.saver
    delete_method = getattr(saver, "adelete_thread", None)
    if delete_method is None:
        return SessionDeleteResponse(session_id=session_id, deleted=False)
    try:
        await delete_method(session_id)
    except Exception:  # pragma: no cover - backend-specific
        return SessionDeleteResponse(session_id=session_id, deleted=False)
    return SessionDeleteResponse(session_id=session_id, deleted=True)
