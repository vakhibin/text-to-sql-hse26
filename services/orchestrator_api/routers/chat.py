"""``POST /chat`` — one conversational turn.

Each turn is wrapped in a Langfuse root span (``orchestrator_chat``) so the
trace shows the user message as ``input`` and the assistant reply as
``output``. The Langfuse LangChain ``CallbackHandler`` is plugged into the
LangGraph ``ainvoke`` config; it transparently emits child spans for the
``agent`` LLM call and for each tool invocation, all attached to the
current OTEL span (= our root span) via OpenTelemetry context propagation.

The Langfuse ``session_id`` mirrors the orchestrator session id so that all
turns of one chat collapse into a single Langfuse session in the UI.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

from services.orchestrator_api.message_utils import as_text, message_to_chat
from services.orchestrator_api.schemas import ChatRequest, ChatResponse
from text_to_sql_agent.tools.observability import (
    flush_langfuse,
    get_langfuse_langchain_handler,
    safe_state_snapshot,
    start_langfuse_span,
    update_langfuse_span,
)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    config: dict[str, Any] = {"configurable": {"thread_id": req.session_id}}

    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=req.message)],
        "session_id": req.session_id,
        "user_id": req.user_id,
    }
    if req.active_db_id is not None:
        input_state["active_db_id"] = req.active_db_id

    handler = get_langfuse_langchain_handler()
    if handler is not None:
        config["callbacks"] = [handler]

    # Open the chat-turn root span so per-tool / per-LLM child observations
    # produced by the LangChain handler attach to it via OTEL context. Updates
    # must happen INSIDE the ``with`` block (the SDK closes the span on exit).
    with start_langfuse_span(
        name="orchestrator_chat",
        trace_id=req.session_id,
        input_payload=safe_state_snapshot(
            {
                "message": req.message,
                "session_id": req.session_id,
                "user_id": req.user_id,
                "active_db_id": req.active_db_id,
            }
        ),
        metadata={
            "session_id": req.session_id,
            "user_id": req.user_id,
            "active_db_id": req.active_db_id,
        },
        as_type="chain",
    ):
        try:
            result = await graph.ainvoke(input_state, config=config)
        except Exception as exc:  # pragma: no cover - safety net for LLM errors
            update_langfuse_span(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
                metadata={"session_id": req.session_id, "user_id": req.user_id},
            )
            flush_langfuse()
            raise HTTPException(status_code=502, detail=f"agent invocation failed: {exc}") from exc

        messages = cast(list[Any], result.get("messages") or [])
        ai_message = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        reply = as_text(ai_message.content) if ai_message is not None else ""

        last_human_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        turn_raw = messages[last_human_idx:] if last_human_idx is not None else messages
        turn_messages = [
            chat_msg
            for chat_msg in (message_to_chat(m) for m in turn_raw)
            if chat_msg is not None
        ]

        # Summarize tool activity for the trace metadata: name + count and
        # whether the turn produced new SQL. Keeps the output panel readable
        # while still surfacing what happened during the turn.
        tool_calls_summary = _summarize_tool_calls(turn_raw)
        update_langfuse_span(
            output=safe_state_snapshot(
                {
                    "reply": reply,
                    "active_db_id": result.get("active_db_id"),
                    "last_sql": result.get("last_sql"),
                    "tool_calls": tool_calls_summary,
                }
            ),
            metadata={
                "session_id": req.session_id,
                "user_id": req.user_id,
                "active_db_id": result.get("active_db_id"),
                "tool_call_count": len(tool_calls_summary),
            },
        )

    flush_langfuse()
    return ChatResponse(
        session_id=req.session_id,
        user_id=req.user_id,
        reply=reply,
        messages_delta=turn_messages,
        active_db_id=result.get("active_db_id"),
        last_sql=result.get("last_sql"),
        warnings=[],
    )


def _summarize_tool_calls(turn_messages: list[Any]) -> list[dict[str, Any]]:
    """Return ``[{"name": ..., "args": ...}, ...]`` for AI ``tool_calls`` in this turn.

    Used only as compact metadata for the Langfuse root span; the per-tool
    child observations created by ``CallbackHandler`` are the source of
    truth for tool inputs/outputs.
    """
    summary: list[dict[str, Any]] = []
    for msg in turn_messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in getattr(msg, "tool_calls", []) or []:
            summary.append(
                {
                    "name": call.get("name"),
                    "args": call.get("args"),
                }
            )
    return summary
