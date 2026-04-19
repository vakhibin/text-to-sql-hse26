"""``POST /chat`` — one conversational turn."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

from services.orchestrator_api.message_utils import as_text, message_to_chat
from services.orchestrator_api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
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

    return ChatResponse(
        session_id=req.session_id,
        user_id=req.user_id,
        reply=reply,
        messages_delta=turn_messages,
        active_db_id=result.get("active_db_id"),
        last_sql=result.get("last_sql"),
        warnings=[],
    )
