"""Tests for user-confirmed write-SQL orchestrator tools."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from orchestrator_agent import agent as agent_mod
from orchestrator_agent.clients.text_to_sql import TextToSQLClient
from orchestrator_agent.graph import build_orchestrator_graph
from orchestrator_agent.memory import create_checkpointer
from orchestrator_agent.tools.write import make_write_tools


class ScriptedToolCallLLM(BaseChatModel):
    def __init__(self, replies: list[AIMessage]) -> None:
        super().__init__()
        object.__setattr__(self, "_call_index", 0)
        object.__setattr__(self, "_replies", replies)

    @property
    def _llm_type(self) -> str:
        return "scripted_tool_call"

    def bind_tools(self, tools: list[Any], **_: Any) -> "ScriptedToolCallLLM":
        return self

    def _generate(
        self, messages: list[Any], stop: list[str] | None = None, **_: Any
    ) -> ChatResult:
        idx = self._call_index  # type: ignore[attr-defined]
        replies: list[AIMessage] = self._replies  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_index", idx + 1)
        message = replies[idx] if idx < len(replies) else AIMessage(content="")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self, messages: list[Any], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)


def _make_client(handler) -> TextToSQLClient:
    return TextToSQLClient(
        base_url="http://test", timeout_s=5.0, transport=httpx.MockTransport(handler)
    )


async def _run_one_turn(
    *,
    replies: list[AIMessage],
    handler,
    input_state: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    client = _make_client(handler)
    handle = await create_checkpointer(sqlite_path=":memory:")
    agent_mod.set_chat_model(ScriptedToolCallLLM(replies))
    try:
        tools = make_write_tools(client)
        graph = build_orchestrator_graph(handle.saver, tools=tools)
        config = {"configurable": {"thread_id": session_id}}
        return await graph.ainvoke(input_state, config=config)
    finally:
        agent_mod.set_chat_model(None)
        await client.aclose()
        await handle.aclose()


def _last_tool_message(state: dict[str, Any]) -> ToolMessage:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, ToolMessage):
            return msg
    raise AssertionError("no ToolMessage in state")


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.mark.asyncio
async def test_propose_write_sql_sets_pending_confirmation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("propose_write_sql must not call upstream")

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "propose_write_sql",
                {
                    "sql": "DELETE FROM students WHERE id = 3",
                    "rationale": "Remove duplicate row",
                },
                "c1",
            ),
            AIMessage(content="confirm?"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="delete row")],
            "active_db_id": "toy",
        },
        session_id="s_write_propose",
    )

    pending = state.get("pending_confirmation") or {}
    assert pending["type"] == "write_sql"
    assert pending["sql"] == "DELETE FROM students WHERE id = 3"
    assert pending["db_id"] == "toy"
    tm = _last_tool_message(state)
    assert "NOT executed" in tm.content


@pytest.mark.asyncio
async def test_propose_write_sql_rejects_read_only_sql() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("read-only proposal should not call upstream")

    state = await _run_one_turn(
        replies=[
            _tool_call("propose_write_sql", {"sql": "SELECT 1"}, "c2"),
            AIMessage(content="bad"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="select")],
            "active_db_id": "toy",
        },
        session_id="s_write_readonly",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "read-only" in tm.content
    assert state.get("pending_confirmation") is None


@pytest.mark.asyncio
async def test_confirm_write_sql_executes_pending_sql_and_clears_confirmation() -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/execute-confirmed"
        payload = json.loads(request.content)
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "db_id": payload["db_id"],
                "sql": payload["sql"],
                "success": True,
                "read_only": False,
                "rows": [],
                "columns": None,
                "row_count": 0,
                "elapsed_s": 0.01,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("confirm_write_sql", {}, "c3"),
            AIMessage(content="executed"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="yes, confirm")],
            "pending_confirmation": {
                "type": "write_sql",
                "sql": "DELETE FROM students WHERE id = 3",
                "db_id": "toy",
            },
        },
        session_id="s_write_confirm",
    )

    assert seen_payloads[0]["confirmation"] == "USER_CONFIRMED_WRITE"
    assert state.get("pending_confirmation") is None
    history = state.get("sql_history") or []
    assert history[-1]["source"] == "write_confirmed"
    assert history[-1]["executed"] is True
    tm = _last_tool_message(state)
    assert "Confirmed write SQL executed" in tm.content


@pytest.mark.asyncio
async def test_confirm_write_sql_without_pending_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no pending confirmation should not call upstream")

    state = await _run_one_turn(
        replies=[_tool_call("confirm_write_sql", {}, "c4"), AIMessage(content="no")],
        handler=handler,
        input_state={"messages": [HumanMessage(content="confirm")]},
        session_id="s_write_no_pending",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "No pending" in tm.content


@pytest.mark.asyncio
async def test_cancel_pending_confirmation_clears_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("cancel should not call upstream")

    state = await _run_one_turn(
        replies=[
            _tool_call("cancel_pending_confirmation", {}, "c5"),
            AIMessage(content="cancelled"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="cancel")],
            "pending_confirmation": {
                "type": "write_sql",
                "sql": "UPDATE students SET age = 21 WHERE id = 1",
                "db_id": "toy",
            },
        },
        session_id="s_write_cancel",
    )

    assert state.get("pending_confirmation") is None
    tm = _last_tool_message(state)
    assert "cancelled" in tm.content
