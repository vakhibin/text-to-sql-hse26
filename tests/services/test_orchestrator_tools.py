"""End-to-end tests for the orchestrator's core tools.

Tools depend on LangGraph runtime state injection (``InjectedState``,
``InjectedToolCallId``) that's only wired up by a compiled graph. So we drive
them through a full mini-graph:

- ``ScriptedToolCallLLM`` fakes the LLM and emits pre-scripted messages with
  tool calls.
- The real ``TextToSQLClient`` is pointed at an ``httpx.MockTransport`` so no
  network or FastAPI app is involved.

Each test builds a one-shot graph, invokes one conversational turn, and
asserts both the final state (artifacts like ``last_sql`` / ``active_db_id``)
and the ``ToolMessage`` text the LLM would see next.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from services.orchestrator_api import agent as agent_mod
from services.orchestrator_api.clients.text_to_sql import TextToSQLClient
from services.orchestrator_api.graph import build_orchestrator_graph
from services.orchestrator_api.memory import create_checkpointer
from services.orchestrator_api.tools.core import make_core_tools


class ScriptedToolCallLLM(BaseChatModel):
    """Fake LLM that emits a pre-scripted sequence of AIMessages.

    Reply ``i`` is consumed on the ``i``-th model call within a session. After
    the scripted replies run out the model returns an empty AI message, which
    ends the tool-calling loop.

    Implementation note: ``replies`` and ``_call_index`` are set via
    ``object.__setattr__`` so they don't clash with pydantic's field machinery
    inherited from ``BaseChatModel``.
    """

    def __init__(self, replies: list[AIMessage]) -> None:
        super().__init__()
        object.__setattr__(self, "_call_index", 0)
        object.__setattr__(self, "_replies", replies)

    @property
    def _llm_type(self) -> str:
        return "scripted_tool_call"

    def bind_tools(self, tools: list[Any], **_: Any) -> "ScriptedToolCallLLM":
        # Tool-bound behaviour is whatever the scripted replies carry.
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
    """Spin up an in-memory graph for one turn and return the final state."""
    client = _make_client(handler)
    handle = await create_checkpointer(sqlite_path=":memory:")
    agent_mod.set_chat_model(ScriptedToolCallLLM(replies))
    try:
        tools = make_core_tools(client)
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
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


@pytest.mark.asyncio
async def test_run_text_to_sql_happy_path_updates_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/run"
        return httpx.Response(
            200,
            json={
                "trace_id": "tr",
                "db_id": "toy",
                "question": "how many?",
                "sql": "SELECT COUNT(*) FROM t",
                "executed": True,
                "rows": [[42]],
                "columns": ["cnt"],
                "row_count": 1,
                "stage_status": {},
                "warnings": [],
                "cost_usd": 0.0,
                "elapsed_s": 0.1,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "run_text_to_sql",
                {"question": "how many?", "db_id": "toy"},
                "c1",
            ),
            AIMessage(content="Got 42."),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="how many rows?")]},
        session_id="s_run_happy",
    )

    assert state["active_db_id"] == "toy"
    assert state["last_sql"] == "SELECT COUNT(*) FROM t"
    assert state["last_rows_preview"] == [[42]]
    assert state["last_rows_columns"] == ["cnt"]
    tm = _last_tool_message(state)
    assert tm.name == "run_text_to_sql"
    assert "42" in tm.content


@pytest.mark.asyncio
async def test_run_text_to_sql_uses_active_db_from_state() -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "trace_id": "tr",
                "db_id": "from_state",
                "question": "q",
                "sql": "SELECT 1",
                "executed": True,
                "rows": [[1]],
                "columns": ["x"],
                "row_count": 1,
                "stage_status": {},
                "warnings": [],
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("run_text_to_sql", {"question": "q"}, "c2"),
            AIMessage(content="done"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="q")],
            "active_db_id": "from_state",
        },
        session_id="s_run_from_state",
    )

    assert seen_payloads[0]["db_id"] == "from_state"
    assert state["active_db_id"] == "from_state"


@pytest.mark.asyncio
async def test_run_text_to_sql_missing_db_returns_tool_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise AssertionError("should not call upstream when db_id missing")

    state = await _run_one_turn(
        replies=[
            _tool_call("run_text_to_sql", {"question": "q"}, "c3"),
            AIMessage(content="please pick a db"),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="no db yet")]},
        session_id="s_no_db",
    )

    assert calls["n"] == 0
    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "database" in tm.content.lower()


@pytest.mark.asyncio
async def test_execute_sql_happy_path_sets_last_sql_and_preview() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/execute"
        return httpx.Response(
            200,
            json={
                "db_id": "toy",
                "sql": "SELECT 1",
                "success": True,
                "rows": [[1]],
                "columns": ["x"],
                "row_count": 1,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("execute_sql", {"sql": "SELECT 1", "db_id": "toy"}, "c4"),
            AIMessage(content="Ran it."),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="run select 1")]},
        session_id="s_exec_ok",
    )

    assert state["last_sql"] == "SELECT 1"
    assert state["active_db_id"] == "toy"
    assert state["last_rows_preview"] == [[1]]
    tm = _last_tool_message(state)
    assert "Rows: 1" in tm.content


@pytest.mark.asyncio
async def test_execute_sql_guardrail_violation_is_reported_to_llm() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "db_id": "toy",
                "sql": "UPDATE t SET x = 1",
                "success": False,
                "error": "Write SQL rejected by guardrail",
                "error_code": "READ_ONLY_VIOLATION",
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "execute_sql", {"sql": "UPDATE t SET x = 1", "db_id": "toy"}, "c5"
            ),
            AIMessage(content="Write was rejected."),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="update please")]},
        session_id="s_guardrail",
    )

    tm = _last_tool_message(state)
    assert "READ_ONLY_VIOLATION" in tm.content
    assert state["last_sql"] == "UPDATE t SET x = 1"


@pytest.mark.asyncio
async def test_explain_sql_returns_explanation_as_tool_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/explain"
        return httpx.Response(
            200,
            json={
                "trace_id": "t",
                "sql": "SELECT 1",
                "explanation": "Returns a constant value.",
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("explain_sql", {"sql": "SELECT 1"}, "c6"),
            AIMessage(content="Here."),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="explain")],
            "active_db_id": "toy",
        },
        session_id="s_explain",
    )

    tm = _last_tool_message(state)
    assert tm.content == "Returns a constant value."


@pytest.mark.asyncio
async def test_tool_reports_http_failure_gracefully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "run_text_to_sql", {"question": "q", "db_id": "toy"}, "c7"
            ),
            AIMessage(content="Upstream failed."),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="q")]},
        session_id="s_http_err",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "500" in tm.content
