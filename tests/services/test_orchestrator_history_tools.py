"""End-to-end tests for the orchestrator's SQL manipulation + history tools.

Same pattern as ``test_orchestrator_tools.py`` / ``test_orchestrator_discovery_tools.py``:
each test drives a one-turn mini-graph through a scripted LLM and an
``httpx.MockTransport`` so no real network or LLM is touched.
"""

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
from orchestrator_agent.tools.history import make_history_tools


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
        tools = make_history_tools(client)
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


# ---- fix_sql -------------------------------------------------------------


@pytest.mark.asyncio
async def test_fix_sql_happy_path_updates_last_sql_and_history() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/refine"
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "trace_id": "t",
                "db_id": payload["db_id"],
                "original_sql": payload["sql"],
                "refined_sql": "SELECT COUNT(*) FROM t",
                "changed": True,
                "executed": True,
                "success": True,
                "warnings": [],
                "cost_usd": 0.0,
                "elapsed_s": 0.1,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("fix_sql", {}, "c1"),
            AIMessage(content="fixed"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="please fix")],
            "active_db_id": "toy",
            "last_sql": "SELECT COUNT( FROM t",
        },
        session_id="s_fix_ok",
    )

    assert state["last_sql"] == "SELECT COUNT(*) FROM t"
    history = state.get("sql_history") or []
    assert len(history) == 1
    assert history[0]["source"] == "fix"
    assert history[0]["executed"] is True
    tm = _last_tool_message(state)
    assert "Refined SQL" in tm.content


@pytest.mark.asyncio
async def test_fix_sql_without_last_sql_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach upstream without sql")

    state = await _run_one_turn(
        replies=[_tool_call("fix_sql", {}, "c2"), AIMessage(content="ack")],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="fix")],
            "active_db_id": "toy",
        },
        session_id="s_fix_no_sql",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "No SQL to fix" in tm.content


@pytest.mark.asyncio
async def test_fix_sql_reports_still_failing_refinement() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "trace_id": "t",
                "db_id": payload["db_id"],
                "original_sql": payload["sql"],
                "refined_sql": "SELECT COUNT(*) FROM t",
                "changed": True,
                "executed": False,
                "success": False,
                "error": "no such column: x",
                "warnings": [],
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("fix_sql", {"sql": "SELECT x FROM t"}, "c3"),
            AIMessage(content="still broken"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="x")],
            "active_db_id": "toy",
        },
        session_id="s_fix_still_broken",
    )

    tm = _last_tool_message(state)
    assert tm.status != "error"
    assert "no such column: x" in tm.content
    history = state.get("sql_history") or []
    assert history[-1]["executed"] is False


# ---- modify_sql ----------------------------------------------------------


@pytest.mark.asyncio
async def test_modify_sql_happy_path_updates_last_sql() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/modify"
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "trace_id": "t",
                "db_id": payload["db_id"],
                "original_sql": payload["sql"],
                "modified_sql": "SELECT * FROM t WHERE year = 2023",
                "changed": True,
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "modify_sql",
                {"instruction": "only 2023", "sql": "SELECT * FROM t"},
                "c4",
            ),
            AIMessage(content="done"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="only 2023")],
            "active_db_id": "toy",
        },
        session_id="s_modify_ok",
    )

    assert state["last_sql"] == "SELECT * FROM t WHERE year = 2023"
    history = state.get("sql_history") or []
    assert history[-1]["source"] == "modify"
    assert history[-1]["executed"] is False


@pytest.mark.asyncio
async def test_modify_sql_unchanged_is_reported_not_as_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "trace_id": "t",
                "db_id": payload["db_id"],
                "original_sql": payload["sql"],
                "modified_sql": payload["sql"],
                "changed": False,
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "modify_sql",
                {"instruction": "???", "sql": "SELECT 1"},
                "c5",
            ),
            AIMessage(content="nothing"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="x")],
            "active_db_id": "toy",
        },
        session_id="s_modify_noop",
    )

    tm = _last_tool_message(state)
    assert tm.status != "error"
    assert "No changes" in tm.content


@pytest.mark.asyncio
async def test_modify_sql_uses_last_sql_from_state() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "trace_id": "t",
                "db_id": payload["db_id"],
                "original_sql": payload["sql"],
                "modified_sql": "SELECT 2",
                "changed": True,
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            },
        )

    await _run_one_turn(
        replies=[
            _tool_call("modify_sql", {"instruction": "bump"}, "c6"),
            AIMessage(content="ok"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="bump")],
            "active_db_id": "toy",
            "last_sql": "SELECT 1",
        },
        session_id="s_modify_from_state",
    )

    assert seen[0]["sql"] == "SELECT 1"


# ---- list_recent --------------------------------------------------------


@pytest.mark.asyncio
async def test_list_recent_empty_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("list_recent should not call upstream")

    state = await _run_one_turn(
        replies=[_tool_call("list_recent", {}, "c7"), AIMessage(content="ok")],
        handler=handler,
        input_state={"messages": [HumanMessage(content="history?")]},
        session_id="s_list_empty",
    )

    tm = _last_tool_message(state)
    assert tm.status != "error"
    assert "No SQL" in tm.content


@pytest.mark.asyncio
async def test_list_recent_returns_reversed_history() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("list_recent should not call upstream")

    history = [
        {"sql": "SELECT 1", "db_id": "toy", "source": "execute", "executed": True, "row_count": 1},
        {"sql": "SELECT 2", "db_id": "toy", "source": "execute", "executed": True, "row_count": 1},
        {"sql": "SELECT 3", "db_id": "toy", "source": "execute", "executed": True, "row_count": 1},
    ]
    state = await _run_one_turn(
        replies=[
            _tool_call("list_recent", {"limit": 2}, "c8"),
            AIMessage(content="ok"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="history?")],
            "sql_history": history,
        },
        session_id="s_list_recent",
    )

    tm = _last_tool_message(state)
    assert tm.status != "error"
    # Newest first: [1] SELECT 3, [2] SELECT 2
    content = tm.content
    assert content.index("SELECT 3") < content.index("SELECT 2")
    assert "SELECT 1" not in content
    assert "[1]" in content and "[2]" in content


# ---- rerun --------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerun_default_reexecutes_most_recent_history_entry() -> None:
    seen_sql: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/execute"
        payload = json.loads(request.content)
        seen_sql.append(payload["sql"])
        return httpx.Response(
            200,
            json={
                "db_id": payload["db_id"],
                "sql": payload["sql"],
                "success": True,
                "rows": [[42]],
                "columns": ["n"],
                "row_count": 1,
                "elapsed_s": 0.0,
            },
        )

    history = [
        {"sql": "SELECT 1", "db_id": "toy", "source": "execute", "executed": True},
        {"sql": "SELECT COUNT(*) FROM t", "db_id": "toy", "source": "run", "executed": True},
    ]
    state = await _run_one_turn(
        replies=[_tool_call("rerun", {}, "c9"), AIMessage(content="done")],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="again")],
            "active_db_id": "toy",
            "sql_history": history,
        },
        session_id="s_rerun_default",
    )

    assert seen_sql == ["SELECT COUNT(*) FROM t"]
    assert state["last_sql"] == "SELECT COUNT(*) FROM t"
    assert state.get("last_rows_preview") == [[42]]
    new_history = state.get("sql_history") or []
    assert new_history[-1]["source"] == "rerun"


@pytest.mark.asyncio
async def test_rerun_specific_index_uses_entry_db_id() -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "db_id": payload["db_id"],
                "sql": payload["sql"],
                "success": True,
                "rows": [[1]],
                "columns": ["x"],
                "row_count": 1,
                "elapsed_s": 0.0,
            },
        )

    history = [
        {"sql": "SELECT old", "db_id": "db_a", "source": "execute", "executed": True},
        {"sql": "SELECT new", "db_id": "db_b", "source": "execute", "executed": True},
    ]
    await _run_one_turn(
        replies=[
            _tool_call("rerun", {"index": 2}, "c10"),
            AIMessage(content="ok"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="2nd")],
            "active_db_id": "db_b",
            "sql_history": history,
        },
        session_id="s_rerun_idx",
    )

    assert seen_payloads[0]["sql"] == "SELECT old"
    assert seen_payloads[0]["db_id"] == "db_a"


@pytest.mark.asyncio
async def test_rerun_out_of_range_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("rerun oob should not call upstream")

    state = await _run_one_turn(
        replies=[
            _tool_call("rerun", {"index": 99}, "c11"),
            AIMessage(content="ack"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="oob")],
            "active_db_id": "toy",
            "sql_history": [
                {"sql": "SELECT 1", "db_id": "toy", "source": "execute", "executed": True}
            ],
        },
        session_id="s_rerun_oob",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "out of range" in tm.content


@pytest.mark.asyncio
async def test_rerun_falls_back_to_last_sql_when_no_history() -> None:
    seen_sql: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_sql.append(payload["sql"])
        return httpx.Response(
            200,
            json={
                "db_id": payload["db_id"],
                "sql": payload["sql"],
                "success": True,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "elapsed_s": 0.0,
            },
        )

    await _run_one_turn(
        replies=[_tool_call("rerun", {}, "c12"), AIMessage(content="ok")],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="again")],
            "active_db_id": "toy",
            "last_sql": "SELECT 7",
        },
        session_id="s_rerun_fallback",
    )

    assert seen_sql == ["SELECT 7"]


@pytest.mark.asyncio
async def test_rerun_propagates_upstream_failure_as_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "db_id": payload["db_id"],
                "sql": payload["sql"],
                "success": False,
                "error": "no such table: t",
                "error_code": "DB_ERROR",
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[_tool_call("rerun", {}, "c13"), AIMessage(content="bad")],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="r")],
            "active_db_id": "toy",
            "sql_history": [
                {"sql": "SELECT * FROM t", "db_id": "toy", "source": "execute", "executed": False}
            ],
        },
        session_id="s_rerun_fail",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "DB_ERROR" in tm.content
