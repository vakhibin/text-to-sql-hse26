"""End-to-end tests for the orchestrator's discovery tools.

Mirror of ``test_orchestrator_tools.py`` but scoped to tools built by
``make_discovery_tools``. Each test drives a one-turn mini-graph with a
scripted LLM and an ``httpx.MockTransport`` — no real network, no real LLM.
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
from orchestrator_agent.tools.discovery import make_discovery_tools


class ScriptedToolCallLLM(BaseChatModel):
    """Emits a pre-scripted sequence of AIMessages; see core-tools test docstring."""

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
        tools = make_discovery_tools(client)
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


# ---- list_databases ------------------------------------------------------


@pytest.mark.asyncio
async def test_list_databases_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/databases"
        return httpx.Response(
            200,
            json={
                "schema_root": "/fake/root",
                "databases": [
                    {"db_id": "toy", "db_path": None, "num_tables": 3},
                    {"db_id": "shop", "db_path": None, "num_tables": 5},
                ],
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("list_databases", {}, "c1"),
            AIMessage(content="picked toy"),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="what dbs?")]},
        session_id="s_list",
    )

    tm = _last_tool_message(state)
    assert tm.name == "list_databases"
    assert "toy" in tm.content
    assert "shop" in tm.content
    assert "3 tables" in tm.content


# ---- describe_database ---------------------------------------------------


_SCHEMA_PAYLOAD = {
    "db_id": "toy",
    "db_path": None,
    "tables": [
        {
            "name": "users",
            "columns": [
                {"name": "id", "type": "INTEGER", "sample_values": []},
                {"name": "name", "type": "TEXT", "sample_values": []},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
        },
        {
            "name": "orders",
            "columns": [
                {"name": "id", "type": "INTEGER", "sample_values": []},
                {"name": "user_id", "type": "INTEGER", "sample_values": []},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [
                {"column": "user_id", "ref_table": "users", "ref_column": "id"}
            ],
        },
    ],
}


@pytest.mark.asyncio
async def test_describe_database_uses_active_db_from_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/databases/toy/schema"
        return httpx.Response(200, json=_SCHEMA_PAYLOAD)

    state = await _run_one_turn(
        replies=[
            _tool_call("describe_database", {}, "c2"),
            AIMessage(content="ok"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="what is in the db?")],
            "active_db_id": "toy",
        },
        session_id="s_describe_active",
    )

    tm = _last_tool_message(state)
    assert "users" in tm.content
    assert "orders" in tm.content
    assert "PK=[id]" in tm.content
    assert "FK=[user_id->users.id]" in tm.content


@pytest.mark.asyncio
async def test_describe_database_404_hints_list_databases() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    state = await _run_one_turn(
        replies=[
            _tool_call("describe_database", {"db_id": "ghost"}, "c3"),
            AIMessage(content="missing"),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="describe ghost")]},
        session_id="s_describe_404",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "ghost" in tm.content
    assert "list_databases" in tm.content


@pytest.mark.asyncio
async def test_describe_database_does_not_change_active_db() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_SCHEMA_PAYLOAD)

    state = await _run_one_turn(
        replies=[
            _tool_call("describe_database", {"db_id": "toy"}, "c4"),
            AIMessage(content="ok"),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="peek")]},
        session_id="s_describe_no_switch",
    )

    assert state.get("active_db_id") is None


# ---- switch_database -----------------------------------------------------


@pytest.mark.asyncio
async def test_switch_database_updates_state_and_clears_last_sql() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/databases/toy/schema"
        return httpx.Response(200, json=_SCHEMA_PAYLOAD)

    state = await _run_one_turn(
        replies=[
            _tool_call("switch_database", {"db_id": "toy"}, "c5"),
            AIMessage(content="switched"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="use toy")],
            "active_db_id": "other",
            "last_sql": "SELECT 1",
            "last_rows_preview": [[1]],
            "last_rows_columns": ["x"],
        },
        session_id="s_switch_ok",
    )

    assert state["active_db_id"] == "toy"
    assert state.get("last_sql") is None
    assert state.get("last_rows_preview") is None
    assert state.get("last_rows_columns") is None
    tm = _last_tool_message(state)
    assert "Active database is now 'toy'" in tm.content
    assert "users" in tm.content and "orders" in tm.content


@pytest.mark.asyncio
async def test_switch_database_404_does_not_touch_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    state = await _run_one_turn(
        replies=[
            _tool_call("switch_database", {"db_id": "ghost"}, "c6"),
            AIMessage(content="missed"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="use ghost")],
            "active_db_id": "keep_me",
        },
        session_id="s_switch_404",
    )

    assert state["active_db_id"] == "keep_me"
    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "ghost" in tm.content


# ---- sample_table --------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_table_happy_path() -> None:
    seen_sql: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/execute"
        payload = json.loads(request.content)
        seen_sql.append(payload["sql"])
        return httpx.Response(
            200,
            json={
                "db_id": "toy",
                "sql": payload["sql"],
                "success": True,
                "rows": [[1, "alice"], [2, "bob"]],
                "columns": ["id", "name"],
                "row_count": 2,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call("sample_table", {"table_name": "users", "limit": 2}, "c7"),
            AIMessage(content="seen"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="peek users")],
            "active_db_id": "toy",
        },
        session_id="s_sample_ok",
    )

    assert seen_sql == ["SELECT * FROM users LIMIT 2"]
    tm = _last_tool_message(state)
    assert tm.name == "sample_table"
    assert "alice" in tm.content
    assert "Rows: 2" in tm.content


@pytest.mark.asyncio
async def test_sample_table_rejects_unsafe_identifier() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach upstream on invalid identifier")

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "sample_table", {"table_name": "users; DROP TABLE t"}, "c8"
            ),
            AIMessage(content="rejected"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="evil")],
            "active_db_id": "toy",
        },
        session_id="s_sample_bad_ident",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "Invalid table name" in tm.content


@pytest.mark.asyncio
async def test_sample_table_errors_when_no_db_selected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach upstream when db_id missing")

    state = await _run_one_turn(
        replies=[
            _tool_call("sample_table", {"table_name": "users"}, "c9"),
            AIMessage(content="need db"),
        ],
        handler=handler,
        input_state={"messages": [HumanMessage(content="peek")]},
        session_id="s_sample_no_db",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "database" in tm.content.lower()


# ---- search_table_values -------------------------------------------------


@pytest.mark.asyncio
async def test_search_table_values_happy_path() -> None:
    seen_sql: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/execute"
        payload = json.loads(request.content)
        seen_sql.append(payload["sql"])
        return httpx.Response(
            200,
            json={
                "db_id": "toy",
                "sql": payload["sql"],
                "success": True,
                "rows": [["Alice"], ["Alicia"]],
                "columns": ["name"],
                "row_count": 2,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "search_table_values",
                {
                    "table_name": "users",
                    "column_name": "name",
                    "search_term": "ali",
                },
                "c10",
            ),
            AIMessage(content="found"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="lookup")],
            "active_db_id": "toy",
        },
        session_id="s_search_ok",
    )

    assert len(seen_sql) == 1
    sql = seen_sql[0]
    assert "SELECT DISTINCT name FROM users" in sql
    assert "LIKE '%ali%'" in sql
    tm = _last_tool_message(state)
    assert "Alice" in tm.content and "Alicia" in tm.content


@pytest.mark.asyncio
async def test_search_table_values_no_matches_is_reported_as_tool_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "db_id": "toy",
                "sql": payload["sql"],
                "success": True,
                "rows": [],
                "columns": ["name"],
                "row_count": 0,
                "elapsed_s": 0.0,
            },
        )

    state = await _run_one_turn(
        replies=[
            _tool_call(
                "search_table_values",
                {
                    "table_name": "users",
                    "column_name": "name",
                    "search_term": "zzz",
                },
                "c11",
            ),
            AIMessage(content="none"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="lookup")],
            "active_db_id": "toy",
        },
        session_id="s_search_empty",
    )

    tm = _last_tool_message(state)
    assert tm.status != "error"
    assert "No values" in tm.content
    assert "'zzz'" in tm.content


@pytest.mark.asyncio
async def test_search_table_values_escapes_single_quote() -> None:
    seen_sql: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_sql.append(payload["sql"])
        return httpx.Response(
            200,
            json={
                "db_id": "toy",
                "sql": payload["sql"],
                "success": True,
                "rows": [],
                "columns": ["name"],
                "row_count": 0,
                "elapsed_s": 0.0,
            },
        )

    await _run_one_turn(
        replies=[
            _tool_call(
                "search_table_values",
                {
                    "table_name": "users",
                    "column_name": "name",
                    "search_term": "O'Brien",
                },
                "c12",
            ),
            AIMessage(content="done"),
        ],
        handler=handler,
        input_state={
            "messages": [HumanMessage(content="lookup")],
            "active_db_id": "toy",
        },
        session_id="s_search_quote",
    )

    assert "'%O''Brien%'" in seen_sql[0]
