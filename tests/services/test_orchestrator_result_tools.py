"""Tests for deterministic result UX tools."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from orchestrator_agent import agent as agent_mod
from orchestrator_agent.graph import build_orchestrator_graph
from orchestrator_agent.memory import create_checkpointer
from orchestrator_agent.tools.results import make_result_tools


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


async def _run_one_turn(
    *,
    replies: list[AIMessage],
    input_state: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    handle = await create_checkpointer(sqlite_path=":memory:")
    agent_mod.set_chat_model(ScriptedToolCallLLM(replies))
    try:
        graph = build_orchestrator_graph(handle.saver, tools=make_result_tools())
        config = {"configurable": {"thread_id": session_id}}
        return await graph.ainvoke(input_state, config=config)
    finally:
        agent_mod.set_chat_model(None)
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
async def test_summarize_results_renders_latest_preview() -> None:
    state = await _run_one_turn(
        replies=[
            _tool_call("summarize_results", {"limit": 1}, "c1"),
            AIMessage(content="summary"),
        ],
        input_state={
            "messages": [HumanMessage(content="summarize")],
            "active_db_id": "toy",
            "last_sql": "SELECT name, age FROM students",
            "last_rows_columns": ["name", "age"],
            "last_rows_preview": [["Alice", 20], ["Bob", 21]],
            "last_row_count": 12,
        },
        session_id="s_result_summary",
    )

    tm = _last_tool_message(state)
    assert tm.status != "error"
    assert "Latest result summary" in tm.content
    assert "first 2 of 12 rows" in tm.content
    assert "| name | age |" in tm.content
    assert "Alice" in tm.content
    assert "Bob" not in tm.content


@pytest.mark.asyncio
async def test_summarize_results_without_rows_returns_error() -> None:
    state = await _run_one_turn(
        replies=[_tool_call("summarize_results", {}, "c2"), AIMessage(content="no rows")],
        input_state={"messages": [HumanMessage(content="summarize")]},
        session_id="s_result_summary_empty",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "No result rows" in tm.content


@pytest.mark.asyncio
async def test_export_results_csv_updates_last_export() -> None:
    state = await _run_one_turn(
        replies=[
            _tool_call("export_results", {"format": "csv"}, "c3"),
            AIMessage(content="csv"),
        ],
        input_state={
            "messages": [HumanMessage(content="export")],
            "last_rows_columns": ["name", "note"],
            "last_rows_preview": [["Alice", "a,b"], ["Bob", None]],
            "last_row_count": 2,
        },
        session_id="s_result_export_csv",
    )

    tm = _last_tool_message(state)
    assert tm.status != "error"
    assert "as csv" in tm.content
    assert "name,note" in tm.content
    assert 'Alice,"a,b"' in tm.content
    assert state["last_result_export"] == 'name,note\r\nAlice,"a,b"\r\nBob,'


@pytest.mark.asyncio
async def test_export_results_json_records() -> None:
    state = await _run_one_turn(
        replies=[
            _tool_call("export_results", {"format": "json"}, "c4"),
            AIMessage(content="json"),
        ],
        input_state={
            "messages": [HumanMessage(content="export json")],
            "last_rows_columns": ["n"],
            "last_rows_preview": [[1], [2]],
        },
        session_id="s_result_export_json",
    )

    tm = _last_tool_message(state)
    assert '"n": 1' in tm.content
    assert state["last_result_export"].startswith("[")


@pytest.mark.asyncio
async def test_export_results_bad_format_returns_error() -> None:
    state = await _run_one_turn(
        replies=[
            _tool_call("export_results", {"format": "xlsx"}, "c5"),
            AIMessage(content="bad format"),
        ],
        input_state={
            "messages": [HumanMessage(content="export xlsx")],
            "last_rows_columns": ["n"],
            "last_rows_preview": [[1]],
        },
        session_id="s_result_export_bad",
    )

    tm = _last_tool_message(state)
    assert tm.status == "error"
    assert "Unsupported export format" in tm.content
