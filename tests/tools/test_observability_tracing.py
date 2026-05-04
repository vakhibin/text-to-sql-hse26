"""Tests for Langfuse span helpers and the LangGraph stage-tracing wrapper.

These tests do not talk to a real Langfuse backend. They patch
``get_langfuse_client`` to return a stub so we can assert that:

- spans are opened with the expected name / metadata / input
- node return values are forwarded to ``span.update`` as ``output``
- node exceptions still propagate but the span is marked ERROR
- helpers degrade to no-ops when Langfuse is disabled
- ``safe_state_snapshot`` truncates strings/lists deterministically
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any

import pytest

from text_to_sql_agent.graph import tracing as tracing_module
from text_to_sql_agent.tools import observability as obs_module


# ---------------------------------------------------------------------------
# safe_state_snapshot
# ---------------------------------------------------------------------------


def test_safe_state_snapshot_truncates_long_strings() -> None:
    long_text = "x" * 5000
    out = obs_module.safe_state_snapshot({"sql": long_text}, max_str=100)
    assert isinstance(out, dict)
    assert out["sql"].startswith("x" * 100)
    assert "[+4900 chars]" in out["sql"]


def test_safe_state_snapshot_truncates_long_lists() -> None:
    candidates = [f"SELECT {i}" for i in range(120)]
    out = obs_module.safe_state_snapshot(
        {"candidates": candidates}, max_list=10, max_str=200
    )
    assert isinstance(out["candidates"], list)
    assert len(out["candidates"]) == 11  # 10 items + truncation marker
    assert out["candidates"][-1] == "... [+110 more]"


def test_safe_state_snapshot_handles_nested_mappings() -> None:
    payload = {
        "schema": {
            "students": {"columns": ["id", "name"]},
            "courses": {"columns": ["id", "title"]},
        },
        "trace_id": "abc",
        "missing": None,
    }
    out = obs_module.safe_state_snapshot(payload)
    assert out["schema"]["students"]["columns"] == ["id", "name"]
    assert out["trace_id"] == "abc"
    assert out["missing"] is None


def test_safe_state_snapshot_falls_back_to_str_for_non_serializable() -> None:
    class Custom:
        def __repr__(self) -> str:
            return "<Custom-instance>"

    out = obs_module.safe_state_snapshot({"obj": Custom()})
    assert out["obj"] == "<Custom-instance>"


# ---------------------------------------------------------------------------
# start_langfuse_span / update_langfuse_span (no client)
# ---------------------------------------------------------------------------


def test_start_langfuse_span_returns_nullcontext_when_client_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: None)
    span, ctx = obs_module.start_langfuse_span(
        name="selector", trace_id="t-1", input_payload={"q": "x"}
    )
    assert span is None
    with ctx:
        pass


def test_update_langfuse_span_no_op_on_none() -> None:
    obs_module.update_langfuse_span(None, output={"x": 1}, level="ERROR")


# ---------------------------------------------------------------------------
# start_langfuse_span with stubbed client
# ---------------------------------------------------------------------------


class _RecordingSpan:
    """Minimal LangfuseSpan stub that captures update() calls."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self.name = name
        self.params = params
        self.updates: list[dict[str, Any]] = []
        self.entered = False
        self.exited = False

    def __enter__(self) -> "_RecordingSpan":
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        self.exited = True
        return False

    def update(self, **kwargs: Any) -> "_RecordingSpan":
        self.updates.append(kwargs)
        return self


class _RecordingClient:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def start_as_current_observation(self, **params: Any):
        span = _RecordingSpan(name=params["name"], params=params)
        self.spans.append(span)
        return span


def test_start_langfuse_span_passes_name_input_and_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingClient()
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: client)

    span, ctx = obs_module.start_langfuse_span(
        name="selector",
        trace_id="trace-42",
        input_payload={"question": "hi"},
        metadata={"db_id": "toy"},
        as_type="span",
    )
    assert span is client.spans[0]
    with ctx:
        pass

    recorded = client.spans[0]
    assert recorded.params["name"] == "selector"
    assert recorded.params["as_type"] == "span"
    assert recorded.params["input"] == {"question": "hi"}
    assert recorded.params["metadata"] == {"session_id": "trace-42", "db_id": "toy"}


def test_update_langfuse_span_forwards_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingClient()
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: client)

    span, ctx = obs_module.start_langfuse_span(
        name="generator", trace_id=None, input_payload={"x": 1}
    )
    with ctx:
        pass
    obs_module.update_langfuse_span(
        span,
        output={"candidates": ["SELECT 1"]},
        level="ERROR",
        status_message="boom",
        metadata={"stage": "generator"},
    )

    assert client.spans[0].updates == [
        {
            "output": {"candidates": ["SELECT 1"]},
            "level": "ERROR",
            "status_message": "boom",
            "metadata": {"stage": "generator"},
        }
    ]


def test_start_langfuse_span_returns_nullcontext_when_client_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenClient:
        def start_as_current_observation(self, **_: Any):
            raise RuntimeError("langfuse exploded")

    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: _BrokenClient())
    span, ctx = obs_module.start_langfuse_span(
        name="x", trace_id="t", input_payload=None
    )
    assert span is None
    with ctx:
        pass


# ---------------------------------------------------------------------------
# trace_pipeline_stage wrapper
# ---------------------------------------------------------------------------


@contextmanager
def _patch_recording_client(monkeypatch: pytest.MonkeyPatch) -> _RecordingClient:
    client = _RecordingClient()
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: client)
    yield client


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question": "How many students?",
        "db_id": "toy",
        "trace_id": "trace-7",
        "stage_status": {"selector": "pending"},
    }
    state.update(overrides)
    return state


def test_trace_pipeline_stage_records_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingClient()
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: client)

    async def fake_node(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "filtered_schema": "students(id,name)",
            "stage_status": {"selector": "success"},
        }

    wrapped = tracing_module.trace_pipeline_stage("selector", fake_node)
    state = _base_state()
    result = asyncio.run(wrapped(state))

    assert result["filtered_schema"] == "students(id,name)"

    span = client.spans[0]
    assert span.params["name"] == "selector"
    assert span.params["input"]["question"] == "How many students?"
    assert span.params["input"]["db_id"] == "toy"
    assert span.params["metadata"]["stage"] == "selector"
    assert span.params["metadata"]["db_id"] == "toy"

    assert span.entered and span.exited
    assert len(span.updates) == 1
    update = span.updates[0]
    assert update["output"]["filtered_schema"] == "students(id,name)"
    assert update["metadata"]["stage_status"] == "success"


def test_trace_pipeline_stage_marks_error_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingClient()
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: client)

    async def boom(state: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("schema missing")

    wrapped = tracing_module.trace_pipeline_stage("selector", boom)
    with pytest.raises(ValueError, match="schema missing"):
        asyncio.run(wrapped(_base_state()))

    span = client.spans[0]
    assert len(span.updates) == 1
    update = span.updates[0]
    assert update["level"] == "ERROR"
    assert "ValueError" in update["status_message"]
    assert "schema missing" in update["status_message"]


def test_trace_pipeline_stage_runs_node_when_langfuse_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: None)

    async def fake_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"final_sql": "SELECT 1"}

    wrapped = tracing_module.trace_pipeline_stage("voting", fake_node)
    result = asyncio.run(wrapped(_base_state()))
    assert result == {"final_sql": "SELECT 1"}


def test_trace_pipeline_stage_filters_input_and_output_by_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-specific keys keep Langfuse panes lean and route service fields to metadata."""
    client = _RecordingClient()
    monkeypatch.setattr(obs_module, "get_langfuse_client", lambda: client)

    async def fake_selector(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "filtered_schema": "students(id,name)",
            "retrieved_schema_context": "ctx",
            "stage_status": {"selector": "success"},
            "stage_timings": {"selector": 0.42},
            "warnings": ["minor warning"],
            "llm_usage": [{"model": "x", "tokens": 100}],
        }

    wrapped = tracing_module.trace_pipeline_stage(
        "selector",
        fake_selector,
        input_keys=("question", "db_id", "evidence"),
        output_keys=("filtered_schema", "retrieved_schema_context"),
    )
    state = _base_state(evidence="extra hint", llm_usage=[])
    asyncio.run(wrapped(state))

    span = client.spans[0]

    # Input panel must be restricted to declared keys only.
    assert set(span.params["input"].keys()) == {"question", "db_id", "evidence"}

    # Output panel carries only stage-specific result fields.
    update = span.updates[0]
    assert set(update["output"].keys()) == {"filtered_schema", "retrieved_schema_context"}

    # Service fields the node also wrote land in metadata.patch_extras.
    extras = update["metadata"]["patch_extras"]
    assert set(extras.keys()) == {
        "stage_status",
        "stage_timings",
        "warnings",
        "llm_usage",
    }
    assert extras["stage_timings"] == {"selector": 0.42}
    assert update["metadata"]["stage_status"] == "success"
