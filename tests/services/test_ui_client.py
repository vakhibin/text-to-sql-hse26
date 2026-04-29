"""Tests for the Streamlit UI HTTP client and helpers."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from services.ui.client import (
    DEFAULT_ORCHESTRATOR_URL,
    OrchestratorUIClient,
    OrchestratorUIError,
    format_history_label,
    latest_result_from_session,
    normalize_base_url,
    session_extra,
    sql_history_from_session,
    visible_messages,
)


def _json_response(body: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, json=body)


def test_normalize_base_url() -> None:
    assert normalize_base_url(" http://localhost:8002/ ") == "http://localhost:8002"
    assert normalize_base_url("") == DEFAULT_ORCHESTRATOR_URL
    assert normalize_base_url(None) == DEFAULT_ORCHESTRATOR_URL


def test_visible_messages_filters_tool_and_empty_messages() -> None:
    out = visible_messages(
        [
            {"role": "human", "content": "hi"},
            {"role": "tool", "content": "SQL:\nSELECT 1"},
            {"role": "ai", "content": "hello"},
            {"role": "ai", "content": "   "},
        ]
    )
    assert out == [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "hello"},
    ]


def test_sql_history_from_session_newest_first() -> None:
    session = {
        "extra": {
            "sql_history": [
                {"sql": "SELECT 1", "source": "run"},
                {"sql": "SELECT 2", "source": "execute"},
            ]
        }
    }
    history = sql_history_from_session(session)
    assert [entry["sql"] for entry in history] == ["SELECT 2", "SELECT 1"]


def test_latest_result_from_session_normalizes_preview_rows() -> None:
    session = {
        "extra": {
            "last_rows_columns": ["name", "age"],
            "last_rows_preview": [["Ann", 30], ["Bob", 25]],
            "last_row_count": 2,
        }
    }

    result = latest_result_from_session(session)

    assert result == {
        "row_count": 2,
        "columns": ["name", "age"],
        "rows": [{"name": "Ann", "age": 30}, {"name": "Bob", "age": 25}],
    }


def test_session_extra_handles_missing_or_malformed_extra() -> None:
    assert session_extra(None) == {}
    assert session_extra({"extra": []}) == {}


def test_format_history_label_truncates_long_sql() -> None:
    label = format_history_label(
        1,
        {
            "source": "run",
            "db_id": "toy",
            "sql": "SELECT " + ", ".join(f"col_{i}" for i in range(30)),
        },
    )
    assert label.startswith("1. [run] toy: SELECT")
    assert label.endswith("...")
    assert len(label) < 110


def test_ui_client_chat_roundtrip() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        assert request.method == "POST"
        assert request.url.path == "/chat"
        captured["payload"] = json.loads(request.content)
        return _json_response(
            {
                "session_id": "s1",
                "user_id": "u1",
                "reply": "ok",
                "messages_delta": [
                    {"role": "human", "content": "hi"},
                    {"role": "ai", "content": "ok"},
                ],
                "active_db_id": "toy",
                "last_sql": "SELECT 1",
                "warnings": [],
            }
        )

    client = OrchestratorUIClient(
        "http://test", transport=httpx.MockTransport(handler)
    )
    try:
        response = client.chat(
            session_id="s1",
            user_id="u1",
            message="hi",
            active_db_id="toy",
        )
    finally:
        client.close()

    assert captured["payload"] == {
        "session_id": "s1",
        "user_id": "u1",
        "message": "hi",
        "active_db_id": "toy",
    }
    assert response["reply"] == "ok"


def test_ui_client_get_session_404_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"detail":"missing"}')

    client = OrchestratorUIClient(
        "http://test", transport=httpx.MockTransport(handler)
    )
    try:
        assert client.get_session("missing") is None
    finally:
        client.close()


def test_ui_client_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = OrchestratorUIClient(
        "http://test", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(OrchestratorUIError) as exc_info:
            client.health()
    finally:
        client.close()

    assert "HTTP 500" in str(exc_info.value)
