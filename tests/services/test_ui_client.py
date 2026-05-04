"""Tests for the Streamlit UI HTTP client and helpers."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from services.ui.client import (
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_TEXT_TO_SQL_URL,
    OrchestratorUIClient,
    OrchestratorUIError,
    TextToSQLUIClient,
    TextToSQLUIError,
    database_options,
    format_history_label,
    latest_result_from_session,
    latest_run_meta_from_session,
    normalize_base_url,
    normalize_text_to_sql_url,
    schema_tables,
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


def test_latest_run_meta_from_session_normalizes_pipeline_metadata() -> None:
    session = {
        "extra": {
            "last_run_meta": {
                "trace_id": "trace-1",
                "stage_status": {"selector": "success", "generator": "failed"},
                "warnings": ["selector: fallback"],
                "cost_usd": "0.1234",
                "elapsed_s": "2.5",
                "executed": True,
                "error": None,
            }
        }
    }

    meta = latest_run_meta_from_session(session)

    assert meta == {
        "trace_id": "trace-1",
        "stage_status": {"selector": "success", "generator": "failed"},
        "warnings": ["selector: fallback"],
        "cost_usd": 0.1234,
        "elapsed_s": 2.5,
        "executed": True,
        "error": None,
    }


def test_latest_run_meta_from_session_handles_missing_or_malformed_meta() -> None:
    assert latest_run_meta_from_session(None)["stage_status"] == {}
    assert latest_run_meta_from_session({"extra": {"last_run_meta": []}}) == {
        "trace_id": None,
        "stage_status": {},
        "warnings": [],
        "cost_usd": 0.0,
        "elapsed_s": 0.0,
        "executed": False,
        "error": None,
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


def test_normalize_text_to_sql_url() -> None:
    assert normalize_text_to_sql_url(" http://localhost:8001/ ") == "http://localhost:8001"
    assert normalize_text_to_sql_url("") == DEFAULT_TEXT_TO_SQL_URL
    assert normalize_text_to_sql_url(None) == DEFAULT_TEXT_TO_SQL_URL


def test_database_options_normalizes_and_sorts() -> None:
    catalog = {
        "schema_root": "/data",
        "databases": [
            {"db_id": "world_1", "num_tables": 3},
            {"db_id": "concert_singer", "num_tables": "4"},
            {"db_id": "", "num_tables": 1},
            "garbage",
            {"db_id": "car_1", "num_tables": None},
        ],
    }
    options = database_options(catalog)
    assert [opt["db_id"] for opt in options] == ["car_1", "concert_singer", "world_1"]
    by_id = {opt["db_id"]: opt for opt in options}
    assert by_id["concert_singer"]["num_tables"] == 4
    assert by_id["car_1"]["num_tables"] == 0
    assert by_id["concert_singer"]["label"] == "concert_singer (4 tables)"


def test_database_options_handles_empty_or_missing_catalog() -> None:
    assert database_options(None) == []
    assert database_options({}) == []
    assert database_options({"databases": []}) == []


def test_schema_tables_normalizes_columns_and_keys() -> None:
    schema = {
        "db_id": "toy",
        "tables": [
            {
                "name": "students",
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "name", "type": "TEXT"},
                ],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            {
                "name": "courses",
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "student_id", "type": "INTEGER"},
                ],
                "primary_keys": ["id"],
                "foreign_keys": [
                    {"column": "student_id", "ref_table": "students", "ref_column": "id"}
                ],
            },
            {"name": "", "columns": []},
        ],
    }
    tables = schema_tables(schema)
    assert [t["name"] for t in tables] == ["courses", "students"]
    courses = next(t for t in tables if t["name"] == "courses")
    assert courses["primary_keys"] == ["id"]
    assert courses["foreign_keys"] == [
        {"column": "student_id", "ref_table": "students", "ref_column": "id"}
    ]
    students = next(t for t in tables if t["name"] == "students")
    assert students["columns"] == [
        {"name": "id", "type": "INTEGER"},
        {"name": "name", "type": "TEXT"},
    ]


def test_schema_tables_handles_missing_or_malformed_payload() -> None:
    assert schema_tables(None) == []
    assert schema_tables({}) == []
    assert schema_tables({"tables": "garbage"}) == []


def test_text_to_sql_client_lists_databases() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/databases"
        return _json_response(
            {
                "schema_root": "/data/spider",
                "databases": [
                    {"db_id": "concert_singer", "num_tables": 4},
                    {"db_id": "car_1", "num_tables": 6},
                ],
            }
        )

    client = TextToSQLUIClient("http://test", transport=httpx.MockTransport(handler))
    try:
        catalog = client.list_databases()
    finally:
        client.close()
    assert catalog["schema_root"] == "/data/spider"
    assert {d["db_id"] for d in catalog["databases"]} == {"concert_singer", "car_1"}


def test_text_to_sql_client_get_schema_returns_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/databases/toy/schema"
        return _json_response(
            {
                "db_id": "toy",
                "tables": [
                    {
                        "name": "students",
                        "columns": [{"name": "id", "type": "INTEGER"}],
                        "primary_keys": ["id"],
                        "foreign_keys": [],
                    }
                ],
            }
        )

    client = TextToSQLUIClient("http://test", transport=httpx.MockTransport(handler))
    try:
        schema = client.get_schema("toy")
    finally:
        client.close()
    assert schema is not None
    assert schema["db_id"] == "toy"
    assert schema["tables"][0]["name"] == "students"


def test_text_to_sql_client_get_schema_404_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"detail":"unknown"}')

    client = TextToSQLUIClient("http://test", transport=httpx.MockTransport(handler))
    try:
        assert client.get_schema("missing") is None
    finally:
        client.close()


def test_text_to_sql_client_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = TextToSQLUIClient("http://test", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(TextToSQLUIError) as exc_info:
            client.list_databases()
    finally:
        client.close()
    assert "HTTP 500" in str(exc_info.value)
