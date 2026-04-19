"""Tests for the orchestrator's HTTP client to ``text_to_sql_api``.

The suite runs against an ``httpx.MockTransport`` — no FastAPI app, no network
— so it pins the wire contract the orchestrator depends on: payload keys,
query params, and response-model validation.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from orchestrator_agent.clients.text_to_sql import (
    TextToSQLAPIError,
    TextToSQLClient,
)


def _json_response(body: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, json=body)


def _make_client(handler) -> TextToSQLClient:
    transport = httpx.MockTransport(handler)
    return TextToSQLClient(
        base_url="http://testserver", timeout_s=5.0, transport=transport
    )


@pytest.mark.asyncio
async def test_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return _json_response(
            {"status": "ok", "service": "text_to_sql_api", "version": "0.1.0"}
        )

    client = _make_client(handler)
    try:
        out = await client.health()
        assert out["service"] == "text_to_sql_api"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_databases_roundtrip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/databases"
        return _json_response(
            {
                "schema_root": "/tmp/spider",
                "databases": [
                    {"db_id": "concert_singer", "db_path": "/tmp/x.sqlite", "num_tables": 4}
                ],
            }
        )

    client = _make_client(handler)
    try:
        resp = await client.list_databases()
        assert resp.schema_root == "/tmp/spider"
        assert resp.databases[0].db_id == "concert_singer"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_run_sends_expected_payload() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content
        import json

        parsed = json.loads(request.content)
        captured["json"] = parsed
        return _json_response(
            {
                "trace_id": "tr1",
                "db_id": parsed["db_id"],
                "question": parsed["question"],
                "sql": "SELECT 1",
                "executed": True,
                "rows": [[1]],
                "columns": ["n"],
                "row_count": 1,
                "stage_status": {},
                "warnings": [],
                "cost_usd": 0.01,
                "elapsed_s": 0.1,
            }
        )

    client = _make_client(handler)
    try:
        resp = await client.run(question="how many?", db_id="toy", evidence="hint")
        assert captured["path"] == "/run"
        assert captured["json"] == {
            "question": "how many?",
            "db_id": "toy",
            "evidence": "hint",
        }
        assert resp.sql == "SELECT 1"
        assert resp.executed is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_execute_success_and_error_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        if payload["sql"].upper().startswith("SELECT"):
            return _json_response(
                {
                    "db_id": payload["db_id"],
                    "sql": payload["sql"],
                    "success": True,
                    "rows": [[1]],
                    "columns": ["x"],
                    "row_count": 1,
                    "elapsed_s": 0.01,
                }
            )
        return _json_response(
            {
                "db_id": payload["db_id"],
                "sql": payload["sql"],
                "success": False,
                "error": "Write SQL rejected by guardrail",
                "error_code": "READ_ONLY_VIOLATION",
                "elapsed_s": 0.0,
            }
        )

    client = _make_client(handler)
    try:
        ok = await client.execute(sql="SELECT 1", db_id="toy")
        assert ok.success is True
        bad = await client.execute(sql="UPDATE t SET x=1", db_id="toy")
        assert bad.success is False
        assert bad.error_code == "READ_ONLY_VIOLATION"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_explain_roundtrip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/explain"
        return _json_response(
            {
                "trace_id": "t1",
                "sql": "SELECT 1",
                "explanation": "Returns a constant.",
                "cost_usd": 0.0,
                "elapsed_s": 0.01,
            }
        )

    client = _make_client(handler)
    try:
        resp = await client.explain(sql="SELECT 1")
        assert "constant" in resp.explanation
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_http_error_is_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500, text="boom")

    client = _make_client(handler)
    try:
        with pytest.raises(TextToSQLAPIError) as exc_info:
            await client.run(question="q", db_id="x")
        assert exc_info.value.status_code == 500
        assert "boom" in str(exc_info.value)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_404_from_schema_is_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=404, text='{"detail":"not found"}')

    client = _make_client(handler)
    try:
        with pytest.raises(TextToSQLAPIError) as exc_info:
            await client.get_schema("missing")
        assert exc_info.value.status_code == 404
    finally:
        await client.aclose()
