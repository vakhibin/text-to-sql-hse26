"""Contract tests for the text-to-sql FastAPI service.

Covers:

- ``/databases`` and ``/databases/{db_id}/schema`` (real schema loader, fake dataset)
- ``/execute`` (real executor + guardrail, fake SQLite)
- ``/run``, ``/refine``, ``/explain`` (pipeline adapter monkey-patched to
  avoid real LLM calls; we verify the HTTP contract, not pipeline behavior)
"""

from __future__ import annotations

import httpx
import pytest

from services.text_to_sql_api import pipeline_adapter
from text_to_sql_agent.tools.sql_executor import SQLExecutionResult


# ---- /databases ----------------------------------------------------------


@pytest.mark.asyncio
async def test_list_databases(api_client: httpx.AsyncClient, fake_schema_root: str) -> None:
    resp = await api_client.get("/databases", params={"schema_root": fake_schema_root})
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_root"].endswith("spider_fake")
    db_ids = {db["db_id"] for db in body["databases"]}
    assert "toy" in db_ids
    toy = next(db for db in body["databases"] if db["db_id"] == "toy")
    assert toy["num_tables"] == 2
    assert toy["db_path"] and toy["db_path"].endswith("toy.sqlite")


@pytest.mark.asyncio
async def test_get_schema(api_client: httpx.AsyncClient, fake_schema_root: str) -> None:
    resp = await api_client.get(
        "/databases/toy/schema", params={"schema_root": fake_schema_root}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_id"] == "toy"
    assert {t["name"] for t in body["tables"]} == {"students", "courses"}
    students = next(t for t in body["tables"] if t["name"] == "students")
    col_names = {c["name"] for c in students["columns"]}
    assert col_names == {"id", "name", "age"}
    assert students["primary_keys"] == ["id"]
    courses = next(t for t in body["tables"] if t["name"] == "courses")
    assert any(
        fk["column"] == "student_id"
        and fk["ref_table"] == "students"
        and fk["ref_column"] == "id"
        for fk in courses["foreign_keys"]
    )


@pytest.mark.asyncio
async def test_get_schema_unknown_db_returns_404(
    api_client: httpx.AsyncClient, fake_schema_root: str
) -> None:
    resp = await api_client.get(
        "/databases/not_real/schema", params={"schema_root": fake_schema_root}
    )
    assert resp.status_code == 404


# ---- /execute ------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_select_success(
    api_client: httpx.AsyncClient, fake_schema_root: str
) -> None:
    resp = await api_client.post(
        "/execute",
        json={
            "sql": "SELECT name, age FROM students ORDER BY id",
            "db_id": "toy",
            "schema_root": fake_schema_root,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["read_only"] is True
    assert body["columns"] == ["name", "age"]
    assert body["rows"] == [["Alice", 20], ["Bob", 22], ["Eve", 19]]
    assert body["row_count"] == 3


@pytest.mark.asyncio
async def test_execute_write_is_rejected(
    api_client: httpx.AsyncClient, fake_schema_root: str
) -> None:
    resp = await api_client.post(
        "/execute",
        json={
            "sql": "DELETE FROM students",
            "db_id": "toy",
            "schema_root": fake_schema_root,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "READ_ONLY_VIOLATION"
    assert body["rows"] is None

    check = await api_client.post(
        "/execute",
        json={
            "sql": "SELECT COUNT(*) FROM students",
            "db_id": "toy",
            "schema_root": fake_schema_root,
        },
    )
    assert check.status_code == 200
    assert check.json()["rows"] == [[3]]


@pytest.mark.asyncio
async def test_execute_confirmed_write_requires_marker(
    api_client: httpx.AsyncClient, fake_schema_root: str
) -> None:
    resp = await api_client.post(
        "/execute-confirmed",
        json={
            "sql": "DELETE FROM students WHERE id = 3",
            "db_id": "toy",
            "schema_root": fake_schema_root,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_execute_confirmed_write_mutates_database(
    api_client: httpx.AsyncClient, fake_schema_root: str
) -> None:
    resp = await api_client.post(
        "/execute-confirmed",
        json={
            "sql": "DELETE FROM students WHERE id = 3",
            "db_id": "toy",
            "schema_root": fake_schema_root,
            "confirmation": "USER_CONFIRMED_WRITE",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["read_only"] is False

    check = await api_client.post(
        "/execute",
        json={
            "sql": "SELECT COUNT(*) FROM students",
            "db_id": "toy",
            "schema_root": fake_schema_root,
        },
    )
    assert check.status_code == 200
    assert check.json()["rows"] == [[2]]


@pytest.mark.asyncio
async def test_execute_unknown_db(api_client: httpx.AsyncClient, fake_schema_root: str) -> None:
    resp = await api_client.post(
        "/execute",
        json={
            "sql": "SELECT 1",
            "db_id": "missing_db",
            "schema_root": fake_schema_root,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "DB_ERROR"
    assert "not found" in (body["error"] or "").lower()


@pytest.mark.asyncio
async def test_execute_bad_sql_surfaces_db_error(
    api_client: httpx.AsyncClient, fake_schema_root: str
) -> None:
    resp = await api_client.post(
        "/execute",
        json={
            "sql": "SELECT no_such_col FROM students",
            "db_id": "toy",
            "schema_root": fake_schema_root,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "DB_ERROR"


@pytest.mark.asyncio
async def test_execute_validation_error_422(api_client: httpx.AsyncClient) -> None:
    resp = await api_client.post("/execute", json={"sql": "", "db_id": "toy"})
    assert resp.status_code == 422


# ---- /run (pipeline mocked) ---------------------------------------------


@pytest.mark.asyncio
async def test_run_uses_pipeline_and_executes_final_sql(
    api_client: httpx.AsyncClient,
    fake_schema_root: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_pipeline(**kwargs: object) -> dict[str, object]:
        return {
            "final_sql": "SELECT name FROM students ORDER BY id LIMIT 1",
            "best_sql": "SELECT name FROM students ORDER BY id LIMIT 1",
            "error_message": None,
            "warnings": ["selector: trimmed to 1 table"],
            "stage_status": {"selector": "success", "generator": "success"},
            "total_cost_usd": 0.0123,
            "trace_id": "trace-xyz",
        }

    monkeypatch.setattr(pipeline_adapter, "run_pipeline", fake_run_pipeline)

    resp = await api_client.post(
        "/run",
        json={
            "question": "Who is the first student alphabetically?",
            "db_id": "toy",
            "schema_root": fake_schema_root,
            "trace_id": "trace-xyz",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == "trace-xyz"
    assert body["sql"] == "SELECT name FROM students ORDER BY id LIMIT 1"
    assert body["executed"] is True
    assert body["rows"] == [["Alice"]]
    assert body["columns"] == ["name"]
    assert body["row_count"] == 1
    assert body["cost_usd"] == pytest.approx(0.0123)
    assert body["stage_status"]["selector"] == "success"


@pytest.mark.asyncio
async def test_run_pipeline_error_is_surfaced(
    api_client: httpx.AsyncClient,
    fake_schema_root: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_pipeline(**kwargs: object) -> dict[str, object]:
        return {
            "final_sql": "",
            "best_sql": "",
            "error_message": "selector: no candidate tables",
            "warnings": [],
            "stage_status": {"selector": "failed"},
            "total_cost_usd": 0.0,
            "trace_id": "trace-fail",
        }

    monkeypatch.setattr(pipeline_adapter, "run_pipeline", fake_run_pipeline)

    resp = await api_client.post(
        "/run",
        json={
            "question": "???",
            "db_id": "toy",
            "schema_root": fake_schema_root,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sql"] is None
    assert body["executed"] is False
    assert body["error"] == "selector: no candidate tables"


# ---- /refine (adapter mocked) -------------------------------------------


@pytest.mark.asyncio
async def test_refine_returns_refined_sql(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_refine(**kwargs: object) -> dict[str, object]:
        return {
            "trace_id": "trace-ref",
            "refined_state": {
                "final_sql": "SELECT name FROM students WHERE age > 20",
                "best_sql": "SELECT name FROM students WHERE age > 20",
                "execution_result": "[(\"Bob\",)]",
                "error_message": None,
                "warnings": ["refiner: ast_repair applied 1 fix(es)"],
                "total_cost_usd": 0.002,
            },
        }

    monkeypatch.setattr(pipeline_adapter, "refine_sql_standalone", fake_refine)

    resp = await api_client.post(
        "/refine",
        json={
            "sql": "SELECT naem FROM students WHERE age > 20",
            "db_id": "toy",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_sql"] == "SELECT naem FROM students WHERE age > 20"
    assert body["refined_sql"] == "SELECT name FROM students WHERE age > 20"
    assert body["changed"] is True
    assert body["executed"] is True
    assert body["success"] is True
    assert body["trace_id"] == "trace-ref"


@pytest.mark.asyncio
async def test_refine_reports_failure(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_refine(**kwargs: object) -> dict[str, object]:
        return {
            "trace_id": "trace-ref-fail",
            "refined_state": {
                "final_sql": "SELECT 1",
                "best_sql": "SELECT 1",
                "execution_result": None,
                "error_message": "refiner: out of attempts",
                "warnings": [],
                "total_cost_usd": 0.001,
            },
        }

    monkeypatch.setattr(pipeline_adapter, "refine_sql_standalone", fake_refine)

    resp = await api_client.post(
        "/refine",
        json={"sql": "SELECT 1", "db_id": "toy"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["executed"] is False
    assert body["error"] == "refiner: out of attempts"


@pytest.mark.asyncio
async def test_refine_unknown_db_404(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_refine(**kwargs: object) -> dict[str, object]:
        raise ValueError("db_id 'missing' not found in tables.json")

    monkeypatch.setattr(pipeline_adapter, "refine_sql_standalone", fake_refine)

    resp = await api_client.post(
        "/refine",
        json={"sql": "SELECT 1", "db_id": "missing"},
    )
    assert resp.status_code == 404


# ---- /explain (adapter mocked) ------------------------------------------


@pytest.mark.asyncio
async def test_explain_returns_explanation(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_explain(**kwargs: object) -> dict[str, object]:
        return {
            "trace_id": "trace-exp",
            "explanation": "Returns the oldest student's name.",
            "cost_usd": 0.0007,
            "elapsed_s": 0.42,
        }

    monkeypatch.setattr(pipeline_adapter, "explain_sql", fake_explain)

    resp = await api_client.post(
        "/explain",
        json={
            "sql": "SELECT name FROM students ORDER BY age DESC LIMIT 1",
            "db_id": "toy",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == "trace-exp"
    assert "oldest" in body["explanation"].lower()
    assert body["cost_usd"] == pytest.approx(0.0007)


@pytest.mark.asyncio
async def test_modify_returns_modified_sql(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_modify(**kwargs: object) -> dict[str, object]:
        return {
            "trace_id": "trace-mod",
            "db_id": kwargs["db_id"],
            "original_sql": kwargs["sql"],
            "modified_sql": "SELECT * FROM students WHERE year = 2023",
            "changed": True,
            "cost_usd": 0.001,
            "elapsed_s": 0.3,
        }

    monkeypatch.setattr(pipeline_adapter, "modify_sql_standalone", fake_modify)

    resp = await api_client.post(
        "/modify",
        json={
            "sql": "SELECT * FROM students",
            "instruction": "only 2023",
            "db_id": "toy",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == "trace-mod"
    assert body["changed"] is True
    assert "2023" in body["modified_sql"]


@pytest.mark.asyncio
async def test_modify_unchanged_when_llm_returns_same_sql(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_modify(**kwargs: object) -> dict[str, object]:
        return {
            "trace_id": "t",
            "db_id": kwargs["db_id"],
            "original_sql": kwargs["sql"],
            "modified_sql": kwargs["sql"],
            "changed": False,
            "cost_usd": 0.0,
            "elapsed_s": 0.1,
        }

    monkeypatch.setattr(pipeline_adapter, "modify_sql_standalone", fake_modify)

    resp = await api_client.post(
        "/modify",
        json={"sql": "SELECT 1", "instruction": "no-op", "db_id": "toy"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["changed"] is False
    assert body["modified_sql"] == "SELECT 1"


@pytest.mark.asyncio
async def test_modify_unknown_db_returns_404(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_modify(**kwargs: object) -> dict[str, object]:
        raise ValueError("unknown db_id='ghost'")

    monkeypatch.setattr(pipeline_adapter, "modify_sql_standalone", fake_modify)

    resp = await api_client.post(
        "/modify",
        json={"sql": "SELECT 1", "instruction": "x", "db_id": "ghost"},
    )
    assert resp.status_code == 404


# ---- /modify: fence stripping unit ---------------------------------------


def test_strip_sql_fences_removes_markdown_block() -> None:
    text = "```sql\nSELECT 1\n```"
    assert pipeline_adapter._strip_sql_fences(text) == "SELECT 1"


def test_strip_sql_fences_trims_trailing_semicolon() -> None:
    assert pipeline_adapter._strip_sql_fences("SELECT 1 ;  ") == "SELECT 1"


def test_strip_sql_fences_passthrough_plain_sql() -> None:
    assert pipeline_adapter._strip_sql_fences("  SELECT 1  ") == "SELECT 1"


# ---- /execute helper sanity --------------------------------------------


def test_sql_execution_result_carries_columns() -> None:
    """Regression check: SQLExecutionResult now exposes a columns field."""
    result = SQLExecutionResult(success=True, rows=[(1,)], columns=["a"])
    assert result.columns == ["a"]
