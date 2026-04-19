"""FastAPI application exposing the text-to-SQL pipeline as a microservice.

Endpoints:

- ``GET  /health``                          — liveness probe
- ``GET  /databases``                       — list available databases
- ``GET  /databases/{db_id}/schema``        — schema dump for a database
- ``POST /run``                             — run the full LangGraph pipeline
- ``POST /execute``                         — execute a given SQL (read-only)
- ``POST /refine``                          — refine a given SQL (AST repair + tool-augmented LLM)
- ``POST /explain``                         — generate a natural-language explanation of SQL

HTTP status codes:

- ``200`` — request processed; check ``success``/``error`` in the body
- ``404`` — database or resource not found
- ``422`` — request validation error (FastAPI default)
- ``500`` — unexpected server failure
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from services.text_to_sql_api import databases as db_service
from services.text_to_sql_api import pipeline_adapter as pipeline
from services.text_to_sql_api.schemas import (
    DatabasesResponse,
    ExecuteRequest,
    ExecuteResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    RefineRequest,
    RefineResponse,
    RunRequest,
    RunResponse,
    SchemaResponse,
)

SERVICE_NAME = "text_to_sql_api"
SERVICE_VERSION = "0.1.0"

app = FastAPI(title="Text-to-SQL API", version=SERVICE_VERSION)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


# ---- databases -----------------------------------------------------------


@app.get("/databases", response_model=DatabasesResponse)
async def list_databases(
    schema_root: str | None = Query(default=None, description="Override Spider/BIRD root."),
) -> DatabasesResponse:
    return await db_service.get_databases(schema_root=schema_root)


@app.get("/databases/{db_id}/schema", response_model=SchemaResponse)
async def get_database_schema(
    db_id: str,
    schema_root: str | None = Query(default=None, description="Override Spider/BIRD root."),
) -> SchemaResponse:
    schema = await db_service.get_schema(db_id, schema_root=schema_root)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"db_id {db_id!r} not found")
    return schema


# ---- execute -------------------------------------------------------------


def _serialize_rows(rows: list[Any] | None) -> list[list[Any]] | None:
    if rows is None:
        return None
    serialized: list[list[Any]] = []
    for row in rows:
        try:
            serialized.append([_coerce_cell(v) for v in row])
        except TypeError:
            serialized.append([str(row)])
    return serialized


def _coerce_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return str(value)
    except Exception:
        return repr(value)


@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    started = time.perf_counter()
    result, _ = await pipeline.execute_sql_read_only(
        sql=req.sql,
        db_id=req.db_id,
        schema_root=req.schema_root,
        timeout_seconds=req.timeout_seconds,
    )
    elapsed = round(time.perf_counter() - started, 4)

    if result.success:
        rows = _serialize_rows(result.rows)
        return ExecuteResponse(
            db_id=req.db_id,
            sql=req.sql,
            success=True,
            rows=rows,
            columns=result.columns,
            row_count=len(rows) if rows is not None else 0,
            elapsed_s=elapsed,
        )

    error_code: str = "DB_ERROR"
    error_msg = result.error or "unknown error"
    if "guardrail" in error_msg.lower() or "write sql rejected" in error_msg.lower():
        error_code = "READ_ONLY_VIOLATION"
    elif "timeout" in error_msg.lower():
        error_code = "TIMEOUT"

    return ExecuteResponse(
        db_id=req.db_id,
        sql=req.sql,
        success=False,
        error=error_msg,
        error_code=error_code,  # type: ignore[arg-type]
        elapsed_s=elapsed,
    )


# ---- run -----------------------------------------------------------------


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    started = time.perf_counter()
    try:
        state = await pipeline.run_pipeline(
            question=req.question,
            db_id=req.db_id,
            evidence=req.evidence,
            schema_root=req.schema_root,
            trace_id=req.trace_id,
        )
    except Exception as exc:  # pragma: no cover - last-resort safety
        elapsed = round(time.perf_counter() - started, 4)
        return RunResponse(
            trace_id=req.trace_id or "",
            db_id=req.db_id,
            question=req.question,
            sql=None,
            executed=False,
            error=f"pipeline error: {exc}",
            elapsed_s=elapsed,
        )

    final_sql = (state.get("final_sql") or state.get("best_sql") or "").strip() or None
    error_msg = state.get("error_message")
    warnings = list(state.get("warnings") or [])
    stage_status = {k: str(v) for k, v in (state.get("stage_status") or {}).items()}
    cost_usd = float(state.get("total_cost_usd") or 0.0)
    trace_id = str(state.get("trace_id") or req.trace_id or "")

    rows: list[list[Any]] | None = None
    columns: list[str] | None = None
    row_count: int | None = None
    executed = False

    if final_sql:
        exec_result, _ = await pipeline.execute_sql_read_only(
            sql=final_sql,
            db_id=req.db_id,
            schema_root=req.schema_root,
        )
        if exec_result.success:
            executed = True
            rows = _serialize_rows(exec_result.rows)
            columns = exec_result.columns
            row_count = len(rows) if rows is not None else 0
        else:
            if not error_msg:
                error_msg = exec_result.error

    elapsed = round(time.perf_counter() - started, 4)
    return RunResponse(
        trace_id=trace_id,
        db_id=req.db_id,
        question=req.question,
        sql=final_sql,
        executed=executed,
        rows=rows,
        columns=columns,
        row_count=row_count,
        error=error_msg,
        stage_status=stage_status,
        warnings=warnings,
        cost_usd=cost_usd,
        elapsed_s=elapsed,
    )


# ---- refine --------------------------------------------------------------


@app.post("/refine", response_model=RefineResponse)
async def refine(req: RefineRequest) -> RefineResponse:
    started = time.perf_counter()
    try:
        adapter_out = await pipeline.refine_sql_standalone(
            sql=req.sql,
            db_id=req.db_id,
            question=req.question,
            evidence=req.evidence,
            schema_root=req.schema_root,
            error_hint=req.error_hint,
            trace_id=req.trace_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    state = adapter_out["refined_state"]
    refined_sql = (state.get("final_sql") or state.get("best_sql") or req.sql).strip()
    error_msg = state.get("error_message")
    warnings = list(state.get("warnings") or [])
    cost_usd = float(state.get("total_cost_usd") or 0.0)
    executed = state.get("execution_result") is not None and not error_msg
    success = executed
    elapsed = round(time.perf_counter() - started, 4)

    return RefineResponse(
        trace_id=str(adapter_out["trace_id"]),
        db_id=req.db_id,
        original_sql=req.sql,
        refined_sql=refined_sql,
        changed=(refined_sql.strip() != req.sql.strip()),
        executed=executed,
        success=success,
        error=error_msg,
        warnings=warnings,
        cost_usd=cost_usd,
        elapsed_s=elapsed,
    )


# ---- explain -------------------------------------------------------------


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest) -> ExplainResponse:
    try:
        out = await pipeline.explain_sql(
            sql=req.sql,
            db_id=req.db_id,
            schema_root=req.schema_root,
            trace_id=req.trace_id,
        )
    except Exception as exc:  # pragma: no cover - surfaced to caller
        return JSONResponse(
            status_code=500,
            content={"detail": f"explain failed: {exc}"},
        )
    return ExplainResponse(
        trace_id=str(out["trace_id"]),
        sql=req.sql,
        explanation=str(out["explanation"]),
        cost_usd=float(out.get("cost_usd") or 0.0),
        elapsed_s=float(out.get("elapsed_s") or 0.0),
    )
