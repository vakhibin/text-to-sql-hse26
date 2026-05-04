"""SQL endpoints: /run, /execute, /refine, /explain.

Shared row-serialisation helpers live here rather than in ``main`` so the
FastAPI app stays a thin composition root. All endpoints return HTTP 200
with structured success/error bodies; 404 is reserved for missing db_id.
"""

from __future__ import annotations

import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from services.text_to_sql_api import pipeline_adapter as pipeline
from services.text_to_sql_api.schemas import (
    ExecuteConfirmedRequest,
    ExecuteRequest,
    ExecuteResponse,
    ExplainRequest,
    ExplainResponse,
    ModifyRequest,
    ModifyResponse,
    RefineRequest,
    RefineResponse,
    RunRequest,
    RunResponse,
)

router = APIRouter(tags=["sql"])


def _coerce_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return str(value)
    except Exception:
        return repr(value)


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


def _selected_tables_from_mschema(mschema: str | None) -> list[str] | None:
    """Extract table names from compact mSchema lines (`table(col:type, ...)`)."""
    if not mschema:
        return None
    tables: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*)\(", mschema, re.MULTILINE):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            tables.append(name)
    return tables or None


def _classify_exec_error(message: str) -> str:
    lowered = message.lower()
    if "guardrail" in lowered or "write sql rejected" in lowered:
        return "READ_ONLY_VIOLATION"
    if "timeout" in lowered:
        return "TIMEOUT"
    return "DB_ERROR"


@router.post("/execute", response_model=ExecuteResponse)
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

    error_msg = result.error or "unknown error"
    return ExecuteResponse(
        db_id=req.db_id,
        sql=req.sql,
        success=False,
        error=error_msg,
        error_code=_classify_exec_error(error_msg),  # type: ignore[arg-type]
        elapsed_s=elapsed,
    )


@router.post("/execute-confirmed", response_model=ExecuteResponse)
async def execute_confirmed(req: ExecuteConfirmedRequest) -> ExecuteResponse:
    """Execute SQL after explicit user confirmation.

    This endpoint is intentionally separate from ``/execute`` so the default
    agent/tool path remains read-only. The request schema requires the literal
    confirmation marker ``USER_CONFIRMED_WRITE``.
    """
    started = time.perf_counter()
    result, _ = await pipeline.execute_sql_user_confirmed(
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
            read_only=False,
            rows=rows,
            columns=result.columns,
            row_count=len(rows) if rows is not None else 0,
            elapsed_s=elapsed,
        )

    error_msg = result.error or "unknown error"
    return ExecuteResponse(
        db_id=req.db_id,
        sql=req.sql,
        success=False,
        read_only=False,
        error=error_msg,
        error_code=_classify_exec_error(error_msg),  # type: ignore[arg-type]
        elapsed_s=elapsed,
    )


@router.post("/run", response_model=RunResponse)
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
    selected_tables = _selected_tables_from_mschema(str(state.get("filtered_schema") or ""))
    query_sketch_text = str(state.get("query_sketch_text") or "").strip() or None

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
        elif not error_msg:
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
        selected_tables=selected_tables,
        query_sketch_text=query_sketch_text,
        stage_status=stage_status,
        warnings=warnings,
        cost_usd=cost_usd,
        elapsed_s=elapsed,
        langfuse_trace_url=state.get("langfuse_trace_url"),
    )


@router.post("/refine", response_model=RefineResponse)
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
    except (FileNotFoundError, ValueError) as exc:
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


@router.post("/modify", response_model=ModifyResponse)
async def modify(req: ModifyRequest) -> ModifyResponse:
    try:
        out = await pipeline.modify_sql_standalone(
            sql=req.sql,
            instruction=req.instruction,
            db_id=req.db_id,
            schema_root=req.schema_root,
            trace_id=req.trace_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to caller
        return JSONResponse(
            status_code=500,
            content={"detail": f"modify failed: {exc}"},
        )
    return ModifyResponse(
        trace_id=str(out["trace_id"]),
        db_id=str(out["db_id"]),
        original_sql=str(out["original_sql"]),
        modified_sql=str(out["modified_sql"]),
        changed=bool(out["changed"]),
        cost_usd=float(out.get("cost_usd") or 0.0),
        elapsed_s=float(out.get("elapsed_s") or 0.0),
    )


@router.post("/explain", response_model=ExplainResponse)
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
