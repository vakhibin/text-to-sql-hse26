"""Execution filter node for validating generated SQL candidates."""

from __future__ import annotations

import asyncio
import time

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.tools.sql_executor import execute_sql


async def run_execution_filter(state: SQLAgentState) -> SQLAgentState:
    """Execute candidate SQL queries and keep only valid ones."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    stage_status["execution_filter"] = "running"

    candidates = state.get("candidates", [])
    if not candidates:
        stage_status["execution_filter"] = "success"
        return {
            **state,
            "valid_candidates": [],
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "execution_filter": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "execution_filter: no candidates to validate"],
        }

    db_path = str(state.get("full_schema", {}).get("db_path", "")).strip()
    if not db_path:
        stage_status["execution_filter"] = "failed"
        return {
            **state,
            "valid_candidates": [],
            "error_message": "execution_filter: db_path is missing in full_schema",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "execution_filter": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "execution_filter: db_path missing"],
        }

    async def _run_one(sql: str):
        result = await execute_sql(
            db_path,
            sql,
            timeout_seconds=settings.execution_timeout_seconds,
        )
        return sql, result

    try:
        results = await asyncio.gather(*[_run_one(sql) for sql in candidates])
        valid_candidates: list[str] = []
        failed = 0
        for sql, execution in results:
            if execution.success:
                valid_candidates.append(sql)
            else:
                failed += 1
                if execution.error:
                    warnings.append(f"execution_filter: {execution.error}")

        if failed > 0:
            warnings.append(f"execution_filter: {failed}/{len(candidates)} candidates failed")
        if not valid_candidates:
            warnings.append("execution_filter: no valid candidates after execution")

        stage_status["execution_filter"] = "success"
        return {
            **state,
            "valid_candidates": valid_candidates,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "execution_filter": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
        }
    except Exception as exc:
        stage_status["execution_filter"] = "failed"
        return {
            **state,
            "valid_candidates": [],
            "error_message": str(exc),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "execution_filter": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"execution_filter_error: {exc}"],
        }

