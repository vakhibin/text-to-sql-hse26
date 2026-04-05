"""Execution filter node for validating generated SQL candidates."""

from __future__ import annotations

import asyncio
import re
import time

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.tools.sql_executor import execute_sql
from text_to_sql_agent.tools.sql_candidate_analysis import (
    analyze_sql_candidate,
    summarize_candidate_analysis,
)
from text_to_sql_agent.tools.sql_schema_validator import validate_sql_schema_references

_REFUSAL_PATTERNS = re.compile(
    r"^\s*SELECT\s+'[^']*("
    r"cannot|can't|unable|sorry|not possible|no answer|impossible"
    r")[^']*'\s*;?\s*$",
    re.IGNORECASE,
)


def _is_refusal_sql(sql: str) -> bool:
    """Detect SQL that is really an LLM refusal wrapped in a SELECT literal."""
    if _REFUSAL_PATTERNS.match(sql):
        return True
    cleaned = sql.strip().rstrip(";").strip()
    upper = cleaned.upper()
    if upper.startswith("SELECT") and "FROM" not in upper:
        tokens = cleaned.split(None, 1)
        if len(tokens) == 2 and tokens[1].startswith(("'", '"')):
            return True
    return False


def _eligible_for_simple_cheap_path(
    state: SQLAgentState,
    candidate_diagnostic: dict[str, object] | None,
) -> bool:
    """Allow the simple cheap path based on actual SQL structure, not LLM risk flags."""
    if not candidate_diagnostic:
        return False
    if not bool(candidate_diagnostic.get("execution_success")):
        return False
    if candidate_diagnostic.get("schema_errors"):
        return False

    analysis = candidate_diagnostic.get("analysis", {})
    if not isinstance(analysis, dict):
        return False
    if analysis.get("parse_error"):
        return False
    if int(analysis.get("join_count", 0)) > 0:
        return False
    if bool(analysis.get("has_subquery")):
        return False
    if str(analysis.get("set_operation", "none")) != "none":
        return False
    return True


def _simple_candidate_score(state: SQLAgentState, candidate_diagnostic: dict[str, object]) -> tuple[object, ...]:
    """Prefer the simplest valid candidate that matches the requested output width."""
    risk_flags = state.get("decomposition_risk_flags", {})
    target_projection_width = int(risk_flags.get("projection_width", 1) or 1)
    analysis = candidate_diagnostic.get("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}
    projection_count = int(analysis.get("projection_count", 0) or 0)
    return (
        abs(projection_count - target_projection_width),
        int(analysis.get("join_count", 0) or 0),
        int(analysis.get("table_count", 0) or 0),
        bool(analysis.get("has_aggregation")),
        bool(analysis.get("has_group_by")),
        bool(analysis.get("has_subquery")),
        str(analysis.get("set_operation", "none")) != "none",
        len(candidate_diagnostic.get("schema_warnings", [])),
        int(candidate_diagnostic.get("candidate_index", 0) or 0),
    )


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
        candidate_diagnostics: list[dict[str, object]] = []
        failed = 0
        refusals = 0
        for idx, (sql, execution) in enumerate(results):
            if _is_refusal_sql(sql):
                refusals += 1
                candidate_diagnostics.append({
                    "candidate_index": idx,
                    "sql": sql,
                    "execution_success": False,
                    "execution_error": "refusal_sql",
                    "schema_valid": False,
                    "schema_errors": ["refusal_sql"],
                    "schema_warnings": [],
                    "analysis": analyze_sql_candidate(sql),
                    "analysis_summary": "refusal",
                })
                continue

            validation = validate_sql_schema_references(sql, state.get("full_schema", {}))
            analysis = analyze_sql_candidate(sql)
            diagnostic = {
                "candidate_index": idx,
                "sql": sql,
                "execution_success": execution.success,
                "execution_error": execution.error or "",
                "schema_valid": validation.is_valid,
                "schema_errors": validation.errors,
                "schema_warnings": validation.warnings,
                "analysis": analysis,
                "analysis_summary": summarize_candidate_analysis(analysis),
            }
            candidate_diagnostics.append(diagnostic)
            if execution.success:
                valid_candidates.append(sql)
            else:
                failed += 1
                if execution.error:
                    warnings.append(f"execution_filter: {execution.error}")

        if refusals:
            warnings.append(f"execution_filter: {refusals}/{len(candidates)} candidates were refusal SQL")

        if failed > 0:
            warnings.append(f"execution_filter: {failed}/{len(candidates)} candidates failed")
        if not valid_candidates:
            warnings.append("execution_filter: no valid candidates after execution")

        best_sql = state.get("best_sql", "")
        judge_reasoning = state.get("judge_reasoning", "")
        valid_diagnostics = [
            diagnostic for diagnostic in candidate_diagnostics if bool(diagnostic.get("execution_success"))
        ]
        safe_simple_diagnostics = [
            diagnostic
            for diagnostic in valid_diagnostics
            if _eligible_for_simple_cheap_path(
                state,
                diagnostic,
            )
        ]
        if (
            state.get("complexity") == "simple"
            and settings.simple_skip_judge_when_valid
            and safe_simple_diagnostics
        ):
            best_diagnostic = min(
                safe_simple_diagnostics,
                key=lambda diagnostic: _simple_candidate_score(state, diagnostic),
            )
            best_sql = str(best_diagnostic.get("sql", valid_candidates[0]))
            judge_reasoning = "Skipped judge for safe simple query after execution validation."
            stage_status["judge"] = "skipped"
            warnings.append("execution_filter: skipped judge for safe simple query with valid candidate")

        stage_status["execution_filter"] = "success"
        return {
            **state,
            "valid_candidates": valid_candidates,
            "candidate_diagnostics": candidate_diagnostics,
            "best_sql": best_sql,
            "judge_reasoning": judge_reasoning,
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

