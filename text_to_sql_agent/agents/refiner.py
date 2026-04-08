"""Refiner agent: execute best SQL and iteratively fix on failures."""

from __future__ import annotations

import re
import time

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.refiner import build_refiner_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole
from text_to_sql_agent.tools.sql_candidate_analysis import (
    analyze_sql_candidate,
    summarize_candidate_analysis,
)
from text_to_sql_agent.tools.sql_schema_validator import validate_sql_schema_references
from text_to_sql_agent.tools.sql_executor import execute_sql


def _extract_sql(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned and not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


async def run_refiner(state: SQLAgentState) -> SQLAgentState:
    """Run final SQL execution and prepare next refinement attempt if needed."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))
    stage_status["refiner"] = "running"

    db_path = str(state.get("full_schema", {}).get("db_path", "")).strip()
    if not db_path:
        stage_status["refiner"] = "failed"
        return {
            **state,
            "error_message": "refiner: db_path is missing in full_schema",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "refiner": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "refiner: db_path missing"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    current_sql = state.get("final_sql") or state.get("best_sql", "")
    if not current_sql:
        stage_status["refiner"] = "failed"
        return {
            **state,
            "error_message": "refiner: no SQL available to execute",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "refiner": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "refiner: empty SQL input"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    selected_candidate_diagnostic = dict(state.get("selected_candidate_diagnostic", {}))
    selected_candidate_summary = str(
        selected_candidate_diagnostic.get("analysis_summary")
        or summarize_candidate_analysis(analyze_sql_candidate(current_sql))
    )
    judge_issues = list(state.get("judge_issues", []))
    failed_candidate_summaries = [
        f"candidate {item.get('candidate_index', '-')}: "
        f"execution_error={item.get('execution_error', '-')} | "
        f"schema_errors={item.get('schema_errors', [])} | "
        f"schema_warnings={item.get('schema_warnings', [])} | "
        f"{item.get('analysis_summary', '-')}"
        for item in state.get("candidate_diagnostics", [])
        if (
            not bool(item.get("execution_success"))
            or not bool(item.get("schema_valid", True))
        )
    ][:3]

    validation = validate_sql_schema_references(current_sql, state.get("full_schema", {}))
    if validation.warnings:
        warnings.append(validation.warning_message())
    if not validation.is_valid:
        execution = None
        warnings.append(validation.error_message())
        execution_error = validation.error_message()
    else:
        execution = await execute_sql(
            db_path,
            current_sql,
            timeout_seconds=settings.execution_timeout_seconds,
        )
        execution_error = execution.error or "Unknown execution error."

    should_attempt_refine = execution is None or not execution.success

    if execution and execution.success and not should_attempt_refine:
        stage_status["refiner"] = "success"
        return {
            **state,
            "final_sql": current_sql,
            "execution_result": str(execution.rows),
            "error_message": None,
            "judge_needs_refine": False,
            "judge_issues": [],
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "refiner": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    # Prepare one refinement step after schema validation or execution failure.
    attempts = int(state.get("refine_attempts", 0)) + 1
    next_sql = current_sql
    refine_trigger = execution_error

    try:
        prompt = build_refiner_prompt(
            question=state.get("question", ""),
            evidence=state.get("evidence"),
            filtered_schema=state.get("filtered_schema", ""),
            retrieved_schema_context=state.get("retrieved_schema_context", ""),
            query_sketch_text=state.get("query_sketch_text", ""),
            failed_sql=current_sql,
            execution_error=refine_trigger,
            judge_reasoning=state.get("judge_reasoning", ""),
            judge_confidence=state.get("judge_confidence", "unknown"),
            judge_issues=judge_issues,
            validation_errors=validation.errors,
            validation_warnings=validation.warnings,
            selected_candidate_summary=selected_candidate_summary,
            failed_candidate_summaries=failed_candidate_summaries,
        )
        router = LLMRouter()
        response = await router.ainvoke_with_metadata(
            role=ModelRole.REFINER,
            messages=[("system", "Return only corrected SQL."), ("user", prompt)],
            temperature_override=0.0,
            trace_id=state.get("trace_id"),
            db_id=state.get("db_id"),
            stage="refiner",
        )
        llm_usage.append(response.usage)
        total_cost_usd += float(response.usage.get("cost_usd", 0.0))
        parsed = _extract_sql(response.text)
        if parsed:
            next_sql = parsed
        else:
            warnings.append("refiner: LLM returned empty fix, keeping previous SQL")
    except Exception as exc:  # pragma: no cover - network/runtime path
        warnings.append(f"refiner_error: {exc}")

    stage_status["refiner"] = "success"
    return {
        **state,
        "final_sql": next_sql,
        "execution_result": None,
        "refine_attempts": attempts,
        "error_message": refine_trigger,
        "judge_needs_refine": False,
        "judge_issues": [],
        "stage_status": stage_status,
        "stage_timings": {
            **stage_timings,
            "refiner": round(time.perf_counter() - started, 4),
        },
        "warnings": warnings,
        "llm_usage": llm_usage,
        "total_cost_usd": total_cost_usd,
    }

