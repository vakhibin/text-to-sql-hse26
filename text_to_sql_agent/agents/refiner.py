"""Refiner agent: execute best SQL and iteratively fix on failures."""

from __future__ import annotations

import re
import time

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.refiner import build_refiner_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole
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
        }

    execution = await execute_sql(
        db_path,
        current_sql,
        timeout_seconds=settings.execution_timeout_seconds,
    )
    if execution.success:
        stage_status["refiner"] = "success"
        return {
            **state,
            "final_sql": current_sql,
            "execution_result": str(execution.rows),
            "error_message": None,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "refiner": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
        }

    # Execution failed: prepare one refinement step.
    attempts = int(state.get("refine_attempts", 0)) + 1
    execution_error = execution.error or "Unknown execution error."
    next_sql = current_sql

    try:
        prompt = build_refiner_prompt(
            question=state.get("question", ""),
            filtered_schema=state.get("filtered_schema", ""),
            failed_sql=current_sql,
            execution_error=execution_error,
        )
        router = LLMRouter()
        fixed = await router.ainvoke(
            role=ModelRole.REFINER,
            messages=[("system", "Return only corrected SQL."), ("user", prompt)],
            temperature_override=0.0,
        )
        parsed = _extract_sql(fixed)
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
        "error_message": execution_error,
        "stage_status": stage_status,
        "stage_timings": {
            **stage_timings,
            "refiner": round(time.perf_counter() - started, 4),
        },
        "warnings": warnings,
    }

