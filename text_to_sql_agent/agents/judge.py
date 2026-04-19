"""Judge agent: select best SQL from filtered candidate set."""

from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel, Field

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.judge import build_judge_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole

_VALID_CONFIDENCE = {"low", "medium", "high"}
_VALID_ISSUES = {
    "projection_order_mismatch",
    "wrong_output_shape",
    "unnecessary_join",
    "literal_value_risk",
    "duplicate_row_risk",
    "aggregation_shape_risk",
    "table_selection_risk",
}


class JudgeStructuredOutput(BaseModel):
    best_index: int
    reasoning: str = ""
    confidence: str = "medium"
    needs_refine: bool = False
    issues: list[str] = Field(default_factory=list)


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _normalize_confidence(raw_value: object) -> str:
    value = str(raw_value or "medium").strip().lower()
    return value if value in _VALID_CONFIDENCE else "medium"


def _normalize_issues(raw_value: object) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    issues = []
    for item in raw_value:
        value = str(item).strip()
        if value in _VALID_ISSUES and value not in issues:
            issues.append(value)
    return issues


def _select_pool_diagnostics(
    pool: list[str],
    candidate_diagnostics: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split diagnostics into selectable candidates and rejected candidates."""
    selectable: list[dict[str, object]] = []
    remaining = list(candidate_diagnostics)
    for sql in pool:
        match_index = next(
            (idx for idx, diagnostic in enumerate(remaining) if diagnostic.get("sql") == sql),
            None,
        )
        if match_index is None:
            selectable.append(
                {
                    "candidate_index": len(selectable),
                    "sql": sql,
                    "execution_success": True,
                    "execution_error": "",
                    "schema_errors": [],
                    "schema_warnings": [],
                    "analysis": {},
                    "analysis_summary": "-",
                }
            )
            continue
        selectable.append(remaining.pop(match_index))
    rejected = [diagnostic for diagnostic in remaining if not bool(diagnostic.get("execution_success"))]
    return selectable, rejected


def _parse_judge_response(
    response_text: str,
    n_candidates: int,
) -> tuple[int | None, str, str, bool, list[str], str | None]:
    try:
        payload = json.loads(_extract_json_blob(response_text))
    except Exception:
        return None, "", "medium", False, [], "judge: failed to parse JSON response"

    raw_index = payload.get("best_index")
    if not isinstance(raw_index, int):
        return None, "", "medium", False, [], "judge: best_index missing or not int"
    if raw_index < 0 or raw_index >= n_candidates:
        return None, "", "medium", False, [], "judge: best_index out of range"

    reasoning = str(payload.get("reasoning", "")).strip()
    confidence = _normalize_confidence(payload.get("confidence"))
    needs_refine = bool(payload.get("needs_refine", False))
    issues = _normalize_issues(payload.get("issues"))
    return raw_index, reasoning or "LLM chose this candidate as best.", confidence, needs_refine, issues, None


async def run_judge(state: SQLAgentState) -> SQLAgentState:
    """Pick best SQL candidate using LLM-as-judge with fallback policy."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))
    stage_status["judge"] = "running"

    preferred = state.get("valid_candidates", [])
    fallback = state.get("candidates", [])
    pool = preferred if preferred else fallback
    selectable_candidates, _rejected = _select_pool_diagnostics(
        pool,
        list(state.get("candidate_diagnostics", [])),
    )
    if not pool:
        stage_status["judge"] = "failed"
        return {
            **state,
            "best_sql": "",
            "judge_reasoning": "",
            "judge_confidence": "unknown",
            "judge_needs_refine": False,
            "judge_issues": [],
            "selected_candidate_diagnostic": {},
            "error_message": "judge: no candidates available",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "judge": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "judge: no candidates to evaluate"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    try:
        prompt = build_judge_prompt(
            question=state["question"],
            evidence=state.get("evidence"),
            filtered_schema=state.get("filtered_schema", ""),
            selectable_candidates=selectable_candidates,
        )
        router = LLMRouter()
        response = await router.ainvoke_with_metadata(
            role=ModelRole.JUDGE,
            messages=[("system", "Return strict JSON only."), ("user", prompt)],
            temperature_override=0.0,
            trace_id=state.get("trace_id"),
            db_id=state.get("db_id"),
            stage="judge",
            structured_output=JudgeStructuredOutput,
        )
        response_text = response.text
        llm_usage.append(response.usage)
        total_cost_usd += float(response.usage.get("cost_usd", 0.0))

        if response.structured is not None:
            out = response.structured
            best_idx = out.best_index
            reasoning = str(out.reasoning or "").strip()
            confidence = _normalize_confidence(out.confidence)
            needs_refine = bool(out.needs_refine)
            issues = _normalize_issues(out.issues)
            parse_warning = None
            if not isinstance(best_idx, int) or best_idx < 0 or best_idx >= len(pool):
                parse_warning = "judge: best_index missing or out of range"
                best_idx = None
        else:
            best_idx, reasoning, confidence, needs_refine, issues, parse_warning = _parse_judge_response(
                response_text,
                len(pool),
            )
        if best_idx is None:
            best_idx = 0
            if parse_warning:
                warnings.append(parse_warning)
            reasoning = "Fallback to first candidate due to judge parsing issue."
            confidence = "low"
            needs_refine = False
            issues = []

        selected_candidate_diagnostic = (
            selectable_candidates[best_idx] if best_idx < len(selectable_candidates) else {}
        )
        needs_refine = needs_refine or bool(issues)

        stage_status["judge"] = "success"
        return {
            **state,
            "best_sql": pool[best_idx],
            "judge_reasoning": reasoning,
            "judge_confidence": confidence,
            "judge_needs_refine": needs_refine,
            "judge_issues": issues,
            "selected_candidate_diagnostic": selected_candidate_diagnostic,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "judge": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }
    except Exception as exc:
        # Controlled fallback: keep pipeline moving with deterministic choice.
        stage_status["judge"] = "success"
        return {
            **state,
            "best_sql": pool[0],
            "judge_reasoning": "Fallback to first candidate due to judge invocation failure.",
            "judge_confidence": "low",
            "judge_needs_refine": False,
            "judge_issues": [],
            "selected_candidate_diagnostic": selectable_candidates[0] if selectable_candidates else {},
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "judge": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"judge_error: {exc}"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

