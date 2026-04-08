"""Decomposer agent: complexity classification + sub-question decomposition."""

from __future__ import annotations

import json
import re
import time

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import ComplexityLevel, SQLAgentState
from text_to_sql_agent.prompts.decomposer import build_decomposer_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole

_VALID_COMPLEXITIES: set[ComplexityLevel] = {"simple", "moderate", "complex", "unknown"}
_DEFAULT_RISK_FLAGS = {
    "needs_join": False,
    "needs_aggregation": False,
    "projection_width": 1,
    "literal_filter_risk": False,
    "bag_semantics_risk": False,
}


def _extract_json_blob(text: str) -> str:
    """Extract likely JSON payload from raw model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _normalize_risk_flags(raw_flags: object) -> dict[str, object]:
    """Normalize decomposition risk flags with conservative defaults."""
    flags = dict(_DEFAULT_RISK_FLAGS)
    if not isinstance(raw_flags, dict):
        return flags

    for key in ("needs_join", "needs_aggregation", "literal_filter_risk", "bag_semantics_risk"):
        value = raw_flags.get(key)
        if isinstance(value, bool):
            flags[key] = value

    projection_width = raw_flags.get("projection_width")
    if isinstance(projection_width, int):
        flags["projection_width"] = max(1, min(projection_width, 12))
    elif isinstance(projection_width, float):
        flags["projection_width"] = max(1, min(int(projection_width), 12))
    return flags


def _parse_decomposition(
    response_text: str,
) -> tuple[ComplexityLevel, list[str], dict[str, object], str, str | None]:
    """Parse LLM response into normalized decomposition fields."""
    try:
        blob = _extract_json_blob(response_text)
        payload = json.loads(blob)
    except Exception:
        return "unknown", [], dict(_DEFAULT_RISK_FLAGS), "", "decomposer: failed to parse JSON response"

    raw_complexity = str(payload.get("complexity", "unknown")).lower().strip()
    complexity: ComplexityLevel = raw_complexity if raw_complexity in _VALID_COMPLEXITIES else "unknown"  # type: ignore[assignment]

    raw_questions = payload.get("sub_questions", [])
    sub_questions = []
    if isinstance(raw_questions, list):
        sub_questions = [str(item).strip() for item in raw_questions if str(item).strip()]

    risk_flags = _normalize_risk_flags(payload.get("risk_flags"))
    reasoning = str(payload.get("reasoning", "")).strip()

    if complexity == "complex" and not sub_questions:
        return (
            "complex",
            [],
            risk_flags,
            reasoning,
            "decomposer: complex classified but no sub_questions returned",
        )

    if complexity == "simple":
        return "simple", [], risk_flags, reasoning, None

    return complexity, sub_questions, risk_flags, reasoning, None


async def run_decomposer(state: SQLAgentState) -> SQLAgentState:
    """Classify complexity and decompose question into sub-questions."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))

    if not settings.decomposer_enabled:
        stage_status["decomposer"] = "skipped"
        return {
            **state,
            "complexity": "unknown",
            "sub_questions": [],
            "decomposition_reasoning": "",
            "decomposition_risk_flags": dict(_DEFAULT_RISK_FLAGS),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "decomposer": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "decomposer: skipped via DECOMPOSER_ENABLED=false"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    stage_status["decomposer"] = "running"

    try:
        prompt = build_decomposer_prompt(
            question=state["question"],
            evidence=state.get("evidence"),
        )
        router = LLMRouter()
        response = await router.ainvoke_with_metadata(
            role=ModelRole.GENERATOR_PRIMARY,
            messages=[
                ("system", "Return strict JSON only."),
                ("user", prompt),
            ],
            temperature_override=0.0,
            trace_id=state.get("trace_id"),
            db_id=state.get("db_id"),
            stage="decomposer",
        )
        response_text = response.text
        llm_usage.append(response.usage)
        total_cost_usd += float(response.usage.get("cost_usd", 0.0))

        complexity, sub_questions, risk_flags, reasoning, parse_warning = _parse_decomposition(response_text)
        if parse_warning:
            warnings.append(parse_warning)

        stage_status["decomposer"] = "success"
        return {
            **state,
            "complexity": complexity,
            "sub_questions": sub_questions,
            "decomposition_reasoning": reasoning,
            "decomposition_risk_flags": risk_flags,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "decomposer": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }
    except Exception as exc:
        stage_status["decomposer"] = "failed"
        return {
            **state,
            "complexity": "unknown",
            "sub_questions": [],
            "decomposition_reasoning": "",
            "decomposition_risk_flags": dict(_DEFAULT_RISK_FLAGS),
            "error_message": str(exc),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "decomposer": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"decomposer_error: {exc}"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

