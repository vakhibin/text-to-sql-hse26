"""Decomposer agent: complexity classification + sub-question decomposition."""

from __future__ import annotations

import json
import re
import time

from text_to_sql_agent.graph.state import ComplexityLevel, SQLAgentState
from text_to_sql_agent.prompts.decomposer import build_decomposer_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole

_VALID_COMPLEXITIES: set[ComplexityLevel] = {"simple", "moderate", "complex", "unknown"}


def _extract_json_blob(text: str) -> str:
    """Extract likely JSON payload from raw model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _parse_decomposition(response_text: str) -> tuple[ComplexityLevel, list[str], str | None]:
    """Parse LLM response into normalized decomposition fields."""
    try:
        blob = _extract_json_blob(response_text)
        payload = json.loads(blob)
    except Exception:
        return "unknown", [], "decomposer: failed to parse JSON response"

    raw_complexity = str(payload.get("complexity", "unknown")).lower().strip()
    complexity: ComplexityLevel = raw_complexity if raw_complexity in _VALID_COMPLEXITIES else "unknown"  # type: ignore[assignment]

    raw_questions = payload.get("sub_questions", [])
    sub_questions = []
    if isinstance(raw_questions, list):
        sub_questions = [str(item).strip() for item in raw_questions if str(item).strip()]

    if complexity == "complex" and not sub_questions:
        return "complex", [], "decomposer: complex classified but no sub_questions returned"

    if complexity == "simple":
        return "simple", [], None

    return complexity, sub_questions, None


async def run_decomposer(state: SQLAgentState) -> SQLAgentState:
    """Classify complexity and decompose question into sub-questions."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    stage_status["decomposer"] = "running"

    try:
        prompt = build_decomposer_prompt(
            question=state["question"],
            evidence=state.get("evidence"),
        )
        router = LLMRouter()
        response_text = await router.ainvoke(
            role=ModelRole.GENERATOR_PRIMARY,
            messages=[
                ("system", "Return strict JSON only."),
                ("user", prompt),
            ],
            temperature_override=0.0,
        )

        complexity, sub_questions, parse_warning = _parse_decomposition(response_text)
        if parse_warning:
            warnings.append(parse_warning)

        stage_status["decomposer"] = "success"
        return {
            **state,
            "complexity": complexity,
            "sub_questions": sub_questions,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "decomposer": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
        }
    except Exception as exc:
        stage_status["decomposer"] = "failed"
        return {
            **state,
            "complexity": "unknown",
            "sub_questions": [],
            "error_message": str(exc),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "decomposer": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"decomposer_error: {exc}"],
        }

