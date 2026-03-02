"""Judge agent: select best SQL from filtered candidate set."""

from __future__ import annotations

import json
import re
import time

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.judge import build_judge_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _parse_judge_response(response_text: str, n_candidates: int) -> tuple[int | None, str]:
    try:
        payload = json.loads(_extract_json_blob(response_text))
    except Exception:
        return None, "judge: failed to parse JSON response"

    raw_index = payload.get("best_index")
    if not isinstance(raw_index, int):
        return None, "judge: best_index missing or not int"
    if raw_index < 0 or raw_index >= n_candidates:
        return None, "judge: best_index out of range"

    reasoning = str(payload.get("reasoning", "")).strip()
    return raw_index, reasoning or "LLM chose this candidate as best."


async def run_judge(state: SQLAgentState) -> SQLAgentState:
    """Pick best SQL candidate using LLM-as-judge with fallback policy."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    stage_status["judge"] = "running"

    preferred = state.get("valid_candidates", [])
    fallback = state.get("candidates", [])
    pool = preferred if preferred else fallback
    if not pool:
        stage_status["judge"] = "failed"
        return {
            **state,
            "best_sql": "",
            "judge_reasoning": "",
            "error_message": "judge: no candidates available",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "judge": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "judge: no candidates to evaluate"],
        }

    try:
        prompt = build_judge_prompt(
            question=state["question"],
            filtered_schema=state.get("filtered_schema", ""),
            candidates=pool,
        )
        router = LLMRouter()
        response_text = await router.ainvoke(
            role=ModelRole.JUDGE,
            messages=[("system", "Return strict JSON only."), ("user", prompt)],
            temperature_override=0.0,
        )

        best_idx, reasoning = _parse_judge_response(response_text, len(pool))
        if best_idx is None:
            best_idx = 0
            warnings.append(reasoning)
            reasoning = "Fallback to first candidate due to judge parsing issue."

        stage_status["judge"] = "success"
        return {
            **state,
            "best_sql": pool[best_idx],
            "judge_reasoning": reasoning,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "judge": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
        }
    except Exception as exc:
        # Controlled fallback: keep pipeline moving with deterministic choice.
        stage_status["judge"] = "success"
        return {
            **state,
            "best_sql": pool[0],
            "judge_reasoning": "Fallback to first candidate due to judge invocation failure.",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "judge": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"judge_error: {exc}"],
        }

