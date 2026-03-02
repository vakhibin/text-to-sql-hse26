"""Ensemble generator agent."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Sequence

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.generator import build_generator_prompt
from text_to_sql_agent.tools.few_shot import load_few_shot_pool, sample_examples_for_candidate
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole


def _extract_sql(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned and not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


def _build_messages(prompt: str) -> Sequence[tuple[str, str]]:
    return [
        ("system", "Output only SQL."),
        ("user", prompt),
    ]


async def run_generator(state: SQLAgentState) -> SQLAgentState:
    """Generate N SQL candidates asynchronously with 5/3 role split."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    stage_status["generator"] = "running"

    try:
        router = LLMRouter()
        pool = await load_few_shot_pool()
        if not pool:
            warnings.append("generator: few-shot pool unavailable; using zero-shot prompts")

        roles = router.generator_roles()
        num_candidates = settings.num_candidates

        async def _run_one(idx: int) -> str:
            role = roles[idx % len(roles)]
            examples = sample_examples_for_candidate(
                pool=pool,
                candidate_index=idx,
                k=settings.few_shot_examples_per_candidate,
                seed=settings.few_shot_seed,
                target_db_id=state.get("db_id"),
            )
            prompt = build_generator_prompt(
                question=state["question"],
                filtered_schema=state.get("filtered_schema", ""),
                complexity=state.get("complexity", "unknown"),
                sub_questions=state.get("sub_questions", []),
                few_shot_examples=examples,
            )
            response = await router.ainvoke(
                role=role,
                messages=_build_messages(prompt),
            )
            return _extract_sql(response)

        candidates_raw = await asyncio.gather(*[_run_one(i) for i in range(num_candidates)])
        candidates = [c for c in candidates_raw if c]
        if not candidates:
            warnings.append("generator: no SQL candidates produced")

        stage_status["generator"] = "success"
        return {
            **state,
            "candidates": candidates,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "generator": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
        }
    except Exception as exc:
        stage_status["generator"] = "failed"
        return {
            **state,
            "candidates": [],
            "error_message": str(exc),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "generator": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"generator_error: {exc}"],
        }

