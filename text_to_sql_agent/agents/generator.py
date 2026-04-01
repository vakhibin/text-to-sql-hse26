"""Ensemble generator agent."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Sequence

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.generator import build_generator_prompt
from text_to_sql_agent.tools.few_shot import load_few_shot_pool, retrieve_examples_for_candidate
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
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))
    stage_status["generator"] = "running"

    try:
        router = LLMRouter()
        pool = await load_few_shot_pool()
        if not pool:
            warnings.append("generator: few-shot pool unavailable; using zero-shot prompts")

        complexity = state.get("complexity", "unknown")
        num_candidates, roles = router.generator_plan_for_complexity(complexity)
        if complexity == "moderate":
            warnings.append(
                "generator: using reduced ensemble budget for moderate complexity"
            )

        async def _run_one(idx: int) -> tuple[str, dict[str, object]]:
            role = roles[idx % len(roles)]
            examples = await retrieve_examples_for_candidate(
                pool=pool,
                question=state["question"],
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
                query_sketch_text=state.get("query_sketch_text", ""),
                few_shot_examples=examples,
            )
            response = await router.ainvoke_with_metadata(
                role=role,
                messages=_build_messages(prompt),
                trace_id=state.get("trace_id"),
                db_id=state.get("db_id"),
                stage="generator",
            )
            return _extract_sql(response.text), response.usage

        generation_results = await asyncio.gather(*[_run_one(i) for i in range(num_candidates)])
        candidates = [sql for sql, _usage in generation_results if sql]
        llm_usage.extend(usage for _sql, usage in generation_results)
        total_cost_usd += sum(float(usage.get("cost_usd", 0.0)) for _sql, usage in generation_results)
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
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
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
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

