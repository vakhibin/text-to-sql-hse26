"""Self-consistency voting: select best SQL by majority execution result."""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.tools.sql_candidate_analysis import (
    analyze_sql_candidate,
    summarize_candidate_analysis,
)


def _canonical_result(rows: list[tuple[Any, ...]] | None) -> tuple[tuple[Any, ...], ...]:
    """Convert execution rows into a hashable canonical form for comparison.

    Uses sorted string representation of rows so that column-order-preserving
    bag equality is sufficient for grouping.  ``None`` (failed execution) maps
    to a sentinel that will never match real output.
    """
    if rows is None:
        return (("__NO_RESULT__",),)
    return tuple(sorted(tuple(str(c) for c in row) for row in rows))


def _sql_simplicity_score(sql: str) -> tuple[int, int]:
    """Lower is simpler: (JOIN count, length). Counts JOIN tokens with word boundaries."""
    join_count = len(re.findall(r"\bJOIN\b", sql, flags=re.IGNORECASE))
    return (join_count, len(sql))


def _pick_best_from_group(sqls: list[str]) -> str:
    """Among SQLs that produce the same result, prefer the simplest one."""
    return min(sqls, key=_sql_simplicity_score)


def _build_voting_reasoning(
    groups: dict[tuple, list[str]],
    winner_key: tuple,
    total: int,
) -> str:
    winner_size = len(groups[winner_key])
    parts = [f"Majority voting: {winner_size}/{total} candidates produced the same result."]
    if len(groups) > 1:
        other_sizes = sorted(
            (len(sqls) for key, sqls in groups.items() if key != winner_key),
            reverse=True,
        )
        parts.append(f"Other groups: {other_sizes}.")
    return " ".join(parts)


async def run_voting(state: SQLAgentState) -> SQLAgentState:
    """Pick best SQL candidate via self-consistency majority voting.

    Groups valid candidates by their canonical execution result and selects
    the SQL from the largest group.  Ties are broken by SQL simplicity.
    Falls back to the first valid candidate when diagnostics lack rows.
    """
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))
    stage_status["voting"] = "running"

    preferred = state.get("valid_candidates", [])
    fallback = state.get("candidates", [])
    pool = preferred if preferred else fallback

    if not pool:
        stage_status["voting"] = "failed"
        return {
            **state,
            "best_sql": "",
            "selection_reasoning": "",
            "selection_confidence": "unknown",
            "selection_method": "voting",
            "selected_candidate_diagnostic": {},
            "error_message": "voting: no candidates available",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "voting": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, "voting: no candidates to evaluate"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    diagnostics = list(state.get("candidate_diagnostics", []))
    sql_to_rows: dict[str, list[tuple[Any, ...]] | None] = {}
    sql_to_diag: dict[str, dict[str, Any]] = {}
    for diag in diagnostics:
        sql = diag.get("sql", "")
        if sql in pool:
            sql_to_rows[sql] = diag.get("execution_rows")
            sql_to_diag[sql] = diag

    groups: dict[tuple, list[str]] = {}
    for sql in pool:
        key = _canonical_result(sql_to_rows.get(sql))
        groups.setdefault(key, []).append(sql)

    winner_key = max(groups, key=lambda k: len(groups[k]))
    winner_sqls = groups[winner_key]
    best_sql = _pick_best_from_group(winner_sqls)
    reasoning = _build_voting_reasoning(groups, winner_key, len(pool))

    majority_ratio = len(winner_sqls) / len(pool) if pool else 0.0
    if majority_ratio >= 0.6:
        confidence = "high"
    elif majority_ratio >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    needs_refine = confidence == "low"

    selected_diag = sql_to_diag.get(best_sql, {})
    if not selected_diag:
        analysis = analyze_sql_candidate(best_sql)
        selected_diag = {
            "sql": best_sql,
            "execution_success": True,
            "analysis": analysis,
            "analysis_summary": summarize_candidate_analysis(analysis),
        }

    stage_status["voting"] = "success"
    return {
        **state,
        "best_sql": best_sql,
        "selection_reasoning": reasoning,
        "selection_confidence": confidence,
        "selection_method": "voting",
        "selection_needs_refine": needs_refine,
        "selected_candidate_diagnostic": selected_diag,
        "stage_status": stage_status,
        "stage_timings": {
            **stage_timings,
            "voting": round(time.perf_counter() - started, 4),
        },
        "warnings": warnings,
        "llm_usage": llm_usage,
        "total_cost_usd": total_cost_usd,
    }
