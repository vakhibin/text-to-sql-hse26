"""Query-sketcher agent: schema-grounded plan before SQL generation."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.query_sketcher import build_query_sketcher_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(item).strip() for item in value if str(item).strip()]
    return items


def _normalize_table_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        table = str(item.get("table", "")).strip()
        if not table:
            continue
        normalized.append(
            {
                "table": table,
                "columns": _normalize_string_list(item.get("columns", [])),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return normalized


def _normalize_join_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        left_table = str(item.get("left_table", "")).strip()
        right_table = str(item.get("right_table", "")).strip()
        if not left_table or not right_table:
            continue
        normalized.append(
            {
                "left_table": left_table,
                "right_table": right_table,
                "join_keys": _normalize_string_list(item.get("join_keys", [])),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return normalized


def _format_query_sketch_text(sketch: dict[str, Any]) -> str:
    lines: list[str] = []
    intent = str(sketch.get("intent", "")).strip()
    task_summary = str(sketch.get("task_summary", "")).strip()
    if intent:
        lines.append(f"Intent: {intent}")
    if task_summary:
        lines.append(f"Task summary: {task_summary}")

    candidate_tables = sketch.get("candidate_tables", [])
    if candidate_tables:
        lines.append("Candidate tables:")
        for item in candidate_tables:
            table = item.get("table", "")
            columns = ", ".join(item.get("columns", []))
            reason = item.get("reason", "")
            detail = f"- {table}"
            if columns:
                detail += f" | columns: {columns}"
            if reason:
                detail += f" | reason: {reason}"
            lines.append(detail)

    join_plan = sketch.get("join_plan", [])
    if join_plan:
        lines.append("Join plan:")
        for item in join_plan:
            join_keys = ", ".join(item.get("join_keys", []))
            reason = item.get("reason", "")
            detail = f"- {item.get('left_table', '')} -> {item.get('right_table', '')}"
            if join_keys:
                detail += f" | keys: {join_keys}"
            if reason:
                detail += f" | reason: {reason}"
            lines.append(detail)

    for label, key in (
        ("Filters", "filters"),
        ("Aggregations", "aggregations"),
        ("Grouping", "grouping"),
        ("Ordering", "ordering"),
        ("Ambiguities", "ambiguities"),
        ("Risks", "risks"),
        ("Generation hints", "generation_hints"),
    ):
        items = sketch.get(key, [])
        if items:
            lines.append(f"{label}:")
            lines.extend(f"- {item}" for item in items)

    limit = str(sketch.get("limit", "")).strip()
    if limit:
        lines.append(f"Limit: {limit}")
    lines.append(f"Subquery needed: {bool(sketch.get('subquery_needed', False))}")
    set_operation = str(sketch.get("set_operation", "none")).strip()
    if set_operation:
        lines.append(f"Set operation: {set_operation}")
    return "\n".join(lines).strip()


def _parse_query_sketch(response_text: str) -> tuple[dict[str, Any], str, str | None]:
    try:
        payload = json.loads(_extract_json_blob(response_text))
    except Exception:
        return {}, "", "query_sketcher: failed to parse JSON response"

    sketch = {
        "intent": str(payload.get("intent", "other")).strip() or "other",
        "task_summary": str(payload.get("task_summary", "")).strip(),
        "candidate_tables": _normalize_table_items(payload.get("candidate_tables", [])),
        "join_plan": _normalize_join_items(payload.get("join_plan", [])),
        "filters": _normalize_string_list(payload.get("filters", [])),
        "aggregations": _normalize_string_list(payload.get("aggregations", [])),
        "grouping": _normalize_string_list(payload.get("grouping", [])),
        "ordering": _normalize_string_list(payload.get("ordering", [])),
        "limit": str(payload.get("limit", "none")).strip() or "none",
        "subquery_needed": bool(payload.get("subquery_needed", False)),
        "set_operation": str(payload.get("set_operation", "none")).strip() or "none",
        "ambiguities": _normalize_string_list(payload.get("ambiguities", [])),
        "risks": _normalize_string_list(payload.get("risks", [])),
        "generation_hints": _normalize_string_list(payload.get("generation_hints", [])),
    }
    return sketch, _format_query_sketch_text(sketch), None


async def run_query_sketcher(state: SQLAgentState) -> SQLAgentState:
    """Build a compact query-generation plan from question and schema."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))
    stage_status["sketcher"] = "running"

    try:
        prompt = build_query_sketcher_prompt(
            question=state["question"],
            evidence=state.get("evidence"),
            complexity=state.get("complexity", "unknown"),
            sub_questions=state.get("sub_questions", []),
            filtered_schema=state.get("filtered_schema", ""),
            retrieved_schema_context=state.get("retrieved_schema_context", ""),
        )
        router = LLMRouter()
        response = await router.ainvoke_with_metadata(
            role=ModelRole.GENERATOR_PRIMARY,
            messages=[
                ("system", "Return strict JSON only. Do not output SQL."),
                ("user", prompt),
            ],
            temperature_override=0.0,
            trace_id=state.get("trace_id"),
            db_id=state.get("db_id"),
            stage="sketcher",
        )
        llm_usage.append(response.usage)
        total_cost_usd += float(response.usage.get("cost_usd", 0.0))

        sketch, sketch_text, parse_warning = _parse_query_sketch(response.text)
        if parse_warning:
            warnings.append(parse_warning)

        if not sketch:
            warnings.append("query_sketcher: empty sketch, generator will fall back to schema + decomposition only")

        stage_status["sketcher"] = "success"
        return {
            **state,
            "query_sketch": sketch,
            "query_sketch_text": sketch_text,
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "sketcher": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }
    except Exception as exc:
        stage_status["sketcher"] = "failed"
        return {
            **state,
            "query_sketch": {},
            "query_sketch_text": "",
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "sketcher": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"query_sketcher_error: {exc}"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }
