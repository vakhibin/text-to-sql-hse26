"""Query-sketcher agent: schema-grounded plan before SQL generation."""

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.query_sketcher import build_query_sketcher_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole
from text_to_sql_agent.tools.value_linker import format_column_hints, format_value_hints


class QuerySketchTableItem(BaseModel):
    table: str
    columns: list[str] = Field(default_factory=list)
    reason: str = ""


class QuerySketchJoinItem(BaseModel):
    left_table: str
    right_table: str
    join_keys: list[str] = Field(default_factory=list)
    reason: str = ""


class QuerySketchSchema(BaseModel):
    intent: str = "other"
    task_summary: str = ""
    candidate_tables: list[QuerySketchTableItem] = Field(default_factory=list)
    join_plan: list[QuerySketchJoinItem] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    aggregations: list[str] = Field(default_factory=list)
    grouping: list[str] = Field(default_factory=list)
    ordering: list[str] = Field(default_factory=list)
    limit: str = "none"
    subquery_needed: bool = False
    set_operation: str = "none"
    ambiguities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    generation_hints: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(
        default_factory=list,
        description="Question terms with no matching column in the provided schema (e.g. horsepower).",
    )


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _json_candidate_variants(text: str) -> list[str]:
    blob = _extract_json_blob(text)
    normalized = (
        blob.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .strip()
    )
    variants = [normalized]
    variants.append(re.sub(r",(\s*[}\]])", r"\1", normalized))
    variants.append(re.sub(r"\bTrue\b", "true", normalized))
    variants.append(re.sub(r"\bFalse\b", "false", normalized))
    variants.append(re.sub(r"\bNone\b", "null", normalized))
    variants.append(
        re.sub(
            r"\bNone\b",
            "null",
            re.sub(r"\bFalse\b", "false", re.sub(r"\bTrue\b", "true", variants[-1])),
        )
    )
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        value = variant.strip()
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _load_jsonish_payload(text: str) -> dict[str, Any] | None:
    for candidate in _json_candidate_variants(text):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        try:
            payload = ast.literal_eval(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


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

    missing = sketch.get("missing_entities", [])
    if missing:
        lines.append("Missing entities (no column in primary schema):")
        lines.extend(f"- {item}" for item in missing)

    limit = str(sketch.get("limit", "")).strip()
    if limit:
        lines.append(f"Limit: {limit}")
    lines.append(f"Subquery needed: {bool(sketch.get('subquery_needed', False))}")
    set_operation = str(sketch.get("set_operation", "none")).strip()
    if set_operation:
        lines.append(f"Set operation: {set_operation}")
    return "\n".join(lines).strip()


def _build_minimal_fallback_sketch(state: SQLAgentState) -> tuple[dict[str, Any], str]:
    filtered_schema = state.get("filtered_schema", "")
    table_names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", filtered_schema)
    unique_tables: list[str] = []
    for table in table_names:
        if table not in unique_tables:
            unique_tables.append(table)

    question = state.get("question", "").strip()
    question_lower = question.lower()
    candidate_tables = [
        {
            "table": table,
            "columns": [],
            "reason": "Selected schema table from selector context.",
        }
        for table in unique_tables[:3]
    ]

    aggregations: list[str] = []
    grouping: list[str] = []
    ordering: list[str] = []
    filters: list[str] = []
    generation_hints = [
        "Use only selected schema tables and columns.",
        "Prefer the simplest SQL that satisfies the question.",
    ]
    if any(keyword in question_lower for keyword in ("count", "many", "number of")):
        aggregations.append("count-related aggregation may be needed")
    if any(keyword in question_lower for keyword in ("average", "avg", "minimum", "maximum", "sum", "total")):
        aggregations.append("aggregate function likely needed")
    if "each" in question_lower or "per " in question_lower:
        grouping.append("grouping may be needed")
    if any(keyword in question_lower for keyword in ("order", "sorted", "highest", "lowest", "youngest", "oldest")):
        ordering.append("ordering may be needed")
    if any(keyword in question_lower for keyword in ("after", "before", "between", "not", "only", "from", "in")):
        filters.append("apply the question's filter conditions carefully")
    if candidate_tables:
        generation_hints.append("Start from the first candidate table unless the question clearly needs a join.")

    set_operation = "none"
    if "intersect" in question_lower or "both" in question_lower:
        set_operation = "intersect"
    elif "except" in question_lower or "not in" in question_lower:
        set_operation = "except"
    elif "union" in question_lower or "either" in question_lower:
        set_operation = "union"

    sketch = {
        "intent": "other",
        "task_summary": question or "Fallback sketch from question and selected schema.",
        "candidate_tables": candidate_tables,
        "join_plan": [],
        "filters": filters,
        "aggregations": aggregations,
        "grouping": grouping,
        "ordering": ordering,
        "limit": "none",
        "subquery_needed": set_operation != "none" or any(
            keyword in question_lower for keyword in ("most", "least", "than", "not", "both")
        ),
        "set_operation": set_operation,
        "ambiguities": [],
        "risks": ["Fallback sketch used because query-sketcher output was unavailable or malformed."],
        "generation_hints": generation_hints[:6],
        "missing_entities": [],
    }
    return sketch, _format_query_sketch_text(sketch)


def _normalize_query_sketch_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
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
        "missing_entities": _normalize_string_list(payload.get("missing_entities", [])),
    }
    return sketch, _format_query_sketch_text(sketch)


def _parse_query_sketch(response_text: str) -> tuple[dict[str, Any], str, str | None]:
    payload = _load_jsonish_payload(response_text)
    if payload is None:
        return {}, "", "query_sketcher: failed to parse JSON response"
    sketch, sketch_text = _normalize_query_sketch_payload(payload)[:2]
    return sketch, sketch_text, None


async def _repair_query_sketch_with_structured_output(
    *,
    router: LLMRouter,
    response_text: str,
    state: SQLAgentState,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    usage_records: list[dict[str, Any]] = []
    llm = router.get_chat_model(ModelRole.QUERY_SKETCHER, temperature_override=0.0)
    structured_llm = llm.with_structured_output(QuerySketchSchema, include_raw=True)
    repaired = await structured_llm.ainvoke(
        [
            ("system", "Convert the draft into valid structured output only. Do not output SQL."),
            (
                "user",
                "Normalize this draft query sketch into the target schema. "
                "Keep it compact, schema-grounded, and valid for the structured schema.\n\n"
                f"Question: {state.get('question', '')}\n"
                f"Selected schema:\n{state.get('filtered_schema', '')}\n\n"
                f"Draft sketch:\n{response_text}",
            ),
        ]
    )

    raw_response = repaired.get("raw") if isinstance(repaired, dict) else None
    parsed = repaired.get("parsed") if isinstance(repaired, dict) else None
    parsing_error = repaired.get("parsing_error") if isinstance(repaired, dict) else None
    if raw_response is not None:
        usage = router._extract_usage(
            response=raw_response,
            model_name=router.model_for_role(ModelRole.QUERY_SKETCHER),
            stage="sketcher_repair",
        )
        usage_records.append(usage.as_dict())
    if parsing_error is not None:
        warnings.append(f"query_sketcher: structured repair failed ({parsing_error})")
    if parsed is None:
        return {}, "", usage_records, warnings
    if isinstance(parsed, BaseModel):
        payload = parsed.model_dump()
    elif isinstance(parsed, dict):
        payload = parsed
    else:
        return {}, "", usage_records, [*warnings, "query_sketcher: structured repair returned no payload"]
    sketch, sketch_text = _normalize_query_sketch_payload(payload)[:2]
    return sketch, sketch_text, usage_records, warnings


async def run_query_sketcher(state: SQLAgentState) -> SQLAgentState:
    """Build a compact query-generation plan from question and schema."""
    from text_to_sql_agent.config import settings as _cfg

    if not _cfg.query_sketcher_enabled:
        return {
            "query_sketch": {},
            "query_sketch_text": "",
            "missing_entities": [],
            "stage_status": {**dict(state.get("stage_status", {})), "sketcher": "skipped"},
            "warnings": list(state.get("warnings", [])),
        }

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
            filtered_schema=state.get("filtered_schema", ""),
            retrieved_schema_context=state.get("retrieved_schema_context", ""),
            value_hints_text=format_value_hints(state.get("value_hints", [])),
            column_hints_text=format_column_hints(state.get("column_hints", [])),
        )
        router = LLMRouter()
        response = await router.ainvoke_with_metadata(
            role=ModelRole.QUERY_SKETCHER,
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
            try:
                repaired_sketch, repaired_text, repair_usage, repair_warnings = (
                    await _repair_query_sketch_with_structured_output(
                        router=router,
                        response_text=response.text,
                        state=state,
                    )
                )
                if repair_usage:
                    llm_usage.extend(repair_usage)
                    total_cost_usd += sum(float(item.get("cost_usd", 0.0)) for item in repair_usage)
                warnings.extend(repair_warnings)
                if repaired_sketch:
                    sketch, sketch_text = repaired_sketch, repaired_text
                    warnings.append("query_sketcher: repaired malformed sketch with structured-output fallback")
            except Exception as repair_exc:
                warnings.append(f"query_sketcher: structured repair unavailable ({repair_exc})")

        if not sketch:
            sketch, sketch_text = _build_minimal_fallback_sketch(state)
            warnings.append("query_sketcher: used deterministic fallback sketch")

        missing_entities = _normalize_string_list(sketch.get("missing_entities", []))
        sketch["missing_entities"] = missing_entities
        sketch_text = _format_query_sketch_text(sketch)
        loops = int(state.get("sketcher_selector_loops") or 0)
        if missing_entities and loops >= _cfg.max_sketcher_selector_recovery:
            warnings.append(
                "sketcher: missing_entities set but selector recovery budget exhausted; continuing to generator"
            )

        stage_status["sketcher"] = "success"
        return {
            **state,
            "query_sketch": sketch,
            "query_sketch_text": sketch_text,
            "missing_entities": missing_entities,
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
            "missing_entities": [],
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "sketcher": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"query_sketcher_error: {exc}"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }
