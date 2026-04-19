"""Value & column linker node: pre-generation grounding to DB schema and values."""

from __future__ import annotations

import re
import time

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.tools.value_linker import link_columns, link_values


def _extract_selected_tables(filtered_schema_mschema: str) -> list[str]:
    """Parse table names from mSchema text like 'TableName(col1 type, ...)'."""
    return re.findall(r"^(\w+)\(", filtered_schema_mschema, re.MULTILINE)


async def run_value_linker(state: SQLAgentState) -> SQLAgentState:
    """Link question entities to actual database values and schema columns."""
    started = time.perf_counter()
    warnings = list(state.get("warnings", []))

    full_schema = state.get("full_schema", {})
    if not full_schema:
        warnings.append("value_linker: skipped — no schema available")
        return {"value_hints": [], "column_hints": [], "warnings": warnings}

    question = state.get("question", "")
    evidence = state.get("evidence") or ""
    query = f"{question} {evidence}".strip() if evidence else question

    selected_tables = _extract_selected_tables(state.get("filtered_schema", ""))

    value_hint_dicts: list[dict] = []
    try:
        hints = link_values(
            question=query,
            full_schema=full_schema,
            selected_tables=selected_tables or None,
        )
        value_hint_dicts = [h.as_dict() for h in hints]
    except Exception as exc:
        warnings.append(f"value_linker: value error — {exc}")

    column_hint_dicts: list[dict] = []
    try:
        col_hints = link_columns(
            question=query,
            full_schema=full_schema,
            selected_tables=selected_tables or None,
        )
        column_hint_dicts = [h.as_dict() for h in col_hints]
    except Exception as exc:
        warnings.append(f"value_linker: column error — {exc}")

    if value_hint_dicts:
        warnings.append(f"value_linker: found {len(value_hint_dicts)} value hint(s)")
    if column_hint_dicts:
        warnings.append(f"value_linker: found {len(column_hint_dicts)} column hint(s)")

    return {
        "value_hints": value_hint_dicts,
        "column_hints": column_hint_dicts,
        "stage_timings": {
            **dict(state.get("stage_timings", {})),
            "value_linker": round(time.perf_counter() - started, 4),
        },
        "warnings": warnings,
    }
