"""Shared helpers used across tool families (core, discovery, history, results).

Kept private to the ``orchestrator_agent.tools`` package — nothing outside
should import from here. The helpers handle three concerns every tool cares
about:

- resolving ``db_id`` from explicit arg or session state,
- rendering a row preview compact enough for an LLM context window,
- emitting a ``Command`` that carries a ``ToolMessage`` (success or error)
  back to the agent.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

ROW_PREVIEW_LIMIT = 10
SQL_HISTORY_CAP = 20


def clear_result_artifacts() -> dict[str, Any]:
    """Return state updates that invalidate the current row result.

    SQL-producing tools that do not execute should call this so a later
    ``export_results`` cannot accidentally export rows from an older query.
    """
    return {
        "last_rows_preview": None,
        "last_rows_columns": None,
        "last_row_count": None,
        "last_result_export": None,
    }


def append_history(
    state: dict[str, Any] | None,
    *,
    sql: str,
    db_id: str | None,
    source: str,
    executed: bool,
    row_count: int | None = None,
    cap: int = SQL_HISTORY_CAP,
) -> list[dict[str, Any]]:
    """Return a new ``sql_history`` list with one entry appended and capped.

    Tools read the current list from injected state, call this, and include
    the returned list in their ``Command(update=...)``. There is no reducer
    so updates replace the whole list — read-append-write is sequential and
    safe because tool calls run one at a time.
    """
    current = list((state or {}).get("sql_history") or [])
    entry: dict[str, Any] = {
        "sql": sql,
        "db_id": db_id,
        "source": source,
        "executed": bool(executed),
    }
    if row_count is not None:
        entry["row_count"] = row_count
    current.append(entry)
    if len(current) > cap:
        current = current[-cap:]
    return current


def resolve_db_id(
    state: dict[str, Any] | None, explicit: str | None
) -> tuple[str | None, str | None]:
    """Return ``(db_id, error)`` — error is non-None when no db can be resolved."""
    if explicit:
        return explicit, None
    if state and state.get("active_db_id"):
        return state["active_db_id"], None
    return None, (
        "No database selected. Ask the user which database to use "
        "or call a discovery tool to list databases, then retry."
    )


def format_rows_preview(
    columns: list[str] | None,
    rows: list[list[Any]] | None,
    limit: int = ROW_PREVIEW_LIMIT,
) -> str:
    if not rows:
        return "(no rows)"
    header = " | ".join(columns) if columns else ""
    preview_rows = rows[:limit]
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in row) for row in preview_rows
    )
    truncated = "" if len(rows) <= limit else f"\n... ({len(rows) - limit} more rows)"
    return (f"{header}\n{body}" if header else body) + truncated


def format_markdown_table(
    columns: list[str] | None,
    rows: list[list[Any]] | None,
    *,
    limit: int | None = None,
) -> str:
    """Render rows as a compact Markdown table."""
    if not rows:
        return "(no rows)"
    preview = rows if limit is None else rows[:limit]
    col_count = len(columns or []) or max((len(row) for row in preview), default=0)
    header = list(columns or [f"col_{i + 1}" for i in range(col_count)])

    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in preview:
        padded = list(row) + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(cell(v) for v in padded[: len(header)]) + " |")
    if limit is not None and len(rows) > limit:
        lines.append(f"\n... ({len(rows) - limit} more rows in preview)")
    return "\n".join(lines)


def format_csv(
    columns: list[str] | None,
    rows: list[list[Any]] | None,
) -> str:
    """Render rows as CSV text."""
    output = io.StringIO()
    writer = csv.writer(output)
    if columns:
        writer.writerow(columns)
    for row in rows or []:
        writer.writerow(["" if value is None else value for value in row])
    return output.getvalue().strip()


def format_json_rows(
    columns: list[str] | None,
    rows: list[list[Any]] | None,
) -> str:
    """Render rows as JSON records when columns are known, otherwise arrays."""
    if columns:
        data = [
            {columns[i]: row[i] if i < len(row) else None for i in range(len(columns))}
            for row in (rows or [])
        ]
    else:
        data = rows or []
    return json.dumps(data, ensure_ascii=False, indent=2)


def tool_error(tool_call_id: str, name: str, message: str) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=message,
                    tool_call_id=tool_call_id,
                    name=name,
                    status="error",
                )
            ]
        }
    )


def tool_message(
    tool_call_id: str,
    name: str,
    content: str,
    *,
    extra_updates: dict[str, Any] | None = None,
) -> Command:
    """Emit a successful ``ToolMessage`` plus optional state updates."""
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id, name=name)]
    }
    if extra_updates:
        update.update(extra_updates)
    return Command(update=update)
