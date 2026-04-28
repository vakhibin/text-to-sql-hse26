"""Result UX tools: summarize and export the latest row preview.

These tools are deterministic and local to the orchestrator state. They do
not call ``text_to_sql_api`` because all required data is already captured by
``run_text_to_sql``, ``execute_sql``, or ``rerun`` as a small row preview.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from orchestrator_agent.tools._shared import (
    ROW_PREVIEW_LIMIT,
    format_csv,
    format_json_rows,
    format_markdown_table,
    tool_error,
    tool_message,
)

_SUMMARY_DEFAULT_LIMIT = 5
_EXPORT_FORMATS = ("markdown", "csv", "json")


def _has_rows(state: dict[str, Any]) -> bool:
    return state.get("last_rows_preview") is not None


def _preview_scope(row_count: int | None, preview_count: int) -> str:
    if row_count is None:
        return f"showing {preview_count} preview rows"
    if row_count <= preview_count:
        return f"all {row_count} rows"
    return f"first {preview_count} of {row_count} rows"


def make_result_tools() -> list:
    """Build deterministic result UX tools."""

    @tool
    async def summarize_results(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        limit: int = _SUMMARY_DEFAULT_LIMIT,
    ) -> Command:
        """Summarize the latest query result preview in a readable form.

        Args:
            limit: Maximum preview rows to include, clamped to 1..10.

        Use this after ``run_text_to_sql``, ``execute_sql``, or ``rerun`` when
        the user asks "what did we get?" or wants a compact result recap.
        """
        if not _has_rows(state):
            return tool_error(
                tool_call_id,
                "summarize_results",
                "No result rows are available in this session. Run or rerun a query first.",
            )

        rows = list(state.get("last_rows_preview") or [])
        columns = list(state.get("last_rows_columns") or [])
        row_count = state.get("last_row_count")
        clamped = max(1, min(int(limit), ROW_PREVIEW_LIMIT))
        preview_rows = rows[:clamped]

        lines = [
            "Latest result summary:",
            f"- Database: {state.get('active_db_id') or '?'}",
            f"- SQL: {state.get('last_sql') or '(unknown)'}",
            f"- Rows: {_preview_scope(row_count, len(rows))}",
            f"- Columns: {', '.join(columns) if columns else '(unknown)'}",
            "",
            "Preview:",
            format_markdown_table(columns, preview_rows),
        ]
        return tool_message(tool_call_id, "summarize_results", "\n".join(lines))

    @tool
    async def export_results(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        format: str = "markdown",
    ) -> Command:
        """Export the latest query result preview as Markdown, CSV, or JSON.

        This exports only the preview stored in session state, not the full DB
        result set. Re-run the SQL with a narrower LIMIT when the user needs a
        larger or different export slice.
        """
        if not _has_rows(state):
            return tool_error(
                tool_call_id,
                "export_results",
                "No result rows are available to export. Run or rerun a query first.",
            )

        fmt = str(format).lower()
        if fmt not in _EXPORT_FORMATS:
            return tool_error(
                tool_call_id,
                "export_results",
                f"Unsupported export format {format!r}. Use markdown, csv, or json.",
            )

        rows = list(state.get("last_rows_preview") or [])
        columns = list(state.get("last_rows_columns") or [])
        if fmt == "markdown":
            rendered = format_markdown_table(columns, rows)
        elif fmt == "csv":
            rendered = format_csv(columns, rows)
        else:
            rendered = format_json_rows(columns, rows)

        row_count = state.get("last_row_count")
        header = (
            f"Exported {len(rows)} preview rows"
            + (f" from {row_count} total rows" if row_count is not None else "")
            + f" as {fmt}:"
        )
        return tool_message(
            tool_call_id,
            "export_results",
            f"{header}\n\n{rendered}",
            extra_updates={"last_result_export": rendered},
        )

    return [summarize_results, export_results]
