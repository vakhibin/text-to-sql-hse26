"""Shared helpers used across tool families (core, discovery, history, ...).

Kept private to the ``orchestrator_agent.tools`` package — nothing outside
should import from here. The helpers handle three concerns every tool cares
about:

- resolving ``db_id`` from explicit arg or session state,
- rendering a row preview compact enough for an LLM context window,
- emitting a ``Command`` that carries a ``ToolMessage`` (success or error)
  back to the agent.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

ROW_PREVIEW_LIMIT = 10


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
