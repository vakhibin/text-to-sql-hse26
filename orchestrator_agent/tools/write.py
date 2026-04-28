"""User-confirmed write-SQL tools.

The default ``execute_sql`` tool remains read-only. This family handles the
separate write flow:

1. ``propose_write_sql`` stores a pending write request and asks the user to
   confirm.
2. ``confirm_write_sql`` executes the pending SQL through the explicit
   ``/execute-confirmed`` endpoint only after the user confirms in chat.
3. ``cancel_pending_confirmation`` clears the pending request.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from orchestrator_agent.clients.text_to_sql import (
    TextToSQLAPIError,
    TextToSQLClient,
)
from orchestrator_agent.tools._shared import (
    append_history,
    clear_result_artifacts,
    format_rows_preview,
    resolve_db_id,
    tool_error,
    tool_message,
)
from text_to_sql_agent.tools.sql_guardrail import is_read_only

_WRITE_CONFIRMATION_TYPE = "write_sql"


def _pending_payload(*, sql: str, db_id: str, rationale: str | None) -> dict[str, Any]:
    return {
        "type": _WRITE_CONFIRMATION_TYPE,
        "sql": sql,
        "db_id": db_id,
        "rationale": rationale,
    }


def _confirmation_text(sql: str, db_id: str, rationale: str | None) -> str:
    lines = [
        "Write SQL prepared but NOT executed.",
        f"Database: {db_id}",
        "SQL:",
        sql,
    ]
    if rationale:
        lines.extend(["Reason:", rationale])
    lines.append(
        "Ask the user to explicitly confirm before calling confirm_write_sql. "
        "If they decline, call cancel_pending_confirmation."
    )
    return "\n".join(lines)


def make_write_tools(client: TextToSQLClient) -> list:
    """Build write-confirmation tools bound to a concrete HTTP client."""

    @tool
    async def propose_write_sql(
        sql: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        db_id: str | None = None,
        rationale: str | None = None,
    ) -> Command:
        """Prepare a write/DDL SQL statement for explicit user confirmation.

        This tool never executes SQL. Use it when the user asks to INSERT,
        UPDATE, DELETE, CREATE, DROP, ALTER, or otherwise mutate the database.
        For SELECT queries, use ``execute_sql`` instead.
        """
        resolved, err = resolve_db_id(state, db_id)
        if err:
            return tool_error(tool_call_id, "propose_write_sql", err)
        if is_read_only(sql):
            return tool_error(
                tool_call_id,
                "propose_write_sql",
                "This SQL is read-only. Use execute_sql instead of the write confirmation flow.",
            )

        pending = _pending_payload(sql=sql, db_id=resolved, rationale=rationale)
        return tool_message(
            tool_call_id,
            "propose_write_sql",
            _confirmation_text(sql, resolved, rationale),
            extra_updates={
                "active_db_id": resolved,
                "last_sql": sql,
                "pending_confirmation": pending,
                **clear_result_artifacts(),
            },
        )

    @tool
    async def confirm_write_sql(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Execute the pending write SQL after explicit user confirmation.

        Only call this when the latest user message clearly confirms the
        pending write. Do not call it for ambiguous replies.
        """
        pending = dict(state.get("pending_confirmation") or {})
        if pending.get("type") != _WRITE_CONFIRMATION_TYPE:
            return tool_error(
                tool_call_id,
                "confirm_write_sql",
                "No pending write SQL confirmation. Call propose_write_sql first.",
            )
        sql = str(pending.get("sql") or "").strip()
        db_id = str(pending.get("db_id") or "").strip()
        if not sql or not db_id:
            return tool_error(
                tool_call_id,
                "confirm_write_sql",
                "Pending confirmation is incomplete; cancel it and propose again.",
            )

        try:
            resp = await client.execute_confirmed(sql=sql, db_id=db_id)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return tool_error(
                tool_call_id,
                "confirm_write_sql",
                f"text_to_sql_api confirmed execution failed: {exc}",
            )

        history = append_history(
            state,
            sql=sql,
            db_id=db_id,
            source="write_confirmed",
            executed=resp.success,
            row_count=resp.row_count,
        )
        update: dict[str, Any] = {
            "active_db_id": db_id,
            "last_sql": sql,
            "pending_confirmation": None,
            "sql_history": history,
        }
        if resp.success and resp.rows is not None:
            update["last_rows_preview"] = resp.rows
            update["last_rows_columns"] = resp.columns
            update["last_row_count"] = resp.row_count
            update["last_result_export"] = None
        else:
            update.update(clear_result_artifacts())

        if resp.success:
            content = (
                f"Confirmed write SQL executed.\nSQL:\n{sql}\n"
                f"Rows returned: {resp.row_count}.\n"
                f"Preview:\n{format_rows_preview(resp.columns, resp.rows)}"
            )
            return tool_message(
                tool_call_id, "confirm_write_sql", content, extra_updates=update
            )

        code = resp.error_code or "ERROR"
        return tool_error(
            tool_call_id,
            "confirm_write_sql",
            f"Confirmed write SQL failed ({code}): {resp.error or 'unknown error'}",
        )

    @tool
    async def cancel_pending_confirmation(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Cancel any pending user-confirmation request."""
        pending = state.get("pending_confirmation")
        if not pending:
            return tool_message(
                tool_call_id,
                "cancel_pending_confirmation",
                "No pending confirmation to cancel.",
            )
        return tool_message(
            tool_call_id,
            "cancel_pending_confirmation",
            "Pending confirmation cancelled. No SQL was executed.",
            extra_updates={"pending_confirmation": None},
        )

    return [propose_write_sql, confirm_write_sql, cancel_pending_confirmation]
