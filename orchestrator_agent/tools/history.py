"""SQL manipulation + history tools.

Built by ``make_history_tools(client)`` so the HTTP client is injected at
graph-build time. Tools share the ``Command``-returning pattern used by
the core and discovery families.

Tools exposed here:

- ``fix_sql`` — ask the refiner to repair SQL, usually ``last_sql`` from
  state; delegates to ``POST /refine``.
- ``modify_sql`` — apply a natural-language edit to SQL (not an error repair);
  delegates to ``POST /modify`` (single LLM call with schema context).
- ``list_recent`` — dump the in-session ``sql_history`` back to the LLM so it
  can reason about past queries without re-running anything.
- ``rerun`` — re-execute a SQL from history by 1-based index (``1`` = most
  recent). Falls back to ``last_sql`` when no history is stored. Uses the
  db_id stored alongside the history entry as the authoritative target.

All tools that touch SQL write an entry to ``sql_history`` via
``append_history`` so the next turn (and ``list_recent``) sees a consistent
view.
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
    ROW_PREVIEW_LIMIT,
    append_history,
    clear_result_artifacts,
    format_rows_preview,
    resolve_db_id,
    tool_error,
    tool_message,
)

_LIST_RECENT_DEFAULT = 10
_LIST_RECENT_MAX = 20


def _format_history_entry(idx: int, entry: dict[str, Any]) -> str:
    """Render a history entry as a compact line for the LLM."""
    source = entry.get("source", "?")
    db = entry.get("db_id") or "?"
    executed = entry.get("executed")
    row_count = entry.get("row_count")
    status_bits = [f"source={source}", f"db={db}"]
    if executed is True:
        rc = "?" if row_count is None else row_count
        status_bits.append(f"ran OK, rows={rc}")
    elif executed is False:
        status_bits.append("not executed / failed")
    sql = str(entry.get("sql") or "").strip()
    return f"[{idx}] ({', '.join(status_bits)})\n{sql}"


def make_history_tools(client: TextToSQLClient) -> list:
    """Build SQL manipulation + history tools bound to a concrete HTTP client."""

    @tool
    async def fix_sql(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        sql: str | None = None,
        db_id: str | None = None,
        error_hint: str | None = None,
    ) -> Command:
        """Repair a SQL query that failed or looks wrong, using the refiner.

        Args:
            sql: SQL to fix. Optional — defaults to ``last_sql`` from the session.
            db_id: Target database. Optional — defaults to the active database.
            error_hint: Optional execution/error text to seed the refiner with.
                When omitted the service will execute the SQL itself and use
                whatever error (if any) comes back.

        Updates ``last_sql`` to the refined SQL and records the fix in
        ``sql_history``. Does NOT execute the refined SQL beyond what the
        refiner does internally — call ``execute_sql`` to get rows.
        """
        target_sql = sql or state.get("last_sql")
        if not target_sql:
            return tool_error(
                tool_call_id,
                "fix_sql",
                "No SQL to fix. Pass an explicit 'sql' argument or run a "
                "query first so last_sql is populated.",
            )
        resolved, err = resolve_db_id(state, db_id)
        if err:
            return tool_error(tool_call_id, "fix_sql", err)

        try:
            resp = await client.refine(
                sql=target_sql, db_id=resolved, error_hint=error_hint
            )
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return tool_error(
                tool_call_id, "fix_sql", f"text_to_sql_api call failed: {exc}"
            )

        lines = [f"Original SQL:\n{resp.original_sql}", f"Refined SQL:\n{resp.refined_sql}"]
        if not resp.changed:
            lines.append("No changes were applied by the refiner.")
        elif resp.success and resp.executed:
            lines.append("Refined SQL executed successfully.")
        elif resp.error:
            lines.append(f"Refined SQL still failing: {resp.error}")
        else:
            lines.append("Refined SQL was produced but not re-executed.")
        summary = "\n\n".join(lines)

        history = append_history(
            state,
            sql=resp.refined_sql,
            db_id=resolved,
            source="fix",
            executed=bool(resp.executed and resp.success),
        )
        return tool_message(
            tool_call_id,
            "fix_sql",
            summary,
            extra_updates={
                "last_sql": resp.refined_sql,
                "active_db_id": resolved,
                "sql_history": history,
                **clear_result_artifacts(),
            },
        )

    @tool
    async def modify_sql(
        instruction: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        sql: str | None = None,
        db_id: str | None = None,
    ) -> Command:
        """Apply a natural-language edit to a SQL query.

        Use this for user-driven changes like "add a WHERE clause for 2023"
        or "group by year instead of month". For error repair, use ``fix_sql``.

        Args:
            instruction: What to change, in plain English.
            sql: SQL to modify. Optional — defaults to ``last_sql``.
            db_id: Target database. Optional — defaults to the active database.

        Updates ``last_sql`` to the modified SQL and records the edit in
        ``sql_history``. Does NOT execute — call ``execute_sql`` (or ``rerun``)
        to see results.
        """
        target_sql = sql or state.get("last_sql")
        if not target_sql:
            return tool_error(
                tool_call_id,
                "modify_sql",
                "No SQL to modify. Pass an explicit 'sql' argument or run a "
                "query first so last_sql is populated.",
            )
        resolved, err = resolve_db_id(state, db_id)
        if err:
            return tool_error(tool_call_id, "modify_sql", err)

        try:
            resp = await client.modify(
                sql=target_sql, instruction=instruction, db_id=resolved
            )
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            if isinstance(exc, TextToSQLAPIError) and exc.status_code == 404:
                return tool_error(
                    tool_call_id,
                    "modify_sql",
                    f"Database {resolved!r} not found. Use list_databases to "
                    "see options.",
                )
            return tool_error(
                tool_call_id, "modify_sql", f"text_to_sql_api call failed: {exc}"
            )

        lines = [f"Instruction:\n{instruction}", f"Modified SQL:\n{resp.modified_sql}"]
        if not resp.changed:
            lines.append("No changes were applied (instruction may be unsupported).")
        summary = "\n\n".join(lines)

        history = append_history(
            state,
            sql=resp.modified_sql,
            db_id=resolved,
            source="modify",
            executed=False,
        )
        return tool_message(
            tool_call_id,
            "modify_sql",
            summary,
            extra_updates={
                "last_sql": resp.modified_sql,
                "active_db_id": resolved,
                "sql_history": history,
                **clear_result_artifacts(),
            },
        )

    @tool
    async def list_recent(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        limit: int = _LIST_RECENT_DEFAULT,
    ) -> Command:
        """Return the most recent SQL queries produced or executed this session.

        Args:
            limit: How many entries to return, newest first. Clamped to 1..20.

        Output is a numbered list where ``[1]`` is the most recent entry, so
        ``rerun(index=1)`` replays the last query.
        """
        history = list(state.get("sql_history") or [])
        if not history:
            return tool_message(
                tool_call_id,
                "list_recent",
                "No SQL has been produced or executed in this session yet.",
            )
        clamped = max(1, min(int(limit), _LIST_RECENT_MAX))
        # Most-recent first.
        recent = list(reversed(history))[:clamped]
        body = "\n\n".join(
            _format_history_entry(i + 1, entry) for i, entry in enumerate(recent)
        )
        header = f"Last {len(recent)} SQL queries (newest first):"
        return tool_message(tool_call_id, "list_recent", f"{header}\n\n{body}")

    @tool
    async def rerun(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        index: int = 1,
    ) -> Command:
        """Re-execute a past SQL query by its position in ``list_recent``.

        Args:
            index: 1-based index, ``1`` = most recent (default). Out-of-range
                values return a tool error.

        Uses the db_id stored on the history entry (not ``active_db_id``), so
        reruns stay consistent even if the active database has since changed.
        """
        history = list(state.get("sql_history") or [])
        idx = int(index)
        if idx < 1:
            return tool_error(
                tool_call_id, "rerun", "index must be >= 1 (1 = most recent)."
            )

        if history:
            if idx > len(history):
                return tool_error(
                    tool_call_id,
                    "rerun",
                    f"index={idx} is out of range (history has {len(history)} entries).",
                )
            entry = list(reversed(history))[idx - 1]
            target_sql = str(entry.get("sql") or "")
            target_db = entry.get("db_id") or state.get("active_db_id")
        else:
            # Fallback: no history yet, rerun last_sql if present (treated as idx=1).
            if idx != 1:
                return tool_error(
                    tool_call_id,
                    "rerun",
                    "No SQL history yet. Only index=1 is valid, and requires a "
                    "previous query in this session.",
                )
            target_sql = state.get("last_sql") or ""
            target_db = state.get("active_db_id")

        if not target_sql:
            return tool_error(
                tool_call_id, "rerun", "No SQL available to rerun."
            )
        if not target_db:
            return tool_error(
                tool_call_id,
                "rerun",
                "No database associated with this SQL; call switch_database first.",
            )

        try:
            resp = await client.execute(sql=target_sql, db_id=target_db)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return tool_error(
                tool_call_id, "rerun", f"text_to_sql_api call failed: {exc}"
            )

        if resp.success:
            summary = (
                f"Re-ran:\n{target_sql}\n\n"
                f"Rows: {resp.row_count}.\n"
                f"Preview:\n{format_rows_preview(resp.columns, resp.rows)}"
            )
            new_history = append_history(
                state,
                sql=target_sql,
                db_id=target_db,
                source="rerun",
                executed=True,
                row_count=resp.row_count,
            )
            update: dict[str, Any] = {
                "last_sql": target_sql,
                "active_db_id": target_db,
                "sql_history": new_history,
            }
            if resp.rows is not None:
                update["last_rows_preview"] = resp.rows[:ROW_PREVIEW_LIMIT]
                update["last_rows_columns"] = resp.columns
                update["last_row_count"] = resp.row_count
                update["last_result_export"] = None
            else:
                update.update(clear_result_artifacts())
            return tool_message(tool_call_id, "rerun", summary, extra_updates=update)

        code = resp.error_code or "ERROR"
        return tool_error(
            tool_call_id,
            "rerun",
            f"Rerun failed ({code}): {resp.error or 'unknown error'}",
        )

    return [fix_sql, modify_sql, list_recent, rerun]
