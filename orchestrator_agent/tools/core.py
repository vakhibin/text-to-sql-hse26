"""Core tools: ``run_text_to_sql``, ``execute_sql``, ``explain_sql``.

The tools are built by ``make_core_tools(client)`` so the HTTP client is
injected via closure at graph-build time. Each tool returns a LangGraph
``Command`` that:

- appends a ``ToolMessage`` carrying a compact, LLM-friendly summary
- updates conversation artifacts in state (``active_db_id``, ``last_sql``,
  and row previews for subsequent turns)

Failures coming back from ``text_to_sql_api`` (HTTP 200 with ``success=False``
or explicit error fields) are converted to human-readable ``ToolMessage``
strings rather than raised, so the LLM can react inside the same turn.
Unexpected HTTP failures or transport errors are surfaced as tool-error
messages too — the goal is that the conversation never crashes because a
downstream call failed.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from orchestrator_agent.clients.text_to_sql import (
    TextToSQLAPIError,
    TextToSQLClient,
)
from orchestrator_agent.tools._shared import (
    ROW_PREVIEW_LIMIT as _ROW_PREVIEW_LIMIT,
    append_history as _append_history,
    clear_result_artifacts as _clear_result_artifacts,
    format_rows_preview as _format_rows_preview,
    resolve_db_id as _resolve_db_id,
    tool_error as _tool_error,
)
from services.text_to_sql_api.schemas import (
    ExecuteResponse,
    ExplainResponse,
    RunResponse,
)


def _run_summary(resp: RunResponse) -> str:
    if not resp.sql:
        err = resp.error or "pipeline returned no SQL"
        return f"Pipeline produced no SQL. Error: {err}"
    lines = [f"SQL:\n{resp.sql}"]
    if resp.executed:
        lines.append(
            f"Executed successfully. Rows: {resp.row_count}.\n"
            f"Preview:\n{_format_rows_preview(resp.columns, resp.rows)}"
        )
    elif resp.error:
        lines.append(f"SQL was produced but failed to execute: {resp.error}")
    else:
        lines.append("SQL was produced but not executed.")
    if resp.warnings:
        lines.append(f"Warnings: {', '.join(resp.warnings)}")
    return "\n".join(lines)


def _run_meta(resp: RunResponse) -> dict[str, Any]:
    """Return compact run metadata for UI rendering and session inspection."""
    return {
        "trace_id": resp.trace_id,
        "stage_status": dict(resp.stage_status or {}),
        "warnings": list(resp.warnings or []),
        "cost_usd": float(resp.cost_usd or 0.0),
        "elapsed_s": float(resp.elapsed_s or 0.0),
        "executed": bool(resp.executed),
        "error": resp.error,
    }


def _execute_summary(resp: ExecuteResponse) -> str:
    if resp.success:
        return (
            f"Query executed. Rows: {resp.row_count}.\n"
            f"Preview:\n{_format_rows_preview(resp.columns, resp.rows)}"
        )
    code = resp.error_code or "ERROR"
    return f"Query failed ({code}): {resp.error or 'unknown error'}"


def make_core_tools(client: TextToSQLClient) -> list:
    """Build the core tool set bound to a concrete HTTP client.

    The tools read the active database from state (via ``InjectedState``) so
    the LLM does not have to repeat ``db_id`` on every call once it's chosen.
    """

    @tool
    async def run_text_to_sql(
        question: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        db_id: str | None = None,
        evidence: str | None = None,
    ) -> Command:
        """Run the full text-to-SQL pipeline for a natural-language question.

        Args:
            question: User's question in plain English.
            db_id: Target database id. Optional — defaults to the active database
                for this session when omitted.
            evidence: Optional BIRD-style hint (e.g. domain glossary).

        Returns a ``ToolMessage`` summarising the SQL produced, whether it ran,
        and a preview of the result rows. Updates ``active_db_id`` and ``last_sql``.
        """
        resolved, err = _resolve_db_id(state, db_id)
        if err:
            return _tool_error(tool_call_id, "run_text_to_sql", err)

        try:
            resp = await client.run(
                question=question, db_id=resolved, evidence=evidence
            )
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return _tool_error(
                tool_call_id,
                "run_text_to_sql",
                f"text_to_sql_api call failed: {exc}",
            )

        summary = _run_summary(resp)
        update: dict[str, Any] = {
            "active_db_id": resolved,
            "last_run_meta": _run_meta(resp),
            "messages": [
                ToolMessage(
                    content=summary, tool_call_id=tool_call_id, name="run_text_to_sql"
                )
            ],
        }
        if resp.sql:
            update["last_sql"] = resp.sql
            update["sql_history"] = _append_history(
                state,
                sql=resp.sql,
                db_id=resolved,
                source="run",
                executed=resp.executed,
                row_count=resp.row_count,
            )
        if resp.rows is not None:
            update["last_rows_preview"] = resp.rows[:_ROW_PREVIEW_LIMIT]
            update["last_rows_columns"] = resp.columns
            update["last_row_count"] = resp.row_count
            update["last_result_export"] = None
        else:
            update.update(_clear_result_artifacts())
        return Command(update=update)

    @tool
    async def execute_sql(
        sql: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        db_id: str | None = None,
    ) -> Command:
        """Execute a given SELECT statement against the target database.

        This is a READ-ONLY path: the downstream service rejects writes at the
        SQL-AST level. For write-style statements, ask the user to confirm via
        the write-SQL flow (not available in this phase).

        Args:
            sql: SQL to run.
            db_id: Target database id. Optional — defaults to the active database.
        """
        resolved, err = _resolve_db_id(state, db_id)
        if err:
            return _tool_error(tool_call_id, "execute_sql", err)

        try:
            resp = await client.execute(sql=sql, db_id=resolved)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return _tool_error(
                tool_call_id,
                "execute_sql",
                f"text_to_sql_api call failed: {exc}",
            )

        summary = _execute_summary(resp)
        update: dict[str, Any] = {
            "active_db_id": resolved,
            "last_sql": sql,
            "sql_history": _append_history(
                state,
                sql=sql,
                db_id=resolved,
                source="execute",
                executed=resp.success,
                row_count=resp.row_count,
            ),
            "messages": [
                ToolMessage(
                    content=summary, tool_call_id=tool_call_id, name="execute_sql"
                )
            ],
        }
        if resp.success and resp.rows is not None:
            update["last_rows_preview"] = resp.rows[:_ROW_PREVIEW_LIMIT]
            update["last_rows_columns"] = resp.columns
            update["last_row_count"] = resp.row_count
            update["last_result_export"] = None
        else:
            update.update(_clear_result_artifacts())
        return Command(update=update)

    @tool
    async def explain_sql(
        sql: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        db_id: str | None = None,
    ) -> Command:
        """Explain a given SQL statement in plain English (2-4 sentences).

        Args:
            sql: SQL to explain.
            db_id: Optional database id for schema-aware explanation. Defaults
                to the active database when omitted.
        """
        resolved, _ = _resolve_db_id(state, db_id)
        try:
            resp: ExplainResponse = await client.explain(sql=sql, db_id=resolved)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return _tool_error(
                tool_call_id,
                "explain_sql",
                f"text_to_sql_api call failed: {exc}",
            )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=resp.explanation,
                        tool_call_id=tool_call_id,
                        name="explain_sql",
                    )
                ]
            }
        )

    return [run_text_to_sql, execute_sql, explain_sql]
