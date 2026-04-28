"""Catalog exploration tools: discover databases, inspect schema, sample rows.

Built by ``make_discovery_tools(client)`` so the HTTP client is injected at
graph-build time. All tools share the ``Command``-returning pattern from the
core tool family and use ``orchestrator_agent.tools._shared`` helpers for
consistent db-id resolution and error reporting.

Tools exposed here:

- ``list_databases`` — catalog overview (db_ids + table counts)
- ``describe_database`` — schema dump for one database
- ``switch_database`` — set ``active_db_id`` for subsequent tool calls
- ``sample_table`` — preview first N rows of a table
- ``search_table_values`` — look up real literal values in a column so the
  LLM can use correct casing/spelling in ``WHERE`` clauses

The last two tools are implemented on top of ``execute_sql`` on the server
side (``POST /execute``), so no new FastAPI endpoints are needed. Parameters
are parameterised where possible; identifiers are validated locally before
they reach SQLite so a hallucinated table name fails fast with a clear
error message.
"""

from __future__ import annotations

import re
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
    clear_result_artifacts,
    format_rows_preview,
    resolve_db_id,
    tool_error,
    tool_message,
)
from services.text_to_sql_api.schemas import (
    DatabasesResponse,
    ExecuteResponse,
    SchemaResponse,
)

# Conservative SQLite-compatible identifier pattern. We avoid quoting because
# sqlglot/SQLite accepts both bare and ``"double-quoted"`` identifiers but the
# DB files in Spider/BIRD use bare names; allowing quotes would open the door
# to smuggling arbitrary SQL via a hallucinated identifier.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SEARCH_VALUES_LIMIT = 10
_SAMPLE_DEFAULT_LIMIT = 5
_SAMPLE_MAX_LIMIT = 50


def _format_databases(resp: DatabasesResponse) -> str:
    if not resp.databases:
        return f"No databases found under {resp.schema_root!r}."
    lines = [f"Databases under {resp.schema_root}:"]
    for db in resp.databases:
        lines.append(f"- {db.db_id} ({db.num_tables} tables)")
    return "\n".join(lines)


def _format_schema(schema: SchemaResponse, *, max_columns_per_table: int = 20) -> str:
    lines = [f"Database: {schema.db_id}"]
    for table in schema.tables:
        cols = table.columns[:max_columns_per_table]
        col_texts = [f"{c.name}: {c.type}" for c in cols]
        if len(table.columns) > max_columns_per_table:
            col_texts.append(f"... (+{len(table.columns) - max_columns_per_table} more)")
        pk_text = f" PK=[{', '.join(table.primary_keys)}]" if table.primary_keys else ""
        fk_text = ""
        if table.foreign_keys:
            fk_pairs = [
                f"{fk.column}->{fk.ref_table}.{fk.ref_column}" for fk in table.foreign_keys
            ]
            fk_text = f" FK=[{', '.join(fk_pairs)}]"
        lines.append(f"- {table.name}({', '.join(col_texts)}){pk_text}{fk_text}")
    return "\n".join(lines)


def _require_identifier(name: str, *, kind: str) -> str | None:
    """Return an error string if ``name`` is not a safe bare identifier."""
    if not _IDENTIFIER_RE.match(name):
        return (
            f"Invalid {kind} {name!r}: only letters, digits, and underscores are "
            f"allowed (must start with a letter or underscore)."
        )
    return None


def _execute_response_summary(resp: ExecuteResponse) -> tuple[bool, str]:
    if resp.success:
        return True, (
            f"Rows: {resp.row_count}.\n"
            f"Preview:\n{format_rows_preview(resp.columns, resp.rows)}"
        )
    code = resp.error_code or "ERROR"
    return False, f"Query failed ({code}): {resp.error or 'unknown error'}"


def make_discovery_tools(client: TextToSQLClient) -> list:
    """Build the discovery tool set bound to a concrete HTTP client."""

    @tool
    async def list_databases(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """List all databases available to the agent.

        Returns db_ids along with table counts so the LLM can pick or suggest
        one before running deeper queries.
        """
        try:
            resp = await client.list_databases()
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return tool_error(
                tool_call_id, "list_databases", f"text_to_sql_api call failed: {exc}"
            )
        return tool_message(tool_call_id, "list_databases", _format_databases(resp))

    @tool
    async def describe_database(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        db_id: str | None = None,
    ) -> Command:
        """Describe the schema of a database: tables, columns, PKs, FKs.

        Args:
            db_id: Target database. Optional — falls back to the active database.

        Does NOT change the active database; call ``switch_database`` for that.
        """
        resolved, err = resolve_db_id(state, db_id)
        if err:
            return tool_error(tool_call_id, "describe_database", err)
        try:
            schema = await client.get_schema(resolved)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            if isinstance(exc, TextToSQLAPIError) and exc.status_code == 404:
                return tool_error(
                    tool_call_id,
                    "describe_database",
                    f"Database {resolved!r} not found. Use list_databases to see options.",
                )
            return tool_error(
                tool_call_id,
                "describe_database",
                f"text_to_sql_api call failed: {exc}",
            )
        return tool_message(tool_call_id, "describe_database", _format_schema(schema))

    @tool
    async def switch_database(
        db_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Set the active database for subsequent tool calls.

        Validates that ``db_id`` exists by fetching its schema. On success,
        updates ``active_db_id`` and clears any ``last_sql`` / row preview so
        follow-up tool calls do not mix contexts.
        """
        try:
            schema = await client.get_schema(db_id)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            if isinstance(exc, TextToSQLAPIError) and exc.status_code == 404:
                return tool_error(
                    tool_call_id,
                    "switch_database",
                    f"Database {db_id!r} not found. Use list_databases to see options.",
                )
            return tool_error(
                tool_call_id,
                "switch_database",
                f"text_to_sql_api call failed: {exc}",
            )

        tables = ", ".join(t.name for t in schema.tables) or "(no tables)"
        content = f"Active database is now {db_id!r}. Tables: {tables}."
        return tool_message(
            tool_call_id,
            "switch_database",
            content,
            extra_updates={
                "active_db_id": db_id,
                "last_sql": None,
                **clear_result_artifacts(),
            },
        )

    @tool
    async def sample_table(
        table_name: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        db_id: str | None = None,
        limit: int = _SAMPLE_DEFAULT_LIMIT,
    ) -> Command:
        """Preview the first ``limit`` rows of a table (read-only).

        Useful for showing the user example data or for the LLM to confirm
        column semantics before writing a longer query.

        Args:
            table_name: Target table.
            db_id: Optional target database. Defaults to the active database.
            limit: Row cap, clamped to 1..50.
        """
        resolved, err = resolve_db_id(state, db_id)
        if err:
            return tool_error(tool_call_id, "sample_table", err)
        id_err = _require_identifier(table_name, kind="table name")
        if id_err:
            return tool_error(tool_call_id, "sample_table", id_err)
        clamped = max(1, min(int(limit), _SAMPLE_MAX_LIMIT))
        sql = f"SELECT * FROM {table_name} LIMIT {clamped}"
        try:
            resp = await client.execute(sql=sql, db_id=resolved)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return tool_error(
                tool_call_id, "sample_table", f"text_to_sql_api call failed: {exc}"
            )
        ok, summary = _execute_response_summary(resp)
        if not ok:
            return tool_error(tool_call_id, "sample_table", summary)
        return tool_message(
            tool_call_id,
            "sample_table",
            f"Sample of {table_name} (limit={clamped}):\n{summary}",
        )

    @tool
    async def search_table_values(
        table_name: str,
        column_name: str,
        search_term: str,
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        db_id: str | None = None,
    ) -> Command:
        """Find real literal values in a column matching a substring (read-only).

        Use this when you are about to write ``WHERE col = 'Value'`` but are
        unsure about exact spelling, casing, or whitespace. Returns up to
        10 distinct matches.

        Args:
            table_name: Table to search in.
            column_name: Column to search in.
            search_term: Substring to match (case-insensitive via LIKE).
            db_id: Optional target database.
        """
        resolved, err = resolve_db_id(state, db_id)
        if err:
            return tool_error(tool_call_id, "search_table_values", err)
        for name, kind in ((table_name, "table name"), (column_name, "column name")):
            id_err = _require_identifier(name, kind=kind)
            if id_err:
                return tool_error(tool_call_id, "search_table_values", id_err)
        safe_term = search_term.replace("'", "''")
        sql = (
            f"SELECT DISTINCT {column_name} FROM {table_name} "
            f"WHERE CAST({column_name} AS TEXT) LIKE '%{safe_term}%' "
            f"LIMIT {_SEARCH_VALUES_LIMIT}"
        )
        try:
            resp = await client.execute(sql=sql, db_id=resolved)
        except (TextToSQLAPIError, httpx.HTTPError) as exc:
            return tool_error(
                tool_call_id,
                "search_table_values",
                f"text_to_sql_api call failed: {exc}",
            )
        ok, summary = _execute_response_summary(resp)
        if not ok:
            return tool_error(tool_call_id, "search_table_values", summary)
        row_count = resp.row_count or 0
        if row_count == 0:
            content = (
                f"No values in {table_name}.{column_name} match "
                f"{search_term!r}."
            )
        else:
            values = [str(row[0]) for row in (resp.rows or [])[:ROW_PREVIEW_LIMIT]]
            content = (
                f"Values in {table_name}.{column_name} matching {search_term!r} "
                f"(up to {_SEARCH_VALUES_LIMIT}):\n- "
                + "\n- ".join(values)
            )
        return tool_message(tool_call_id, "search_table_values", content)

    return [
        list_databases,
        describe_database,
        switch_database,
        sample_table,
        search_table_values,
    ]
