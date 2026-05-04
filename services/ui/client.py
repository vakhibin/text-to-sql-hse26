"""Small synchronous client for the Streamlit UI.

Streamlit scripts run top-to-bottom on each interaction, so a tiny sync
``httpx.Client`` wrapper is simpler than sharing the async orchestrator client.
The functions here are intentionally UI-framework agnostic and covered by
unit tests; ``app.py`` handles Streamlit rendering only.

Two clients live here:

- ``OrchestratorUIClient`` talks to ``orchestrator_api`` and is used for the
  conversational flow (``/chat``, ``/sessions``).
- ``TextToSQLUIClient`` talks to ``text_to_sql_api`` directly. The Streamlit
  UI uses it only for read-only catalog browsing (``/databases``,
  ``/databases/{db_id}/schema``); SQL execution still goes through the
  orchestrator so the agent can update conversation state and audit logs.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_ORCHESTRATOR_URL = "http://localhost:8002"
DEFAULT_TEXT_TO_SQL_URL = "http://localhost:8001"
DEFAULT_UI_TIMEOUT_S = 300.0
DEFAULT_BROWSE_TIMEOUT_S = 30.0


class OrchestratorUIError(RuntimeError):
    """Raised when the UI cannot talk to ``orchestrator_api``."""


class TextToSQLUIError(RuntimeError):
    """Raised when the UI cannot talk to ``text_to_sql_api``."""


def normalize_base_url(url: str | None) -> str:
    """Return a non-empty base URL without a trailing slash."""
    cleaned = (url or DEFAULT_ORCHESTRATOR_URL).strip().rstrip("/")
    return cleaned or DEFAULT_ORCHESTRATOR_URL


def visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Filter API messages down to chat bubbles shown to the user."""
    visible: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        if role not in {"human", "ai"}:
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        visible.append({"role": role, "content": content})
    return visible


def session_extra(session: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``extra`` artifact payload from a session response."""
    extra = (session or {}).get("extra") or {}
    return extra if isinstance(extra, dict) else {}


def sql_history_from_session(session: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract newest-first SQL history from a session response."""
    history = list(session_extra(session).get("sql_history") or [])
    return list(reversed(history))


def latest_result_from_session(session: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized latest result metadata and preview rows."""
    extra = session_extra(session)
    columns = [str(col) for col in (extra.get("last_rows_columns") or [])]
    raw_rows = list(extra.get("last_rows_preview") or [])
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if isinstance(raw, dict):
            rows.append(raw)
            continue
        values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        row_columns = columns or [f"col_{i + 1}" for i in range(len(values))]
        rows.append(dict(zip(row_columns, values)))
    return {
        "row_count": extra.get("last_row_count"),
        "columns": columns,
        "rows": rows,
    }


def latest_run_meta_from_session(session: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized metadata for the latest full text-to-SQL run."""
    raw = session_extra(session).get("last_run_meta") or {}
    if not isinstance(raw, dict):
        return {
            "trace_id": None,
            "stage_status": {},
            "warnings": [],
            "cost_usd": 0.0,
            "elapsed_s": 0.0,
            "executed": False,
            "error": None,
            "langfuse_trace_url": None,
        }
    stage_status = raw.get("stage_status") or {}
    if not isinstance(stage_status, dict):
        stage_status = {}
    warnings = raw.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    try:
        cost_usd = float(raw.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost_usd = 0.0
    try:
        elapsed_s = float(raw.get("elapsed_s") or 0.0)
    except (TypeError, ValueError):
        elapsed_s = 0.0
    trace_url = raw.get("langfuse_trace_url")
    return {
        "trace_id": raw.get("trace_id"),
        "stage_status": {str(k): str(v) for k, v in stage_status.items()},
        "warnings": [str(w) for w in warnings],
        "cost_usd": cost_usd,
        "elapsed_s": elapsed_s,
        "executed": bool(raw.get("executed")),
        "error": raw.get("error"),
        "selected_tables": [
            str(table) for table in (raw.get("selected_tables") or []) if str(table).strip()
        ],
        "query_sketch_text": str(raw.get("query_sketch_text") or "").strip() or None,
        "langfuse_trace_url": str(trace_url) if trace_url else None,
    }


def format_history_label(index: int, entry: dict[str, Any]) -> str:
    """Human-readable label for sidebar SQL history."""
    source = entry.get("source") or "?"
    db_id = entry.get("db_id") or "?"
    sql = " ".join(str(entry.get("sql") or "").split())
    if len(sql) > 80:
        sql = sql[:77] + "..."
    return f"{index}. [{source}] {db_id}: {sql}"


class OrchestratorUIClient:
    """Sync HTTP wrapper used by the Streamlit app."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_s: float = DEFAULT_UI_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OrchestratorUIError(f"orchestrator_api request failed: {exc}") from exc
        if response.status_code >= 400:
            raise OrchestratorUIError(
                f"orchestrator_api HTTP {response.status_code}: {response.text}"
            )
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def chat(
        self,
        *,
        session_id: str,
        user_id: str,
        message: str,
        active_db_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
        }
        if active_db_id:
            payload["active_db_id"] = active_db_id
        return self._request("POST", "/chat", json=payload)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/sessions/{session_id}")
        except OrchestratorUIError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def reset_session(self, session_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/sessions/{session_id}")


def normalize_text_to_sql_url(url: str | None) -> str:
    """Return a non-empty base URL for ``text_to_sql_api`` without a trailing slash."""
    cleaned = (url or DEFAULT_TEXT_TO_SQL_URL).strip().rstrip("/")
    return cleaned or DEFAULT_TEXT_TO_SQL_URL


class TextToSQLUIClient:
    """Sync HTTP wrapper around ``text_to_sql_api`` used for catalog browsing."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_s: float = DEFAULT_BROWSE_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_text_to_sql_url(base_url)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TextToSQLUIError(f"text_to_sql_api request failed: {exc}") from exc
        if response.status_code >= 400:
            raise TextToSQLUIError(
                f"text_to_sql_api HTTP {response.status_code}: {response.text}"
            )
        return response.json()

    def list_databases(self) -> dict[str, Any]:
        """Return the catalog payload (``{schema_root, databases: [...]}``)."""
        return self._request("GET", "/databases")

    def get_schema(self, db_id: str) -> dict[str, Any] | None:
        """Return schema for ``db_id`` or ``None`` when the db is unknown."""
        try:
            return self._request("GET", f"/databases/{db_id}/schema")
        except TextToSQLUIError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise


def database_options(catalog: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return a normalized list of database descriptors from ``/databases``.

    Each item carries ``db_id``, ``num_tables``, and ``label`` fit for a
    Streamlit selectbox. The list is sorted by ``db_id`` for stable display.
    """
    if not catalog:
        return []
    raw = catalog.get("databases") or []
    items: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        db_id = str(entry.get("db_id") or "").strip()
        if not db_id:
            continue
        num_tables = entry.get("num_tables")
        try:
            num_tables_int = int(num_tables) if num_tables is not None else 0
        except (TypeError, ValueError):
            num_tables_int = 0
        items.append(
            {
                "db_id": db_id,
                "num_tables": num_tables_int,
                "label": f"{db_id} ({num_tables_int} tables)",
            }
        )
    items.sort(key=lambda item: item["db_id"])
    return items


def schema_tables(schema: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return a normalized list of tables for the schema browser.

    Each table dict contains ``name``, ``columns`` (list of ``{name, type}``),
    ``primary_keys`` (list of column names), and ``foreign_keys`` (list of
    ``{column, ref_table, ref_column}``). Missing fields are filled with
    safe defaults so the renderer does not need to know about API quirks.
    """
    if not schema:
        return []
    tables_raw = schema.get("tables") or []
    tables: list[dict[str, Any]] = []
    for table in tables_raw:
        if not isinstance(table, dict):
            continue
        name = str(table.get("name") or "").strip()
        if not name:
            continue
        cols_raw = table.get("columns") or []
        columns: list[dict[str, str]] = []
        for col in cols_raw:
            if not isinstance(col, dict):
                continue
            col_name = str(col.get("name") or "").strip()
            if not col_name:
                continue
            columns.append(
                {
                    "name": col_name,
                    "type": str(col.get("type") or ""),
                }
            )
        pks = [str(pk) for pk in (table.get("primary_keys") or []) if pk]
        fks_raw = table.get("foreign_keys") or []
        fks: list[dict[str, str]] = []
        for fk in fks_raw:
            if not isinstance(fk, dict):
                continue
            fks.append(
                {
                    "column": str(fk.get("column") or ""),
                    "ref_table": str(fk.get("ref_table") or ""),
                    "ref_column": str(fk.get("ref_column") or ""),
                }
            )
        tables.append(
            {
                "name": name,
                "columns": columns,
                "primary_keys": pks,
                "foreign_keys": fks,
            }
        )
    tables.sort(key=lambda item: item["name"])
    return tables
