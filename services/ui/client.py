"""Small synchronous client for the Streamlit UI.

Streamlit scripts run top-to-bottom on each interaction, so a tiny sync
``httpx.Client`` wrapper is simpler than sharing the async orchestrator client.
The functions here are intentionally UI-framework agnostic and covered by
unit tests; ``app.py`` handles Streamlit rendering only.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_ORCHESTRATOR_URL = "http://localhost:8002"
DEFAULT_UI_TIMEOUT_S = 300.0


class OrchestratorUIError(RuntimeError):
    """Raised when the UI cannot talk to ``orchestrator_api``."""


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
