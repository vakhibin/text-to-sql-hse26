"""Append-only audit logging for sensitive orchestrator actions."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_LOG_PATH = ".cache/orchestrator/writes.jsonl"


def audit_log_path() -> Path:
    """Return the configured append-only write audit log path."""
    return Path(os.getenv("ORCH_AUDIT_LOG_PATH", DEFAULT_AUDIT_LOG_PATH))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


async def append_write_audit_event(
    *,
    session_id: str | None,
    user_id: str | None,
    db_id: str,
    sql: str,
    success: bool,
    row_count: int | None = None,
    error: str | None = None,
    tool_call_id: str | None = None,
) -> None:
    """Append a write-SQL audit event.

    The helper intentionally does not return anything. Callers may treat audit
    logging as best-effort, but the normal path is deterministic and covered by
    tests.
    """
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "db_id": db_id,
        "sql": sql,
        "success": bool(success),
        "row_count": row_count,
        "error": error,
        "tool_call_id": tool_call_id,
    }
    await asyncio.to_thread(_append_jsonl, audit_log_path(), payload)
