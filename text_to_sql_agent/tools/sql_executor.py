"""Async SQL execution helpers for execution filter and refiner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


@dataclass
class SQLExecutionResult:
    success: bool
    rows: Optional[list[tuple[Any, ...]]] = None
    error: Optional[str] = None


def _to_sqlalchemy_url(db_path_or_url: str) -> str:
    if "://" in db_path_or_url:
        return db_path_or_url
    normalized = db_path_or_url.strip()
    return f"sqlite+aiosqlite:///{normalized}"


async def execute_sql(
    db_path_or_url: str,
    sql: str,
    *,
    timeout_seconds: int = 20,
) -> SQLExecutionResult:
    """Execute one SQL statement via SQLAlchemy in isolated try/except."""
    engine = create_async_engine(_to_sqlalchemy_url(db_path_or_url), future=True)
    try:
        async with engine.connect() as conn:
            result = await asyncio.wait_for(conn.execute(text(sql)), timeout=timeout_seconds)
            rows = list(result.fetchall()) if result.returns_rows else []
        return SQLExecutionResult(success=True, rows=rows, error=None)
    except Exception as exc:  # pragma: no cover - runtime/db path
        return SQLExecutionResult(success=False, rows=None, error=str(exc))
    finally:
        await engine.dispose()

