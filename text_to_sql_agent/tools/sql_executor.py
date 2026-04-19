"""Async SQL execution helpers for execution filter and refiner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from text_to_sql_agent.tools.sql_guardrail import (
    WriteSQLRejected,
    assert_read_only,
)


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


def _apply_sqlite_text_factory(dbapi_connection: object) -> None:
    """Attach forgiving UTF-8 decoding to the real sqlite3.Connection.

    SQLAlchemy ``sqlite+aiosqlite`` passes ``AsyncAdapt_aiosqlite_connection`` into the
    ``connect`` event, not a raw ``sqlite3.Connection``. The underlying DBAPI handle is
    ``driver_connection._conn`` (``aiosqlite`` wraps ``sqlite3``).
    """
    decode = lambda b: b.decode("utf-8", errors="ignore")
    driver = getattr(dbapi_connection, "driver_connection", None)
    if driver is not None:
        sqlite_conn = getattr(driver, "_conn", None)
        if sqlite_conn is not None:
            sqlite_conn.text_factory = decode
            return
    if hasattr(dbapi_connection, "text_factory"):
        dbapi_connection.text_factory = decode


async def execute_sql(
    db_path_or_url: str,
    sql: str,
    *,
    timeout_seconds: int = 20,
    allow_write: bool = False,
) -> SQLExecutionResult:
    """Execute one SQL statement via SQLAlchemy in isolated try/except.

    By default a read-only guardrail rejects any non-SELECT SQL before it
    reaches SQLite. Pass ``allow_write=True`` only from user-confirmed code
    paths (e.g. the future ``/execute_user_confirmed`` endpoint).
    """
    if not allow_write:
        try:
            assert_read_only(sql)
        except WriteSQLRejected as exc:
            return SQLExecutionResult(
                success=False,
                rows=None,
                error=f"write SQL rejected by guardrail: {exc}",
            )

    engine = create_async_engine(_to_sqlalchemy_url(db_path_or_url), future=True)

    @event.listens_for(engine.sync_engine, "connect")
    def _on_sqlite_connect(dbapi_connection: object, _connection_record: object) -> None:
        _apply_sqlite_text_factory(dbapi_connection)

    try:
        if allow_write:
            async with engine.begin() as conn:
                result = await asyncio.wait_for(
                    conn.execute(text(sql)), timeout=timeout_seconds
                )
                rows = list(result.fetchall()) if result.returns_rows else []
        else:
            async with engine.connect() as conn:
                result = await asyncio.wait_for(
                    conn.execute(text(sql)), timeout=timeout_seconds
                )
                rows = list(result.fetchall()) if result.returns_rows else []
        return SQLExecutionResult(success=True, rows=rows, error=None)
    except Exception as exc:  # pragma: no cover - runtime/db path
        return SQLExecutionResult(success=False, rows=None, error=str(exc))
    finally:
        await engine.dispose()

