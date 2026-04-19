"""Integration tests for the executor + guardrail wiring."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from text_to_sql_agent.tools.sql_executor import execute_sql


@pytest.fixture()
def sqlite_db(tmp_path: Path) -> str:
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT);
            INSERT INTO students (id, name) VALUES (1, 'Alice'), (2, 'Bob');
            """
        )
        conn.commit()
    finally:
        conn.close()
    return str(db_path)


@pytest.mark.asyncio
async def test_select_succeeds(sqlite_db: str) -> None:
    result = await execute_sql(sqlite_db, "SELECT id, name FROM students ORDER BY id")
    assert result.success
    assert result.rows == [(1, "Alice"), (2, "Bob")]


@pytest.mark.asyncio
async def test_insert_blocked_by_guardrail(sqlite_db: str) -> None:
    result = await execute_sql(sqlite_db, "INSERT INTO students (id, name) VALUES (3, 'Eve')")
    assert result.success is False
    assert result.error is not None
    assert "write SQL rejected by guardrail" in result.error

    check = await execute_sql(sqlite_db, "SELECT COUNT(*) FROM students")
    assert check.success
    assert check.rows == [(2,)]


@pytest.mark.asyncio
async def test_multi_statement_blocked(sqlite_db: str) -> None:
    result = await execute_sql(
        sqlite_db, "SELECT 1; DROP TABLE students"
    )
    assert result.success is False
    assert "guardrail" in (result.error or "")


@pytest.mark.asyncio
async def test_allow_write_opt_in(sqlite_db: str) -> None:
    """allow_write=True bypasses the guardrail for user-confirmed paths."""
    result = await execute_sql(
        sqlite_db,
        "INSERT INTO students (id, name) VALUES (3, 'Eve')",
        allow_write=True,
    )
    assert result.success
    check = await execute_sql(sqlite_db, "SELECT COUNT(*) FROM students")
    assert check.success
    assert check.rows == [(3,)]
