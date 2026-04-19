"""Shared fixtures for service tests.

Builds a tiny self-contained Spider-shaped dataset under ``tmp_path`` so
``/databases``, ``/databases/{id}/schema``, and ``/execute`` can exercise the
real schema loader and executor without touching real Spider data.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest


@pytest.fixture()
def fake_schema_root(tmp_path: Path) -> str:
    root = tmp_path / "spider_fake"
    root.mkdir()

    tables = [
        {
            "db_id": "toy",
            "table_names_original": ["students", "courses"],
            "table_names": ["students", "courses"],
            "column_names_original": [
                [-1, "*"],
                [0, "id"],
                [0, "name"],
                [0, "age"],
                [1, "id"],
                [1, "title"],
                [1, "student_id"],
            ],
            "column_names": [
                [-1, "*"],
                [0, "id"],
                [0, "name"],
                [0, "age"],
                [1, "id"],
                [1, "title"],
                [1, "student_id"],
            ],
            "column_types": ["text", "integer", "text", "integer", "integer", "text", "integer"],
            "primary_keys": [1, 4],
            "foreign_keys": [[6, 1]],
        }
    ]
    (root / "tables.json").write_text(json.dumps(tables), encoding="utf-8")

    db_dir = root / "database" / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
            CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, student_id INTEGER);
            INSERT INTO students (id, name, age) VALUES
                (1, 'Alice', 20), (2, 'Bob', 22), (3, 'Eve', 19);
            INSERT INTO courses (id, title, student_id) VALUES
                (10, 'Math', 1), (11, 'SQL', 2);
            """
        )
        conn.commit()
    finally:
        conn.close()

    return str(root)


@pytest.fixture()
async def api_client() -> AsyncIterator[httpx.AsyncClient]:
    """Async HTTP client bound to the text-to-sql FastAPI app in-process."""
    from services.text_to_sql_api.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
