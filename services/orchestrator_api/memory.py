"""Checkpointer factory for orchestrator session memory.

Phase 3 uses SQLite by default (``langgraph-checkpoint-sqlite``). The
Postgres path is stubbed behind ``ORCH_CHECKPOINTER_BACKEND=postgres`` and
will be activated in Phase 10 when Docker Compose brings up a Postgres
container. Tests always use an in-memory SQLite database.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class CheckpointerHandle:
    """Owns the underlying aiosqlite connection and the checkpointer.

    Stored on ``app.state`` so the FastAPI lifespan can close the connection
    cleanly on shutdown. Only the ``saver`` attribute is used by LangGraph.
    """

    def __init__(self, saver: AsyncSqliteSaver, conn: aiosqlite.Connection) -> None:
        self.saver = saver
        self._conn = conn

    async def aclose(self) -> None:
        await self._conn.close()


def _default_sqlite_path() -> str:
    raw = os.getenv("ORCH_SQLITE_PATH", ".cache/orchestrator/sessions.sqlite")
    if raw == ":memory:":
        return raw
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


async def create_checkpointer(
    *,
    backend: str | None = None,
    sqlite_path: str | None = None,
    **_: Any,
) -> CheckpointerHandle:
    """Build and initialise the checkpointer selected by ``backend``.

    Precedence: explicit ``backend`` arg > ``ORCH_CHECKPOINTER_BACKEND`` env
    > default ``"sqlite"``. ``postgres`` is reserved for Phase 10.
    """
    resolved = (backend or os.getenv("ORCH_CHECKPOINTER_BACKEND") or "sqlite").lower()
    if resolved == "postgres":
        raise NotImplementedError(
            "postgres checkpointer backend lands in Phase 10; "
            "run with ORCH_CHECKPOINTER_BACKEND=sqlite for now."
        )
    if resolved != "sqlite":
        raise ValueError(f"unknown checkpointer backend: {resolved!r}")

    path = sqlite_path or _default_sqlite_path()
    conn = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return CheckpointerHandle(saver=saver, conn=conn)
