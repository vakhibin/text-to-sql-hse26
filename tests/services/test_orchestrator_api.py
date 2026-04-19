"""Tests for the orchestrator FastAPI service.

Covers:

- ``/health``
- ``/chat`` with an in-memory SQLite checkpointer and a fake LLM that
  reports how many human turns it has seen; this verifies both round-trip
  correctness and that memory persists across turns within a session and is
  isolated between sessions.
- ``/sessions/{id}`` (state snapshot + 404 for unknown session)
- ``/sessions/{id}`` delete (gracefully reports ``deleted=False`` when the
  SQLite checkpointer has no thread-delete API)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from orchestrator_agent import agent as agent_mod
from orchestrator_agent.graph import build_orchestrator_graph
from orchestrator_agent.memory import create_checkpointer
from services.orchestrator_api.main import app


class MemoryEchoLLM(BaseChatModel):
    """Deterministic fake LLM used to verify session memory.

    Reply format::

        turn=<N> last=<last_user_message>

    ``N`` is the number of ``HumanMessage`` instances the model sees. With a
    working checkpointer, ``N`` grows within a session and resets when the
    session id changes.
    """

    @property
    def _llm_type(self) -> str:
        return "memory_echo_test"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **_: Any,
    ) -> ChatResult:
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        last = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            "",
        )
        content = f"turn={human_count} last={last}"
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture()
async def orch_client() -> AsyncIterator[httpx.AsyncClient]:
    """FastAPI client with in-memory SQLite checkpointer and a fake LLM."""
    handle = await create_checkpointer(sqlite_path=":memory:")
    graph = build_orchestrator_graph(handle.saver)
    app.state.checkpointer = handle
    app.state.graph = graph
    agent_mod.set_chat_model(MemoryEchoLLM())

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        agent_mod.set_chat_model(None)
        await handle.aclose()


@pytest.mark.asyncio
async def test_health(orch_client: httpx.AsyncClient) -> None:
    resp = await orch_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "orchestrator_api"


@pytest.mark.asyncio
async def test_chat_round_trip(orch_client: httpx.AsyncClient) -> None:
    resp = await orch_client.post(
        "/chat",
        json={"session_id": "s1", "user_id": "u1", "message": "hello"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert body["user_id"] == "u1"
    assert body["reply"].startswith("turn=1")
    assert "hello" in body["reply"]
    assert len(body["messages_delta"]) == 2
    assert body["messages_delta"][0]["role"] == "human"
    assert body["messages_delta"][1]["role"] == "ai"


@pytest.mark.asyncio
async def test_chat_memory_persists_within_session(orch_client: httpx.AsyncClient) -> None:
    await orch_client.post("/chat", json={"session_id": "s2", "message": "first"})
    await orch_client.post("/chat", json={"session_id": "s2", "message": "second"})
    resp3 = await orch_client.post(
        "/chat", json={"session_id": "s2", "message": "third"}
    )
    assert resp3.status_code == 200
    assert resp3.json()["reply"].startswith("turn=3")


@pytest.mark.asyncio
async def test_chat_sessions_are_isolated(orch_client: httpx.AsyncClient) -> None:
    await orch_client.post("/chat", json={"session_id": "sA", "message": "one"})
    await orch_client.post("/chat", json={"session_id": "sA", "message": "two"})
    resp_b = await orch_client.post(
        "/chat", json={"session_id": "sB", "message": "hi"}
    )
    assert resp_b.json()["reply"].startswith("turn=1")

    resp_a = await orch_client.post(
        "/chat", json={"session_id": "sA", "message": "three"}
    )
    assert resp_a.json()["reply"].startswith("turn=3")


@pytest.mark.asyncio
async def test_chat_propagates_active_db_id(orch_client: httpx.AsyncClient) -> None:
    resp = await orch_client.post(
        "/chat",
        json={"session_id": "sdb", "message": "use this db", "active_db_id": "concert_singer"},
    )
    assert resp.status_code == 200
    assert resp.json()["active_db_id"] == "concert_singer"


@pytest.mark.asyncio
async def test_get_session_history(orch_client: httpx.AsyncClient) -> None:
    await orch_client.post(
        "/chat",
        json={"session_id": "hist", "message": "hi", "active_db_id": "toy"},
    )
    await orch_client.post("/chat", json={"session_id": "hist", "message": "more"})

    resp = await orch_client.get("/sessions/hist")
    assert resp.status_code == 200
    body = resp.json()
    roles = [m["role"] for m in body["messages"]]
    assert roles.count("human") == 2
    assert roles.count("ai") == 2
    assert body["active_db_id"] == "toy"


@pytest.mark.asyncio
async def test_get_session_unknown_returns_404(orch_client: httpx.AsyncClient) -> None:
    resp = await orch_client.get("/sessions/does_not_exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_best_effort(orch_client: httpx.AsyncClient) -> None:
    """SQLite checkpointer has no thread-delete API; endpoint reports deleted=False."""
    await orch_client.post("/chat", json={"session_id": "del_me", "message": "hi"})
    resp = await orch_client.delete("/sessions/del_me")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "del_me"
    assert isinstance(body["deleted"], bool)


@pytest.mark.asyncio
async def test_chat_validation_error(orch_client: httpx.AsyncClient) -> None:
    resp = await orch_client.post("/chat", json={"session_id": "", "message": "x"})
    assert resp.status_code == 422

    resp = await orch_client.post("/chat", json={"session_id": "s", "message": ""})
    assert resp.status_code == 422
