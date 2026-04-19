"""FastAPI application for the conversational orchestrator.

Composition root: creates the app, owns the lifespan (HTTP client,
checkpointer, compiled LangGraph), mounts routers.

Endpoints (see routers/ for implementations):

- ``GET    /health``
- ``POST   /chat``
- ``GET    /sessions/{session_id}``
- ``DELETE /sessions/{session_id}``

The compiled graph, its checkpointer, and the HTTP client to
``text_to_sql_api`` are created at startup and stored on ``app.state`` so
routers can reach them via ``request.app.state``. Session isolation is
provided by LangGraph's ``thread_id`` mechanism (``thread_id == session_id``),
backed by the checkpointer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from orchestrator_agent.clients.text_to_sql import TextToSQLClient
from orchestrator_agent.config import settings
from orchestrator_agent.graph import build_orchestrator_graph
from orchestrator_agent.memory import create_checkpointer
from orchestrator_agent.tools import make_all_tools
from services.orchestrator_api.routers import chat as chat_router
from services.orchestrator_api.routers import health as health_router
from services.orchestrator_api.routers import sessions as sessions_router

SERVICE_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    handle = await create_checkpointer()
    client = TextToSQLClient(
        base_url=settings.text_to_sql_api_url,
        timeout_s=settings.text_to_sql_api_timeout_s,
    )
    tools = make_all_tools(client)
    graph = build_orchestrator_graph(handle.saver, tools=tools)

    app.state.checkpointer = handle
    app.state.text_to_sql_client = client
    app.state.tools = tools
    app.state.graph = graph
    try:
        yield
    finally:
        await client.aclose()
        await handle.aclose()


app = FastAPI(title="Orchestrator API", version=SERVICE_VERSION, lifespan=lifespan)

app.include_router(health_router.router)
app.include_router(chat_router.router)
app.include_router(sessions_router.router)
