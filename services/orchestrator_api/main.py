"""FastAPI application for the conversational orchestrator.

Composition root: creates the app, owns the lifespan (checkpointer +
compiled LangGraph), mounts routers.

Endpoints (see routers/ for implementations):

- ``GET    /health``
- ``POST   /chat``
- ``GET    /sessions/{session_id}``
- ``DELETE /sessions/{session_id}``

The single compiled LangGraph and its checkpointer are created at startup
and stored on ``app.state`` so routers can reach them via ``request.app.state``.
Session isolation is provided by LangGraph's ``thread_id`` mechanism
(``thread_id == session_id``), backed by the checkpointer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.orchestrator_api.graph import build_orchestrator_graph
from services.orchestrator_api.memory import create_checkpointer
from services.orchestrator_api.routers import chat as chat_router
from services.orchestrator_api.routers import health as health_router
from services.orchestrator_api.routers import sessions as sessions_router

SERVICE_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    handle = await create_checkpointer()
    graph = build_orchestrator_graph(handle.saver)
    app.state.checkpointer = handle
    app.state.graph = graph
    try:
        yield
    finally:
        await handle.aclose()


app = FastAPI(title="Orchestrator API", version=SERVICE_VERSION, lifespan=lifespan)

app.include_router(health_router.router)
app.include_router(chat_router.router)
app.include_router(sessions_router.router)
