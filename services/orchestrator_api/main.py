"""FastAPI application exposing the conversational orchestrator agent.

Phase 0: only the health endpoint is wired up. The /chat endpoint and
tool-calling agent are added in Phase 3+.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

SERVICE_NAME = "orchestrator_api"
SERVICE_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


app = FastAPI(title="Orchestrator API", version=SERVICE_VERSION)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)
