"""FastAPI application exposing the text-to-SQL pipeline as a microservice.

Phase 0: only the health endpoint is wired up. Pipeline endpoints are added in Phase 2.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

SERVICE_NAME = "text_to_sql_api"
SERVICE_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


app = FastAPI(title="Text-to-SQL API", version=SERVICE_VERSION)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)
