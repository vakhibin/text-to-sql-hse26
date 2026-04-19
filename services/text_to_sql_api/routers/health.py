"""Liveness probe router for the text-to-SQL service."""

from __future__ import annotations

from fastapi import APIRouter

from services.text_to_sql_api.schemas import HealthResponse

SERVICE_NAME = "text_to_sql_api"
SERVICE_VERSION = "0.1.0"

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)
