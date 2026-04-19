"""Health-endpoint contract tests for all FastAPI services.

These tests run the ASGI app in-process via httpx.ASGITransport so no real
network port is needed. They catch obvious import/regression issues and
serve as the baseline test harness that later phases extend.
"""

from __future__ import annotations

import httpx
import pytest

from services.orchestrator_api.main import app as orchestrator_app
from services.text_to_sql_api.main import app as text_to_sql_app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("app", "expected_service"),
    [
        (text_to_sql_app, "text_to_sql_api"),
        (orchestrator_app, "orchestrator_api"),
    ],
)
async def test_health_endpoint(app, expected_service: str) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == expected_service
    assert payload["version"]
