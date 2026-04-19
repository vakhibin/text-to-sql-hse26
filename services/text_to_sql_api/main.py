"""FastAPI application for the text-to-SQL microservice.

Composition root: creates the app, mounts routers, owns no business logic.

Endpoints (see routers/ for implementations):

- ``GET  /health``
- ``GET  /databases``
- ``GET  /databases/{db_id}/schema``
- ``POST /run``
- ``POST /execute`` (read-only, guardrail-enforced)
- ``POST /refine``
- ``POST /explain``

HTTP status codes:

- ``200`` — request processed; check ``success``/``error`` in the body
- ``404`` — database or resource not found
- ``422`` — request validation error (FastAPI default)
- ``500`` — unexpected server failure
"""

from __future__ import annotations

from fastapi import FastAPI

from services.text_to_sql_api.routers import databases as databases_router
from services.text_to_sql_api.routers import health as health_router
from services.text_to_sql_api.routers import sql as sql_router

SERVICE_VERSION = "0.1.0"

app = FastAPI(title="Text-to-SQL API", version=SERVICE_VERSION)

app.include_router(health_router.router)
app.include_router(databases_router.router)
app.include_router(sql_router.router)
