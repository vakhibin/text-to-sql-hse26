"""Database discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.text_to_sql_api import databases as db_service
from services.text_to_sql_api.schemas import DatabasesResponse, SchemaResponse

router = APIRouter(prefix="/databases", tags=["databases"])


@router.get("", response_model=DatabasesResponse)
async def list_databases(
    schema_root: str | None = Query(default=None, description="Override Spider/BIRD root."),
) -> DatabasesResponse:
    return await db_service.get_databases(schema_root=schema_root)


@router.get("/{db_id}/schema", response_model=SchemaResponse)
async def get_database_schema(
    db_id: str,
    schema_root: str | None = Query(default=None, description="Override Spider/BIRD root."),
) -> SchemaResponse:
    schema = await db_service.get_schema(db_id, schema_root=schema_root)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"db_id {db_id!r} not found")
    return schema
