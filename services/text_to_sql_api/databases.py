"""Database discovery helpers for the text-to-SQL service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from text_to_sql_agent.config import settings
from text_to_sql_agent.tools.schema_loader import (
    list_database_ids,
    load_schema,
)

from services.text_to_sql_api.schemas import (
    ColumnInfo,
    DatabaseInfo,
    DatabasesResponse,
    ForeignKey,
    SchemaResponse,
    TableInfo,
)


def _resolve_schema_root(schema_root: str | None) -> str:
    return schema_root or settings.spider_root


async def get_databases(schema_root: str | None = None) -> DatabasesResponse:
    root = _resolve_schema_root(schema_root)
    db_ids = list_database_ids(root)
    databases: list[DatabaseInfo] = []
    for db_id in db_ids:
        try:
            schema = await load_schema(db_id, spider_root=root, with_sample_values=False)
        except FileNotFoundError:
            databases.append(DatabaseInfo(db_id=db_id, db_path=None, num_tables=0))
            continue
        db_path = schema.get("db_path")
        db_path_str = str(db_path) if db_path and Path(db_path).exists() else None
        databases.append(
            DatabaseInfo(
                db_id=db_id,
                db_path=db_path_str,
                num_tables=len(schema.get("tables", [])),
            )
        )
    return DatabasesResponse(schema_root=str(Path(root).resolve()), databases=databases)


async def get_schema(db_id: str, schema_root: str | None = None) -> SchemaResponse | None:
    root = _resolve_schema_root(schema_root)
    try:
        raw = await load_schema(db_id, spider_root=root, with_sample_values=True)
    except (ValueError, FileNotFoundError):
        return None
    return _to_schema_response(raw)


def _to_schema_response(raw: dict[str, Any]) -> SchemaResponse:
    tables: list[TableInfo] = []
    for table in raw.get("tables", []):
        columns = [
            ColumnInfo(
                name=str(col.get("name", "")),
                type=str(col.get("type", "")),
                sample_values=[str(v) for v in col.get("sample_values", []) or []],
            )
            for col in table.get("columns", [])
        ]
        fks = [
            ForeignKey(
                column=str(fk.get("column", "")),
                ref_table=str(fk.get("ref_table", "")),
                ref_column=str(fk.get("ref_column", "")),
            )
            for fk in table.get("foreign_keys", []) or []
        ]
        tables.append(
            TableInfo(
                name=str(table.get("name", "")),
                columns=columns,
                primary_keys=[str(pk) for pk in table.get("primary_keys", []) or []],
                foreign_keys=fks,
            )
        )
    db_path = raw.get("db_path")
    return SchemaResponse(
        db_id=str(raw.get("db_id", "")),
        db_path=str(db_path) if db_path else None,
        tables=tables,
    )
