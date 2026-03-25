"""Schema loading and mSchema formatting for Spider and BIRD-like datasets."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

from text_to_sql_agent.config import settings

SchemaLayout = Literal["spider", "bird"]


def _default_spider_root() -> Path:
    return Path(settings.spider_root)


def _sample_values(db_path: Path, table: str, column: str, limit: int = 3) -> list[str]:
    if not db_path.exists():
        return []
    query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit};"
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(query).fetchall()
        return [str(row[0]) for row in rows if row and row[0] is not None]
    except Exception:
        return []


def _spider_tables_path(dataset_root: Path) -> Path:
    tables_path = dataset_root / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"Spider tables.json not found at: {tables_path}")
    return tables_path


def _candidate_bird_tables_paths(dataset_root: Path) -> list[Path]:
    # Layouts supported:
    # 1) databases/bird/dev/dev_tables.json + databases/bird/dev_databases/...
    # 2) databases/bird/dev_tables.json + databases/bird/dev_databases/...
    # 3) legacy: databases/bird/dev/dev_databases/... (nested sqlite folders)
    candidates = [
        dataset_root / "dev" / "dev_tables.json",
        dataset_root / "dev_tables.json",
        dataset_root / "train" / "train_tables.json",
        dataset_root / "train_tables.json",
    ]
    return [p for p in candidates if p.exists()]


def _load_tables_json(tables_path: Path) -> list[dict[str, Any]]:
    with tables_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bird_sqlite_path_candidates(dataset_root: Path, db_id: str) -> list[Path]:
    # Prefer existing sqlite first; also support legacy nesting.
    candidates = [
        dataset_root / "dev_databases" / db_id / f"{db_id}.sqlite",
        dataset_root / "train_databases" / db_id / f"{db_id}.sqlite",
        dataset_root / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite",
        dataset_root / "train" / "train_databases" / db_id / f"{db_id}.sqlite",
    ]
    return candidates


def _sqlite_path_for_dataset(dataset_root: Path, db_id: str, layout: SchemaLayout) -> Path:
    if layout == "spider":
        return dataset_root / "database" / db_id / f"{db_id}.sqlite"

    for cand in _bird_sqlite_path_candidates(dataset_root, db_id):
        if cand.exists():
            return cand

    # Fallback to the most likely location (dev_databases) even if it doesn't exist.
    return dataset_root / "dev_databases" / db_id / f"{db_id}.sqlite"


def _build_schema_for_db(
    *,
    db_id: str,
    dataset_root: Path,
    layout: SchemaLayout,
    with_sample_values: bool,
    sample_limit: int,
) -> dict[str, Any]:
    if layout == "spider":
        tables_path = _spider_tables_path(dataset_root)
        all_schemas = _load_tables_json(tables_path)
        schema = next((item for item in all_schemas if item.get("db_id") == db_id), None)
        if schema is None:
            raise ValueError(f"db_id '{db_id}' not found in {tables_path.name}")
    else:
        candidates = _candidate_bird_tables_paths(dataset_root)
        if not candidates:
            raise FileNotFoundError(
                f"BIRD dev/train tables.json not found under dataset root: {dataset_root}"
            )
        schema = None
        used_tables_path: Path | None = None
        for tables_path in candidates:
            all_schemas = _load_tables_json(tables_path)
            schema = next((item for item in all_schemas if item.get("db_id") == db_id), None)
            if schema is not None:
                used_tables_path = tables_path
                break
        if schema is None:
            used = used_tables_path.name if used_tables_path else "tables.json"
            raise ValueError(f"db_id '{db_id}' not found in {used} (BIRD)")

    table_names = schema.get("table_names_original", schema.get("table_names", []))
    column_names = schema.get("column_names_original", schema.get("column_names", []))
    column_types = schema.get("column_types", [])
    primary_keys = set(schema.get("primary_keys", []))
    foreign_keys = schema.get("foreign_keys", [])

    db_path = _sqlite_path_for_dataset(dataset_root, db_id, layout)
    if layout == "bird" and not db_path.exists():
        # Execution filter needs a real DB; better fail fast for inference.
        raise FileNotFoundError(f"BIRD sqlite not found at: {db_path}")

    tables: list[dict[str, Any]] = [
        {"name": table_name, "columns": [], "primary_keys": [], "foreign_keys": []}
        for table_name in table_names
    ]

    column_index_to_ref: dict[int, tuple[str, str]] = {}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            continue
        table_name = table_names[table_idx]
        col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
        samples = _sample_values(db_path, table_name, col_name, sample_limit) if with_sample_values else []
        column_record = {
            "name": col_name,
            "type": str(col_type).upper(),
            "sample_values": samples,
        }
        tables[table_idx]["columns"].append(column_record)
        column_index_to_ref[col_idx] = (table_name, col_name)
        if col_idx in primary_keys:
            tables[table_idx]["primary_keys"].append(col_name)

    table_map = {tbl["name"]: tbl for tbl in tables}
    for from_idx, to_idx in foreign_keys:
        if from_idx not in column_index_to_ref or to_idx not in column_index_to_ref:
            continue
        from_table, from_col = column_index_to_ref[from_idx]
        to_table, to_col = column_index_to_ref[to_idx]
        table_map[from_table]["foreign_keys"].append(
            {"column": from_col, "ref_table": to_table, "ref_column": to_col}
        )

    return {
        "db_id": db_id,
        "tables": tables,
        "db_path": str(db_path),
    }


async def load_schema(
    db_id: str,
    *,
    spider_root: str | Path | None = None,
    layout: SchemaLayout = "spider",
    with_sample_values: bool = True,
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Load Spider or BIRD schema for database id with optional sample values."""
    root = Path(spider_root) if spider_root else _default_spider_root()
    return await asyncio.to_thread(
        _build_schema_for_db,
        db_id=db_id,
        dataset_root=root,
        layout=layout,
        with_sample_values=with_sample_values,
        sample_limit=sample_limit,
    )


def list_db_ids(dataset_root: str | Path) -> list[str]:
    """Return sorted db_id values for either Spider (`tables.json`) or BIRD (`*_tables.json`)."""
    root = Path(dataset_root)
    if (root / "tables.json").exists():
        all_schemas = _load_tables_json(root / "tables.json")
        return sorted({str(item["db_id"]) for item in all_schemas if item.get("db_id")})

    db_ids: set[str] = set()
    for tables_path in _candidate_bird_tables_paths(root):
        all_schemas = _load_tables_json(tables_path)
        for item in all_schemas:
            if item.get("db_id"):
                db_ids.add(str(item["db_id"]))
    return sorted(db_ids)


def to_mschema(schema: dict[str, Any]) -> str:
    """Convert schema to compact mSchema-like representation."""
    lines: list[str] = []
    for table in schema.get("tables", []):
        columns: list[str] = []
        for col in table.get("columns", []):
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "TEXT")
            samples = col.get("sample_values", [])
            sample_part = f" sample={samples}" if samples else ""
            columns.append(f"{col_name}:{col_type}{sample_part}")

        fk_values = table.get("foreign_keys", [])
        fk_part = ""
        if fk_values:
            links = [f"{fk['column']}->{fk['ref_table']}.{fk['ref_column']}" for fk in fk_values]
            fk_part = f" fk=[{'; '.join(links)}]"

        pk_values = table.get("primary_keys", [])
        pk_part = f" pk={pk_values}" if pk_values else ""
        lines.append(f"{table.get('name')}({', '.join(columns)}){pk_part}{fk_part}")
    return "\n".join(lines)

