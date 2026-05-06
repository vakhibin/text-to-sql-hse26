"""Schema loading and mSchema formatting for Spider-like datasets."""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from text_to_sql_agent.config import settings


_tables_json_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
_schema_cache: dict[tuple[str, str, bool, int, str], dict[str, Any]] = {}
_mschema_lines_cache: dict[tuple[str, str, bool, int, str], tuple[tuple[str, str], ...]] = {}
_cache_lock = Lock()

# Increment when SQLite discovery / db_path semantics change (invalidates in-process caches).
_SCHEMA_RESOLUTION_REVISION = 1


def _default_spider_root() -> Path:
    return Path(settings.spider_root)


def _root_key(spider_root: Path) -> str:
    return str(spider_root.resolve())


def _schema_cache_key(
    *,
    db_id: str,
    spider_root: Path,
    with_sample_values: bool,
    sample_limit: int,
    spider_schema_variant: str,
) -> tuple[str, str, bool, int, str, int]:
    return (
        _root_key(spider_root),
        db_id,
        with_sample_values,
        sample_limit,
        spider_schema_variant,
        _SCHEMA_RESOLUTION_REVISION,
    )


def _load_tables_json(spider_root: Path, *, spider_schema_variant: str = "default") -> list[dict[str, Any]]:
    cache_key = (_root_key(spider_root), spider_schema_variant)
    with _cache_lock:
        cached = _tables_json_cache.get(cache_key)
    if cached is not None:
        return cached

    if spider_schema_variant == "test":
        tables_path = spider_root / "test_tables.json"
        if not tables_path.is_file():
            raise FileNotFoundError(f"Spider test schema file not found: {tables_path}")
    else:
        candidates = [spider_root / "tables.json", spider_root / "dev_tables.json"]
        tables_path = next((path for path in candidates if path.exists()), None)
        if tables_path is None:
            raise FileNotFoundError(
                "Schema tables file not found. Tried:\n" + "\n".join(str(path) for path in candidates)
            )
    with tables_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    with _cache_lock:
        _tables_json_cache[cache_key] = data
    return data


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


def _repair_db_parent_if_placeholder_sqlite_parent(db_sqlite_path: Path) -> None:
    """If ``db_sqlite_path`` parent exists as a file (often 0-byte ZIP dir bug), fix layout."""
    parent = db_sqlite_path.parent
    if not parent.exists():
        return
    if parent.is_dir():
        return
    if not parent.is_file():
        return
    try:
        if parent.stat().st_size == 0:
            parent.unlink()
            parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _pick_existing_sqlite(db_candidates: list[Path]) -> Path:
    """Return first candidate path that exists as a file; repair common broken test_database layout."""
    for path in db_candidates:
        _repair_db_parent_if_placeholder_sqlite_parent(path)
        if path.is_file():
            return path
        parent = path.parent
        if parent.is_dir():
            for alt in sorted(parent.glob("*.sqlite")):
                if alt.is_file():
                    return alt
    return db_candidates[0]


def _build_schema_for_db(
    *,
    db_id: str,
    spider_root: Path,
    with_sample_values: bool,
    sample_limit: int,
    spider_schema_variant: str = "default",
) -> dict[str, Any]:
    all_schemas = _load_tables_json(spider_root, spider_schema_variant=spider_schema_variant)
    schema = next((item for item in all_schemas if item.get("db_id") == db_id), None)
    if schema is None:
        hint = "test_tables.json" if spider_schema_variant == "test" else "tables.json"
        raise ValueError(f"db_id '{db_id}' not found in {hint}")

    table_names = schema.get("table_names_original", schema.get("table_names", []))
    column_names = schema.get("column_names_original", schema.get("column_names", []))
    column_types = schema.get("column_types", [])
    raw_primary_keys = schema.get("primary_keys", [])
    primary_keys: set[int] = set()
    for pk in raw_primary_keys:
        if isinstance(pk, int):
            primary_keys.add(pk)
        elif isinstance(pk, (list, tuple)):
            primary_keys.update(idx for idx in pk if isinstance(idx, int))
    foreign_keys = schema.get("foreign_keys", [])

    if spider_schema_variant == "test":
        db_candidates = [
            spider_root / "test_database" / db_id / f"{db_id}.sqlite",
            spider_root / "database" / db_id / f"{db_id}.sqlite",
            spider_root / "dev_databases" / db_id / f"{db_id}.sqlite",
            spider_root / "mini_dev_databases" / db_id / f"{db_id}.sqlite",
            spider_root / "train_databases" / db_id / f"{db_id}.sqlite",
        ]
    else:
        db_candidates = [
            spider_root / "database" / db_id / f"{db_id}.sqlite",
            spider_root / "dev_databases" / db_id / f"{db_id}.sqlite",
            spider_root / "mini_dev_databases" / db_id / f"{db_id}.sqlite",
            spider_root / "train_databases" / db_id / f"{db_id}.sqlite",
        ]
        # BIRD-style layouts: schema JSON under e.g. bird/dev/, SQLite under bird/dev_databases/.
        parent = spider_root.parent
        if str(parent) != str(spider_root):
            db_candidates.extend(
                [
                    parent / "dev_databases" / db_id / f"{db_id}.sqlite",
                    parent / "train_databases" / db_id / f"{db_id}.sqlite",
                    parent / "mini_dev_databases" / db_id / f"{db_id}.sqlite",
                ]
            )
    db_path = _pick_existing_sqlite(db_candidates)
    db_path_str = str(db_path.resolve()) if db_path.exists() else str(db_path)

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
        "db_path": db_path_str,
    }


async def load_schema(
    db_id: str,
    *,
    spider_root: str | Path | None = None,
    with_sample_values: bool = True,
    sample_limit: int = 3,
    spider_schema_variant: str = "default",
) -> dict[str, Any]:
    """Load Spider schema for database id with optional sample values."""
    root = Path(spider_root) if spider_root else _default_spider_root()
    cache_key = _schema_cache_key(
        db_id=db_id,
        spider_root=root,
        with_sample_values=with_sample_values,
        sample_limit=sample_limit,
        spider_schema_variant=spider_schema_variant,
    )
    with _cache_lock:
        cached = _schema_cache.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    schema = await asyncio.to_thread(
        _build_schema_for_db,
        db_id=db_id,
        spider_root=root,
        with_sample_values=with_sample_values,
        sample_limit=sample_limit,
        spider_schema_variant=spider_schema_variant,
    )
    with _cache_lock:
        _schema_cache[cache_key] = copy.deepcopy(schema)
    return schema


def _render_table_line(table: dict[str, Any]) -> str:
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
    return f"{table.get('name')}({', '.join(columns)}){pk_part}{fk_part}"


def schema_to_mschema(
    schema: dict[str, Any],
    *,
    schema_root: str | Path | None = None,
    with_sample_values: bool = True,
    sample_limit: int = 3,
    spider_schema_variant: str = "default",
) -> str:
    """Convert schema to compact mSchema-like representation with cache support."""
    db_id = str(schema.get("db_id", "")).strip()
    root = Path(schema_root) if schema_root else _default_spider_root()
    if not db_id:
        return to_mschema(schema)

    cache_key = _schema_cache_key(
        db_id=db_id,
        spider_root=root,
        with_sample_values=with_sample_values,
        sample_limit=sample_limit,
        spider_schema_variant=spider_schema_variant,
    )
    with _cache_lock:
        cached = _mschema_lines_cache.get(cache_key)

    if cached is None:
        cached = tuple(
            (str(table.get("name", "")), _render_table_line(table))
            for table in schema.get("tables", [])
            if str(table.get("name", "")).strip()
        )
        with _cache_lock:
            _mschema_lines_cache[cache_key] = cached

    selected_names = [str(table.get("name", "")) for table in schema.get("tables", [])]
    line_map = dict(cached)
    return "\n".join(line_map[name] for name in selected_names if name in line_map)


def to_mschema(schema: dict[str, Any]) -> str:
    """Convert schema to compact mSchema-like representation."""
    return "\n".join(_render_table_line(table) for table in schema.get("tables", []))

