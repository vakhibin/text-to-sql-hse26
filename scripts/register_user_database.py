"""Register a user-provided SQLite database with the Text-to-SQL agent.

Generates a Spider-style ``tables.json`` entry from the SQLite schema
(via ``sqlite_master``, ``PRAGMA table_info``, ``PRAGMA foreign_key_list``)
and copies the SQLite file into the layout expected by
``text_to_sql_agent.tools.schema_loader``::

    <root>/
        tables.json                       # Spider-style schema catalogue
        database/
            <db_id>/
                <db_id>.sqlite            # the actual database file

After registration, point ``SPIDER_ROOT`` in ``.env`` at ``<root>`` and
restart the services. The new database will appear in
``GET /databases``, in the Streamlit UI database selector, and become
available to the orchestrator agent's ``run_text_to_sql`` /
``switch_database`` / ``describe_database`` tools.

Usage:

    uv run python scripts/register_user_database.py \\
        --sqlite /path/to/my.sqlite \\
        --db-id my_db \\
        --root databases/user_dbs

Re-register an existing db_id (overwrites tables.json entry and the
copied sqlite file):

    uv run python scripts/register_user_database.py ... --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

_NUMBER_TYPES = {
    "INT",
    "INTEGER",
    "REAL",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "NUMERIC",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "MEDIUMINT",
}
_TIME_TYPES = {"DATE", "DATETIME", "TIMESTAMP", "TIME"}
_BOOL_TYPES = {"BOOL", "BOOLEAN"}
_BLOB_TYPES = {"BLOB"}


def _classify_type(sql_type: str) -> str:
    """Map a raw SQLite column type string to a Spider-style coarse type."""
    raw = (sql_type or "").upper().split("(")[0].strip()
    if not raw:
        return "text"
    if raw in _NUMBER_TYPES:
        return "number"
    if raw in _TIME_TYPES:
        return "time"
    if raw in _BOOL_TYPES:
        return "boolean"
    if raw in _BLOB_TYPES:
        return "others"
    for kw in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE"):
        if kw in raw:
            return "number"
    for kw in ("DATE", "TIME"):
        if kw in raw:
            return "time"
    return "text"


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table', 'view') "
        "AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [str(r[0]) for r in rows]


def build_tables_entry(db_id: str, sqlite_path: Path) -> dict[str, Any]:
    """Build a Spider-style tables.json entry from a SQLite database file."""
    with sqlite3.connect(str(sqlite_path)) as conn:
        table_names = _list_tables(conn)
        column_names_original: list[list[Any]] = [[-1, "*"]]
        column_names: list[list[Any]] = [[-1, "*"]]
        column_types: list[str] = ["text"]
        primary_keys: list[int] = []
        col_lookup: dict[tuple[str, str], int] = {}

        for table_idx, table in enumerate(table_names):
            cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            # (cid, name, type, notnull, dflt_value, pk)
            for col in cols:
                col_name = str(col[1])
                col_type = str(col[2]) if col[2] is not None else ""
                pk_flag = int(col[5]) if col[5] is not None else 0
                abs_idx = len(column_names_original)
                column_names_original.append([table_idx, col_name])
                column_names.append([table_idx, col_name])
                column_types.append(_classify_type(col_type))
                col_lookup[(table, col_name)] = abs_idx
                if pk_flag:
                    primary_keys.append(abs_idx)

        foreign_keys: list[list[int]] = []
        for table in table_names:
            fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
            # (id, seq, parent_table, from_col, to_col, on_update, on_delete, match)
            for fk in fks:
                parent_table = str(fk[2])
                from_col = str(fk[3])
                to_col = str(fk[4])
                from_key = (table, from_col)
                to_key = (parent_table, to_col)
                if from_key not in col_lookup or to_key not in col_lookup:
                    continue
                foreign_keys.append([col_lookup[from_key], col_lookup[to_key]])

    return {
        "db_id": db_id,
        "table_names_original": table_names,
        "table_names": table_names,
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }


def _load_existing_catalogue(tables_json: Path) -> list[dict[str, Any]]:
    if not tables_json.exists():
        return []
    with tables_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(
            f"Expected a JSON array at {tables_json}, got {type(data).__name__}"
        )
    return data


def _upsert_entry(
    catalogue: list[dict[str, Any]],
    entry: dict[str, Any],
    *,
    force: bool,
) -> list[dict[str, Any]]:
    db_id = entry["db_id"]
    existing_idx = next(
        (i for i, e in enumerate(catalogue) if e.get("db_id") == db_id), None
    )
    if existing_idx is None:
        return catalogue + [entry]
    if not force:
        raise SystemExit(
            f"db_id '{db_id}' already exists in tables.json. "
            f"Re-run with --force to overwrite."
        )
    updated = list(catalogue)
    updated[existing_idx] = entry
    return updated


def _copy_sqlite(sqlite_src: Path, root: Path, db_id: str, *, force: bool) -> Path:
    dst_dir = root / "database" / db_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{db_id}.sqlite"
    if dst.exists() and not force:
        raise SystemExit(
            f"Target sqlite already exists at {dst}. Re-run with --force to overwrite."
        )
    shutil.copyfile(sqlite_src, dst)
    return dst


def _print_summary(
    *,
    root: Path,
    db_id: str,
    sqlite_dst: Path,
    table_count: int,
    column_count: int,
    fk_count: int,
) -> None:
    abs_root = root.resolve()
    print()
    print(f"Registered '{db_id}'")
    print(f"  root:        {abs_root}")
    print(f"  sqlite:      {sqlite_dst.resolve()}")
    print(f"  tables.json: {(root / 'tables.json').resolve()}")
    print(f"  tables:      {table_count}")
    print(f"  columns:     {column_count}")
    print(f"  foreign keys: {fk_count}")
    print()
    print("Next steps:")
    print(f"  1. Set SPIDER_ROOT={abs_root} in your .env")
    print(f"     (or pass evidence/db_id directly when calling /run)")
    print("  2. Restart the services (docker compose restart or rerun uvicorn)")
    print(f"  3. The new database will appear as '{db_id}' in GET /databases")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sqlite",
        required=True,
        type=Path,
        help="Path to the source SQLite database file to register.",
    )
    parser.add_argument(
        "--db-id",
        required=True,
        help="Identifier for the database (used in API responses and SQL prompts).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("databases/user_dbs"),
        help="Catalogue root directory. Will be created if missing. "
        "Set SPIDER_ROOT to this path after registration. "
        "Default: databases/user_dbs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing db_id entry in tables.json and the copied sqlite.",
    )
    args = parser.parse_args(argv)

    sqlite_src: Path = args.sqlite
    if not sqlite_src.is_file():
        parser.error(f"SQLite file not found: {sqlite_src}")

    db_id: str = args.db_id.strip()
    if not db_id:
        parser.error("--db-id must not be empty")

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    tables_json = root / "tables.json"

    entry = build_tables_entry(db_id, sqlite_src)
    if not entry["table_names_original"]:
        parser.error(
            f"No user tables found in {sqlite_src}. "
            "Spider/BIRD-style tables.json requires at least one table."
        )

    catalogue = _load_existing_catalogue(tables_json)
    catalogue = _upsert_entry(catalogue, entry, force=args.force)
    with tables_json.open("w", encoding="utf-8") as f:
        json.dump(catalogue, f, ensure_ascii=False, indent=2)

    sqlite_dst = _copy_sqlite(sqlite_src, root, db_id, force=args.force)

    table_count = len(entry["table_names_original"])
    column_count = len(entry["column_names_original"]) - 1
    fk_count = len(entry["foreign_keys"])
    _print_summary(
        root=root,
        db_id=db_id,
        sqlite_dst=sqlite_dst,
        table_count=table_count,
        column_count=column_count,
        fk_count=fk_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
