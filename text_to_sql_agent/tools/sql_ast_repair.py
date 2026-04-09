"""Deterministic SQL repair via fuzzy-matching identifiers against schema."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import traverse_scope


@dataclass
class RepairResult:
    original_sql: str
    repaired_sql: str
    changes: list[str] = field(default_factory=list)

    @property
    def was_repaired(self) -> bool:
        return self.repaired_sql != self.original_sql


def _build_schema_index(
    full_schema: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Build case-insensitive lookup indexes from schema.

    Returns:
        table_index: lower(name) -> original_name
        column_index: lower(table_name) -> {lower(col_name) -> original_col_name}
    """
    table_index: dict[str, str] = {}
    column_index: dict[str, dict[str, str]] = {}

    for table in full_schema.get("tables", []):
        tname = str(table.get("name", "")).strip()
        if not tname:
            continue
        table_index[tname.lower()] = tname
        cols: dict[str, str] = {}
        for col in table.get("columns", []):
            cname = str(col.get("name", "")).strip()
            if cname:
                cols[cname.lower()] = cname
        column_index[tname.lower()] = cols

    return table_index, column_index


def _all_column_names(column_index: dict[str, dict[str, str]]) -> dict[str, str]:
    """Flatten all columns across tables for unqualified column repair."""
    merged: dict[str, str] = {}
    for cols in column_index.values():
        for lower_name, original in cols.items():
            merged.setdefault(lower_name, original)
    return merged


def _fuzzy_match(
    target: str,
    candidates: list[str],
    *,
    cutoff: float = 0.6,
) -> str | None:
    """Find closest match for target among candidates."""
    matches = get_close_matches(target.lower(), candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def _replace_identifier(sql: str, old: str, new: str) -> str:
    """Replace an identifier in SQL preserving word boundaries.

    Handles both quoted and unquoted identifiers.
    """
    pattern = re.compile(
        r'(?<![.\w])' + re.escape(old) + r'(?![.\w])',
        re.IGNORECASE,
    )
    return pattern.sub(new, sql)


def repair_sql_schema_references(
    sql: str,
    full_schema: dict[str, Any],
) -> RepairResult:
    """Try to fix table and column references by fuzzy-matching against schema.

    Repair strategy (conservative, ordered):
    1. Case-insensitive exact match for tables
    2. Fuzzy match for unknown tables (cutoff=0.6)
    3. Case-insensitive exact match for columns within resolved tables
    4. Fuzzy match for unknown columns within the correct table
    5. Fuzzy match for unknown unqualified columns across all tables
    """
    if not sql.strip() or not full_schema.get("tables"):
        return RepairResult(original_sql=sql, repaired_sql=sql)

    table_index, column_index = _build_schema_index(full_schema)
    all_columns = _all_column_names(column_index)
    table_names_lower = list(table_index.keys())
    changes: list[str] = []

    try:
        expression = parse_one(sql, read="sqlite")
    except (ParseError, Exception):
        return RepairResult(original_sql=sql, repaired_sql=sql)

    repaired = sql

    table_alias_to_real: dict[str, str] = {}
    for scope in traverse_scope(expression):
        for alias, source in scope.sources.items():
            if not isinstance(source, exp.Table):
                continue
            source_name = str(source.name or "").strip()
            if not source_name:
                continue

            real_alias = str(alias).strip()
            lower_name = source_name.lower()

            if lower_name in table_index:
                correct = table_index[lower_name]
                table_alias_to_real[real_alias] = correct
                if source_name != correct:
                    repaired = _replace_identifier(repaired, source_name, correct)
                    changes.append(f"table case fix: {source_name} -> {correct}")
                continue

            match = _fuzzy_match(source_name, table_names_lower)
            if match:
                correct = table_index[match]
                table_alias_to_real[real_alias] = correct
                repaired = _replace_identifier(repaired, source_name, correct)
                changes.append(f"table fuzzy fix: {source_name} -> {correct}")

    try:
        expression2 = parse_one(repaired, read="sqlite")
    except (ParseError, Exception):
        expression2 = expression

    for scope in traverse_scope(expression2):
        for column in scope.columns:
            col_name = str(column.name or "").strip()
            qualifier = str(column.table or "").strip()
            if not col_name or col_name == "*":
                continue

            if qualifier:
                real_table = table_alias_to_real.get(qualifier)
                if not real_table:
                    real_table = table_index.get(qualifier.lower())
                if not real_table:
                    continue

                table_cols = column_index.get(real_table.lower(), {})
                table_col_names_lower = list(table_cols.keys())

                if col_name.lower() in table_cols:
                    correct = table_cols[col_name.lower()]
                    if col_name != correct:
                        old_ref = f"{qualifier}.{col_name}"
                        new_ref = f"{qualifier}.{correct}"
                        repaired = repaired.replace(old_ref, new_ref)
                        changes.append(f"column case fix: {old_ref} -> {new_ref}")
                    continue

                match = _fuzzy_match(col_name, table_col_names_lower)
                if match:
                    correct = table_cols[match]
                    old_ref = f"{qualifier}.{col_name}"
                    new_ref = f"{qualifier}.{correct}"
                    repaired = repaired.replace(old_ref, new_ref)
                    changes.append(f"column fuzzy fix: {old_ref} -> {new_ref}")
            else:
                if col_name.lower() in all_columns:
                    correct = all_columns[col_name.lower()]
                    if col_name != correct:
                        repaired = _replace_identifier(repaired, col_name, correct)
                        changes.append(f"unqualified column case fix: {col_name} -> {correct}")
                    continue

                all_col_names_lower = list(all_columns.keys())
                match = _fuzzy_match(col_name, all_col_names_lower)
                if match:
                    correct = all_columns[match]
                    repaired = _replace_identifier(repaired, col_name, correct)
                    changes.append(f"unqualified column fuzzy fix: {col_name} -> {correct}")

    return RepairResult(
        original_sql=sql,
        repaired_sql=repaired,
        changes=changes,
    )
