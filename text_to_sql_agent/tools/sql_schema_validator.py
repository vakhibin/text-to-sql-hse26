"""Lightweight SQL-to-schema validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, traverse_scope


@dataclass
class SQLSchemaValidationResult:
    """Validation result for table/column references against loaded schema."""

    blocking_errors: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.blocking_errors

    @property
    def errors(self) -> list[str]:
        return self.blocking_errors

    def error_message(self) -> str:
        if not self.blocking_errors:
            return ""
        unique_errors = list(dict.fromkeys(self.blocking_errors))
        return "schema_validation: " + "; ".join(unique_errors)

    def warning_message(self) -> str:
        if not self.warnings:
            return ""
        unique_warnings = list(dict.fromkeys(self.warnings))
        return "schema_validation_warning: " + "; ".join(unique_warnings)


def _schema_columns_by_table(schema: dict[str, Any]) -> dict[str, set[str]]:
    return {
        str(table.get("name", "")): {
            str(column.get("name", ""))
            for column in table.get("columns", [])
            if str(column.get("name", "")).strip()
        }
        for table in schema.get("tables", [])
        if str(table.get("name", "")).strip()
    }


def _projected_columns(scope: Scope) -> set[str]:
    expression = getattr(scope, "expression", None)
    selects = getattr(expression, "selects", []) or []
    return {
        str(select.output_name).strip()
        for select in selects
        if str(getattr(select, "output_name", "")).strip()
    }


def _explicit_projection_aliases(scope: Scope) -> set[str]:
    expression = getattr(scope, "expression", None)
    selects = getattr(expression, "selects", []) or []
    return {
        str(select.alias).strip()
        for select in selects
        if str(getattr(select, "alias", "")).strip()
    }


def _source_columns(
    source: Scope | exp.Expression,
    schema_columns_by_table: dict[str, set[str]],
) -> tuple[str | None, set[str]]:
    if isinstance(source, Scope):
        return "derived", _projected_columns(source)

    if isinstance(source, exp.Table):
        table_name = str(source.name or "").strip()
        return table_name, set(schema_columns_by_table.get(table_name, set()))

    expression = getattr(source, "expression", None)
    selects = getattr(expression, "selects", []) or []
    derived_columns = {
        str(select.output_name).strip()
        for select in selects
        if str(getattr(select, "output_name", "")).strip()
    }
    return "derived", derived_columns


def validate_sql_schema_references(
    sql: str,
    schema: dict[str, Any],
) -> SQLSchemaValidationResult:
    """Validate referenced tables and columns against loaded schema."""
    schema_columns_by_table = _schema_columns_by_table(schema)
    if not sql.strip() or not schema_columns_by_table:
        return SQLSchemaValidationResult(blocking_errors=[], warnings=[])

    try:
        expression = parse_one(sql, read="sqlite")
    except ParseError as exc:
        return SQLSchemaValidationResult(blocking_errors=[f"parse error: {exc}"], warnings=[])
    except Exception as exc:
        return SQLSchemaValidationResult(blocking_errors=[f"unexpected parse error: {exc}"], warnings=[])

    blocking_errors: list[str] = []
    warnings: list[str] = []
    for scope in traverse_scope(expression):
        available_sources: dict[str, set[str]] = {}
        projected_aliases = _explicit_projection_aliases(scope)

        for alias, source in scope.sources.items():
            source_name, source_columns = _source_columns(source, schema_columns_by_table)
            if source_name and source_name != "derived" and source_name not in schema_columns_by_table:
                blocking_errors.append(f"unknown table '{source_name}'")
                continue
            available_sources[str(alias)] = source_columns

        for column in scope.columns:
            column_name = str(column.name or "").strip()
            qualifier = str(column.table or "").strip()
            if not column_name or column_name == "*":
                continue

            if qualifier:
                known_columns = available_sources.get(qualifier)
                if known_columns is None:
                    blocking_errors.append(f"unknown table or alias '{qualifier}' for column '{column_name}'")
                elif column_name not in known_columns:
                    blocking_errors.append(f"unknown column '{qualifier}.{column_name}'")
                continue

            if column_name in projected_aliases:
                continue

            matching_sources = [
                alias for alias, source_columns in available_sources.items() if column_name in source_columns
            ]
            if not matching_sources:
                warnings.append(f"unknown unqualified column '{column_name}'")
            elif len(matching_sources) > 1:
                warnings.append(f"ambiguous column '{column_name}'")

    unique_blocking_errors = list(dict.fromkeys(blocking_errors))
    unique_warnings = list(dict.fromkeys(warnings))
    return SQLSchemaValidationResult(
        blocking_errors=unique_blocking_errors,
        warnings=unique_warnings,
    )
