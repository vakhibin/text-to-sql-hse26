"""Helpers for compact structural analysis of SQL candidates."""

from __future__ import annotations

from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError


def _output_name(select_expr: exp.Expression) -> str:
    output_name = str(getattr(select_expr, "output_name", "")).strip()
    if output_name:
        return output_name
    return select_expr.sql(dialect="sqlite")


def analyze_sql_candidate(sql: str) -> dict[str, Any]:
    """Return compact structural signals used by routing, judge, and refiner."""
    analysis: dict[str, Any] = {
        "table_names": [],
        "table_count": 0,
        "join_count": 0,
        "projection_count": 0,
        "projection_items": [],
        "has_aggregation": False,
        "has_group_by": False,
        "has_distinct": False,
        "has_subquery": False,
        "set_operation": "none",
        "literals": [],
        "parse_error": None,
    }

    if not sql.strip():
        analysis["parse_error"] = "empty_sql"
        return analysis

    try:
        expression = parse_one(sql, read="sqlite")
    except ParseError as exc:
        analysis["parse_error"] = str(exc)
        return analysis
    except Exception as exc:  # pragma: no cover - defensive runtime path
        analysis["parse_error"] = f"unexpected parse error: {exc}"
        return analysis

    table_names = [
        str(table.name).strip()
        for table in expression.find_all(exp.Table)
        if str(table.name).strip()
    ]
    projection_items = [
        _output_name(select_expr)
        for select_expr in getattr(expression, "selects", []) or []
    ]
    set_operation = "none"
    if any(isinstance(node, exp.Union) for node in expression.walk()):
        set_operation = "union"
    elif any(isinstance(node, exp.Intersect) for node in expression.walk()):
        set_operation = "intersect"
    elif any(isinstance(node, exp.Except) for node in expression.walk()):
        set_operation = "except"

    analysis.update(
        {
            "table_names": list(dict.fromkeys(table_names)),
            "table_count": len(set(table_names)),
            "join_count": sum(1 for _ in expression.find_all(exp.Join)),
            "projection_count": len(projection_items),
            "projection_items": projection_items[:8],
            "has_aggregation": any(isinstance(node, exp.AggFunc) for node in expression.walk()),
            "has_group_by": expression.args.get("group") is not None,
            "has_distinct": bool(expression.args.get("distinct")),
            "has_subquery": any(isinstance(node, exp.Subquery) for node in expression.walk()),
            "set_operation": set_operation,
            "literals": [
                literal.this
                for literal in expression.find_all(exp.Literal)
                if not literal.is_int and not literal.is_number
            ][:6],
        }
    )
    return analysis


def summarize_candidate_analysis(analysis: dict[str, Any]) -> str:
    """Render compact summary for prompt consumption."""
    if analysis.get("parse_error"):
        return f"parse_error={analysis['parse_error']}"

    tables = ", ".join(analysis.get("table_names", [])[:5]) or "-"
    projections = ", ".join(analysis.get("projection_items", [])[:5]) or "-"
    literals = ", ".join(repr(item) for item in analysis.get("literals", [])[:4]) or "-"
    return (
        f"tables={tables}; joins={analysis.get('join_count', 0)}; "
        f"projection_count={analysis.get('projection_count', 0)}; "
        f"projection_items={projections}; "
        f"aggregation={analysis.get('has_aggregation', False)}; "
        f"group_by={analysis.get('has_group_by', False)}; "
        f"distinct={analysis.get('has_distinct', False)}; "
        f"subquery={analysis.get('has_subquery', False)}; "
        f"set_op={analysis.get('set_operation', 'none')}; "
        f"literals={literals}"
    )
