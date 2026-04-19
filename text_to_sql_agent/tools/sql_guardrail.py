"""Read-only SQL guardrail.

Deterministic AST-based check that refuses any SQL that could mutate the
database or the session. Used by the executor (so the agent cannot run write
SQL by accident) and later by the `/execute_user_confirmed` endpoint that will
allow writes only when the user explicitly opts in.

Policy (strict by default):

- Allowed root expressions: ``SELECT``, ``SELECT`` wrapped by ``WITH`` CTEs,
  and set operations ``UNION`` / ``INTERSECT`` / ``EXCEPT``.
- Exactly one statement is allowed. ``SELECT ...; DROP TABLE t;`` is rejected.
- Nothing mutating may appear anywhere in the tree (e.g. CTE containing
  ``INSERT ... RETURNING``).
- ``PRAGMA`` / ``ATTACH`` / ``DETACH`` / ``VACUUM`` / transaction control and
  any other non-SELECT command are rejected.
- Unparseable SQL is rejected (fail-closed).

The guardrail does **not** attempt semantic validation (table/column
existence). That is the job of the schema validator and executor.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

__all__ = ["WriteSQLRejected", "assert_read_only", "is_read_only"]


class WriteSQLRejected(ValueError):
    """Raised when SQL is not a pure read-only query."""


_READ_ROOTS: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
)

_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Use,
    exp.Set,
    exp.SetItem,
)


def _is_command_write(node: exp.Command) -> bool:
    """sqlglot falls back to ``exp.Command`` for statements it does not model.

    Any generic command (VACUUM, ANALYZE, REINDEX, ATTACH in some versions,
    etc.) is treated as write. ``SELECT`` is never parsed as a Command, so
    this is safe.
    """
    return True


def is_read_only(sql: str) -> bool:
    """Return ``True`` iff ``sql`` is a single pure read-only query."""
    try:
        assert_read_only(sql)
    except WriteSQLRejected:
        return False
    return True


def assert_read_only(sql: str) -> None:
    """Raise :class:`WriteSQLRejected` unless ``sql`` is a pure read query."""
    if sql is None or not sql.strip():
        raise WriteSQLRejected("SQL is empty")

    try:
        parsed = sqlglot.parse(sql, dialect="sqlite")
    except ParseError as exc:
        raise WriteSQLRejected(f"SQL could not be parsed: {exc}") from exc

    statements = [stmt for stmt in parsed if stmt is not None]
    if len(statements) == 0:
        raise WriteSQLRejected("SQL contains no statements")
    if len(statements) > 1:
        raise WriteSQLRejected(
            f"multi-statement SQL is not allowed (got {len(statements)} statements)"
        )

    root = statements[0]

    if isinstance(root, exp.Command):
        if _is_command_write(root):
            raise WriteSQLRejected(
                f"command statement is not allowed: {root.name or root.sql(dialect='sqlite')!r}"
            )

    if not isinstance(root, _READ_ROOTS):
        raise WriteSQLRejected(
            f"only SELECT statements are allowed, got {type(root).__name__}"
        )

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise WriteSQLRejected(
                f"forbidden node inside SQL: {type(node).__name__}"
            )
        if isinstance(node, exp.Command) and _is_command_write(node):
            raise WriteSQLRejected(
                f"forbidden command inside SQL: {node.name or node.sql(dialect='sqlite')!r}"
            )
