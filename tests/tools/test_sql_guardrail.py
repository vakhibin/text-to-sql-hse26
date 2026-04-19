"""Tests for the read-only SQL guardrail."""

from __future__ import annotations

import pytest

from text_to_sql_agent.tools.sql_guardrail import (
    WriteSQLRejected,
    assert_read_only,
    is_read_only,
)


READ_ONLY_SQL = [
    pytest.param("SELECT 1", id="trivial_select"),
    pytest.param("select * from students", id="lowercase_select"),
    pytest.param("SELECT name FROM students WHERE age > 18", id="select_where"),
    pytest.param(
        "SELECT s.name, c.title FROM students s JOIN courses c ON s.course_id = c.id",
        id="select_join",
    ),
    pytest.param(
        "WITH adults AS (SELECT * FROM students WHERE age >= 18) SELECT name FROM adults",
        id="cte_select",
    ),
    pytest.param(
        "SELECT * FROM a UNION SELECT * FROM b",
        id="union",
    ),
    pytest.param(
        "SELECT * FROM a INTERSECT SELECT * FROM b",
        id="intersect",
    ),
    pytest.param(
        "SELECT * FROM a EXCEPT SELECT * FROM b",
        id="except",
    ),
    pytest.param(
        "SELECT (SELECT COUNT(*) FROM orders o WHERE o.uid = u.id) FROM users u",
        id="subquery_in_projection",
    ),
    pytest.param("SELECT 1;", id="trailing_semicolon"),
    pytest.param("  SELECT 1  ", id="surrounding_whitespace"),
    pytest.param("-- comment\nSELECT 1", id="line_comment_prefix"),
    pytest.param("/* block */ SELECT 1", id="block_comment_prefix"),
    pytest.param(
        "SELECT name FROM students GROUP BY name HAVING COUNT(*) > 1 ORDER BY name LIMIT 10",
        id="group_order_limit",
    ),
]


WRITE_SQL = [
    pytest.param("INSERT INTO students (id, name) VALUES (1, 'a')", id="insert"),
    pytest.param("UPDATE students SET name = 'a' WHERE id = 1", id="update"),
    pytest.param("DELETE FROM students WHERE id = 1", id="delete"),
    pytest.param("CREATE TABLE t (id INT)", id="create_table"),
    pytest.param("DROP TABLE students", id="drop_table"),
    pytest.param("ALTER TABLE students ADD COLUMN email TEXT", id="alter_table"),
    pytest.param("PRAGMA table_info(students)", id="pragma_read_metadata"),
    pytest.param("PRAGMA foreign_keys = ON", id="pragma_mutating"),
    pytest.param("ATTACH DATABASE '/tmp/other.sqlite' AS other", id="attach"),
    pytest.param("DETACH DATABASE other", id="detach"),
    pytest.param("VACUUM", id="vacuum"),
    pytest.param("BEGIN TRANSACTION", id="begin_transaction"),
    pytest.param("COMMIT", id="commit"),
    pytest.param("ROLLBACK", id="rollback"),
    pytest.param("SELECT 1; DROP TABLE students", id="multi_statement_select_then_drop"),
    pytest.param("DROP TABLE students; SELECT 1", id="multi_statement_drop_then_select"),
    pytest.param("SELECT 1; SELECT 2", id="multi_statement_two_selects"),
    pytest.param("", id="empty_string"),
    pytest.param("   \n\t  ", id="whitespace_only"),
    pytest.param(";", id="bare_semicolon"),
    pytest.param("this is not sql", id="unparseable_text"),
    pytest.param("REPLACE INTO students (id) VALUES (1)", id="replace_into"),
]


@pytest.mark.parametrize("sql", READ_ONLY_SQL)
def test_read_only_sql_allowed(sql: str) -> None:
    assert_read_only(sql)
    assert is_read_only(sql) is True


@pytest.mark.parametrize("sql", WRITE_SQL)
def test_write_sql_rejected(sql: str) -> None:
    with pytest.raises(WriteSQLRejected):
        assert_read_only(sql)
    assert is_read_only(sql) is False


def test_none_is_rejected() -> None:
    with pytest.raises(WriteSQLRejected):
        assert_read_only(None)  # type: ignore[arg-type]


def test_cte_with_dml_inside_is_rejected() -> None:
    """CTE wrapping an INSERT ... RETURNING must still be rejected."""
    sql = "WITH x AS (INSERT INTO t (a) VALUES (1) RETURNING a) SELECT * FROM x"
    with pytest.raises(WriteSQLRejected):
        assert_read_only(sql)


def test_error_message_is_informative() -> None:
    with pytest.raises(WriteSQLRejected, match="multi-statement"):
        assert_read_only("SELECT 1; SELECT 2")

    with pytest.raises(WriteSQLRejected, match="only SELECT"):
        assert_read_only("INSERT INTO t VALUES (1)")

    with pytest.raises(WriteSQLRejected, match="empty"):
        assert_read_only("   ")
