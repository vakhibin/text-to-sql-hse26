"""SQL execution tool placeholder."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SQLExecutionResult:
    success: bool
    rows: Optional[list] = None
    error: Optional[str] = None


async def execute_sql(db_url: str, sql: str) -> SQLExecutionResult:
    """Execute SQL via SQLAlchemy (placeholder)."""
    _ = (db_url, sql)
    return SQLExecutionResult(success=False, error="Not implemented")

