"""Prompt templates for refiner agent."""

REFINER_PROMPT = """
Fix SQL query using schema and execution error.
Return only corrected SQL.
""".strip()


def build_refiner_prompt(
    *,
    question: str,
    filtered_schema: str,
    failed_sql: str,
    execution_error: str,
) -> str:
    """Build SQL-fix prompt from failed query and DB error."""
    return f"""
You are an expert SQLite SQL fixer.
Given a failed SQL query and execution error, produce a corrected SQL query.
Return ONLY SQL with no markdown or explanation.

Question:
{question}

mSchema:
{filtered_schema}

Failed SQL:
{failed_sql}

Execution error:
{execution_error}

Output only corrected SQL ending with semicolon.
""".strip()

