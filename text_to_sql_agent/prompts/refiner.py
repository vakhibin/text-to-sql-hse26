"""Prompt templates for refiner agent."""

REFINER_PROMPT = """
Fix SQL query using schema and execution error.
Return only corrected SQL.
""".strip()


def build_refiner_prompt(
    *,
    question: str,
    filtered_schema: str,
    retrieved_schema_context: str,
    failed_sql: str,
    execution_error: str,
) -> str:
    """Build SQL-fix prompt from failed query and DB error."""
    additional_schema_block = ""
    if retrieved_schema_context.strip():
        additional_schema_block = f"""
Retrieved schema candidates:
{retrieved_schema_context}
""".strip()

    return f"""
You are an expert SQLite SQL fixer.
Given a failed SQL query and execution error, produce a corrected SQL query.
Return ONLY SQL with no markdown or explanation.

Question:
{question}

Selected mSchema:
{filtered_schema}

{additional_schema_block}

Failed SQL:
{failed_sql}

Execution error:
{execution_error}

Before returning SQL, validate it against the provided schema:
- every referenced table must exist
- every referenced column must belong to the referenced table
- use column data types to avoid invalid predicates, joins, aggregations, or comparisons
- if the failed SQL uses a missing table or column, replace it only with something present in the schema context above

Output only corrected SQL ending with semicolon.
""".strip()

