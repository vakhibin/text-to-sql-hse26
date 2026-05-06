"""Prompt templates for refiner agent."""

REFINER_PROMPT = """
Fix SQL query using schema and execution error.
Return only corrected SQL.
""".strip()


def build_refiner_prompt(
    *,
    question: str,
    evidence: str | None,
    filtered_schema: str,
    retrieved_schema_context: str,
    query_sketch_text: str,
    failed_sql: str,
    execution_error: str,
    selection_reasoning: str,
    selection_confidence: str,
    validation_errors: list[str],
    validation_warnings: list[str],
    selected_candidate_summary: str,
    failed_candidate_summaries: list[str],
    quote_sql_column_identifiers: bool = False,
) -> str:
    """Build SQL-fix prompt from failed query and DB error."""
    additional_schema_block = ""
    if retrieved_schema_context.strip():
        additional_schema_block = f"""
Retrieved schema candidates:
{retrieved_schema_context}
""".strip()

    evidence_block = evidence.strip() if evidence and evidence.strip() else "- (none)"
    query_sketch_block = query_sketch_text.strip() if query_sketch_text.strip() else "- (none)"
    validation_errors_block = "\n".join(f"- {item}" for item in validation_errors) if validation_errors else "- (none)"
    validation_warnings_block = "\n".join(f"- {item}" for item in validation_warnings) if validation_warnings else "- (none)"
    failed_candidates_block = "\n".join(f"- {item}" for item in failed_candidate_summaries) if failed_candidate_summaries else "- (none)"

    quote_columns_block = ""
    if quote_sql_column_identifiers:
        quote_columns_block = """
- Mandatory: in the corrected SQLite, wrap every column name (SELECT, WHERE, JOIN ON, GROUP BY, ORDER BY, HAVING, expressions) in double quotes (\") exactly as in the schema. Keep literals in single quotes.
""".rstrip()

    return f"""
You are an expert SQLite SQL fixer.
Given a selected SQL query plus execution and validation feedback, produce a corrected SQL query.
Return ONLY SQL with no markdown or explanation.

Question:
{question}

Evidence:
{evidence_block}

Query sketch:
{query_sketch_block}

Selected mSchema:
{filtered_schema}

{additional_schema_block}

Current SQL:
{failed_sql}

Selection context:
- confidence: {selection_confidence}
- reasoning: {selection_reasoning or "- (none)"}

Selected candidate structural summary:
{selected_candidate_summary}

Schema validation errors:
{validation_errors_block}

Schema validation warnings:
{validation_warnings_block}

Execution / refinement trigger:
{execution_error}

Other candidate failure summaries:
{failed_candidates_block}

Repair policy:
- Preserve valid parts of the SQL when possible; change only what is needed.
- If validation errors mention output shape or projection order, fix that before making bigger structural changes.
- Prefer the simpler equivalent query when possible.
- If literals are risky, preserve the exact literal spelling/casing/value from the question, evidence, or schema context.
{quote_columns_block}

Before returning SQL, validate it against the provided schema and question:
- every referenced table must exist
- every referenced column must belong to the referenced table
- use column data types to avoid invalid predicates, joins, aggregations, or comparisons
- if the failed SQL uses a missing table or column, replace it only with something present in the schema context above
- keep the output column order and width faithful to the question
- do not add decorative joins or extra projected columns

Output only corrected SQL ending with semicolon.
""".strip()
