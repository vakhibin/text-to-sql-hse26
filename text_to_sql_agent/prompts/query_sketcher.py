"""Prompt templates for query-sketcher agent."""

QUERY_SKETCHER_PROMPT = """
Build a compact, schema-grounded query sketch for a text-to-SQL task.
Return strict JSON only. Do not output SQL.
""".strip()


def build_query_sketcher_prompt(
    *,
    question: str,
    evidence: str | None,
    filtered_schema: str,
    retrieved_schema_context: str,
    value_hints_text: str = "",
    column_hints_text: str = "",
) -> str:
    """Build a strong schema-grounded prompt for query sketching."""
    evidence_block = f"Evidence:\n{evidence}\n\n" if evidence else ""
    broader_schema_block = (
        f"Additional retrieved schema context:\n{retrieved_schema_context}\n\n"
        if retrieved_schema_context and retrieved_schema_context != filtered_schema
        else ""
    )
    linking_blocks = ""
    if value_hints_text:
        linking_blocks += f"{value_hints_text}\n\n"
    if column_hints_text:
        linking_blocks += f"{column_hints_text}\n\n"
    return f"""
You are `query-sketcher`, a planning stage inside a Text-to-SQL system.

Goal:
- Build a compact, schema-grounded query plan BEFORE SQL generation.
- Return JSON only.
- Do not output SQL.
- Do not output markdown.
- Use short phrases, not paragraphs.

Inputs:
Question:
{question}

{evidence_block}Primary selected schema:
{filtered_schema}

{broader_schema_block}{linking_blocks}Rules:
- Use only tables and columns that appear in the provided schema context.
- When value hints are provided, use the exact db_value spelling in your filters instead of paraphrasing the question wording. Record which hints you used in `generation_hints`.
- When column hints are provided, use the exact table.column identifiers listed there instead of guessing or shortening column names.
- Never invent identifiers, aliases like `T1`/`T2`, or derived columns.
- If uncertain, leave the relevant list short or empty and record the uncertainty in `ambiguities` or `risks`.
- Order `candidate_tables` from most likely to least likely.
- Preserve the output column order implied by the question wording whenever the question names multiple fields.
- Keep `candidate_tables` to at most 4 items.
- Keep `join_plan` to at most 3 items.
- Keep `generation_hints` to 3-6 short items.
- `join_plan` should be empty when no join is clearly needed.
- Prefer the simplest query shape that can answer the question correctly.
- Do not recommend a join when one selected table already contains the requested fields and filters.
- `subquery_needed` should be true only when nesting, set operations, exclusion, or comparison to aggregate values is likely required.
- Use lowercase JSON booleans: `true` / `false`.
- Use double-quoted JSON strings.

Return STRICT JSON with exactly these keys:
{{
  "intent": "lookup|aggregation|comparison|ranking|existence|set_operation|other",
  "task_summary": "one short sentence",
  "candidate_tables": [
    {{
      "table": "table_name",
      "columns": ["col_a", "col_b"],
      "reason": "short reason"
    }}
  ],
  "join_plan": [
    {{
      "left_table": "table_a",
      "right_table": "table_b",
      "join_keys": ["key or key-pair description"],
      "reason": "short reason"
    }}
  ],
  "filters": ["..."],
  "aggregations": ["..."],
  "grouping": ["..."],
  "ordering": ["..."],
  "limit": "none or short phrase",
  "subquery_needed": true,
  "set_operation": "none|union|intersect|except",
  "ambiguities": ["..."],
  "risks": ["..."],
  "generation_hints": ["3-6 short bullets that help SQL generation"]
}}
""".strip()
