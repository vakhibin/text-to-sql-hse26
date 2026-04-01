"""Prompt templates for query-sketcher agent."""

QUERY_SKETCHER_PROMPT = """
Build a compact, schema-grounded query sketch for a text-to-SQL task.
Return strict JSON only. Do not output SQL.
""".strip()


def build_query_sketcher_prompt(
    *,
    question: str,
    evidence: str | None,
    complexity: str,
    sub_questions: list[str],
    filtered_schema: str,
    retrieved_schema_context: str,
) -> str:
    """Build a strong schema-grounded prompt for query sketching."""
    evidence_block = f"Evidence:\n{evidence}\n\n" if evidence else ""
    sub_questions_block = (
        "\n".join(f"- {item}" for item in sub_questions) if sub_questions else "- (none)"
    )
    broader_schema_block = (
        f"Additional retrieved schema context:\n{retrieved_schema_context}\n\n"
        if retrieved_schema_context and retrieved_schema_context != filtered_schema
        else ""
    )
    return f"""
You are `query-sketcher`, a planning agent inside a Text-to-SQL system.

Your job is to create a compact SQL generation plan BEFORE SQL is written.
You must be schema-grounded, conservative, and useful for a downstream SQL generator.
Do NOT write SQL.
Do NOT explain outside JSON.

Question:
{question}

{evidence_block}Complexity:
{complexity}

Decomposition hints:
{sub_questions_block}

Primary selected schema:
{filtered_schema}

{broader_schema_block}Think like a database planner:
1. Identify the likely tables and columns that are actually needed.
2. Infer the likely join path only when it is grounded in the provided schema.
3. List filters, aggregations, grouping, ordering, limit, and whether a subquery is likely needed.
4. Highlight ambiguity or risk instead of hallucinating identifiers.
5. Produce a plan that helps SQL generation stay faithful to the schema and the question wording.

Hard grounding rules:
- Use ONLY schema items present in the provided schema context.
- Never invent tables, columns, foreign keys, aliases, or derived fields.
- If the question wording and schema naming differ, map to the closest real schema item.
- If something is unclear, say it is uncertain in `ambiguities` or `risks` instead of guessing.
- Prefer concise field values over long prose.
- If the query looks simple, the sketch should still mention the essential filter/order/aggregation decisions.
- `candidate_tables` should be ordered from most likely to least likely.
- `join_plan` should be empty when no join is needed or when the join path is too uncertain.
- `subquery_needed` should be true only when the question likely requires nesting, set operations, comparison to aggregates, or exclusion logic.

Return STRICT JSON with exactly this shape:
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
