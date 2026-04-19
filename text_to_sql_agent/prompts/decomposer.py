"""Prompt templates for decomposer agent."""

DECOMPOSER_PROMPT = """
Classify query complexity: simple, moderate, or complex.
If complex, produce sub-questions and a short plan.
Return strict JSON.
""".strip()


def build_decomposer_prompt(question: str, evidence: str | None = None) -> str:
    """Build strict-json decomposition prompt."""
    evidence_block = f"Evidence: {evidence}\n" if evidence else ""
    return f"""
You are a Text-to-SQL decomposition assistant.

Task:
1. Classify question complexity as one of: simple, moderate, complex.
2. Provide concise sub-questions that help SQL generation.
3. Emit compact risk flags that help routing and judge decisions.

Inputs:
Question: {question}
{evidence_block}
Output rules:
- Return STRICT JSON only.
- Keep sub_questions short, concrete, and SQL-relevant.
- If complexity is simple, sub_questions may be an empty list.
- Estimate risk flags conservatively. Use `projection_width` as the number of output columns implied by the question.
- `literal_filter_risk` should be true when correctness depends on exact literal spelling, casing, normalization, or code values.
- `bag_semantics_risk` should be true when duplicates, DISTINCT, set operations, or grouping choices may change the final answer.
- Use this exact schema:
{{
  "complexity": "simple|moderate|complex",
  "sub_questions": ["...", "..."],
  "risk_flags": {{
    "needs_join": false,
    "needs_aggregation": false,
    "projection_width": 1,
    "literal_filter_risk": false,
    "bag_semantics_risk": false
  }},
  "reasoning": "one short sentence"
}}
""".strip()

