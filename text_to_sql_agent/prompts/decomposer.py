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

Inputs:
Question: {question}
{evidence_block}
Output rules:
- Return STRICT JSON only.
- Keep sub_questions short, concrete, and SQL-relevant.
- If complexity is simple, sub_questions may be an empty list.
- Use this exact schema:
{{
  "complexity": "simple|moderate|complex",
  "sub_questions": ["...", "..."],
  "reasoning": "one short sentence"
}}
""".strip()

