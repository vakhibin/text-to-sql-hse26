"""Prompt templates for judge agent."""

JUDGE_PROMPT = """
Choose best SQL candidate from valid list for the question.
Return strict JSON: best_index, reasoning.
""".strip()


def build_judge_prompt(
    *,
    question: str,
    filtered_schema: str,
    candidates: list[str],
) -> str:
    """Build strict-json judge prompt over candidate SQL list."""
    items = [f"{idx}: {sql}" for idx, sql in enumerate(candidates)]
    return f"""
You are an expert SQL evaluator.

Question:
{question}

mSchema:
{filtered_schema}

Candidate SQL queries:
{chr(10).join(items)}

Select the best candidate index for correctness and relevance.
Return STRICT JSON in this exact format:
{{
  "best_index": 0,
  "reasoning": "one short sentence"
}}
Do not include markdown or extra text.
""".strip()

