"""Prompt templates for selector agent."""

SELECTOR_RERANK_PROMPT = """
You are a schema reranker.
Given question and Top-K table candidates, pick 3-5 most relevant tables.
Return strict JSON.
""".strip()


def build_selector_rerank_prompt(question: str, candidates: list[dict]) -> str:
    """Build strict-json reranking prompt for selector stage."""
    candidates_lines = []
    for idx, candidate in enumerate(candidates, start=1):
        score = float(candidate.get("score", 0.0))
        candidates_lines.append(
            f"{idx}. table={candidate.get('table_name')} score={score:.4f}"
        )
    return f"""
You are a database schema reranker.

Question:
{question}

Candidate tables:
{chr(10).join(candidates_lines)}

Select 3 to 5 table names that are most relevant to answer the question.
Return STRICT JSON with exact format:
{{
  "selected_tables": ["table_1", "table_2", "table_3"],
  "reasoning": "short reason"
}}
Do not include markdown or extra text.
""".strip()

