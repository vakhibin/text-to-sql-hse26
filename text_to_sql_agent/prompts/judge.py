"""Prompt templates for judge agent."""

JUDGE_PROMPT = """
Choose best SQL candidate from valid list for the question.
Return strict JSON: best_index, reasoning.
""".strip()

