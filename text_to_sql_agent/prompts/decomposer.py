"""Prompt templates for decomposer agent."""

DECOMPOSER_PROMPT = """
Classify query complexity: simple, moderate, or complex.
If complex, produce sub-questions and a short plan.
Return strict JSON.
""".strip()

