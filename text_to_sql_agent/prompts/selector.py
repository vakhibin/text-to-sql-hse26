"""Prompt templates for selector agent."""

SELECTOR_RERANK_PROMPT = """
You are a schema reranker.
Given question and Top-K table candidates, pick 3-5 most relevant tables.
Return strict JSON.
""".strip()

