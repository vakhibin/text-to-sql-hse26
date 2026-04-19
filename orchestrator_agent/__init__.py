"""Reusable orchestrator agent (LangGraph + tools + memory).

Mirrors the split used by ``text_to_sql_agent``: this package contains the
pure agent runtime (state, graph, tools, checkpointer, HTTP client) with no
FastAPI or transport dependencies. The HTTP surface lives in
``services.orchestrator_api`` and consumes this package as a library.
"""
