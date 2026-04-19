"""Thin adapter between FastAPI endpoints and the LangGraph text-to-SQL pipeline.

Keeps the FastAPI layer free of LangGraph internals and gives us a single
place to mock for contract tests (see ``tests/services/test_text_to_sql_api.py``).

Caches the compiled graph per process. Schema loads are already cached in
``schema_loader``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from text_to_sql_agent.agents.refiner import run_refiner
from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import SQLAgentState, make_initial_state
from text_to_sql_agent.prompts import refiner as _refiner_prompt  # noqa: F401  (ensure prompt module imports cleanly)
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole
from text_to_sql_agent.tools.schema_loader import load_schema, schema_to_mschema
from text_to_sql_agent.tools.sql_executor import SQLExecutionResult, execute_sql

_COMPILED_GRAPH = None
_GRAPH_LOCK = asyncio.Lock()
_LLM_ROUTER: LLMRouter | None = None


async def _get_graph():
    """Return the lazily-built compiled LangGraph (one per process)."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH
    async with _GRAPH_LOCK:
        if _COMPILED_GRAPH is None:
            _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def _get_router() -> LLMRouter:
    global _LLM_ROUTER
    if _LLM_ROUTER is None:
        _LLM_ROUTER = LLMRouter()
    return _LLM_ROUTER


def _resolve_schema_root(schema_root: str | None) -> str:
    return schema_root or settings.spider_root


def _resolve_db_path(db_id: str, schema_root: str) -> Path | None:
    root = Path(schema_root)
    candidates = [
        root / "database" / db_id / f"{db_id}.sqlite",
        root / "dev_databases" / db_id / f"{db_id}.sqlite",
        root / "mini_dev_databases" / db_id / f"{db_id}.sqlite",
        root / "train_databases" / db_id / f"{db_id}.sqlite",
    ]
    return next((p for p in candidates if p.exists()), None)


# ---- /run ---------------------------------------------------------------


async def run_pipeline(
    *,
    question: str,
    db_id: str,
    evidence: str | None = None,
    schema_root: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the full LangGraph pipeline and return the final state dict."""
    graph = await _get_graph()
    state = make_initial_state(
        question=question,
        db_id=db_id,
        evidence=evidence,
        schema_root=_resolve_schema_root(schema_root),
        trace_id=trace_id or str(uuid4()),
    )
    result = await graph.ainvoke(state)
    return dict(result)


# ---- /execute -----------------------------------------------------------


async def execute_sql_read_only(
    *,
    sql: str,
    db_id: str,
    schema_root: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[SQLExecutionResult, str | None]:
    """Execute ``sql`` with the guardrail's read-only default.

    Returns ``(result, db_path_str)`` where ``db_path_str`` is ``None`` when the
    database file cannot be located.
    """
    root = _resolve_schema_root(schema_root)
    db_path = _resolve_db_path(db_id, root)
    if db_path is None:
        return (
            SQLExecutionResult(
                success=False,
                rows=None,
                error=f"database file for db_id={db_id!r} not found under {root!r}",
            ),
            None,
        )
    result = await execute_sql(
        str(db_path),
        sql,
        timeout_seconds=timeout_seconds or settings.execution_timeout_seconds,
    )
    return result, str(db_path)


# ---- /refine ------------------------------------------------------------


async def refine_sql_standalone(
    *,
    sql: str,
    db_id: str,
    question: str | None,
    evidence: str | None,
    schema_root: str | None,
    error_hint: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
    """Run the refiner on an arbitrary SQL string.

    Builds a minimal ``SQLAgentState`` so ``run_refiner`` can do its
    deterministic AST repair + execution check + tool-augmented LLM fix path.
    """
    root = _resolve_schema_root(schema_root)
    schema = await load_schema(db_id, spider_root=root, with_sample_values=True)
    trace = trace_id or str(uuid4())
    state: SQLAgentState = make_initial_state(
        question=question or "",
        db_id=db_id,
        evidence=evidence,
        schema_root=root,
        trace_id=trace,
    )
    state["full_schema"] = schema
    state["best_sql"] = sql
    state["final_sql"] = sql
    state["selection_needs_refine"] = True
    if error_hint:
        state["error_message"] = error_hint
    refined = await run_refiner(state)
    return {
        "trace_id": trace,
        "refined_state": dict(refined),
    }


# ---- /explain -----------------------------------------------------------

_EXPLAIN_SYSTEM = (
    "You are a concise SQL explainer. Explain the given SQL in 2-4 clear "
    "sentences of plain English for a non-technical analyst. Mention what "
    "tables are read, what the query returns, and any notable filters, joins, "
    "grouping, or ordering. Do not repeat the SQL."
)


async def explain_sql(
    *,
    sql: str,
    db_id: str | None,
    schema_root: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
    """Produce a natural-language explanation of ``sql`` using the refiner model."""
    started = time.perf_counter()
    schema_text = ""
    if db_id:
        try:
            schema = await load_schema(
                db_id,
                spider_root=_resolve_schema_root(schema_root),
                with_sample_values=False,
            )
            schema_text = schema_to_mschema(
                schema,
                schema_root=_resolve_schema_root(schema_root),
                with_sample_values=False,
            )
        except (FileNotFoundError, ValueError):
            schema_text = ""

    user_parts: list[str] = []
    if schema_text:
        user_parts.append(f"Schema (compact):\n{schema_text}")
    user_parts.append(f"SQL:\n{sql}")
    user_parts.append("Explain in 2-4 sentences.")
    user_prompt = "\n\n".join(user_parts)

    router = _get_router()
    trace = trace_id or str(uuid4())
    result = await router.ainvoke_with_metadata(
        ModelRole.REFINER,
        messages=[("system", _EXPLAIN_SYSTEM), ("user", user_prompt)],
        trace_id=trace,
        db_id=db_id,
        stage="explain",
    )
    elapsed = round(time.perf_counter() - started, 4)
    return {
        "trace_id": trace,
        "explanation": result.text,
        "cost_usd": float(result.usage.get("cost_usd") or 0.0),
        "elapsed_s": elapsed,
    }
