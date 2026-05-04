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
from text_to_sql_agent.tools.observability import (
    flush_langfuse,
    safe_state_snapshot,
    start_langfuse_span,
    update_langfuse_span,
)
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
    """Invoke the full LangGraph pipeline and return the final state dict.

    The whole graph invocation is wrapped in a Langfuse root span so that
    every stage span and LLM generation produced inside lands in the same
    trace. Tracing is best-effort: when Langfuse is disabled, the wrapper
    is a no-op and pipeline behavior is unchanged.
    """
    graph = await _get_graph()
    resolved_trace = trace_id or str(uuid4())
    state = make_initial_state(
        question=question,
        db_id=db_id,
        evidence=evidence,
        schema_root=_resolve_schema_root(schema_root),
        trace_id=resolved_trace,
    )
    # Span updates write to the *current* OTEL span, so they must happen
    # INSIDE the ``with`` block — after ``__exit__`` the span is closed and
    # the update would land on the wrong (or no) current span.
    with start_langfuse_span(
        name="text_to_sql_run",
        trace_id=resolved_trace,
        input_payload=safe_state_snapshot(
            {
                "question": question,
                "db_id": db_id,
                "evidence": evidence,
                "schema_root": state.get("schema_root"),
                "trace_id": resolved_trace,
            }
        ),
        metadata={
            "trace_id": resolved_trace,
            "db_id": db_id,
            "schema_root": state.get("schema_root"),
        },
        as_type="chain",
    ):
        try:
            result = await graph.ainvoke(state)
        except Exception as exc:
            update_langfuse_span(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
                metadata={"trace_id": resolved_trace, "db_id": db_id},
            )
            flush_langfuse()
            raise

        final = dict(result)
        update_langfuse_span(
            output=safe_state_snapshot(
                {
                    "final_sql": final.get("final_sql"),
                    "best_sql": final.get("best_sql"),
                    "stage_status": final.get("stage_status"),
                    "stage_timings": final.get("stage_timings"),
                    "warnings": final.get("warnings"),
                    "error_message": final.get("error_message"),
                    "total_cost_usd": final.get("total_cost_usd"),
                }
            ),
            metadata={
                "trace_id": resolved_trace,
                "db_id": db_id,
                "total_cost_usd": final.get("total_cost_usd"),
            },
        )
    flush_langfuse()
    return final


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


async def execute_sql_user_confirmed(
    *,
    sql: str,
    db_id: str,
    schema_root: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[SQLExecutionResult, str | None]:
    """Execute ``sql`` after an upstream user-confirmation flow.

    This bypasses the read-only guardrail but keeps all execution inside the
    same timeout/error envelope. Only call from explicit confirmation paths.
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
        allow_write=True,
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


# ---- /modify -------------------------------------------------------------

_MODIFY_SYSTEM = (
    "You are a SQLite SQL rewriter. Given an existing SQL query, a database "
    "schema, and a natural-language instruction describing a desired change, "
    "produce a single modified SQL query that satisfies the instruction.\n\n"
    "Hard rules:\n"
    "- Output ONLY the SQL. No prose, no markdown fences, no comments.\n"
    "- Stay in SQLite dialect.\n"
    "- Do not invent tables or columns that are not present in the provided "
    "schema; prefer the original identifiers when the instruction is "
    "ambiguous.\n"
    "- If the instruction is unsafe (writes/DDL) or impossible given the "
    "schema, return the original SQL unchanged."
)


def _strip_sql_fences(text: str) -> str:
    """Strip ```sql fences and surrounding whitespace from an LLM response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped.rstrip(";").strip()


async def modify_sql_standalone(
    *,
    sql: str,
    instruction: str,
    db_id: str,
    schema_root: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
    """Apply a natural-language modification to ``sql`` via a single LLM call.

    Unlike ``/refine`` (which is for error repair), this endpoint is for
    user-driven edits like "add a WHERE clause for 2023" or "group by year
    instead of month".
    """
    started = time.perf_counter()
    root = _resolve_schema_root(schema_root)
    try:
        schema = await load_schema(db_id, spider_root=root, with_sample_values=False)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    schema_text = schema_to_mschema(schema, schema_root=root, with_sample_values=False)

    user_prompt = (
        f"Schema (compact):\n{schema_text}\n\n"
        f"Current SQL:\n{sql}\n\n"
        f"Instruction:\n{instruction}\n\n"
        "Return ONLY the modified SQL."
    )

    router = _get_router()
    trace = trace_id or str(uuid4())
    result = await router.ainvoke_with_metadata(
        ModelRole.REFINER,
        messages=[("system", _MODIFY_SYSTEM), ("user", user_prompt)],
        trace_id=trace,
        db_id=db_id,
        stage="modify",
    )
    modified = _strip_sql_fences(result.text)
    if not modified:
        modified = sql
    elapsed = round(time.perf_counter() - started, 4)
    return {
        "trace_id": trace,
        "db_id": db_id,
        "original_sql": sql,
        "modified_sql": modified,
        "changed": modified.strip() != sql.strip(),
        "cost_usd": float(result.usage.get("cost_usd") or 0.0),
        "elapsed_s": elapsed,
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
