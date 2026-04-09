"""Refiner agent: tool-augmented iterative SQL correction."""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.refiner import build_refiner_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole
from text_to_sql_agent.tools.sql_ast_repair import repair_sql_schema_references as ast_repair
from text_to_sql_agent.tools.sql_candidate_analysis import (
    analyze_sql_candidate,
    summarize_candidate_analysis,
)
from text_to_sql_agent.tools.sql_schema_validator import validate_sql_schema_references
from text_to_sql_agent.tools.sql_executor import execute_sql

_MAX_TOOL_STEPS = 3
_TOOL_LOOP_TIMEOUT_S = 60

_REFINER_SYSTEM = """\
You are an expert SQLite SQL fixer with access to tools.
Diagnose the problem, use tools to inspect the schema and test fixes, then return ONLY the corrected SQL.

Strategy:
1. If the error mentions an unknown table or column, use get_table_columns or list_all_tables to find the correct names.
2. If a literal value might be misspelled, use search_column_values to find the correct spelling.
3. After making changes, use validate_sql to check schema validity.
4. Use execute_sql to verify your fix actually runs.
5. When confident, return ONLY the final corrected SQL (no markdown, no explanation)."""


def _extract_sql(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned and not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


def _build_refiner_tools(
    db_path: str,
    full_schema: dict[str, Any],
) -> list:
    """Create LangChain tools bound to the current DB context."""

    @langchain_tool
    async def execute_sql_check(sql: str) -> str:
        """Execute SQL against the database. Returns row count + preview on success, or the error message."""
        result = await execute_sql(db_path, sql, timeout_seconds=settings.execution_timeout_seconds)
        if result.success:
            n = len(result.rows or [])
            preview = (result.rows or [])[:5]
            return f"SUCCESS: {n} rows. Preview: {preview}"
        return f"ERROR: {result.error}"

    @langchain_tool
    def get_table_columns(table_name: str) -> str:
        """Get column names, types, sample values and foreign keys for a table. Use when you need correct column names."""
        for table in full_schema.get("tables", []):
            if table["name"].lower() == table_name.lower():
                cols = []
                for c in table["columns"]:
                    samples = c.get("sample_values", [])
                    s = f" samples={samples}" if samples else ""
                    cols.append(f"  {c['name']} ({c['type']}){s}")
                pks = table.get("primary_keys", [])
                fks = table.get("foreign_keys", [])
                parts = [f"Table '{table['name']}':"] + cols
                if pks:
                    parts.append(f"Primary keys: {pks}")
                if fks:
                    parts.extend(
                        f"  FK: {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}"
                        for fk in fks
                    )
                return "\n".join(parts)
        available = [t["name"] for t in full_schema.get("tables", [])]
        return f"Table '{table_name}' not found. Available tables: {available}"

    @langchain_tool
    def list_all_tables() -> str:
        """List all tables with their columns. Use to understand the full schema."""
        lines = []
        for t in full_schema.get("tables", []):
            cols = [c["name"] for c in t.get("columns", [])]
            pk = t.get("primary_keys", [])
            pk_s = f" pk={pk}" if pk else ""
            lines.append(f"{t['name']}({', '.join(cols)}){pk_s}")
        return "\n".join(lines)

    @langchain_tool
    def validate_sql(sql: str) -> str:
        """Validate SQL against the database schema. Returns errors and warnings about unknown tables/columns."""
        result = validate_sql_schema_references(sql, full_schema)
        parts = []
        if result.errors:
            parts.append(f"ERRORS: {result.errors}")
        if result.warnings:
            parts.append(f"WARNINGS: {result.warnings}")
        if not parts:
            return "VALID: no schema issues detected"
        return "\n".join(parts)

    @langchain_tool
    def search_column_values(table_name: str, column_name: str, search_term: str) -> str:
        """Search for actual values in a column matching a term. Use to find correct spelling/casing for WHERE literals."""
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    f"SELECT DISTINCT [{column_name}] FROM [{table_name}] "
                    f"WHERE CAST([{column_name}] AS TEXT) LIKE ? LIMIT 10",
                    (f"%{search_term}%",),
                ).fetchall()
            if rows:
                return f"Matching values: {[str(r[0]) for r in rows]}"
            return f"No values matching '{search_term}' in {table_name}.{column_name}"
        except Exception as e:
            return f"Search error: {e}"

    return [execute_sql_check, get_table_columns, list_all_tables, validate_sql, search_column_values]


async def _run_tool_loop(
    router: LLMRouter,
    tools: list,
    user_prompt: str,
    *,
    trace_id: str | None = None,
    db_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run tool-calling loop with the refiner model. Returns (final_text, usage_dicts)."""
    model_name = router.model_for_role(ModelRole.REFINER)
    chat_model = router.get_chat_model(ModelRole.REFINER, temperature_override=0.0)
    model_with_tools = chat_model.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: list = [
        SystemMessage(content=_REFINER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    usage_dicts: list[dict[str, Any]] = []

    for step in range(_MAX_TOOL_STEPS):
        response: AIMessage = await model_with_tools.ainvoke(messages)
        usage_record = router._extract_usage(
            response=response, model_name=model_name, stage=f"refiner_tool_step_{step}",
        )
        usage_dicts.append(usage_record.as_dict())
        messages.append(response)

        if not response.tool_calls:
            return (response.content or ""), usage_dicts

        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            if fn is None:
                messages.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))
                continue
            try:
                if asyncio.iscoroutinefunction(getattr(fn, "coroutine", None)):
                    result = await fn.ainvoke(tc["args"])
                else:
                    result = await asyncio.to_thread(fn.invoke, tc["args"])
            except Exception as exc:
                result = f"Tool error: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content, usage_dicts
    return "", usage_dicts


async def run_refiner(state: SQLAgentState) -> SQLAgentState:
    """Three-phase refiner: deterministic repair → execute → tool-augmented LLM fix."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))
    stage_status["refiner"] = "running"

    db_path = str(state.get("full_schema", {}).get("db_path", "")).strip()
    if not db_path:
        stage_status["refiner"] = "failed"
        return {
            **state,
            "error_message": "refiner: db_path is missing in full_schema",
            "stage_status": stage_status,
            "stage_timings": {**stage_timings, "refiner": round(time.perf_counter() - started, 4)},
            "warnings": [*warnings, "refiner: db_path missing"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    current_sql = state.get("final_sql") or state.get("best_sql", "")
    if not current_sql:
        stage_status["refiner"] = "failed"
        return {
            **state,
            "error_message": "refiner: no SQL available to execute",
            "stage_status": stage_status,
            "stage_timings": {**stage_timings, "refiner": round(time.perf_counter() - started, 4)},
            "warnings": [*warnings, "refiner: empty SQL input"],
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    full_schema = state.get("full_schema", {})

    # --- Phase 1: deterministic AST repair ---
    validation = validate_sql_schema_references(current_sql, full_schema)
    if validation.warnings:
        warnings.append(validation.warning_message())

    working_sql = current_sql
    if not validation.is_valid:
        repair = ast_repair(current_sql, full_schema)
        if repair.was_repaired:
            warnings.append(f"refiner: ast_repair applied {len(repair.changes)} fix(es): {repair.changes}")
            revalidation = validate_sql_schema_references(repair.repaired_sql, full_schema)
            if revalidation.is_valid:
                working_sql = repair.repaired_sql
                validation = revalidation
            else:
                warnings.append("refiner: ast_repair did not fully resolve schema errors")

    # --- Phase 2: execute (possibly repaired) SQL ---
    if not validation.is_valid:
        execution = None
        execution_error = validation.error_message()
        warnings.append(execution_error)
    else:
        execution = await execute_sql(db_path, working_sql, timeout_seconds=settings.execution_timeout_seconds)
        execution_error = execution.error or "Unknown execution error."

    if execution and execution.success:
        stage_status["refiner"] = "success"
        return {
            **state,
            "final_sql": working_sql,
            "execution_result": str(execution.rows),
            "error_message": None,
            "selection_needs_refine": False,
            "stage_status": stage_status,
            "stage_timings": {**stage_timings, "refiner": round(time.perf_counter() - started, 4)},
            "warnings": warnings,
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }

    # --- Phase 3: tool-augmented LLM refinement ---
    attempts = int(state.get("refine_attempts", 0)) + 1
    next_sql = working_sql
    refine_trigger = execution_error

    selected_candidate_diagnostic = dict(state.get("selected_candidate_diagnostic", {}))
    selected_candidate_summary = str(
        selected_candidate_diagnostic.get("analysis_summary")
        or summarize_candidate_analysis(analyze_sql_candidate(current_sql))
    )
    failed_candidate_summaries = [
        f"candidate {item.get('candidate_index', '-')}: "
        f"execution_error={item.get('execution_error', '-')} | "
        f"schema_errors={item.get('schema_errors', [])} | "
        f"{item.get('analysis_summary', '-')}"
        for item in state.get("candidate_diagnostics", [])
        if not bool(item.get("execution_success")) or not bool(item.get("schema_valid", True))
    ][:3]

    try:
        user_prompt = build_refiner_prompt(
            question=state.get("question", ""),
            evidence=state.get("evidence"),
            filtered_schema=state.get("filtered_schema", ""),
            retrieved_schema_context=state.get("retrieved_schema_context", ""),
            query_sketch_text=state.get("query_sketch_text", ""),
            failed_sql=working_sql,
            execution_error=refine_trigger,
            selection_reasoning=state.get("selection_reasoning", ""),
            selection_confidence=state.get("selection_confidence", "unknown"),
            validation_errors=validation.errors,
            validation_warnings=validation.warnings,
            selected_candidate_summary=selected_candidate_summary,
            failed_candidate_summaries=failed_candidate_summaries,
        )

        router = LLMRouter()
        tools = _build_refiner_tools(db_path, full_schema)
        try:
            raw_text, step_usages = await asyncio.wait_for(
                _run_tool_loop(
                    router, tools, user_prompt,
                    trace_id=state.get("trace_id"), db_id=state.get("db_id"),
                ),
                timeout=_TOOL_LOOP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raw_text = ""
            step_usages = []
            warnings.append(f"refiner: tool loop timed out after {_TOOL_LOOP_TIMEOUT_S}s")

        for u in step_usages:
            llm_usage.append(u)
            total_cost_usd += float(u.get("cost_usd", 0.0))

        parsed = _extract_sql(raw_text)
        if parsed:
            next_sql = parsed
        else:
            warnings.append("refiner: LLM returned empty fix, keeping previous SQL")
    except Exception as exc:
        warnings.append(f"refiner_error: {exc}")

    stage_status["refiner"] = "success"
    return {
        **state,
        "final_sql": next_sql,
        "execution_result": None,
        "refine_attempts": attempts,
        "error_message": refine_trigger,
        "selection_needs_refine": False,
        "stage_status": stage_status,
        "stage_timings": {**stage_timings, "refiner": round(time.perf_counter() - started, 4)},
        "warnings": warnings,
        "llm_usage": llm_usage,
        "total_cost_usd": total_cost_usd,
    }
