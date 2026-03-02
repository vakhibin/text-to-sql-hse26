"""Selector agent: retrieval + LLM reranking."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.prompts.selector import build_selector_rerank_prompt
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole
from text_to_sql_agent.tools.schema_loader import load_schema, to_mschema
from text_to_sql_agent.tools.vector_store import build_vector_store


@lru_cache(maxsize=1)
def _get_vector_store():
    return build_vector_store(collection_name=settings.chroma_collection_selector)


def _filter_schema_tables(schema: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    selected_set = set(selected)
    tables = [tbl for tbl in schema.get("tables", []) if tbl.get("name") in selected_set]
    if not tables:
        tables = schema.get("tables", [])
    return {"db_id": schema.get("db_id"), "tables": tables, "db_path": schema.get("db_path")}


def _safe_parse_selected_tables(response_text: str) -> list[str]:
    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict) and isinstance(payload.get("selected_tables"), list):
            return [str(name) for name in payload["selected_tables"] if str(name).strip()]
    except json.JSONDecodeError:
        pass
    return []


async def run_selector(state: SQLAgentState) -> SQLAgentState:
    """Select relevant schema subset using Chroma retrieval and LLM rerank."""
    started = time.perf_counter()
    stage_status = dict(state.get("stage_status", {}))
    stage_timings = dict(state.get("stage_timings", {}))
    warnings = list(state.get("warnings", []))
    stage_status["selector"] = "running"

    try:
        question = state["question"]
        db_id = state["db_id"]
        schema = await load_schema(db_id)

        vector_store = _get_vector_store()
        await vector_store.index_schema(db_id=db_id, schema=schema)
        candidates = await vector_store.query_tables(
            query=question,
            db_id=db_id,
            top_k=settings.selector_top_k_tables,
        )

        selected_tables: list[str] = []
        if candidates:
            prompt = build_selector_rerank_prompt(question=question, candidates=candidates)
            router = LLMRouter()
            response_text = await router.ainvoke(
                role=ModelRole.GENERATOR_PRIMARY,
                messages=[("system", "Return strict JSON only."), ("user", prompt)],
                temperature_override=0.0,
            )
            selected_tables = _safe_parse_selected_tables(response_text)[: settings.selector_target_tables_max]
            if len(selected_tables) < settings.selector_target_tables_min:
                fallback = [c.get("table_name", "") for c in candidates[: settings.selector_target_tables_min]]
                selected_tables = [name for name in fallback if name]
                warnings.append("selector: fallback to top vector candidates")
        else:
            warnings.append("selector: no vector candidates found; using full schema")

        filtered_schema = _filter_schema_tables(schema, selected_tables)
        stage_status["selector"] = "success"
        return {
            **state,
            "full_schema": schema,
            "filtered_schema": to_mschema(filtered_schema),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "selector": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
        }

    except Exception as exc:
        stage_status["selector"] = "failed"
        return {
            **state,
            "error_message": str(exc),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "selector": round(time.perf_counter() - started, 4),
            },
            "warnings": [*warnings, f"selector_error: {exc}"],
        }

