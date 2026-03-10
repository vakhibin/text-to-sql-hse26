"""Selector agent: retrieval + LLM reranking."""

from __future__ import annotations

import json
import re
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


def _debug(message: str) -> None:
    if settings.selector_debug:
        print(f"[selector-debug] {message}")


def _filter_schema_tables(schema: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    selected_set = set(selected)
    tables = [tbl for tbl in schema.get("tables", []) if tbl.get("name") in selected_set]
    if not tables:
        tables = schema.get("tables", [])
    return {"db_id": schema.get("db_id"), "tables": tables, "db_path": schema.get("db_path")}


def _safe_parse_selected_tables(response_text: str) -> list[str]:
    text = (response_text or "").strip()

    # 1) Try strict JSON first.
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("selected_tables"), list):
            return [str(name) for name in payload["selected_tables"] if str(name).strip()]
    except json.JSONDecodeError:
        pass

    # 2) Try JSON fenced blocks / first JSON object in text.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        try:
            payload = json.loads(fence_match.group(1))
            if isinstance(payload, dict) and isinstance(payload.get("selected_tables"), list):
                return [str(name) for name in payload["selected_tables"] if str(name).strip()]
        except Exception:
            pass

    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        try:
            payload = json.loads(obj_match.group(0))
            if isinstance(payload, dict) and isinstance(payload.get("selected_tables"), list):
                return [str(name) for name in payload["selected_tables"] if str(name).strip()]
        except Exception:
            pass

    # 3) Fallback: accept plain list-like output or line-based table names.
    plain_list_match = re.search(r"\[(.*?)\]", text, flags=re.DOTALL)
    if plain_list_match:
        raw = plain_list_match.group(1)
        items = [item.strip(" '\"\n\t") for item in raw.split(",")]
        items = [item for item in items if item]
        if items:
            return items

    line_items = []
    for line in text.splitlines():
        candidate = re.sub(r"^\s*[-*\d\).\s]+", "", line).strip(" '\"")
        if candidate:
            line_items.append(candidate)
    if line_items:
        return line_items

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
        _debug(f"db_id={db_id} question={question!r}")
        _debug(f"retrieved_candidates={len(candidates)}")

        selected_tables: list[str] = []
        if candidates:
            _debug(f"candidate_names={[c.get('table_name') for c in candidates]}")
            prompt = build_selector_rerank_prompt(question=question, candidates=candidates)
            router = LLMRouter()
            response_text = await router.ainvoke(
                role=ModelRole.GENERATOR_PRIMARY,
                messages=[("system", "Return strict JSON only."), ("user", prompt)],
                temperature_override=0.0,
            )
            _debug(f"reranker_raw_response={response_text!r}")
            selected_tables = _safe_parse_selected_tables(response_text)
            _debug(f"parsed_selected_tables={selected_tables}")
            selected_tables = [name for name in selected_tables if name][: settings.selector_target_tables_max]

            candidate_names = [str(c.get("table_name", "")) for c in candidates if c.get("table_name")]
            # Keep only names that actually exist in retrieved candidates.
            selected_tables = [name for name in selected_tables if name in candidate_names]
            _debug(f"selected_after_filter={selected_tables}")

            if not selected_tables:
                selected_tables = candidate_names[: settings.selector_target_tables_min]
                warnings.append("selector: fallback to top vector candidates")
                _debug(f"fallback_selected={selected_tables}")
            elif len(selected_tables) < settings.selector_target_tables_min:
                # Pad rather than hard-fallback to preserve reranker signal.
                for name in candidate_names:
                    if name not in selected_tables:
                        selected_tables.append(name)
                    if len(selected_tables) >= settings.selector_target_tables_min:
                        break
                warnings.append("selector: padded reranker selection with vector candidates")
                _debug(f"padded_selected={selected_tables}")
        else:
            warnings.append("selector: no vector candidates found; using full schema")
            _debug("no_candidates_found_using_full_schema")

        filtered_schema = _filter_schema_tables(schema, selected_tables)
        _debug(f"final_selected_tables={selected_tables}")
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

