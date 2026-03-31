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
from text_to_sql_agent.tools.schema_loader import load_schema, schema_to_mschema
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


def _build_retrieved_schema_context(schema: dict[str, Any], candidate_names: list[str]) -> dict[str, Any]:
    names = [name for name in candidate_names if name]
    if not names:
        return {"db_id": schema.get("db_id"), "tables": [], "db_path": schema.get("db_path")}
    return _filter_schema_tables(schema, names)


def _identifier_tokens(text: str) -> set[str]:
    tokens = {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token}
    expanded = set(tokens)
    for token in list(tokens):
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def _table_lexical_score(question_tokens: set[str], question_text: str, table: dict[str, Any]) -> float:
    table_name = str(table.get("name", "")).strip()
    if not table_name:
        return 0.0
    table_tokens = _identifier_tokens(table_name)
    column_tokens: set[str] = set()
    for column in table.get("columns", []):
        column_tokens.update(_identifier_tokens(str(column.get("name", ""))))

    flat_table_name = " ".join(token for token in re.split(r"[^a-zA-Z0-9]+", table_name.lower()) if token)
    table_overlap = len(table_tokens & question_tokens)
    column_overlap = len(column_tokens & question_tokens)
    score = 0.0
    if flat_table_name and flat_table_name in question_text:
        score += 3.0
    score += table_overlap * 2.0
    score += min(4.0, float(column_overlap))
    return score


def _build_lexical_candidates(schema: dict[str, Any], question: str, top_k: int) -> list[dict[str, Any]]:
    question_tokens = _identifier_tokens(question)
    question_text = " ".join(token for token in re.split(r"[^a-zA-Z0-9]+", question.lower()) if token)
    scored: list[tuple[float, str]] = []
    for table in schema.get("tables", []):
        table_name = str(table.get("name", "")).strip()
        if not table_name:
            continue
        raw_score = _table_lexical_score(question_tokens, question_text, table)
        if raw_score <= 0:
            continue
        scored.append((raw_score, table_name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    lexical_candidates = []
    for raw_score, table_name in scored[:top_k]:
        lexical_candidates.append(
            {
                "table_name": table_name,
                "db_id": schema.get("db_id", ""),
                "score": round(min(1.1, 0.55 + (raw_score / 8.0)), 4),
                "content": f"table={table_name}",
                "source": "lexical",
            }
        )
    return lexical_candidates


def _merge_candidates(
    vector_candidates: list[dict[str, Any]],
    lexical_candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    source_rank = {"hybrid": 2, "lexical": 1, "vector": 0}

    for candidate in vector_candidates:
        table_name = str(candidate.get("table_name", "")).strip()
        if not table_name:
            continue
        merged[table_name] = {
            **candidate,
            "table_name": table_name,
            "source": "vector",
        }

    for candidate in lexical_candidates:
        table_name = str(candidate.get("table_name", "")).strip()
        if not table_name:
            continue
        existing = merged.get(table_name)
        if existing is None:
            merged[table_name] = candidate
            continue
        existing["score"] = round(max(float(existing.get("score", 0.0)), float(candidate.get("score", 0.0))), 4)
        existing["source"] = "hybrid"

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -source_rank.get(str(item.get("source", "")), 0),
            str(item.get("table_name", "")),
        ),
    )
    return ordered[:limit]


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
    llm_usage = list(state.get("llm_usage", []))
    total_cost_usd = float(state.get("total_cost_usd", 0.0))
    stage_status["selector"] = "running"

    try:
        question = state["question"]
        db_id = state["db_id"]
        schema_root = state.get("schema_root")
        schema = await load_schema(db_id, spider_root=schema_root)

        vector_store = _get_vector_store()
        await vector_store.index_schema(db_id=db_id, schema=schema)
        vector_candidates = await vector_store.query_tables(
            query=question,
            db_id=db_id,
            top_k=settings.selector_top_k_tables,
        )
        lexical_candidates = _build_lexical_candidates(
            schema,
            question,
            top_k=settings.selector_top_k_lexical_tables,
        )
        candidates = _merge_candidates(
            vector_candidates=vector_candidates,
            lexical_candidates=lexical_candidates,
            limit=settings.selector_top_k_tables,
        )
        _debug(f"db_id={db_id} question={question!r}")
        _debug(f"vector_candidates={len(vector_candidates)} lexical_candidates={len(lexical_candidates)}")
        _debug(f"retrieved_candidates={len(candidates)}")

        selected_tables: list[str] = []
        candidate_names: list[str] = []
        if candidates:
            _debug(f"candidate_names={[c.get('table_name') for c in candidates]}")
            candidate_names = [str(c.get("table_name", "")) for c in candidates if c.get("table_name")]
            prompt = build_selector_rerank_prompt(question=question, candidates=candidates)
            router = LLMRouter()
            response = await router.ainvoke_with_metadata(
                role=ModelRole.GENERATOR_PRIMARY,
                messages=[("system", "Return strict JSON only."), ("user", prompt)],
                temperature_override=0.0,
                trace_id=state.get("trace_id"),
                db_id=db_id,
                stage="selector",
            )
            response_text = response.text
            llm_usage.append(response.usage)
            total_cost_usd += float(response.usage.get("cost_usd", 0.0))
            _debug(f"reranker_raw_response={response_text!r}")
            selected_tables = _safe_parse_selected_tables(response_text)
            _debug(f"parsed_selected_tables={selected_tables}")
            selected_tables = [name for name in selected_tables if name][: settings.selector_target_tables_max]

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
        retrieved_schema = _build_retrieved_schema_context(schema, candidate_names)
        _debug(f"final_selected_tables={selected_tables}")
        stage_status["selector"] = "success"
        return {
            **state,
            "full_schema": schema,
            "filtered_schema": schema_to_mschema(filtered_schema, schema_root=schema_root),
            "retrieved_schema_context": schema_to_mschema(retrieved_schema, schema_root=schema_root),
            "stage_status": stage_status,
            "stage_timings": {
                **stage_timings,
                "selector": round(time.perf_counter() - started, 4),
            },
            "warnings": warnings,
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
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
            "llm_usage": llm_usage,
            "total_cost_usd": total_cost_usd,
        }


async def prewarm_selector_cache(
    db_ids: list[str],
    *,
    schema_root: str | None = None,
) -> None:
    """Warm schema cache and Chroma index for benchmark runs."""
    vector_store = _get_vector_store()
    for db_id in dict.fromkeys(db_ids):
        schema = await load_schema(db_id, spider_root=schema_root)
        await vector_store.index_schema(db_id=db_id, schema=schema)

