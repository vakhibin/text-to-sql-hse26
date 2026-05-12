#!/usr/bin/env python3
"""Run the full text-to-sql LangGraph once; append one JSON line per invocation; write detailed log file.

If ``--gold-sql`` is omitted, gold SQL is resolved from Spider ``dev.json``, ``train_spider.json``
(via ``--spider-split train``), or ``test.json`` + ``test_gold.sql`` (via ``--spider-split test``)
by ``db_id`` + question (exact, case-insensitive, then fuzzy ≥ 0.92).

Examples:
  uv run python scripts/run_agent_one.py --question "How many singers?" --db-id concert_singer
  uv run python scripts/run_agent_one.py -q "..." -d car_1 --gold-sql "SELECT ..." --json-out outputs/my_runs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.metrics import (
    exact_match,
    execution_match as official_execution_match,
)
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state
from text_to_sql_agent.tools.agent_run_logger import (
    AgentRunLogger,
    build_prediction_summary,
)
from text_to_sql_agent.tools.observability import flush_langfuse
from text_to_sql_agent.tools.sql_executor import execute_sql


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_question(text: str) -> str:
    return " ".join(text.strip().split())


from text_to_sql_agent.evaluation.spider_split_io import spider_split_json_path


def _spider_split_path(schema_root: Path, split: str) -> Path:
    return spider_split_json_path(schema_root, split)


def lookup_spider_gold_sql(
    *,
    schema_root: Path,
    db_id: str,
    question: str,
    split: str,
    fuzzy_min_ratio: float = 0.92,
) -> tuple[str | None, str | None, str | None]:
    """Return (gold_sql, source_label, note). source_label like ``spider:dev.json`` or None."""
    path = _spider_split_path(schema_root, split)
    if not path.is_file():
        return None, None, f"Spider split file missing: {path}"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    nq = _normalize_question(question)
    nq_cf = nq.casefold()
    same_db: list[tuple[str, str, str]] = []
    for item in data:
        if item.get("db_id") != db_id:
            continue
        iq = _normalize_question(str(item.get("question", "")))
        qsql = str(item.get("query", "")).strip()
        same_db.append((iq, iq.casefold(), qsql))

    for iq, _iq_cf, qsql in same_db:
        if iq == nq:
            return qsql, f"spider:{path.name} (exact question match)", None
    for iq, iq_cf, qsql in same_db:
        if iq_cf == nq_cf:
            return qsql, f"spider:{path.name} (case-insensitive question match)", None

    best_sql: str | None = None
    best_r = 0.0
    for iq, _iq_cf, qsql in same_db:
        r = SequenceMatcher(None, nq_cf, iq.casefold()).ratio()
        if r > best_r:
            best_r = r
            best_sql = qsql
    if best_sql is not None and best_r >= fuzzy_min_ratio:
        return (
            best_sql,
            f"spider:{path.name} (fuzzy question match)",
            f"best similarity vs Spider question = {best_r:.4f} (threshold {fuzzy_min_ratio})",
        )

    return (
        None,
        None,
        f"no Spider row for db_id={db_id!r} with matching question in {path.name} "
        f"(candidates for this db: {len(same_db)}); try --gold-sql or fix wording",
    )


async def _run(
    *,
    question: str,
    db_id: str,
    evidence: str | None,
    schema_root: Path,
    gold_sql: str | None,
    auto_gold: bool,
    spider_split: str,
    trace_id: str | None,
    json_out: Path | None,
    log_file: Path | None,
) -> dict:
    agent_run_id = str(uuid4())
    tid = trace_id or agent_run_id

    log_path = log_file or (Path("outputs/agent_debug_logs") / f"{agent_run_id}.log")
    graph = build_graph()
    initial = make_initial_state(
        question=question,
        db_id=db_id,
        evidence=evidence,
        schema_root=str(schema_root),
        trace_id=tid,
        spider_schema_variant=("test" if spider_split == "test" else "default"),
    )

    merged: dict = dict(initial)
    step = 0

    logger = AgentRunLogger(log_path, agent_run_id=agent_run_id)
    logger.log_run_header(
        trace_id=tid,
        question=question,
        db_id=db_id,
        evidence=evidence,
        schema_root=str(schema_root),
    )
    async for chunk in graph.astream(initial, stream_mode="updates"):
        for node, update in chunk.items():
            step += 1
            merged.update(update)
            logger.log_step(step, node, update)

    summary = build_prediction_summary(merged)
    predicted_sql = summary["predicted_sql"]
    gold_sql_norm = (gold_sql or "").strip()
    gold_sql_source: str | None = None
    gold_lookup_note: str | None = None
    if gold_sql_norm:
        gold_sql_source = "cli (--gold-sql)"
    elif auto_gold:
        found, src, note = lookup_spider_gold_sql(
            schema_root=schema_root, db_id=db_id, question=question, split=spider_split
        )
        gold_lookup_note = note
        if found:
            gold_sql_norm = found.strip()
            gold_sql_source = src
        else:
            gold_sql_source = None
    else:
        gold_lookup_note = "auto gold lookup disabled (--no-auto-gold)"

    db_path_str = str((merged.get("full_schema") or {}).get("db_path") or "").strip()

    execution_match: bool | None = None
    pred_exec = None
    gold_exec = None
    pred_execute_err: str | None = None
    gold_execute_err: str | None = None
    if gold_sql_norm and db_path_str and predicted_sql:
        pred_exec = await execute_sql(db_path_str, predicted_sql)
        gold_exec = await execute_sql(db_path_str, gold_sql_norm)
        if not pred_exec.success:
            pred_execute_err = pred_exec.error
        if not gold_exec.success:
            gold_execute_err = gold_exec.error
        execution_match = (
            pred_exec.success
            and gold_exec.success
            and official_execution_match(
                pred_exec.rows,
                gold_exec.rows,
                gold_sql=gold_sql_norm,
            )
        )
    elif gold_sql_norm and not predicted_sql:
        execution_match = False

    em = exact_match(predicted_sql, gold_sql_norm) if gold_sql_norm else None

    logger.write(
        executive_summary={
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql_norm or None,
            "gold_sql_source": gold_sql_source,
            "gold_lookup_note": gold_lookup_note,
            "execution_match": execution_match,
            "exact_match": em,
            "error_message": merged.get("error_message"),
            "predicted_execute_error": pred_execute_err,
            "gold_execute_error": gold_execute_err,
            "warnings": list(merged.get("warnings") or []),
            "trace_id": merged.get("trace_id"),
            "total_cost_usd": float(merged.get("total_cost_usd") or 0.0),
            "stage_status": dict(merged.get("stage_status") or {}),
        },
        extended_summary={
            "agent_run_id": agent_run_id,
            "log_file": str(log_path.resolve()),
            "gold_sql_source": gold_sql_source,
            "gold_lookup_note": gold_lookup_note,
            **summary,
            "llm_usage": list(merged.get("llm_usage") or []),
            "value_hints": merged.get("value_hints"),
            "column_hints": merged.get("column_hints"),
            "query_sketch": merged.get("query_sketch"),
            "query_sketch_text": merged.get("query_sketch_text"),
            "candidates": merged.get("candidates"),
            "valid_candidates": merged.get("valid_candidates"),
            "candidate_diagnostics": merged.get("candidate_diagnostics"),
        },
    )

    row = {
        "agent_run_id": agent_run_id,
        "log_file": str(log_path.resolve()),
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "schema_root": str(schema_root),
        "predicted_sql": predicted_sql,
        "gold_sql": gold_sql_norm or None,
        "gold_sql_source": gold_sql_source,
        "gold_lookup_note": gold_lookup_note,
        "execution_match": execution_match,
        "exact_match": em,
        "error_message": merged.get("error_message"),
        "warnings": list(merged.get("warnings") or []),
        "trace_id": merged.get("trace_id"),
        "total_cost_usd": float(merged.get("total_cost_usd") or 0.0),
        "stage_status": dict(merged.get("stage_status") or {}),
        "stage_timings": dict(merged.get("stage_timings") or {}),
    }

    if json_out is not None:
        _append_jsonl(json_out, row)

    flush_langfuse()

    # stdout: compact line for quick copy
    print(json.dumps({"agent_run_id": agent_run_id, "log_file": str(log_path.resolve()), "predicted_sql": predicted_sql}, ensure_ascii=False))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-to-sql agent once; append JSONL; write detailed log.")
    parser.add_argument("--question", "-q", required=True, type=str)
    parser.add_argument("--db-id", "-d", required=True, type=str)
    parser.add_argument("--evidence", "-e", default=None, type=str)
    parser.add_argument(
        "--schema-root",
        type=Path,
        default=None,
        help=f"Spider root (default: {settings.spider_root})",
    )
    parser.add_argument(
        "--gold-sql",
        default=None,
        type=str,
        help="Override gold SQL; if omitted, gold is taken from Spider dev/train by question+db_id (--spider-split)",
    )
    parser.add_argument(
        "--spider-split",
        choices=("dev", "train", "test"),
        default="dev",
        help="Which Spider JSON to search for auto gold (default: dev)",
    )
    parser.add_argument(
        "--no-auto-gold",
        action="store_true",
        help="Do not load gold SQL from Spider when --gold-sql is omitted",
    )
    parser.add_argument("--trace-id", default=None, type=str, help="Override state trace_id (default: same as agent_run_id)")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs/agent_runs.jsonl"),
        help="Append one JSON object per line (JSON Lines).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: outputs/agent_debug_logs/<agent_run_id>.log)",
    )
    parser.add_argument("--no-json-append", action="store_true", help="Do not append to --json-out")
    args = parser.parse_args()

    root = args.schema_root or Path(settings.spider_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    json_out = None if args.no_json_append else args.json_out

    asyncio.run(
        _run(
            question=args.question,
            db_id=args.db_id,
            evidence=args.evidence,
            schema_root=root,
            gold_sql=args.gold_sql,
            auto_gold=not args.no_auto_gold,
            spider_split=args.spider_split,
            trace_id=args.trace_id,
            json_out=json_out,
            log_file=args.log_file,
        )
    )


if __name__ == "__main__":
    main()
