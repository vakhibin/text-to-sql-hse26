#!/usr/bin/env python3
"""Run first N BIRD dev examples and write row-level execution diagnostics (no metric aggregation)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import asyncio
import json
from datetime import datetime

from text_to_sql_agent.evaluation.metrics import execution_match as official_execution_match
from text_to_sql_agent.evaluation.run_bird import load_bird_examples
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state
from text_to_sql_agent.tools.sql_executor import execute_sql


def _rows_preview(rows: list | None, *, limit: int = 15):
    if rows is None:
        return None
    return [list(r) for r in rows[:limit]]


async def main_async(*, bird_root: Path, split: str, n: int, out: Path) -> None:
    examples, db_dir, schema_root, json_path = load_bird_examples(bird_root, split=split)
    examples = examples[:n]
    graph = build_graph()
    run_id = f"bird-detail-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    records: list[dict] = []

    for idx, ex in enumerate(examples):
        state = make_initial_state(
            question=ex.question,
            db_id=ex.db_id,
            evidence=ex.evidence,
            schema_root=str(schema_root),
            trace_id=f"{run_id}:{idx}",
            quote_sql_column_identifiers=True,
        )
        result = await graph.ainvoke(state)
        pred = (result.get("final_sql") or result.get("best_sql") or "").strip()
        db_path = db_dir / ex.db_id / f"{ex.db_id}.sqlite"

        pred_exec = await execute_sql(str(db_path), pred) if pred else None
        gold_exec = await execute_sql(str(db_path), ex.gold_sql)

        ex_ok = (
            pred_exec is not None
            and pred_exec.success
            and gold_exec.success
            and official_execution_match(
                pred_exec.rows,
                gold_exec.rows,
                gold_sql=ex.gold_sql,
            )
        )

        records.append(
            {
                "example_index": idx,
                "question_id": ex.question_id,
                "db_id": ex.db_id,
                "difficulty": ex.difficulty,
                "question": ex.question,
                "evidence": ex.evidence,
                "predicted_sql": pred,
                "gold_sql": ex.gold_sql,
                "pred_execute_ok": pred_exec.success if pred_exec else False,
                "pred_execute_error": pred_exec.error if pred_exec else None,
                "gold_execute_ok": gold_exec.success,
                "gold_execute_error": gold_exec.error,
                "execution_match": bool(ex_ok),
                "pred_row_count": len(pred_exec.rows) if pred_exec and pred_exec.rows is not None else None,
                "gold_row_count": len(gold_exec.rows) if gold_exec.rows is not None else None,
                "pred_rows_preview": _rows_preview(pred_exec.rows if pred_exec else None),
                "gold_rows_preview": _rows_preview(gold_exec.rows),
                "error_message": result.get("error_message"),
                "warnings": result.get("warnings", []),
                "trace_id": result.get("trace_id"),
                "total_cost_usd": result.get("total_cost_usd", 0.0),
            }
        )

    payload = {
        "run_id": run_id,
        "bird_root": str(bird_root.resolve()),
        "split": split,
        "json_path": str(json_path.resolve()),
        "num_ran": len(records),
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = sum(1 for r in records if r["execution_match"])
    print(f"Wrote {out} — EX on sample: {hits}/{len(records)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-root", type=Path, default=Path("databases/bird"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: outputs/bird_detail_sample_<timestamp>.json",
    )
    args = ap.parse_args()
    out = args.output
    if out is None:
        out = Path("outputs") / f"bird_detail_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    asyncio.run(main_async(bird_root=args.bird_root, split=args.split, n=args.n, out=out))


if __name__ == "__main__":
    main()
