#!/usr/bin/env python3
"""Build a Spider subset manifest focused on the most problematic databases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.spider_debug_subset import build_spider_subset_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Spider subset from top problematic DBs in a results file")
    parser.add_argument("--results", type=str, required=True, help="Path to Spider results JSON")
    parser.add_argument("--spider-root", type=str, default=settings.spider_root)
    parser.add_argument("--split", choices=["dev", "train", "test"], default="dev")
    parser.add_argument("--subset-id", type=str, default="spider-problem-dbs-v1")
    parser.add_argument("--top-dbs", type=int, default=3, help="Number of problematic DBs to include")
    parser.add_argument(
        "--db-ids",
        nargs="+",
        default=None,
        help="Explicit DB ids to include instead of auto-selecting top problematic DBs",
    )
    parser.add_argument(
        "--questions-json",
        type=str,
        default=None,
        help="Optional JSON file with an array of exact questions to keep",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="If set, keep only rows with outcome_hint=failure inside the selected DBs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/debug/spider_problem_dbs_v1.json",
        help="Where to save the subset manifest",
    )
    args = parser.parse_args()

    spider_root = Path(args.spider_root)
    results_path = Path(args.results)
    output_path = Path(args.output)
    questions_path = Path(args.questions_json) if args.questions_json else None

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError(f"Invalid results payload: {results_path}")

    if args.db_ids:
        focus_dbs = list(dict.fromkeys(db for db in args.db_ids if db))
    else:
        error_db_counts = Counter(row.get("db_id", "") for row in predictions if row.get("error_message"))
        focus_dbs = [db for db, _ in error_db_counts.most_common(args.top_dbs) if db]
    if not focus_dbs:
        raise ValueError("No problematic DBs found in the provided results file")

    selected_questions: set[str] | None = None
    if questions_path is not None:
        raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
        if not isinstance(raw_questions, list) or not all(isinstance(item, str) for item in raw_questions):
            raise ValueError(f"questions-json must be a JSON array of strings: {questions_path}")
        selected_questions = set(raw_questions)

    records = build_spider_subset_records(
        spider_root=spider_root,
        split=args.split,
        results_payload_path=results_path,
    )
    selected = [record for record in records if record.db_id in focus_dbs]
    if args.failures_only:
        selected = [record for record in selected if record.outcome_hint == "failure"]
    if selected_questions is not None:
        selected = [record for record in selected if record.question in selected_questions]

    db_counts = Counter(record.db_id for record in selected)
    difficulty_counts = Counter(record.difficulty for record in selected)
    outcome_counts = Counter(record.outcome_hint for record in selected)

    output_payload = {
        "benchmark": "spider_v1",
        "subset_id": args.subset_id,
        "split": args.split,
        "spider_root": str(spider_root),
        "selection_strategy": "top_problematic_dbs_from_results_v1",
        "source_run": str(results_path),
        "selected_db_ids": focus_dbs,
        "failures_only": bool(args.failures_only),
        "questions_json": str(questions_path) if questions_path is not None else None,
        "total_examples": len(selected),
        "db_counts": dict(sorted(db_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "items": [
            {
                "example_index": record.example_index,
                "db_id": record.db_id,
                "question": record.question,
                "gold_sql": record.gold_sql,
                "difficulty": record.difficulty,
                "outcome_hint": record.outcome_hint,
            }
            for record in selected
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved Spider problem-DB subset: {output_path}")
    print(f"  subset_id: {args.subset_id}")
    print(f"  selected_db_ids: {focus_dbs}")
    print(f"  total_examples: {len(selected)}")
    print(f"  db_counts: {dict(sorted(db_counts.items()))}")
    print(f"  difficulty_counts: {dict(sorted(difficulty_counts.items()))}")
    print(f"  outcome_counts: {dict(sorted(outcome_counts.items()))}")


if __name__ == "__main__":
    main()
