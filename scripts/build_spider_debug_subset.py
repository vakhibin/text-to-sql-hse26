#!/usr/bin/env python3
"""Build a stable balanced Spider debug subset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.spider_debug_subset import (
    build_spider_subset_records,
    select_balanced_debug_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stable Spider debug subset manifest")
    parser.add_argument("--spider-root", type=str, default=settings.spider_root)
    parser.add_argument("--split", choices=["dev", "train", "test"], default="dev")
    parser.add_argument("--subset-id", type=str, default="spider-debug-v1")
    parser.add_argument("--per-tier", type=int, default=50, help="Examples per difficulty tier")
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Optional Spider results JSON used to inject success/failure outcome hints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/debug/spider_dev_subset_v1.json",
        help="Where to save the subset manifest",
    )
    args = parser.parse_args()

    spider_root = Path(args.spider_root)
    output_path = Path(args.output)
    results_path = Path(args.results) if args.results else None

    records = build_spider_subset_records(
        spider_root=spider_root,
        split=args.split,
        results_payload_path=results_path,
    )
    selected = select_balanced_debug_subset(records, per_tier_target=args.per_tier, seed=args.seed)

    pool_counts = Counter(record.difficulty for record in records)
    selected_counts = Counter(record.difficulty for record in selected)
    outcome_counts = Counter(record.outcome_hint for record in selected)

    payload = {
        "benchmark": "spider_v1",
        "subset_id": args.subset_id,
        "split": args.split,
        "spider_root": str(spider_root),
        "selection_strategy": "offline_spider_sql_hardness_v1",
        "seed": args.seed,
        "per_tier_target": args.per_tier,
        "total_examples": len(selected),
        "results_source": str(results_path) if results_path is not None else None,
        "source_pool_counts": dict(sorted(pool_counts.items())),
        "selected_difficulty_counts": dict(sorted(selected_counts.items())),
        "selected_outcome_counts": dict(sorted(outcome_counts.items())),
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
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved Spider debug subset: {output_path}")
    print(f"  subset_id: {args.subset_id}")
    print(f"  total_examples: {len(selected)}")
    print(f"  difficulty_counts: {dict(sorted(selected_counts.items()))}")
    print(f"  outcome_counts: {dict(sorted(outcome_counts.items()))}")


if __name__ == "__main__":
    main()
