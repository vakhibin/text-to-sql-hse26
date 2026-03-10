"""Run Spider v1 benchmark with optional dataset auto-download and smoke mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kagglehub

from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.metrics import BenchmarkMetrics, exact_match
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state
from text_to_sql_agent.tools.sql_executor import execute_sql

KAGGLE_SPIDER_DATASET = "jeromeblanchet/yale-universitys-spider-10-nlp-dataset"


@dataclass
class SpiderExample:
    db_id: str
    question: str
    query: str
    evidence: str | None = None


def _required_spider_files(root: Path) -> list[Path]:
    return [
        root / "tables.json",
        root / "dev.json",
        root / "train_spider.json",
        root / "database",
    ]


def _is_spider_ready(root: Path) -> bool:
    return all(path.exists() for path in _required_spider_files(root))


def _copy_spider_tree(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for name in ["tables.json", "dev.json", "train_spider.json", "database"]:
        src = source_root / name
        dst = target_root / name
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _find_spider_root(downloaded_dir: Path) -> Path | None:
    if _is_spider_ready(downloaded_dir):
        return downloaded_dir
    for candidate in downloaded_dir.rglob("*"):
        if candidate.is_dir() and _is_spider_ready(candidate):
            return candidate
    return None


def ensure_spider_dataset(spider_root: Path, allow_download: bool) -> Path:
    """Ensure Spider files exist locally; optionally download via kagglehub."""
    if _is_spider_ready(spider_root):
        return spider_root
    if not allow_download:
        raise FileNotFoundError(
            f"Spider dataset not found at {spider_root}. "
            "Use --download or set SPIDER_ROOT correctly."
        )

    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_SPIDER_DATASET))
    source_root = _find_spider_root(downloaded_path)
    if source_root is None:
        raise FileNotFoundError(
            f"Downloaded dataset at {downloaded_path}, but Spider files were not detected."
        )
    _copy_spider_tree(source_root, spider_root)
    if not _is_spider_ready(spider_root):
        raise FileNotFoundError("Spider dataset copy completed, but required files are still missing.")
    return spider_root


def load_spider_examples(spider_root: Path, split: str) -> list[SpiderExample]:
    split_file = spider_root / ("dev.json" if split == "dev" else "train_spider.json")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        SpiderExample(
            db_id=item["db_id"],
            question=item["question"],
            query=item["query"],
            evidence=item.get("evidence"),
        )
        for item in data
    ]


async def _evaluate_one(graph, example: SpiderExample, spider_root: Path) -> dict[str, Any]:
    state = make_initial_state(
        question=example.question,
        db_id=example.db_id,
        evidence=example.evidence,
    )
    result = await graph.ainvoke(state)
    predicted_sql = (result.get("final_sql") or result.get("best_sql") or "").strip()

    db_path = spider_root / "database" / example.db_id / f"{example.db_id}.sqlite"
    pred_exec = await execute_sql(str(db_path), predicted_sql) if predicted_sql else None
    gold_exec = await execute_sql(str(db_path), example.query)

    execution_match = (
        pred_exec is not None
        and pred_exec.success
        and gold_exec.success
        and (pred_exec.rows or []) == (gold_exec.rows or [])
    )
    return {
        "db_id": example.db_id,
        "question": example.question,
        "predicted_sql": predicted_sql,
        "gold_sql": example.query,
        "execution_match": bool(execution_match),
        "exact_match": exact_match(predicted_sql, example.query),
        "error_message": result.get("error_message"),
        "warnings": result.get("warnings", []),
    }


async def run_spider_benchmark(
    *,
    spider_root: Path,
    split: str,
    max_examples: int | None,
) -> tuple[BenchmarkMetrics, list[dict[str, Any]]]:
    graph = build_graph()
    examples = load_spider_examples(spider_root=spider_root, split=split)
    if max_examples is not None:
        examples = examples[:max_examples]

    predictions: list[dict[str, Any]] = []
    for example in examples:
        predictions.append(await _evaluate_one(graph, example, spider_root))

    total = len(predictions)
    exec_hits = sum(1 for row in predictions if row["execution_match"])
    em_hits = sum(1 for row in predictions if row["exact_match"])
    valid = sum(1 for row in predictions if bool(row["predicted_sql"]))
    errors = sum(1 for row in predictions if bool(row.get("error_message")))

    metrics = BenchmarkMetrics(
        execution_accuracy=(exec_hits / total) if total else 0.0,
        exact_match=(em_hits / total) if total else 0.0,
        total=total,
        valid_predictions=valid,
        errors=errors,
    )
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spider v1 benchmark")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run on small subset for quick checks")
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--download", action="store_true", default=False, help="Auto-download Spider if missing")
    parser.add_argument("--output", type=str, default="outputs/spider_v1_results.json")
    parser.add_argument("--spider-root", type=str, default=settings.spider_root)
    args = parser.parse_args()

    spider_root = Path(args.spider_root)
    spider_root = ensure_spider_dataset(spider_root=spider_root, allow_download=args.download)

    max_examples = args.max_examples
    if args.smoke:
        max_examples = args.smoke_size

    metrics, predictions = asyncio.run(
        run_spider_benchmark(
            spider_root=spider_root,
            split=args.split,
            max_examples=max_examples,
        )
    )

    from datetime import datetime
    base = Path(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base.with_stem(f"{base.stem}_{stamp}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "spider_root": str(spider_root),
        "smoke": args.smoke,
        "max_examples": max_examples,
        "metrics": {
            "execution_accuracy": metrics.execution_accuracy,
            "exact_match": metrics.exact_match,
            "total": metrics.total,
            "valid_predictions": metrics.valid_predictions,
            "errors": metrics.errors,
        },
        "predictions": predictions,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Spider v1 evaluation completed")
    print(f"  Split: {args.split}")
    print(f"  Total: {metrics.total}")
    print(f"  EX: {metrics.execution_accuracy:.4f}")
    print(f"  EM: {metrics.exact_match:.4f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

