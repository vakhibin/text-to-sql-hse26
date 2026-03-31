"""Run Spider v1 benchmark with optional dataset auto-download and smoke mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import sys

import kagglehub
from tqdm import tqdm

from text_to_sql_agent.agents.selector import prewarm_selector_cache
from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.metrics import BenchmarkMetrics, exact_match
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state
from text_to_sql_agent.tools.observability import flush_langfuse
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


def _aggregate_usage(records: list[dict[str, Any]], total_examples: int) -> dict[str, Any]:
    prompt_tokens = sum(int(record.get("prompt_tokens", 0)) for record in records)
    completion_tokens = sum(int(record.get("completion_tokens", 0)) for record in records)
    total_tokens = sum(int(record.get("total_tokens", 0)) for record in records)
    total_cost_usd = sum(float(record.get("cost_usd", 0.0)) for record in records)
    divisor = total_examples if total_examples else 1
    return {
        "llm_calls": len(records),
        "total_cost_usd": round(total_cost_usd, 8),
        "avg_cost_per_example_usd": round(total_cost_usd / divisor, 8),
        "avg_prompt_tokens": round(prompt_tokens / divisor, 2),
        "avg_completion_tokens": round(completion_tokens / divisor, 2),
        "avg_total_tokens": round(total_tokens / divisor, 2),
    }


def _strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "llm_usage"}


async def _evaluate_one(
    graph,
    example: SpiderExample,
    spider_root: Path,
    *,
    benchmark_run_id: str,
    example_idx: int,
) -> dict[str, Any]:
    state = make_initial_state(
        question=example.question,
        db_id=example.db_id,
        evidence=example.evidence,
        schema_root=str(spider_root),
        trace_id=f"{benchmark_run_id}:{example_idx}",
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
        "trace_id": result.get("trace_id"),
        "llm_usage": result.get("llm_usage", []),
        "total_cost_usd": result.get("total_cost_usd", 0.0),
    }


async def run_spider_benchmark(
    *,
    spider_root: Path,
    split: str,
    max_examples: int | None,
    concurrency: int = 1,
    prewarm: bool = False,
) -> tuple[BenchmarkMetrics, list[dict[str, Any]], dict[str, Any]]:
    benchmark_run_id = f"spider-{split}-{uuid4()}"
    graph = build_graph()
    examples = load_spider_examples(spider_root=spider_root, split=split)
    if max_examples is not None:
        examples = examples[:max_examples]

    prewarm_started = time.perf_counter()
    if prewarm:
        await prewarm_selector_cache(
            [example.db_id for example in examples],
            schema_root=str(spider_root),
        )
    prewarm_time_s = time.perf_counter() - prewarm_started

    semaphore = asyncio.Semaphore(concurrency)
    results_by_index: dict[int, dict[str, Any]] = {}
    exec_hits = 0
    em_hits = 0
    err_count = 0
    lock = asyncio.Lock()
    eval_started = time.perf_counter()

    pbar = tqdm(total=len(examples), desc="Spider eval", unit="q", file=sys.stderr)

    async def _worker(idx: int, example: SpiderExample) -> None:
        nonlocal exec_hits, em_hits, err_count
        async with semaphore:
            result = await _evaluate_one(
                graph,
                example,
                spider_root,
                benchmark_run_id=benchmark_run_id,
                example_idx=idx,
            )

        async with lock:
            results_by_index[idx] = result
            if result["execution_match"]:
                exec_hits += 1
            if result["exact_match"]:
                em_hits += 1
            if result.get("error_message"):
                err_count += 1
                tqdm.write(
                    f"  ERROR [{result['db_id']}] {result['question'][:60]}... "
                    f"-> {result['error_message'][:120]}",
                    file=sys.stderr,
                )
            done = len(results_by_index)
            pbar.set_postfix(EX=f"{exec_hits/done:.0%}", EM=f"{em_hits/done:.0%}", err=err_count)
            pbar.update(1)

    await asyncio.gather(*[_worker(i, ex) for i, ex in enumerate(examples)])
    pbar.close()
    eval_time_s = time.perf_counter() - eval_started

    raw_predictions = [results_by_index[i] for i in range(len(examples))]
    predictions = [_strip_internal_fields(row) for row in raw_predictions]
    total = len(predictions)
    valid = sum(1 for row in predictions if bool(row["predicted_sql"]))
    errors = err_count
    usage_records = [record for row in raw_predictions for record in row.get("llm_usage", [])]

    metrics = BenchmarkMetrics(
        execution_accuracy=(exec_hits / total) if total else 0.0,
        exact_match=(em_hits / total) if total else 0.0,
        total=total,
        valid_predictions=valid,
        errors=errors,
    )
    summary = {
        "benchmark_run_id": benchmark_run_id,
        "prewarm_time_s": round(prewarm_time_s, 4),
        "eval_time_s": round(eval_time_s, 4),
        "avg_time_per_example_s": round((eval_time_s / total) if total else 0.0, 4),
        **_aggregate_usage(usage_records, total),
    }
    return metrics, predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spider v1 benchmark")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run on small subset for quick checks")
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--download", action="store_true", default=False, help="Auto-download Spider if missing")
    parser.add_argument("--output", type=str, default="outputs/spider_v1_results.json")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of examples to evaluate in parallel")
    parser.add_argument("--prewarm", action="store_true", help="Preload schema cache and vector index before scoring")
    parser.add_argument("--spider-root", type=str, default=settings.spider_root)
    args = parser.parse_args()

    spider_root = Path(args.spider_root)
    spider_root = ensure_spider_dataset(spider_root=spider_root, allow_download=args.download)

    max_examples = args.max_examples
    if args.smoke:
        max_examples = args.smoke_size

    metrics, predictions, summary = asyncio.run(
        run_spider_benchmark(
            spider_root=spider_root,
            split=args.split,
            max_examples=max_examples,
            concurrency=args.concurrency,
            prewarm=args.prewarm,
        )
    )

    base = Path(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base.with_stem(f"{base.stem}_{stamp}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "spider_root": str(spider_root),
        "smoke": args.smoke,
        "max_examples": max_examples,
        "prewarm": args.prewarm,
        "metrics": {
            "execution_accuracy": metrics.execution_accuracy,
            "exact_match": metrics.exact_match,
            "total": metrics.total,
            "valid_predictions": metrics.valid_predictions,
            "errors": metrics.errors,
        },
        "timings": {
            "prewarm_time_s": summary["prewarm_time_s"],
            "eval_time_s": summary["eval_time_s"],
            "avg_time_per_example_s": summary["avg_time_per_example_s"],
        },
        "cost": {
            "llm_calls": summary["llm_calls"],
            "total_cost_usd": summary["total_cost_usd"],
            "avg_cost_per_example_usd": summary["avg_cost_per_example_usd"],
            "avg_prompt_tokens": summary["avg_prompt_tokens"],
            "avg_completion_tokens": summary["avg_completion_tokens"],
            "avg_total_tokens": summary["avg_total_tokens"],
        },
        "benchmark_run_id": summary["benchmark_run_id"],
        "predictions": predictions,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    flush_langfuse()

    print("Spider v1 evaluation completed")
    print(f"  Split: {args.split}")
    print(f"  Total: {metrics.total}")
    print(f"  EX: {metrics.execution_accuracy:.4f}")
    print(f"  EM: {metrics.exact_match:.4f}")
    print(f"  Prewarm: {summary['prewarm_time_s']:.2f}s")
    print(f"  Eval: {summary['eval_time_s']:.2f}s")
    print(f"  Avg/example: {summary['avg_time_per_example_s']:.2f}s")
    print(f"  Cost: ${summary['total_cost_usd']:.6f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

