"""Run Spider v1 benchmark with optional dataset auto-download and smoke mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from text_to_sql_agent.evaluation.spider_debug_subset import load_subset_manifest
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


def _build_payload(
    *,
    split: str,
    spider_root: Path,
    smoke: bool,
    max_examples: int | None,
    prewarm: bool,
    subset: dict[str, Any] | None,
    metrics: BenchmarkMetrics,
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    status: str = "completed",
    completed_examples: int | None = None,
) -> dict[str, Any]:
    return {
        "split": split,
        "spider_root": str(spider_root),
        "smoke": smoke,
        "max_examples": max_examples,
        "prewarm": prewarm,
        "subset": subset,
        "status": status,
        "completed_examples": completed_examples if completed_examples is not None else metrics.total,
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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _summarize_partial_results(
    *,
    benchmark_run_id: str,
    prewarm_time_s: float,
    eval_started: float,
    raw_predictions: list[dict[str, Any]],
) -> tuple[BenchmarkMetrics, list[dict[str, Any]], dict[str, Any]]:
    predictions = [_strip_internal_fields(row) for row in raw_predictions]
    total = len(predictions)
    exec_hits = sum(1 for row in predictions if bool(row["execution_match"]))
    em_hits = sum(1 for row in predictions if bool(row["exact_match"]))
    valid = sum(1 for row in predictions if bool(row["predicted_sql"]))
    errors = sum(1 for row in predictions if bool(row.get("error_message")))
    usage_records = [record for row in raw_predictions for record in row.get("llm_usage", [])]
    eval_time_s = time.perf_counter() - eval_started

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


def _apply_subset_manifest(
    *,
    examples: list[SpiderExample],
    split: str,
    subset_manifest_path: Path,
) -> tuple[list[SpiderExample], dict[str, Any]]:
    manifest = load_subset_manifest(subset_manifest_path)
    manifest_split = manifest.get("split")
    if manifest_split and manifest_split != split:
        raise ValueError(
            f"Subset manifest split mismatch: manifest={manifest_split}, requested={split}"
        )

    selected: list[SpiderExample] = []
    seen_indices: set[int] = set()
    for item in manifest["items"]:
        example_index = int(item["example_index"])
        if example_index < 0 or example_index >= len(examples):
            raise IndexError(
                f"Subset manifest references example_index={example_index}, "
                f"but split has {len(examples)} examples"
            )
        if example_index in seen_indices:
            raise ValueError(f"Duplicate example_index in subset manifest: {example_index}")

        example = examples[example_index]
        expected_db_id = item.get("db_id")
        expected_question = item.get("question")
        if expected_db_id and example.db_id != expected_db_id:
            raise ValueError(
                f"Subset manifest db_id mismatch at index {example_index}: "
                f"{expected_db_id} != {example.db_id}"
            )
        if expected_question and example.question != expected_question:
            raise ValueError(
                f"Subset manifest question mismatch at index {example_index}: "
                f"{expected_question!r} != {example.question!r}"
            )

        selected.append(example)
        seen_indices.add(example_index)

    metadata = {
        "enabled": True,
        "subset_id": manifest.get("subset_id"),
        "manifest_path": str(subset_manifest_path),
        "selection_strategy": manifest.get("selection_strategy"),
        "requested_examples": len(manifest["items"]),
        "source_pool_counts": manifest.get("source_pool_counts"),
        "selected_difficulty_counts": manifest.get("selected_difficulty_counts"),
        "selected_outcome_counts": manifest.get("selected_outcome_counts"),
    }
    return selected, metadata


async def run_spider_benchmark(
    *,
    spider_root: Path,
    split: str,
    max_examples: int | None,
    concurrency: int = 1,
    prewarm: bool = False,
    partial_output_path: Path | None = None,
    smoke: bool = False,
    example_timeout_seconds: float | None = None,
    subset_manifest_path: Path | None = None,
) -> tuple[BenchmarkMetrics, list[dict[str, Any]], dict[str, Any]]:
    benchmark_run_id = f"spider-{split}-{uuid4()}"
    graph = build_graph()
    examples = load_spider_examples(spider_root=spider_root, split=split)
    subset_metadata: dict[str, Any] | None = None
    if subset_manifest_path is not None:
        examples, subset_metadata = _apply_subset_manifest(
            examples=examples,
            split=split,
            subset_manifest_path=subset_manifest_path,
        )
    if max_examples is not None:
        examples = examples[:max_examples]
    if subset_metadata is not None:
        subset_metadata["evaluated_examples"] = len(examples)

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
            try:
                coroutine = _evaluate_one(
                    graph,
                    example,
                    spider_root,
                    benchmark_run_id=benchmark_run_id,
                    example_idx=idx,
                )
                if example_timeout_seconds and example_timeout_seconds > 0:
                    result = await asyncio.wait_for(coroutine, timeout=example_timeout_seconds)
                else:
                    result = await coroutine
            except asyncio.TimeoutError:
                result = {
                    "db_id": example.db_id,
                    "question": example.question,
                    "predicted_sql": "",
                    "gold_sql": example.query,
                    "execution_match": False,
                    "exact_match": False,
                    "error_message": f"benchmark_timeout: example exceeded {example_timeout_seconds:.0f}s",
                    "warnings": [],
                    "trace_id": f"{benchmark_run_id}:{idx}",
                    "llm_usage": [],
                    "total_cost_usd": 0.0,
                }
            except Exception as exc:
                result = {
                    "db_id": example.db_id,
                    "question": example.question,
                    "predicted_sql": "",
                    "gold_sql": example.query,
                    "execution_match": False,
                    "exact_match": False,
                    "error_message": f"benchmark_worker_error: {exc}",
                    "warnings": [],
                    "trace_id": f"{benchmark_run_id}:{idx}",
                    "llm_usage": [],
                    "total_cost_usd": 0.0,
                }

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
            if partial_output_path and (done % 25 == 0 or done == len(examples)):
                partial_rows = [results_by_index[i] for i in sorted(results_by_index)]
                partial_metrics, partial_predictions, partial_summary = _summarize_partial_results(
                    benchmark_run_id=benchmark_run_id,
                    prewarm_time_s=prewarm_time_s,
                    eval_started=eval_started,
                    raw_predictions=partial_rows,
                )
                payload = _build_payload(
                    split=split,
                    spider_root=spider_root,
                    smoke=smoke,
                    max_examples=max_examples,
                    prewarm=prewarm,
                    subset=subset_metadata,
                    metrics=partial_metrics,
                    predictions=partial_predictions,
                    summary=partial_summary,
                    status="partial",
                    completed_examples=done,
                )
                await asyncio.to_thread(_write_json_atomic, partial_output_path, payload)

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
    parser.add_argument(
        "--subset-manifest",
        type=str,
        default=None,
        help="Optional path to a fixed Spider subset manifest for cheap debug runs",
    )
    parser.add_argument(
        "--example-timeout-seconds",
        type=float,
        default=None,
        help="Hard timeout for one benchmark example. Defaults to 3x LLM timeout plus execution buffer.",
    )
    args = parser.parse_args()

    spider_root = Path(args.spider_root)
    spider_root = ensure_spider_dataset(spider_root=spider_root, allow_download=args.download)

    max_examples = args.max_examples
    if args.smoke:
        max_examples = args.smoke_size

    base = Path(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base.with_stem(f"{base.stem}_{stamp}")
    default_example_timeout = float((settings.llm_timeout_seconds * settings.retry_attempts) + 120)
    example_timeout_seconds = (
        float(args.example_timeout_seconds)
        if args.example_timeout_seconds is not None
        else default_example_timeout
    )

    metrics, predictions, summary = asyncio.run(
        run_spider_benchmark(
            spider_root=spider_root,
            split=args.split,
            max_examples=max_examples,
            concurrency=args.concurrency,
            prewarm=args.prewarm,
            partial_output_path=output_path,
            smoke=args.smoke,
            example_timeout_seconds=example_timeout_seconds,
            subset_manifest_path=Path(args.subset_manifest) if args.subset_manifest else None,
        )
    )
    subset_payload = None
    if args.subset_manifest:
        subset_payload = load_subset_manifest(Path(args.subset_manifest))
        subset_payload = {
            "enabled": True,
            "subset_id": subset_payload.get("subset_id"),
            "manifest_path": args.subset_manifest,
            "selection_strategy": subset_payload.get("selection_strategy"),
            "requested_examples": len(subset_payload["items"]),
            "source_pool_counts": subset_payload.get("source_pool_counts"),
            "selected_difficulty_counts": subset_payload.get("selected_difficulty_counts"),
            "selected_outcome_counts": subset_payload.get("selected_outcome_counts"),
            "evaluated_examples": metrics.total,
        }
    payload = _build_payload(
        split=args.split,
        spider_root=spider_root,
        smoke=args.smoke,
        max_examples=max_examples,
        prewarm=args.prewarm,
        subset=subset_payload,
        metrics=metrics,
        predictions=predictions,
        summary=summary,
        status="completed",
    )
    _write_json_atomic(output_path, payload)
    flush_langfuse()

    print("Spider v1 evaluation completed")
    print(f"  Split: {args.split}")
    print(f"  Total: {metrics.total}")
    print(f"  EX: {metrics.execution_accuracy:.4f}")
    print(f"  EM: {metrics.exact_match:.4f}")
    if subset_payload:
        print(f"  Subset: {subset_payload['subset_id']} ({subset_payload['evaluated_examples']} examples)")
    print(f"  Prewarm: {summary['prewarm_time_s']:.2f}s")
    print(f"  Eval: {summary['eval_time_s']:.2f}s")
    print(f"  Avg/example: {summary['avg_time_per_example_s']:.2f}s")
    print(f"  Cost: ${summary['total_cost_usd']:.6f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

