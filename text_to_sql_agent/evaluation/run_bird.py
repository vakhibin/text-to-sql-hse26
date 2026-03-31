"""Run BIRD benchmark (EX + R-VES) with optional prewarm and cost summaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tqdm import tqdm

from text_to_sql_agent.agents.selector import prewarm_selector_cache
from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.metrics import BenchmarkMetrics
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state
from text_to_sql_agent.tools.observability import flush_langfuse
from text_to_sql_agent.tools.sql_executor import execute_sql


@dataclass
class BirdExample:
    question_id: int
    db_id: str
    question: str
    gold_sql: str
    evidence: str | None = None
    difficulty: str | None = None


def _detect_bird_layout(root: Path) -> tuple[Path, Path, Path] | None:
    """Detect BIRD dataset layout. Returns (json_path, db_dir, schema_root)."""
    candidate_roots = [
        root,
        root / "MINIDEV",
        root / "data_minidev" / "MINIDEV",
        root / "mini_dev_data",
    ]
    for base in candidate_roots:
        json_candidates = [
            base / "mini_dev_sqlite.json",
            base / "data" / "mini_dev_sqlite-00000-of-00001.json",
            base / "dev.json",
            base / "train.json",
        ]
        db_dir_candidates = [
            base / "mini_dev_databases",
            base / "dev_databases",
            base / "train_databases",
            base / "mini_dev_data" / "dev_databases",
        ]
        schema_candidates = [
            base / "dev_tables.json",
            base / "tables.json",
        ]
        json_path = next((path for path in json_candidates if path.exists()), None)
        db_dir = next((path for path in db_dir_candidates if path.exists()), None)
        schema_file = next((path for path in schema_candidates if path.exists()), None)
        if json_path and db_dir and schema_file:
            return json_path, db_dir, base
    return None


def _is_bird_ready(root: Path) -> bool:
    return _detect_bird_layout(root) is not None


def load_bird_examples(bird_root: Path) -> tuple[list[BirdExample], Path, Path]:
    layout = _detect_bird_layout(bird_root)
    if layout is None:
        raise FileNotFoundError(
            f"BIRD dataset not found at {bird_root}\n"
            f"Expected one of:\n"
            f"  {bird_root}/data_minidev/MINIDEV/mini_dev_sqlite.json + dev_databases/\n"
            f"  {bird_root}/dev.json + dev_databases/\n"
            f"  {bird_root}/mini_dev_sqlite.json + mini_dev_databases/\n"
        )
    json_path, db_dir, schema_root = layout
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples = [
        BirdExample(
            question_id=int(item.get("question_id", idx)),
            db_id=item["db_id"],
            question=item["question"],
            gold_sql=item.get("SQL", item.get("query", "")),
            evidence=item.get("evidence") or None,
            difficulty=item.get("difficulty"),
        )
        for idx, item in enumerate(data)
    ]
    return examples, db_dir, schema_root


def _freeze_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))
    return value


def _freeze_row(row: Any) -> tuple[Any, ...]:
    if isinstance(row, (list, tuple)):
        return tuple(_freeze_value(v) for v in row)
    return (_freeze_value(row),)


def _compute_r_ves(pred_time: float, gold_time: float, execution_match: bool) -> float:
    if not execution_match:
        return 0.0
    if pred_time <= 0:
        return 1.0
    ratio = gold_time / pred_time if pred_time > 0 else 1.0
    return min(1.0, ratio**0.5)


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
    example: BirdExample,
    *,
    db_dir: Path,
    schema_root: Path,
    benchmark_run_id: str,
    example_idx: int,
) -> dict[str, Any]:
    state = make_initial_state(
        question=example.question,
        db_id=example.db_id,
        evidence=example.evidence,
        schema_root=str(schema_root),
        trace_id=f"{benchmark_run_id}:{example_idx}",
    )
    result = await graph.ainvoke(state)
    predicted_sql = (result.get("final_sql") or result.get("best_sql") or "").strip()

    db_path = db_dir / example.db_id / f"{example.db_id}.sqlite"
    pred_time = 0.0
    pred_exec = None
    if predicted_sql:
        started = time.perf_counter()
        pred_exec = await execute_sql(str(db_path), predicted_sql)
        pred_time = time.perf_counter() - started

    started = time.perf_counter()
    gold_exec = await execute_sql(str(db_path), example.gold_sql)
    gold_time = time.perf_counter() - started

    execution_match = (
        pred_exec is not None
        and pred_exec.success
        and gold_exec.success
        and {_freeze_row(row) for row in (pred_exec.rows or [])}
        == {_freeze_row(row) for row in (gold_exec.rows or [])}
    )

    return {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "question": example.question,
        "difficulty": example.difficulty,
        "predicted_sql": predicted_sql,
        "gold_sql": example.gold_sql,
        "execution_match": bool(execution_match),
        "r_ves": round(_compute_r_ves(pred_time, gold_time, bool(execution_match)), 4),
        "pred_time_s": round(pred_time, 4),
        "gold_time_s": round(gold_time, 4),
        "error_message": result.get("error_message"),
        "warnings": result.get("warnings", []),
        "trace_id": result.get("trace_id"),
        "llm_usage": result.get("llm_usage", []),
        "total_cost_usd": result.get("total_cost_usd", 0.0),
    }


async def run_bird_benchmark(
    *,
    bird_root: Path,
    max_examples: int | None,
    concurrency: int = 1,
    prewarm: bool = False,
) -> tuple[BenchmarkMetrics, list[dict[str, Any]], dict[str, Any]]:
    benchmark_run_id = f"bird-{uuid4()}"
    graph = build_graph()
    examples, db_dir, schema_root = load_bird_examples(bird_root)
    if max_examples is not None:
        examples = examples[:max_examples]

    prewarm_started = time.perf_counter()
    if prewarm:
        await prewarm_selector_cache(
            [example.db_id for example in examples],
            schema_root=str(schema_root),
        )
    prewarm_time_s = time.perf_counter() - prewarm_started

    semaphore = asyncio.Semaphore(concurrency)
    results_by_index: dict[int, dict[str, Any]] = {}
    exec_hits = 0
    r_ves_sum = 0.0
    err_count = 0
    lock = asyncio.Lock()
    eval_started = time.perf_counter()

    pbar = tqdm(total=len(examples), desc="BIRD eval", unit="q", file=sys.stderr)

    async def _worker(idx: int, example: BirdExample) -> None:
        nonlocal exec_hits, r_ves_sum, err_count
        async with semaphore:
            result = await _evaluate_one(
                graph,
                example,
                db_dir=db_dir,
                schema_root=schema_root,
                benchmark_run_id=benchmark_run_id,
                example_idx=idx,
            )

        async with lock:
            results_by_index[idx] = result
            if result["execution_match"]:
                exec_hits += 1
            r_ves_sum += result["r_ves"]
            if result.get("error_message"):
                err_count += 1
                tqdm.write(
                    f"  ERROR [{result['db_id']}] {result['question'][:60]}... "
                    f"-> {str(result['error_message'])[:120]}",
                    file=sys.stderr,
                )
            done = len(results_by_index)
            pbar.set_postfix(EX=f"{exec_hits/done:.0%}", RVES=f"{r_ves_sum/done:.2f}", err=err_count)
            pbar.update(1)

    await asyncio.gather(*[_worker(i, ex) for i, ex in enumerate(examples)])
    pbar.close()
    eval_time_s = time.perf_counter() - eval_started

    raw_predictions = [results_by_index[i] for i in range(len(examples))]
    predictions = [_strip_internal_fields(row) for row in raw_predictions]
    total = len(predictions)
    valid = sum(1 for row in predictions if bool(row["predicted_sql"]))
    usage_records = [record for row in raw_predictions for record in row.get("llm_usage", [])]

    metrics = BenchmarkMetrics(
        execution_accuracy=(exec_hits / total) if total else 0.0,
        r_ves=(r_ves_sum / total) if total else 0.0,
        total=total,
        valid_predictions=valid,
        errors=err_count,
    )
    summary = {
        "benchmark_run_id": benchmark_run_id,
        "schema_root": str(schema_root),
        "db_dir": str(db_dir),
        "prewarm_time_s": round(prewarm_time_s, 4),
        "eval_time_s": round(eval_time_s, 4),
        "avg_time_per_example_s": round((eval_time_s / total) if total else 0.0, 4),
        **_aggregate_usage(usage_records, total),
    }
    return metrics, predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIRD benchmark")
    parser.add_argument("--bird-root", type=str, default=settings.bird_mini_root)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run on small subset for quick checks")
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/bird_results.json")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of examples to evaluate in parallel")
    parser.add_argument("--prewarm", action="store_true", help="Preload schema cache and vector index before scoring")
    args = parser.parse_args()

    bird_root = Path(args.bird_root)
    if not _is_bird_ready(bird_root):
        raise FileNotFoundError(
            f"BIRD dataset not found at {bird_root}. "
            "Download and unpack Mini-Dev or full BIRD first."
        )

    max_examples = args.max_examples
    if args.smoke:
        max_examples = args.smoke_size

    metrics, predictions, summary = asyncio.run(
        run_bird_benchmark(
            bird_root=bird_root,
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
        "bird_root": str(bird_root),
        "schema_root": summary["schema_root"],
        "db_dir": summary["db_dir"],
        "smoke": args.smoke,
        "max_examples": max_examples,
        "prewarm": args.prewarm,
        "metrics": {
            "execution_accuracy": metrics.execution_accuracy,
            "r_ves": metrics.r_ves,
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

    print("BIRD evaluation completed")
    print(f"  Total: {metrics.total}")
    print(f"  EX: {metrics.execution_accuracy:.4f}")
    print(f"  R-VES: {metrics.r_ves:.4f}")
    print(f"  Prewarm: {summary['prewarm_time_s']:.2f}s")
    print(f"  Eval: {summary['eval_time_s']:.2f}s")
    print(f"  Avg/example: {summary['avg_time_per_example_s']:.2f}s")
    print(f"  Cost: ${summary['total_cost_usd']:.6f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

