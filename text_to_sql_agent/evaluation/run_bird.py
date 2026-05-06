"""Run BIRD benchmark (EX + R-VES) with optional prewarm, cost summaries, resume, and partial saves."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tqdm import tqdm

from text_to_sql_agent.agents.selector import prewarm_selector_cache
from text_to_sql_agent.tools.few_shot import prewarm_few_shot_cache
from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.metrics import (
    BenchmarkMetrics,
    exact_match,
    execution_match as official_execution_match,
)
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


def _schema_table_filenames_for_split(split: str) -> tuple[str, ...]:
    if split == "train":
        return ("train_tables.json", "tables.json", "dev_tables.json")
    if split == "minidev":
        return ("dev_tables.json", "tables.json", "train_tables.json")
    if split == "dev":
        return ("dev_tables.json", "tables.json", "train_tables.json")
    raise ValueError(f"Unknown BIRD split: {split!r} (expected dev, train, minidev)")


def _triples_for_split(split: str, base: Path) -> list[tuple[Path, Path, Path]]:
    if split == "dev":
        return [
            (base / "dev" / "dev.json", base / "dev_databases", base / "dev"),
            (base / "dev.json", base / "dev_databases", base),
        ]
    if split == "train":
        return [
            (base / "train" / "train.json", base / "train_databases", base / "train"),
            (base / "train.json", base / "train_databases", base),
        ]
    if split == "minidev":
        return [
            (base / "mini_dev_sqlite.json", base / "mini_dev_databases", base),
            (base / "data" / "mini_dev_sqlite-00000-of-00001.json", base / "mini_dev_databases", base),
        ]
    raise ValueError(f"Unknown BIRD split: {split!r} (expected dev, train, minidev)")


def _detect_bird_layout(root: Path, split: str = "dev") -> tuple[Path, Path, Path, Path] | None:
    """Resolve BIRD files for ``split`` under ``root``. Returns json_path, db_dir, schema_root, tables_json."""
    candidate_roots = [
        root,
        root / "MINIDEV",
        root / "data_minidev" / "MINIDEV",
        root / "mini_dev_data",
    ]
    for base in candidate_roots:
        for json_path, db_dir, schema_root in _triples_for_split(split, base):
            if not json_path.is_file() or not db_dir.is_dir():
                continue
            for tables_name in _schema_table_filenames_for_split(split):
                tables_path = schema_root / tables_name
                if tables_path.is_file():
                    return json_path, db_dir, schema_root, tables_path
    return None


def _is_bird_ready(root: Path, split: str = "dev") -> bool:
    return _detect_bird_layout(root, split=split) is not None


def load_bird_examples(bird_root: Path, *, split: str = "dev") -> tuple[list[BirdExample], Path, Path, Path]:
    layout = _detect_bird_layout(bird_root, split=split)
    if layout is None:
        raise FileNotFoundError(
            f"BIRD split {split!r} not found under {bird_root}\n"
            f"--split dev: dev/dev.json or dev.json + dev_databases/ + dev_tables.json; "
            f"train: train/train.json or train.json + train_databases/ + train_tables.json; "
            f"minidev: mini_dev_sqlite.json + mini_dev_databases/."
        )
    json_path, db_dir, schema_root, _tables_path = layout
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
    return examples, db_dir, schema_root, json_path


def _dataset_identity(
    *,
    bird_root: Path,
    json_path: Path,
    num_examples: int,
    max_examples: int | None,
    split: str,
) -> dict[str, Any]:
    return {
        "bird_root": str(bird_root.resolve()),
        "split": split,
        "json_path": str(json_path.resolve()),
        "num_examples": num_examples,
        "max_examples": max_examples,
    }


def _compute_r_ves(pred_time: float, gold_time: float, execution_match: bool) -> float:
    if not execution_match:
        return 0.0
    if pred_time <= 0:
        return 1.0
    ratio = gold_time / pred_time if pred_time > 0 else 1.0
    return min(1.0, ratio**0.5)


def _aggregate_usage_llm(records: list[dict[str, Any]], total_examples: int) -> dict[str, Any]:
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


def _aggregate_cost_for_summary(raw_rows: list[dict[str, Any]], total_examples: int) -> dict[str, Any]:
    llm_records: list[dict[str, Any]] = []
    fallback_usd = 0.0
    for row in raw_rows:
        usage = row.get("llm_usage") or []
        if usage:
            llm_records.extend(usage)
        else:
            fallback_usd += float(row.get("total_cost_usd", 0.0))
    base = _aggregate_usage_llm(llm_records, total_examples)
    if fallback_usd > 0 and not llm_records:
        divisor = total_examples if total_examples else 1
        base["total_cost_usd"] = round(fallback_usd, 8)
        base["avg_cost_per_example_usd"] = round(fallback_usd / divisor, 8)
    elif fallback_usd > 0 and llm_records:
        merged = float(base["total_cost_usd"]) + fallback_usd
        divisor = total_examples if total_examples else 1
        base["total_cost_usd"] = round(merged, 8)
        base["avg_cost_per_example_usd"] = round(merged / divisor, 8)
    return base


def _strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in row.items() if key != "llm_usage"}
    out["example_index"] = row.get("example_index")
    return {k: v for k, v in out.items() if v is not None}


def _internal_row_from_saved(row: dict[str, Any], *, example: BirdExample, example_index: int) -> dict[str, Any]:
    return {
        "question_id": int(row.get("question_id", example.question_id)),
        "db_id": row["db_id"],
        "question": row.get("question", example.question),
        "difficulty": row.get("difficulty", example.difficulty),
        "predicted_sql": row.get("predicted_sql", ""),
        "gold_sql": row.get("gold_sql", example.gold_sql),
        "execution_match": bool(row.get("execution_match")),
        "exact_match": bool(row.get("exact_match")),
        "r_ves": float(row.get("r_ves", 0.0)),
        "pred_time_s": float(row.get("pred_time_s", 0.0)),
        "gold_time_s": float(row.get("gold_time_s", 0.0)),
        "error_message": row.get("error_message"),
        "warnings": row.get("warnings", []),
        "trace_id": row.get("trace_id"),
        "llm_usage": [],
        "total_cost_usd": float(row.get("total_cost_usd", 0.0)),
        "example_index": example_index,
    }


def _metrics_over_full_split(
    *,
    expected_total: int,
    results_by_index: dict[int, dict[str, Any]],
) -> tuple[BenchmarkMetrics, dict[str, int]]:
    if expected_total <= 0:
        z = BenchmarkMetrics(total=0, valid_predictions=0, errors=0)
        return z, {"completed_examples": 0, "incomplete_examples": 0}

    exec_hits = em_hits = 0
    r_ves_sum = 0.0
    valid = 0
    errors = 0
    for idx in range(expected_total):
        row = results_by_index.get(idx)
        if row is None:
            continue
        if row.get("predicted_sql"):
            valid += 1
        if row.get("error_message"):
            errors += 1
        if row.get("execution_match"):
            exec_hits += 1
        if row.get("exact_match"):
            em_hits += 1
        r_ves_sum += float(row.get("r_ves", 0.0))

    completed = len(results_by_index)
    incomplete = expected_total - completed
    metrics = BenchmarkMetrics(
        execution_accuracy=exec_hits / expected_total,
        exact_match=em_hits / expected_total,
        r_ves=r_ves_sum / expected_total,
        total=expected_total,
        valid_predictions=valid,
        errors=errors,
    )
    extras = {"completed_examples": completed, "incomplete_examples": incomplete}
    return metrics, extras


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _should_write_partial(*, completed: int, total: int) -> bool:
    if completed <= 0 or total <= 0:
        return False
    if completed == total:
        return True
    remaining = total - completed
    if remaining <= 5:
        interval = 1
    elif remaining <= 20:
        interval = 5
    else:
        interval = 25
    return completed % interval == 0


def _build_payload(
    *,
    bird_root: Path,
    dataset: dict[str, Any],
    smoke: bool,
    max_examples: int | None,
    prewarm: bool,
    resume_from: str | None,
    retry_errors: bool,
    metrics: BenchmarkMetrics,
    metrics_extra: dict[str, int],
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "bird_root": str(bird_root),
        "split": dataset.get("split", "dev"),
        "quote_sql_column_identifiers": bool(dataset.get("quote_sql_column_identifiers", False)),
        "dataset": dataset,
        "smoke": smoke,
        "max_examples": max_examples,
        "prewarm": prewarm,
        "resume_from": resume_from,
        "retry_errors": retry_errors,
        "status": status,
        "metrics": {
            "execution_accuracy": metrics.execution_accuracy,
            "exact_match": metrics.exact_match,
            "r_ves": metrics.r_ves,
            "total": metrics.total,
            "completed_examples": metrics_extra["completed_examples"],
            "incomplete_examples": metrics_extra["incomplete_examples"],
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
        "schema_root": summary["schema_root"],
        "db_dir": summary["db_dir"],
        "predictions": predictions,
    }


def _validate_resume_dataset(prev: dict[str, Any], current: dict[str, Any]) -> None:
    prev_ds = prev.get("dataset") or {}
    if not prev_ds:
        return
    for key in ("json_path", "num_examples", "max_examples"):
        if prev_ds.get(key) != current.get(key):
            raise ValueError(
                f"Resume file dataset mismatch on {key!r}: "
                f"file={prev_ds.get(key)!r} current={current.get(key)!r}"
            )
    prev_split = prev_ds.get("split")
    if prev_split is not None and prev_split != current.get("split"):
        raise ValueError(
            f"Resume file dataset mismatch on 'split': "
            f"file={prev_split!r} current={current.get('split')!r}"
        )


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
        quote_sql_column_identifiers=True,
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
        and official_execution_match(
            pred_exec.rows,
            gold_exec.rows,
            gold_sql=example.gold_sql,
        )
    )

    row = {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "question": example.question,
        "difficulty": example.difficulty,
        "predicted_sql": predicted_sql,
        "gold_sql": example.gold_sql,
        "execution_match": bool(execution_match),
        "exact_match": exact_match(predicted_sql, example.gold_sql),
        "r_ves": round(_compute_r_ves(pred_time, gold_time, bool(execution_match)), 4),
        "pred_time_s": round(pred_time, 4),
        "gold_time_s": round(gold_time, 4),
        "error_message": result.get("error_message"),
        "warnings": result.get("warnings", []),
        "trace_id": result.get("trace_id"),
        "llm_usage": result.get("llm_usage", []),
        "total_cost_usd": result.get("total_cost_usd", 0.0),
        "example_index": example_idx,
    }
    return row


async def _flush_partial(
    *,
    output_path: Path,
    bird_root: Path,
    dataset: dict[str, Any],
    smoke: bool,
    max_examples: int | None,
    prewarm: bool,
    resume_from: str | None,
    retry_errors: bool,
    results_by_index: dict[int, dict[str, Any]],
    expected_total: int,
    benchmark_run_id: str,
    prewarm_time_s: float,
    eval_started: float,
    schema_root: str,
    db_dir: str,
) -> None:
    raw_ordered = [results_by_index[i] for i in sorted(results_by_index)]
    metrics, metrics_extra = _metrics_over_full_split(
        expected_total=expected_total,
        results_by_index=results_by_index,
    )
    eval_time_s = time.perf_counter() - eval_started
    cost = _aggregate_cost_for_summary(raw_ordered, expected_total)
    completed = len(results_by_index)
    summary = {
        "benchmark_run_id": benchmark_run_id,
        "schema_root": schema_root,
        "db_dir": db_dir,
        "prewarm_time_s": round(prewarm_time_s, 4),
        "eval_time_s": round(eval_time_s, 4),
        "avg_time_per_example_s": round((eval_time_s / completed) if completed else 0.0, 4),
        **cost,
    }
    payload = _build_payload(
        bird_root=bird_root,
        dataset=dataset,
        smoke=smoke,
        max_examples=max_examples,
        prewarm=prewarm,
        resume_from=resume_from,
        retry_errors=retry_errors,
        metrics=metrics,
        metrics_extra=metrics_extra,
        predictions=[_strip_internal_fields(results_by_index[i]) for i in sorted(results_by_index)],
        summary=summary,
        status="partial",
    )
    await asyncio.to_thread(_write_json_atomic, output_path, payload)


def _print_bird_progress_summary(
    *,
    metrics: BenchmarkMetrics,
    metrics_extra: dict[str, int],
    summary: dict[str, Any],
    output_path: Path | None,
    interrupted: bool,
) -> None:
    prefix = "BIRD evaluation interrupted (partial saved)" if interrupted else "BIRD evaluation completed"
    print(prefix, file=sys.stderr)
    print(f"  Total (denominator): {metrics.total}", file=sys.stderr)
    print(f"  Completed rows: {metrics_extra['completed_examples']}", file=sys.stderr)
    if metrics_extra["incomplete_examples"]:
        print(f"  Incomplete: {metrics_extra['incomplete_examples']}", file=sys.stderr)
    print(f"  EX: {metrics.execution_accuracy:.4f}", file=sys.stderr)
    print(f"  EM: {metrics.exact_match:.4f}", file=sys.stderr)
    print(f"  R-VES: {metrics.r_ves:.4f}", file=sys.stderr)
    print(f"  Prewarm: {summary['prewarm_time_s']:.2f}s", file=sys.stderr)
    print(f"  Eval: {summary['eval_time_s']:.2f}s", file=sys.stderr)
    print(f"  Avg/example: {summary['avg_time_per_example_s']:.2f}s", file=sys.stderr)
    print(f"  Cost: ${summary['total_cost_usd']:.6f}", file=sys.stderr)
    if output_path is not None:
        print(f"  Output: {output_path}", file=sys.stderr)


async def run_bird_benchmark(
    *,
    bird_root: Path,
    split: str = "dev",
    max_examples: int | None,
    concurrency: int = 1,
    prewarm: bool = False,
    smoke: bool = False,
    partial_output_path: Path | None = None,
    resume_from: Path | None = None,
    retry_errors: bool = False,
    example_timeout_seconds: float | None = None,
) -> tuple[BenchmarkMetrics, list[dict[str, Any]], dict[str, Any], dict[str, int], dict[str, Any]]:
    examples, db_dir, schema_root, json_path = load_bird_examples(bird_root, split=split)
    if max_examples is not None:
        examples = examples[:max_examples]

    expected_total = len(examples)
    dataset = _dataset_identity(
        bird_root=bird_root,
        json_path=json_path,
        num_examples=expected_total,
        max_examples=max_examples,
        split=split,
    )
    dataset["quote_sql_column_identifiers"] = True

    prev_payload: dict[str, Any] | None = None
    if resume_from is not None and resume_from.is_file():
        with resume_from.open("r", encoding="utf-8") as f:
            prev_payload = json.load(f)
        _validate_resume_dataset(prev_payload, dataset)

    benchmark_run_id = (
        str(prev_payload.get("benchmark_run_id"))
        if prev_payload and prev_payload.get("benchmark_run_id")
        else f"bird-{split}-{uuid4()}"
    )

    results_by_index: dict[int, dict[str, Any]] = {}
    if prev_payload:
        by_qid: dict[int, dict[str, Any]] = {}
        for row in prev_payload.get("predictions", []):
            by_qid[int(row["question_id"])] = row
        for idx, ex in enumerate(examples):
            saved = by_qid.get(ex.question_id)
            if saved is None:
                continue
            if retry_errors and saved.get("error_message"):
                continue
            results_by_index[idx] = _internal_row_from_saved(saved, example=ex, example_index=idx)

    graph = build_graph()
    prewarm_started = time.perf_counter()
    if prewarm:
        await prewarm_selector_cache(
            [example.db_id for example in examples],
            schema_root=str(schema_root),
        )
        await prewarm_few_shot_cache()
    prewarm_time_s = time.perf_counter() - prewarm_started

    pending_indices = [i for i in range(expected_total) if i not in results_by_index]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    eval_started = time.perf_counter()

    pbar = tqdm(
        total=expected_total,
        initial=len(results_by_index),
        desc=f"BIRD eval [{split}]",
        unit="q",
        file=sys.stderr,
    )

    async def _worker(idx: int, example: BirdExample) -> None:
        async with semaphore:
            example_task: asyncio.Task[dict[str, Any]] | None = None
            try:
                coroutine = _evaluate_one(
                    graph,
                    example,
                    db_dir=db_dir,
                    schema_root=schema_root,
                    benchmark_run_id=benchmark_run_id,
                    example_idx=idx,
                )
                example_task = asyncio.create_task(coroutine)
                if example_timeout_seconds and example_timeout_seconds > 0:
                    result = await asyncio.wait_for(asyncio.shield(example_task), timeout=example_timeout_seconds)
                else:
                    result = await example_task
            except asyncio.TimeoutError:
                if example_task is not None and not example_task.done():
                    example_task.cancel()
                result = _timeout_row(example, idx, benchmark_run_id, example_timeout_seconds)
            except Exception as exc:
                if example_task is not None and not example_task.done():
                    example_task.cancel()
                result = _benchmark_error_row(example, idx, benchmark_run_id, exc)

        async with lock:
            results_by_index[idx] = result
            metrics, _me = _metrics_over_full_split(
                expected_total=expected_total,
                results_by_index=results_by_index,
            )
            done = _me["completed_examples"]
            pbar.set_postfix(
                EX=f"{metrics.execution_accuracy:.0%}",
                EM=f"{metrics.exact_match:.0%}",
                RVES=f"{metrics.r_ves:.2f}",
                err=metrics.errors,
            )
            pbar.update(1)

            if partial_output_path and _should_write_partial(completed=done, total=expected_total):
                await _flush_partial(
                    output_path=partial_output_path,
                    bird_root=bird_root,
                    dataset=dataset,
                    smoke=smoke,
                    max_examples=max_examples,
                    prewarm=prewarm,
                    resume_from=str(resume_from) if resume_from else None,
                    retry_errors=retry_errors,
                    results_by_index=dict(results_by_index),
                    expected_total=expected_total,
                    benchmark_run_id=benchmark_run_id,
                    prewarm_time_s=prewarm_time_s,
                    eval_started=eval_started,
                    schema_root=str(schema_root),
                    db_dir=str(db_dir),
                )

    try:
        await asyncio.gather(*[_worker(i, examples[i]) for i in pending_indices])
    except BaseException as exc:
        if partial_output_path:
            snap = dict(results_by_index)
            await _flush_partial(
                output_path=partial_output_path,
                bird_root=bird_root,
                dataset=dataset,
                smoke=smoke,
                max_examples=max_examples,
                prewarm=prewarm,
                resume_from=str(resume_from) if resume_from else None,
                retry_errors=retry_errors,
                results_by_index=snap,
                expected_total=expected_total,
                benchmark_run_id=benchmark_run_id,
                prewarm_time_s=prewarm_time_s,
                eval_started=eval_started,
                schema_root=str(schema_root),
                db_dir=str(db_dir),
            )
            eval_time_s = time.perf_counter() - eval_started
            metrics, metrics_extra = _metrics_over_full_split(
                expected_total=expected_total,
                results_by_index=snap,
            )
            raw_ordered = [snap[i] for i in sorted(snap)]
            cost = _aggregate_cost_for_summary(raw_ordered, expected_total)
            completed = len(snap)
            summary_interrupt = {
                "benchmark_run_id": benchmark_run_id,
                "schema_root": str(schema_root),
                "db_dir": str(db_dir),
                "prewarm_time_s": round(prewarm_time_s, 4),
                "eval_time_s": round(eval_time_s, 4),
                "avg_time_per_example_s": round((eval_time_s / completed) if completed else 0.0, 4),
                **cost,
            }
            _print_bird_progress_summary(
                metrics=metrics,
                metrics_extra=metrics_extra,
                summary=summary_interrupt,
                output_path=partial_output_path,
                interrupted=True,
            )
        raise exc
    finally:
        pbar.close()

    eval_time_s = time.perf_counter() - eval_started
    raw_predictions = [results_by_index[i] for i in range(expected_total)]
    predictions = [_strip_internal_fields(row) for row in raw_predictions]
    metrics, metrics_extra = _metrics_over_full_split(
        expected_total=expected_total,
        results_by_index=results_by_index,
    )
    cost = _aggregate_cost_for_summary(raw_predictions, expected_total)
    summary = {
        "benchmark_run_id": benchmark_run_id,
        "schema_root": str(schema_root),
        "db_dir": str(db_dir),
        "prewarm_time_s": round(prewarm_time_s, 4),
        "eval_time_s": round(eval_time_s, 4),
        "avg_time_per_example_s": round((eval_time_s / expected_total) if expected_total else 0.0, 4),
        **cost,
    }
    return metrics, predictions, summary, metrics_extra, dataset


def _timeout_row(
    example: BirdExample,
    idx: int,
    benchmark_run_id: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    msg = f"benchmark_timeout: example exceeded {timeout_s:.0f}s" if timeout_s else "benchmark_timeout"
    return {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "question": example.question,
        "difficulty": example.difficulty,
        "predicted_sql": "",
        "gold_sql": example.gold_sql,
        "execution_match": False,
        "exact_match": False,
        "r_ves": 0.0,
        "pred_time_s": 0.0,
        "gold_time_s": 0.0,
        "error_message": msg,
        "warnings": [],
        "trace_id": f"{benchmark_run_id}:{idx}",
        "llm_usage": [],
        "total_cost_usd": 0.0,
        "example_index": idx,
    }


def _benchmark_error_row(
    example: BirdExample,
    idx: int,
    benchmark_run_id: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "question": example.question,
        "difficulty": example.difficulty,
        "predicted_sql": "",
        "gold_sql": example.gold_sql,
        "execution_match": False,
        "exact_match": False,
        "r_ves": 0.0,
        "pred_time_s": 0.0,
        "gold_time_s": 0.0,
        "error_message": f"benchmark_worker_error: {exc}",
        "warnings": [],
        "trace_id": f"{benchmark_run_id}:{idx}",
        "llm_usage": [],
        "total_cost_usd": 0.0,
        "example_index": idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIRD benchmark")
    parser.add_argument("--bird-root", type=str, default=settings.bird_root, help="Dataset root (default: BIRD_ROOT)")
    parser.add_argument(
        "--split",
        choices=["dev", "train", "minidev"],
        default="dev",
        help="Which BIRD JSON + DB folders to use (default: dev)",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run on small subset for quick checks")
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/bird_results.json")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Continue from an existing results JSON (same slice / dataset); unfinished and new rows are evaluated",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume-from, re-run examples that have a non-empty error_message",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Number of examples to evaluate in parallel")
    parser.add_argument("--prewarm", action="store_true", help="Preload schema cache and vector index before scoring")
    parser.add_argument(
        "--example-timeout-seconds",
        type=float,
        default=None,
        help="Hard cap per example wall time. Default: LLM timeout * retries + 120s",
    )
    args = parser.parse_args()

    bird_root = Path(args.bird_root)
    if not _is_bird_ready(bird_root, split=args.split):
        raise FileNotFoundError(
            f"BIRD split {args.split!r} not found at {bird_root}. "
            "Check --bird-root and that dev/, train/, or Mini-Dev files are present."
        )

    max_examples = args.max_examples
    if args.smoke:
        max_examples = args.smoke_size

    resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    base = Path(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resume_path if resume_path is not None else base.with_stem(f"{base.stem}_{stamp}")

    default_example_timeout = float((settings.llm_timeout_seconds * settings.retry_attempts) + 120)
    example_timeout_seconds = (
        float(args.example_timeout_seconds)
        if args.example_timeout_seconds is not None
        else default_example_timeout
    )

    metrics, predictions, summary, metrics_extra, dataset = asyncio.run(
        run_bird_benchmark(
            bird_root=bird_root,
            split=args.split,
            max_examples=max_examples,
            concurrency=args.concurrency,
            prewarm=args.prewarm,
            smoke=args.smoke,
            partial_output_path=output_path,
            resume_from=resume_path,
            retry_errors=args.retry_errors,
            example_timeout_seconds=example_timeout_seconds,
        )
    )

    payload = _build_payload(
        bird_root=bird_root,
        dataset=dataset,
        smoke=args.smoke,
        max_examples=max_examples,
        prewarm=args.prewarm,
        resume_from=str(resume_path) if resume_path else None,
        retry_errors=args.retry_errors,
        metrics=metrics,
        metrics_extra=metrics_extra,
        predictions=predictions,
        summary=summary,
        status="completed",
    )
    _write_json_atomic(output_path, payload)
    flush_langfuse()

    print("BIRD evaluation completed")
    print(f"  Split: {args.split}")
    print(f"  Total (denominator): {metrics.total}")
    print(f"  Completed rows: {metrics_extra['completed_examples']}")
    if metrics_extra["incomplete_examples"]:
        print(f"  Incomplete: {metrics_extra['incomplete_examples']}")
    print(f"  EX: {metrics.execution_accuracy:.4f}")
    print(f"  EM: {metrics.exact_match:.4f}")
    print(f"  R-VES: {metrics.r_ves:.4f}")
    print(f"  Prewarm: {summary['prewarm_time_s']:.2f}s")
    print(f"  Eval: {summary['eval_time_s']:.2f}s")
    print(f"  Avg/example: {summary['avg_time_per_example_s']:.2f}s")
    print(f"  Cost: ${summary['total_cost_usd']:.6f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
