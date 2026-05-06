#!/usr/bin/env python3
"""Re-run BIRD benchmark rows that failed with OpenRouter balance (HTTP 402), merge into one full JSON.

Recomputes EX / EM / R-VES and cost over the full split (same denominators as run_bird).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.tools.observability import flush_langfuse
from text_to_sql_agent.evaluation.run_bird import (
    _aggregate_cost_for_summary,
    _benchmark_error_row,
    _build_payload,
    _dataset_identity,
    _evaluate_one,
    _internal_row_from_saved,
    _metrics_over_full_split,
    _strip_internal_fields,
    _timeout_row,
    _validate_resume_dataset,
    _write_json_atomic,
    load_bird_examples,
)


def _is_openrouter_balance_error(message: str | None) -> bool:
    if not message:
        return False
    m = message
    if "402" not in m:
        return False
    low = m.lower()
    if "insufficient" in low or "balance" in low or "credits" in low:
        return True
    if "embeddings http" in low and "402" in m:
        return True
    if "error code: 402" in low or '"code": 402' in low:
        return True
    return False


def _predictions_by_index(predictions: list[dict], expected_total: int) -> dict[int, dict]:
    by_idx: dict[int, dict] = {}
    for row in predictions:
        idx = row.get("example_index")
        if idx is not None:
            by_idx[int(idx)] = row
    if len(by_idx) != expected_total and len(predictions) == expected_total:
        return {i: predictions[i] for i in range(expected_total)}
    if len(by_idx) != expected_total:
        raise ValueError(
            f"Predictions cover {len(by_idx)} indices, expected {expected_total}; "
            "check example_index or prediction count."
        )
    return by_idx


async def _retry_balance_errors_async(
    *,
    source: dict,
    bird_root: Path,
    split: str,
    concurrency: int,
    prewarm: bool,
    example_timeout_seconds: float | None,
) -> tuple[dict[int, dict], list[int], float, float]:
    """Merge kept rows with freshly evaluated balance-error rows.

    Returns ``results_by_index``, ``retry_indices``, ``retry_eval_wall_s``, ``retry_prewarm_s``.
    """
    from text_to_sql_agent.agents.selector import prewarm_selector_cache

    ds = source.get("dataset") or {}
    max_examples = ds.get("max_examples")
    examples, db_dir, schema_root, json_path = load_bird_examples(bird_root, split=split)
    if max_examples is not None:
        examples = examples[: int(max_examples)]

    expected_total = len(examples)
    current = _dataset_identity(
        bird_root=bird_root,
        json_path=json_path,
        num_examples=expected_total,
        max_examples=max_examples if max_examples is None else int(max_examples),
        split=split,
    )
    _validate_resume_dataset(source, current)

    pred_by_idx = _predictions_by_index(source["predictions"], expected_total)
    retry_indices = [i for i in range(expected_total) if _is_openrouter_balance_error(pred_by_idx[i].get("error_message"))]

    results_by_index: dict[int, dict] = {}
    for idx in range(expected_total):
        if idx in retry_indices:
            continue
        results_by_index[idx] = _internal_row_from_saved(
            pred_by_idx[idx], example=examples[idx], example_index=idx
        )

    if not retry_indices:
        return results_by_index, [], 0.0, 0.0

    benchmark_run_id = f"bird-{split}-retrybal-{uuid4()}"
    graph = build_graph()
    default_example_timeout = float((settings.llm_timeout_seconds * settings.retry_attempts) + 120)
    timeout_s = (
        float(example_timeout_seconds)
        if example_timeout_seconds is not None
        else default_example_timeout
    )

    prewarm_s = 0.0
    if prewarm:
        t0 = time.perf_counter()
        await prewarm_selector_cache(
            list({examples[i].db_id for i in retry_indices}),
            schema_root=str(schema_root),
        )
        prewarm_s = time.perf_counter() - t0

    semaphore = asyncio.Semaphore(concurrency)
    eval_started = time.perf_counter()

    pbar = tqdm(total=len(retry_indices), desc="BIRD balance retry", unit="q", file=sys.stderr)

    async def _worker(idx: int) -> None:
        example = examples[idx]
        nonlocal results_by_index
        async with semaphore:
            example_task: asyncio.Task | None = None
            try:
                coro = _evaluate_one(
                    graph,
                    example,
                    db_dir=db_dir,
                    schema_root=schema_root,
                    benchmark_run_id=benchmark_run_id,
                    example_idx=idx,
                )
                example_task = asyncio.create_task(coro)
                if timeout_s > 0:
                    result = await asyncio.wait_for(asyncio.shield(example_task), timeout=timeout_s)
                else:
                    result = await example_task
            except asyncio.TimeoutError:
                if example_task is not None and not example_task.done():
                    example_task.cancel()
                result = _timeout_row(example, idx, benchmark_run_id, timeout_s)
            except Exception as exc:
                if example_task is not None and not example_task.done():
                    example_task.cancel()
                result = _benchmark_error_row(example, idx, benchmark_run_id, exc)

        results_by_index[idx] = result
        pbar.update(1)

    try:
        await asyncio.gather(*[_worker(i) for i in retry_indices])
    finally:
        pbar.close()

    retry_wall = time.perf_counter() - eval_started
    return results_by_index, retry_indices, retry_wall, prewarm_s


async def _run_async(args: argparse.Namespace) -> None:
    source_path = Path(args.source).resolve()
    out_path = Path(args.output).resolve() if args.output else source_path.with_stem(f"{source_path.stem}_merged_balance")

    with source_path.open("r", encoding="utf-8") as f:
        source = json.load(f)

    bird_root = Path(args.bird_root or source.get("dataset", {}).get("bird_root") or source.get("bird_root") or "databases/bird")
    split = args.split or source.get("split") or source.get("dataset", {}).get("split") or "dev"

    max_examples = (source.get("dataset") or {}).get("max_examples")
    examples, db_dir, schema_root, json_path = load_bird_examples(bird_root, split=split)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    expected_total = len(examples)
    pred_by_idx = _predictions_by_index(source["predictions"], expected_total)
    retry_indices = [i for i in range(expected_total) if _is_openrouter_balance_error(pred_by_idx[i].get("error_message"))]

    print(f"Source: {source_path}", file=sys.stderr)
    print(f"OpenRouter balance (402) rows to retry: {len(retry_indices)}", file=sys.stderr)

    prewarm_extra = 0.0
    if retry_indices:
        merged, retried, retry_wall, prewarm_extra = await _retry_balance_errors_async(
            source=source,
            bird_root=bird_root,
            split=split,
            concurrency=args.concurrency,
            prewarm=args.prewarm,
            example_timeout_seconds=(
                float(args.example_timeout_seconds)
                if args.example_timeout_seconds is not None
                else None
            ),
        )
    else:
        merged = {
            i: _internal_row_from_saved(pred_by_idx[i], example=examples[i], example_index=i)
            for i in range(expected_total)
        }
        retried = []
        retry_wall = 0.0

    raw_ordered = [merged[i] for i in range(expected_total)]
    metrics, metrics_extra = _metrics_over_full_split(
        expected_total=expected_total,
        results_by_index=merged,
    )
    cost = _aggregate_cost_for_summary(raw_ordered, expected_total)

    src_tim = source.get("timings") or {}
    source_eval = float(src_tim.get("eval_time_s") or 0.0)
    source_prewarm = float(src_tim.get("prewarm_time_s") or 0.0)
    combined_eval = source_eval + retry_wall
    combined_prewarm = source_prewarm + (prewarm_extra if retry_indices else 0.0)

    dataset = _dataset_identity(
        bird_root=bird_root,
        json_path=json_path,
        num_examples=expected_total,
        max_examples=max_examples if max_examples is None else int(max_examples),
        split=split,
    )

    summary = {
        "benchmark_run_id": f"bird-{split}-merged-{uuid4()}",
        "schema_root": str(schema_root),
        "db_dir": str(db_dir),
        "prewarm_time_s": round(combined_prewarm, 4),
        "eval_time_s": round(combined_eval, 4),
        "avg_time_per_example_s": round((combined_eval / expected_total) if expected_total else 0.0, 4),
        **cost,
    }

    payload = _build_payload(
        bird_root=bird_root,
        dataset=dataset,
        smoke=bool(source.get("smoke", False)),
        max_examples=max_examples if max_examples is None else int(max_examples),
        prewarm=bool(source.get("prewarm", False)),
        resume_from=None,
        retry_errors=False,
        metrics=metrics,
        metrics_extra=metrics_extra,
        predictions=[_strip_internal_fields(merged[i]) for i in range(expected_total)],
        summary=summary,
        status="completed",
    )

    payload["balance_retry_merge"] = {
        "source_file": str(source_path),
        "source_benchmark_run_id": source.get("benchmark_run_id"),
        "retried_indices": retried,
        "retried_count": len(retried),
        "retry_eval_wall_s": round(retry_wall, 4),
        "retry_prewarm_s": round(prewarm_extra, 4) if retry_indices else 0.0,
        "source_timings": src_tim,
    }

    _write_json_atomic(out_path, payload)
    flush_langfuse()

    print("Merged BIRD results (full split metrics)", file=sys.stderr)
    print(f"  EX: {metrics.execution_accuracy:.4f}", file=sys.stderr)
    print(f"  EM: {metrics.exact_match:.4f}", file=sys.stderr)
    print(f"  R-VES: {metrics.r_ves:.4f}", file=sys.stderr)
    print(f"  errors (rows w/ error_message): {metrics.errors}", file=sys.stderr)
    print(f"  Output: {out_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge BIRD results after re-running 402/balance failures")
    ap.add_argument("--source", type=str, required=True, help="Input bird_results JSON")
    ap.add_argument("--output", type=str, default=None, help="Output path (default: <source_stem>_merged_balance.json)")
    ap.add_argument("--bird-root", type=str, default=None)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--prewarm", action="store_true", help="Prewarm selector cache for DB ids in retry set only")
    ap.add_argument(
        "--example-timeout-seconds",
        type=float,
        default=None,
        help="Per-example timeout (default: same as run_bird)",
    )
    args = ap.parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
