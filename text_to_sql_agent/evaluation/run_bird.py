"""Run BIRD benchmark with optional dataset download and smoke mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from text_to_sql_agent.config import settings
from text_to_sql_agent.evaluation.metrics import BenchmarkMetrics, exact_match
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state
from text_to_sql_agent.tools.sql_executor import execute_sql


@dataclass
class BirdExample:
    db_id: str
    question: str
    query: str
    evidence: str | None = None
    difficulty: str | None = None
    question_id: int | None = None


def _resolve_dataset_root(bird_root: Path, split: str) -> Path:
    """Resolve BIRD parent folder containing `dev/` + `dev_databases/` (or `train/` + `train_databases/`)."""
    # If user passes `.../bird/dev`, normalize to parent `.../bird`.
    if bird_root.name in ("dev", "train"):
        bird_root = bird_root.parent

    if split == "dev":
        if (bird_root / "dev" / "dev.json").exists() and (bird_root / "dev_databases").exists():
            return bird_root.resolve()
    elif split == "train":
        if (bird_root / "train" / "train.json").exists() and (bird_root / "train_databases").exists():
            return bird_root.resolve()

    raise FileNotFoundError(
        f"BIRD {split} split not found under {bird_root}. "
        f"Expected {bird_root}/dev/{{dev.json,dev_tables.json}} and dev_databases/ (or train equivalents)."
    )


def _is_bird_ready(dataset_root: Path, split: str) -> bool:
    if split == "dev":
        return (
            (dataset_root / "dev" / "dev.json").exists()
            and (dataset_root / "dev" / "dev_tables.json").exists()
            and (dataset_root / "dev_databases").is_dir()
        )
    if split == "train":
        return (
            (dataset_root / "train" / "train.json").exists()
            and (dataset_root / "train" / "train_tables.json").exists()
            and (dataset_root / "train_databases").is_dir()
        )
    return False


async def _bird_download(bird_parent: Path, split: str) -> None:
    from utils.downloaders.bird_downloader import BirdDatasetDownloader

    bird_parent.mkdir(parents=True, exist_ok=True)
    downloader = BirdDatasetDownloader(str(bird_parent))
    await downloader.download_and_extract(split=split, force=False)


def _infer_bird_download_parent(bird_root: Path) -> Path:
    """Folder where BirdDatasetDownloader creates dev/ and train/ (see utils/downloaders/bird_downloader)."""
    if bird_root.name in ("dev", "train"):
        return bird_root.parent
    return bird_root


def ensure_bird_dataset(bird_root: Path, split: str, allow_download: bool) -> Path:
    """Ensure BIRD files exist; optionally download dev/train zip into databases/bird."""
    dataset_root: Path | None = None
    try:
        dataset_root = _resolve_dataset_root(bird_root, split)
    except FileNotFoundError:
        dataset_root = None

    if dataset_root is not None and _is_bird_ready(dataset_root, split):
        return dataset_root

    if not allow_download:
        raise FileNotFoundError(
            f"BIRD {split} dataset not found under {bird_root}. "
            "Use --download or place dev.json / train.json with tables and sqlite under BIRD_ROOT."
        )

    parent = _infer_bird_download_parent(bird_root)
    asyncio.run(_bird_download(parent, split))

    dataset_root = _resolve_dataset_root(bird_root, split)
    if not _is_bird_ready(dataset_root, split):
        raise FileNotFoundError(
            f"BIRD {split} still incomplete after download. Expected data under {dataset_root}."
        )
    return dataset_root


def load_bird_examples(dataset_root: Path, split: str) -> list[BirdExample]:
    split_file = dataset_root / split / (f"{split}.json")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[BirdExample] = []
    for item in data:
        sql = item.get("SQL") or item.get("query")
        if sql is None:
            continue
        ev = item.get("evidence")
        out.append(
            BirdExample(
                db_id=item["db_id"],
                question=item["question"],
                query=sql,
                evidence=ev if ev else None,
                difficulty=item.get("difficulty"),
                question_id=item.get("question_id"),
            )
        )
    return out


def _bird_sqlite_path(dataset_root: Path, split: str, db_id: str) -> Path:
    sub = "dev_databases" if split == "dev" else "train_databases"
    return dataset_root / sub / db_id / f"{db_id}.sqlite"


async def _evaluate_one(
    graph: Any,
    example: BirdExample,
    dataset_root: Path,
    split: str,
) -> dict[str, Any]:
    state = make_initial_state(
        question=example.question,
        db_id=example.db_id,
        evidence=example.evidence,
        schema_root=str(dataset_root),
        schema_layout="bird",
    )
    result = await graph.ainvoke(state)
    predicted_sql = (result.get("final_sql") or result.get("best_sql") or "").strip()

    db_path = _bird_sqlite_path(dataset_root, split, example.db_id)
    pred_exec = await execute_sql(str(db_path), predicted_sql) if predicted_sql else None
    gold_exec = await execute_sql(str(db_path), example.query)

    execution_match = (
        pred_exec is not None
        and pred_exec.success
        and gold_exec.success
        and (pred_exec.rows or []) == (gold_exec.rows or [])
    )
    return {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "difficulty": example.difficulty,
        "question": example.question,
        "predicted_sql": predicted_sql,
        "gold_sql": example.query,
        "execution_match": bool(execution_match),
        "exact_match": exact_match(predicted_sql, example.query),
        "error_message": result.get("error_message"),
        "warnings": result.get("warnings", []),
    }


async def run_bird_benchmark(
    *,
    dataset_root: Path,
    split: str,
    max_examples: int | None,
    concurrency: int = 1,
) -> tuple[BenchmarkMetrics, list[dict[str, Any]]]:
    graph = build_graph()
    examples = load_bird_examples(dataset_root=dataset_root, split=split)
    if max_examples is not None:
        examples = examples[:max_examples]

    semaphore = asyncio.Semaphore(concurrency)
    results_by_index: dict[int, dict[str, Any]] = {}
    exec_hits = 0
    em_hits = 0
    err_count = 0
    lock = asyncio.Lock()

    pbar = tqdm(total=len(examples), desc="BIRD eval", unit="q", file=sys.stderr)

    async def _worker(idx: int, example: BirdExample) -> None:
        nonlocal exec_hits, em_hits, err_count
        async with semaphore:
            result = await _evaluate_one(graph, example, dataset_root, split)

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
                    f"-> {str(result['error_message'])[:120]}",
                    file=sys.stderr,
                )
            done = len(results_by_index)
            pbar.set_postfix(EX=f"{exec_hits/done:.0%}", EM=f"{em_hits/done:.0%}", err=err_count)
            pbar.update(1)

    await asyncio.gather(*[_worker(i, ex) for i, ex in enumerate(examples)])
    pbar.close()

    predictions = [results_by_index[i] for i in range(len(examples))]
    total = len(predictions)
    valid = sum(1 for row in predictions if bool(row["predicted_sql"]))
    errors = err_count

    metrics = BenchmarkMetrics(
        execution_accuracy=(exec_hits / total) if total else 0.0,
        exact_match=(em_hits / total) if total else 0.0,
        total=total,
        valid_predictions=valid,
        errors=errors,
    )
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIRD benchmark")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run on small subset for quick checks")
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Download BIRD dev/train zip into databases/bird if missing",
    )
    parser.add_argument("--output", type=str, default="outputs/bird_results.json")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel evaluation workers")
    parser.add_argument(
        "--bird-root",
        type=str,
        default=settings.bird_root,
        help="Path to BIRD split folder (e.g. databases/bird/dev) or parent databases/bird",
    )
    args = parser.parse_args()

    bird_root = Path(args.bird_root)
    dataset_root = ensure_bird_dataset(bird_root=bird_root, split=args.split, allow_download=args.download)

    max_examples = args.max_examples
    if args.smoke:
        max_examples = args.smoke_size

    metrics, predictions = asyncio.run(
        run_bird_benchmark(
            dataset_root=dataset_root,
            split=args.split,
            max_examples=max_examples,
            concurrency=args.concurrency,
        )
    )

    base = Path(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base.with_stem(f"{base.stem}_{stamp}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "dataset_root": str(dataset_root),
        "bird_root_arg": str(bird_root),
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

    print("BIRD evaluation completed")
    print(f"  Split: {args.split}")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Total: {metrics.total}")
    print(f"  EX: {metrics.execution_accuracy:.4f}")
    print(f"  EM: {metrics.exact_match:.4f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
