"""Build Spider generator fine-tuning dataset using real agent prompt-building path.

Important: prompts are assembled through existing agent functions, not manual reconstruction:
`run_selector -> run_value_linker -> run_query_sketcher -> build_generator_prompt`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Ensure repository root is importable when script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_to_sql_agent.agents.query_sketcher import run_query_sketcher
from text_to_sql_agent.agents.selector import run_selector
from text_to_sql_agent.agents.value_linker import run_value_linker
from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.state import make_initial_state
from text_to_sql_agent.prompts.generator import build_generator_prompt
from text_to_sql_agent.tools.few_shot import load_few_shot_pool, retrieve_examples_for_candidate
from text_to_sql_agent.tools.value_linker import format_column_hints, format_value_hints


@dataclass
class TrainExample:
    db_id: str
    question: str
    query: str
    evidence: str | None
    source: str
    row_id: int


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _load_train_examples(spider_root: Path) -> list[TrainExample]:
    files = [
        ("train_spider", spider_root / "train_spider.json"),
        ("train_others", spider_root / "train_others.json"),
    ]
    out: list[TrainExample] = []
    for source, path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        rows = _load_json(path)
        for i, row in enumerate(rows):
            question = str(row.get("question", "")).strip()
            query = str(row.get("query", "")).strip()
            db_id = str(row.get("db_id", "")).strip()
            evidence_raw = row.get("evidence")
            evidence = str(evidence_raw).strip() if evidence_raw is not None else None
            if not question or not query or not db_id:
                continue
            out.append(
                TrainExample(
                    db_id=db_id,
                    question=question,
                    query=query,
                    evidence=evidence if evidence else None,
                    source=source,
                    row_id=i,
                )
            )
    return out


async def _build_prompt_via_agents(
    *,
    ex: TrainExample,
    spider_root: Path,
    few_shot_pool: list[dict[str, str]],
    few_shot_k: int,
    candidate_index: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    state = make_initial_state(
        question=ex.question,
        db_id=ex.db_id,
        evidence=ex.evidence,
        schema_root=str(spider_root),
    )

    selector_out = await run_selector(state)
    state = {**state, **selector_out}
    linker_out = await run_value_linker(state)
    state = {**state, **linker_out}
    sketcher_out = await run_query_sketcher(state)
    state = {**state, **sketcher_out}

    few_shot_examples = await retrieve_examples_for_candidate(
        pool=few_shot_pool,
        question=state["question"],
        candidate_index=candidate_index,
        k=few_shot_k,
        seed=seed,
        target_db_id=state.get("db_id"),
    )

    prompt = build_generator_prompt(
        question=state["question"],
        evidence=state.get("evidence"),
        filtered_schema=state.get("filtered_schema", ""),
        query_sketch_text=state.get("query_sketch_text", ""),
        few_shot_examples=few_shot_examples,
        value_hints_text=format_value_hints(state.get("value_hints", [])),
        column_hints_text=format_column_hints(state.get("column_hints", [])),
    )
    meta = {
        "warnings": state.get("warnings", []),
        "stage_status": state.get("stage_status", {}),
        "stage_timings": state.get("stage_timings", {}),
        "llm_usage_count": len(state.get("llm_usage", [])),
        "total_cost_usd": float(state.get("total_cost_usd", 0.0)),
    }
    return prompt, meta


def _normalize_sql_completion(sql: str) -> str:
    s = " ".join(sql.strip().split())
    if s and not s.endswith(";"):
        s += ";"
    return s


async def build_dataset(
    *,
    spider_root: Path,
    output_path: Path,
    output_format: str,
    few_shot_k: int,
    candidate_index: int,
    seed: int,
    max_examples: int | None,
    concurrency: int,
    max_errors: int,
    max_error_rate: float,
    error_sleep_seconds: float,
) -> None:
    examples = _load_train_examples(spider_root)
    if max_examples is not None:
        examples = examples[:max_examples]
    if not examples:
        raise RuntimeError("No examples loaded from train_spider/train_others")

    # Align few-shot source with current agent implementation.
    pool_root = spider_root if spider_root.exists() else Path(settings.spider_root)
    few_shot_pool = await load_few_shot_pool(
        spider_root=pool_root,
        max_pool_size=settings.few_shot_max_pool_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    total = len(examples)
    success_count = 0
    error_count = 0
    write_lock = asyncio.Lock()
    counters_lock = asyncio.Lock()
    queue: asyncio.Queue[tuple[int, TrainExample] | None] = asyncio.Queue()
    stop_event = asyncio.Event()

    pbar = tqdm(total=total, desc="Building finetune dataset", unit="rows")

    async def _worker(worker_id: int) -> None:
        nonlocal success_count, error_count
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            idx, ex = item
            if stop_event.is_set():
                queue.task_done()
                pbar.update(1)
                continue
            try:
                prompt, agent_meta = await _build_prompt_via_agents(
                    ex=ex,
                    spider_root=spider_root,
                    few_shot_pool=few_shot_pool,
                    few_shot_k=few_shot_k,
                    candidate_index=candidate_index,
                    seed=seed,
                )
                completion = _normalize_sql_completion(ex.query)
                if output_format == "chat":
                    row = {
                        "messages": [
                            {"role": "system", "content": "Output only SQL."},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion},
                        ],
                        "metadata": {
                            "db_id": ex.db_id,
                            "source": ex.source,
                            "row_id": ex.row_id,
                            "agent_meta": agent_meta,
                        },
                    }
                else:
                    row = {
                        "prompt": prompt,
                        "completion": completion,
                        "db_id": ex.db_id,
                        "source": ex.source,
                        "row_id": ex.row_id,
                        "question": ex.question,
                        "evidence": ex.evidence,
                        "few_shot_k": few_shot_k,
                        "candidate_index": candidate_index,
                        "agent_meta": agent_meta,
                    }
                async with write_lock:
                    with temp_output.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                async with counters_lock:
                    success_count += 1
            except Exception as exc:
                async with counters_lock:
                    error_count += 1
                    current_errors = error_count
                    current_done = success_count + error_count
                    current_error_rate = (current_errors / current_done) if current_done else 0.0
                tqdm.write(
                    f"[worker {worker_id}] ERROR idx={idx} db_id={ex.db_id}: {exc}",
                )
                # Small backoff after each error to reduce cascading failures.
                await asyncio.sleep(error_sleep_seconds)
                if current_errors >= max_errors or current_error_rate > max_error_rate:
                    stop_event.set()
            finally:
                pbar.update(1)
                queue.task_done()

    for idx, ex in enumerate(examples):
        queue.put_nowait((idx, ex))
    worker_count = max(1, concurrency)
    for _ in range(worker_count):
        queue.put_nowait(None)

    workers = [asyncio.create_task(_worker(i)) for i in range(worker_count)]
    await queue.join()
    await asyncio.gather(*workers)
    pbar.close()

    done = success_count + error_count
    error_rate = (error_count / done) if done else 0.0
    if stop_event.is_set() or error_count >= max_errors or error_rate > max_error_rate:
        if temp_output.exists():
            temp_output.unlink()
        raise RuntimeError(
            f"Build failed due to too many errors: errors={error_count}, done={done}, "
            f"error_rate={error_rate:.2%}, limits: max_errors={max_errors}, max_error_rate={max_error_rate:.2%}"
        )

    temp_output.replace(output_path)
    print(f"Saved {success_count} rows to: {output_path}")
    print(f"Format: {output_format}")
    print(f"Few-shot per row: {few_shot_k}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build generator fine-tuning dataset using real agent prompt path (Spider)."
    )
    parser.add_argument("--spider-root", type=str, default="databases/spider")
    parser.add_argument("--output", type=str, default="outputs/spider_generator_finetune.jsonl")
    parser.add_argument("--output-format", choices=["jsonl", "chat"], default="jsonl")
    parser.add_argument("--few-shot-k", type=int, default=2, help="Few-shot count in prompt (agent retrieval path).")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="Candidate index used by retrieve_examples_for_candidate (generator parity).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers.")
    parser.add_argument("--max-errors", type=int, default=50, help="Fail build when errors reach this count.")
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="Fail build when errors / processed exceeds this value (e.g. 0.05 = 5%%).",
    )
    parser.add_argument(
        "--error-sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep duration after each error to reduce cascading failures.",
    )
    args = parser.parse_args()

    asyncio.run(
        build_dataset(
            spider_root=Path(args.spider_root),
            output_path=Path(args.output),
            output_format=args.output_format,
            few_shot_k=args.few_shot_k,
            candidate_index=args.candidate_index,
            seed=args.seed,
            max_examples=args.max_examples,
            concurrency=args.concurrency,
            max_errors=args.max_errors,
            max_error_rate=args.max_error_rate,
            error_sleep_seconds=args.error_sleep_seconds,
        )
    )


if __name__ == "__main__":
    main()

