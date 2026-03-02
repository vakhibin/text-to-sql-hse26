"""Few-shot example sampling utilities."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from text_to_sql_agent.config import settings


def _read_train_examples_sync(spider_root: str | Path, max_pool_size: int) -> list[dict[str, str]]:
    root = Path(spider_root)
    train_path = root / "train_spider.json"
    if not train_path.exists():
        return []
    with train_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pool: list[dict[str, str]] = []
    for item in raw:
        q = str(item.get("question", "")).strip()
        sql = str(item.get("query", "")).strip()
        db_id = str(item.get("db_id", "")).strip()
        if not q or not sql:
            continue
        pool.append({"question": q, "sql": sql, "db_id": db_id})
        if len(pool) >= max_pool_size:
            break
    return pool


async def load_few_shot_pool(
    *,
    spider_root: str | Path | None = None,
    max_pool_size: int | None = None,
) -> list[dict[str, str]]:
    """Load candidate few-shot examples from Spider train split."""
    root = spider_root or settings.spider_root
    pool_limit = max_pool_size or settings.few_shot_max_pool_size
    return await asyncio.to_thread(_read_train_examples_sync, root, pool_limit)


def sample_examples_for_candidate(
    *,
    pool: list[dict[str, str]],
    candidate_index: int,
    k: int,
    seed: int,
    target_db_id: str | None = None,
) -> list[dict[str, str]]:
    """Sample deterministic few-shot subset for one candidate."""
    if not pool or k <= 0:
        return []

    same_db = [item for item in pool if target_db_id and item.get("db_id") == target_db_id]
    source = same_db if len(same_db) >= k else pool

    rng = random.Random(seed + candidate_index)
    if len(source) <= k:
        return source
    return rng.sample(source, k=k)

