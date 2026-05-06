"""Spider split JSON paths and gold-SQL alignment (dev / train / test)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPIDER_SPLIT_JSON_FILES: dict[str, str] = {
    "dev": "dev.json",
    "train": "train_spider.json",
    "test": "test.json",
}


def spider_split_json_path(spider_root: Path, split: str) -> Path:
    name = SPIDER_SPLIT_JSON_FILES.get(split)
    if not name:
        raise ValueError(f"Unknown Spider split: {split!r}")
    return spider_root / name


def load_test_gold_sql_queries(spider_root: Path) -> list[str]:
    path = spider_root / "test_gold.sql"
    if not path.is_file():
        raise FileNotFoundError(f"Missing test gold file: {path}")
    queries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--"):
            continue
        if "\t" in line:
            line = line.split("\t")[-1].strip()
        queries.append(line)
    return queries


def load_spider_split_records(spider_root: Path, split: str) -> list[dict[str, Any]]:
    """Load Spider examples with ``query`` populated (test joins ``test_gold.sql`` by row index)."""
    split_path = spider_split_json_path(spider_root, split)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Spider split must be a list: {split_path}")
    if split != "test":
        return data
    gold_lines = load_test_gold_sql_queries(spider_root)
    if len(gold_lines) != len(data):
        raise ValueError(
            f"test_gold.sql non-empty query lines ({len(gold_lines)}) "
            f"!= test.json rows ({len(data)})"
        )
    merged: list[dict[str, Any]] = []
    for row, gold_sql in zip(data, gold_lines):
        item = dict(row)
        existing = str(item.get("query") or "").strip()
        item["query"] = existing if existing else gold_sql
        merged.append(item)
    return merged
