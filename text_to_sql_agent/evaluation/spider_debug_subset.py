"""Helpers for stable Spider debug subsets and offline hardness tiers."""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from text_to_sql_agent.evaluation.spider_split_io import load_spider_split_records

DifficultyTier = Literal["simple", "moderate", "complex"]
OutcomeHint = Literal["failure", "execution_only", "exact_match", "unknown"]

_SQL_KEYS = {"select", "from", "where", "groupBy", "having", "orderBy", "limit"}


@dataclass(frozen=True)
class SpiderSubsetRecord:
    example_index: int
    db_id: str
    question: str
    gold_sql: str
    difficulty: DifficultyTier
    outcome_hint: OutcomeHint = "unknown"


def load_spider_split_with_sql(spider_root: Path, split: str) -> list[dict[str, Any]]:
    return load_spider_split_records(spider_root, split)


def _is_sql_dict(node: Any) -> bool:
    return isinstance(node, dict) and _SQL_KEYS.issubset(node.keys())


def _iter_nested_sql_nodes(node: Any) -> list[dict[str, Any]]:
    nested: list[dict[str, Any]] = []
    if isinstance(node, dict):
        for value in node.values():
            if _is_sql_dict(value):
                nested.append(value)
                nested.extend(_iter_nested_sql_nodes(value))
            else:
                nested.extend(_iter_nested_sql_nodes(value))
    elif isinstance(node, list):
        for item in node:
            if _is_sql_dict(item):
                nested.append(item)
                nested.extend(_iter_nested_sql_nodes(item))
            else:
                nested.extend(_iter_nested_sql_nodes(item))
    return nested


def _count_condition_units(conditions: list[Any] | None) -> int:
    if not conditions:
        return 0
    return sum(1 for item in conditions if isinstance(item, list))


def _count_select_aggregations(sql: dict[str, Any]) -> int:
    select_exprs = (sql.get("select") or [False, []])[1]
    count = 0
    for expr in select_exprs:
        if isinstance(expr, list) and expr:
            agg_id = expr[0]
            if isinstance(agg_id, int) and agg_id > 0:
                count += 1
    return count


def classify_spider_sql_difficulty(sql: dict[str, Any]) -> DifficultyTier:
    table_units = ((sql.get("from") or {}).get("table_units") or [])
    joins = max(len(table_units) - 1, 0)
    where_conditions = _count_condition_units(sql.get("where"))
    from_conditions = _count_condition_units((sql.get("from") or {}).get("conds"))
    having_conditions = _count_condition_units(sql.get("having"))
    total_conditions = where_conditions + from_conditions + having_conditions
    has_group = bool(sql.get("groupBy"))
    has_order = bool(sql.get("orderBy"))
    has_limit = sql.get("limit") is not None
    has_having = bool(sql.get("having"))
    select_aggregations = _count_select_aggregations(sql)
    has_set_ops = any(sql.get(key) is not None for key in ("intersect", "union", "except"))
    nested_sql_count = len(_iter_nested_sql_nodes(sql))

    if (
        has_set_ops
        or nested_sql_count > 0
        or joins >= 2
        or total_conditions >= 4
        or (joins >= 1 and has_group and (has_having or select_aggregations > 0))
    ):
        return "complex"

    if (
        joins >= 1
        or has_group
        or has_order
        or has_limit
        or has_having
        or select_aggregations > 0
        or total_conditions >= 2
    ):
        return "moderate"

    return "simple"


def build_spider_subset_records(
    *,
    spider_root: Path,
    split: str,
    results_payload_path: Path | None = None,
) -> list[SpiderSubsetRecord]:
    result_lookup: dict[tuple[str, str, str], OutcomeHint] = {}
    if results_payload_path is not None and results_payload_path.exists():
        with results_payload_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("predictions", []):
            key = (str(row.get("db_id", "")), str(row.get("question", "")), str(row.get("gold_sql", "")))
            execution_match = bool(row.get("execution_match"))
            exact_match = bool(row.get("exact_match"))
            if row.get("error_message") or not execution_match:
                outcome_hint: OutcomeHint = "failure"
            elif exact_match:
                outcome_hint = "exact_match"
            else:
                outcome_hint = "execution_only"
            result_lookup[key] = outcome_hint

    entries = load_spider_split_with_sql(spider_root=spider_root, split=split)
    records: list[SpiderSubsetRecord] = []
    for idx, entry in enumerate(entries):
        gold_sql = str(entry.get("query", ""))
        difficulty = classify_spider_sql_difficulty(entry.get("sql") or {})
        key = (str(entry.get("db_id", "")), str(entry.get("question", "")), gold_sql)
        records.append(
            SpiderSubsetRecord(
                example_index=idx,
                db_id=str(entry.get("db_id", "")),
                question=str(entry.get("question", "")),
                gold_sql=gold_sql,
                difficulty=difficulty,
                outcome_hint=result_lookup.get(key, "unknown"),
            )
        )
    return records


def _round_robin_sample(
    records: list[SpiderSubsetRecord],
    target: int,
    rng: random.Random,
) -> list[SpiderSubsetRecord]:
    if target <= 0 or not records:
        return []

    grouped: dict[str, deque[SpiderSubsetRecord]] = defaultdict(deque)
    db_ids = sorted({record.db_id for record in records})
    rng.shuffle(db_ids)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    for record in shuffled_records:
        grouped[record.db_id].append(record)

    selected: list[SpiderSubsetRecord] = []
    while len(selected) < target and any(grouped.values()):
        for db_id in db_ids:
            queue = grouped[db_id]
            if queue:
                selected.append(queue.popleft())
                if len(selected) == target:
                    break
    return selected


def select_balanced_debug_subset(
    records: list[SpiderSubsetRecord],
    *,
    per_tier_target: int,
    seed: int,
) -> list[SpiderSubsetRecord]:
    rng = random.Random(seed)
    selected: list[SpiderSubsetRecord] = []

    for difficulty in ("simple", "moderate", "complex"):
        tier_records = [record for record in records if record.difficulty == difficulty]
        if len(tier_records) < per_tier_target:
            raise ValueError(
                f"Not enough Spider examples for tier '{difficulty}': "
                f"need {per_tier_target}, have {len(tier_records)}"
            )

        failure = [record for record in tier_records if record.outcome_hint == "failure"]
        execution_only = [record for record in tier_records if record.outcome_hint == "execution_only"]
        exact_match = [record for record in tier_records if record.outcome_hint == "exact_match"]
        unknown = [record for record in tier_records if record.outcome_hint == "unknown"]

        tier_selection: list[SpiderSubsetRecord] = []
        if any(bucket for bucket in (failure, execution_only, exact_match)):
            targets = {
                "failure": int(per_tier_target * 0.4),
                "execution_only": int(per_tier_target * 0.3),
                "exact_match": per_tier_target - int(per_tier_target * 0.4) - int(per_tier_target * 0.3),
            }
            for outcome_name, bucket in (
                ("failure", failure),
                ("execution_only", execution_only),
                ("exact_match", exact_match),
            ):
                tier_selection.extend(_round_robin_sample(bucket, targets[outcome_name], rng))

            used = {record.example_index for record in tier_selection}
            remaining_pool = [record for record in tier_records if record.example_index not in used]
            tier_selection.extend(_round_robin_sample(remaining_pool, per_tier_target - len(tier_selection), rng))
        else:
            tier_selection = _round_robin_sample(tier_records, per_tier_target, rng)

        if len(tier_selection) < per_tier_target:
            used = {record.example_index for record in tier_selection}
            leftovers = [record for record in tier_records if record.example_index not in used]
            tier_selection.extend(_round_robin_sample(leftovers, per_tier_target - len(tier_selection), rng))

        selected.extend(sorted(tier_selection[:per_tier_target], key=lambda record: record.example_index))

    return sorted(selected, key=lambda record: record.example_index)


def load_subset_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid subset manifest: {manifest_path}")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Subset manifest must contain a non-empty 'items' list: {manifest_path}")
    return payload
