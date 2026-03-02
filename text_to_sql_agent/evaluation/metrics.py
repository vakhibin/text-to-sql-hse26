"""Benchmark metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkMetrics:
    execution_accuracy: float = 0.0
    exact_match: float = 0.0
    r_ves: float = 0.0
    success_rate: float = 0.0
    total: int = 0
    valid_predictions: int = 0
    errors: int = 0


def normalize_sql(sql: str) -> str:
    compact = " ".join((sql or "").strip().split())
    if compact.endswith(";"):
        compact = compact[:-1]
    return compact.lower()


def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    return normalize_sql(predicted_sql) == normalize_sql(gold_sql)

