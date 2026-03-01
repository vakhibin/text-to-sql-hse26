"""Benchmark metrics placeholders."""

from dataclasses import dataclass


@dataclass
class BenchmarkMetrics:
    execution_accuracy: float = 0.0
    exact_match: float = 0.0
    r_ves: float = 0.0
    success_rate: float = 0.0

