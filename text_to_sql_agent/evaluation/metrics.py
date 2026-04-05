"""Benchmark metrics helpers.

Execution accuracy uses the official Spider result_eq algorithm:
- Column-order permutations are explored automatically.
- Row order matters only when the gold SQL contains ORDER BY.
- Bag (multiset) semantics are used otherwise.

Reference: https://github.com/taoyds/test-suite-sql-eval/blob/master/exec_eval.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass
class BenchmarkMetrics:
    execution_accuracy: float = 0.0
    exact_match: float = 0.0
    r_ves: float = 0.0
    success_rate: float = 0.0
    total: int = 0
    valid_predictions: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# SQL string helpers
# ---------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    compact = " ".join((sql or "").strip().split())
    if compact.endswith(";"):
        compact = compact[:-1]
    return compact.lower()


def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    return normalize_sql(predicted_sql) == normalize_sql(gold_sql)


# ---------------------------------------------------------------------------
# Official Spider execution-accuracy comparison (result_eq)
# ---------------------------------------------------------------------------

def _permute_tuple(element: tuple[Any, ...], perm: tuple[int, ...]) -> tuple[Any, ...]:
    return tuple(element[i] for i in perm)


def _unorder_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def _quick_rej(
    result1: list[tuple[Any, ...]],
    result2: list[tuple[Any, ...]],
    order_matters: bool,
) -> bool:
    s1 = [_unorder_row(row) for row in result1]
    s2 = [_unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    return set(s1) == set(s2)


def _multiset_eq(l1: list[Any], l2: list[Any]) -> bool:
    if len(l1) != len(l2):
        return False
    d: dict[Any, int] = defaultdict(int)
    for e in l1:
        d[e] += 1
    for e in l2:
        d[e] -= 1
        if d[e] < 0:
            return False
    return True


def _get_constraint_permutation(
    tab1_sets_by_columns: list[set[Any]],
    result2: list[tuple[Any, ...]],
) -> Any:
    num_cols = len(result2[0])
    perm_constraints: list[set[int]] = [set(range(num_cols)) for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)
    rng = random.Random(42)
    for _ in range(20):
        random_tab2_row = rng.choice(result2)
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].discard(tab2_col)
    return product(*perm_constraints)


def result_eq(
    result1: list[tuple[Any, ...]],
    result2: list[tuple[Any, ...]],
    order_matters: bool,
) -> bool:
    """Compare two SQL result sets using official Spider semantics.

    Tries all valid column permutations.  Row order only matters when
    ``order_matters`` is True (i.e. the gold query has ORDER BY).
    """
    if len(result1) == 0 and len(result2) == 0:
        return True
    if len(result1) != len(result2):
        return False
    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        return False
    if not _quick_rej(result1, result2, order_matters):
        return False

    tab1_sets_by_columns: list[set[Any]] = [
        {row[i] for row in result1} for i in range(num_cols)
    ]
    for perm in _get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [_permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and _multiset_eq(result1, result2_perm):
                return True
    return False


def execution_match(
    pred_rows: list[tuple[Any, ...]] | None,
    gold_rows: list[tuple[Any, ...]] | None,
    gold_sql: str,
) -> bool:
    """Determine execution accuracy using official Spider result_eq."""
    if pred_rows is None or gold_rows is None:
        return False
    order_matters = "order by" in (gold_sql or "").lower()
    return result_eq(
        list(pred_rows),
        list(gold_rows),
        order_matters=order_matters,
    )
