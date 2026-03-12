import re
import asyncio
from typing import List, Optional, Tuple, AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from app.core.db import SQLiteDatabase
from app.core.logger import logger

@dataclass
class EvaluationResult:
    """Result of evaluating a single prediction."""
    execution_accuracy: bool  # Results match
    exact_match: bool  # SQL strings match (normalized)
    valid_sql: bool  # Predicted SQL is valid
    execution_error: Optional[str] = None
    predicted_result: Optional[List] = None
    gold_result: Optional[List] = None


@dataclass
class DatasetEvaluationResult:
    """Aggregated evaluation results for a dataset."""
    total: int
    execution_accuracy: float  # EX
    exact_match: float  # EM
    valid_sql_rate: float
    errors: List[dict]

    def __str__(self) -> str:
        return (
            f"Total: {self.total}\n"
            f"Execution Accuracy (EX): {self.execution_accuracy:.2%}\n"
            f"Exact Match (EM): {self.exact_match:.2%}\n"
            f"Valid SQL Rate: {self.valid_sql_rate:.2%}\n"
            f"Errors: {len(self.errors)}"
        )


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.

    - Lowercase
    - Remove extra whitespace
    - Remove trailing semicolon
    - Normalize quotes
    """
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';')
    sql = sql.replace('"', "'")
    return sql


def tokenize_sql(sql: str) -> List[str]:
    """Tokenize SQL for token-level comparison."""
    sql = normalize_sql(sql)
    # Split on whitespace and punctuation, keeping punctuation
    tokens = re.findall(r"[\w]+|[^\s\w]", sql)
    return tokens


def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    """Check if two SQL queries match exactly (after normalization)."""
    return normalize_sql(predicted_sql) == normalize_sql(gold_sql)


def token_match(predicted_sql: str, gold_sql: str) -> bool:
    """Check if two SQL queries have the same tokens (order-independent for some parts)."""
    pred_tokens = set(tokenize_sql(predicted_sql))
    gold_tokens = set(tokenize_sql(gold_sql))
    return pred_tokens == gold_tokens





def compare_results(predicted: List, gold: List) -> bool:
    """
    Compare two query results for equality.

    Uses set comparison for order-independent matching.
    """
    if predicted is None or gold is None:
        return False

    try:
        # Convert to sets of tuples
        pred_set = set(
            tuple(row) if isinstance(row, (list, tuple)) else (row,)
            for row in predicted
        )
        gold_set = set(
            tuple(row) if isinstance(row, (list, tuple)) else (row,)
            for row in gold
        )
        return pred_set == gold_set
    except (TypeError, ValueError):
        # Fallback for unhashable types
        return predicted == gold


async def execution_accuracy(
        predicted_sql: str,
        gold_sql: str,
        db_path: str,
) -> Tuple[bool, Optional[str], Optional[List], Optional[List]]:
    """
    Check if predicted SQL produces the same result as gold SQL asynchronously.

    Args:
        predicted_sql: Predicted SQL query
        gold_sql: Gold SQL query
        db_path: Path to SQLite database

    Returns:
        Tuple of (match, error_message, predicted_result, gold_result)
    """
    predicted_result = None
    gold_result = None
    error = None

    try:
        async with SQLiteDatabase(db_path) as db:
            # Execute predicted SQL
            try:
                predicted_result = await db.execute_query(predicted_sql)
            except Exception as e:
                error = f"Predicted SQL error: {str(e)}"
                return False, error, None, None

            # Execute gold SQL
            try:
                gold_result = await db.execute_query(gold_sql)
            except Exception as e:
                error = f"Gold SQL error: {str(e)}"
                return False, error, predicted_result, None

            # Compare results (order-independent)
            match = compare_results(predicted_result, gold_result)
            return match, None, predicted_result, gold_result
    except Exception as e:
        return False, f"Database error: {str(e)}", None, None


async def evaluate_single(
        predicted_sql: str,
        gold_sql: str,
        db_path: str,
) -> EvaluationResult:
    """
    Evaluate a single prediction asynchronously.

    Args:
        predicted_sql: Predicted SQL query
        gold_sql: Gold SQL query
        db_path: Path to SQLite database

    Returns:
        EvaluationResult with all metrics
    """
    em = exact_match(predicted_sql, gold_sql)

    ex, error, pred_result, gold_result = await execution_accuracy(
        predicted_sql, gold_sql, db_path
    )
    valid = error is None or "Gold SQL error" in (error or "")

    return EvaluationResult(
        execution_accuracy=ex,
        exact_match=em,
        valid_sql=valid,
        execution_error=error,
        predicted_result=pred_result,
        gold_result=gold_result,
    )


def _get_db_path(base_dir: Path, pred: dict, dataset: str, split: str) -> Path:
    """
    Resolve path to SQLite database depending on dataset type.

    - spider: base_dir / "database" / db_id / db_id.sqlite
    - bird:   base_dir / f"{split}_databases" / db_id / db_id.sqlite
    """
    db_id = pred["db_id"]
    if dataset.lower() == "bird":
        return base_dir / f"{split}_databases" / db_id / f"{db_id}.sqlite"
    # default: spider-style layout
    return base_dir / "database" / db_id / f"{db_id}.sqlite"


async def evaluate_dataset(
        predictions: List[dict],
        base_dir: str,
        max_concurrent: int = 10,
        dataset: str = "spider",
        split: str = "dev",
) -> DatasetEvaluationResult:
    """
    Evaluate predictions on a dataset asynchronously with concurrency control.

    Args:
        predictions: List of dicts with keys:
            - predicted_sql: str
            - gold_sql: str
            - db_id: str
            - question: str (optional)
        base_dir:   Path to dataset root directory
        max_concurrent: Maximum number of concurrent evaluations
        dataset:    Dataset type ("spider" or "bird")
        split:      Dataset split (used for BIRD to pick *_{split}_databases)

    Returns:
        DatasetEvaluationResult with aggregated metrics
    """
    base_dir = Path(base_dir)
    total = len(predictions)
    ex_correct = 0
    em_correct = 0
    valid_count = 0
    errors = []

    # Create semaphore to limit concurrent database connections
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_one(i: int, pred: dict) -> tuple:
        async with semaphore:
            db_path = _get_db_path(base_dir, pred, dataset, split)

            result = await evaluate_single(
                predicted_sql=pred["predicted_sql"],
                gold_sql=pred["gold_sql"],
                db_path=str(db_path),
            )

            return i, pred, result

    # Run evaluations concurrently
    tasks = [evaluate_one(i, pred) for i, pred in enumerate(predictions)]

    for task in asyncio.as_completed(tasks):
        i, pred, result = await task

        if result.execution_accuracy:
            ex_correct += 1
        if result.exact_match:
            em_correct += 1
        if result.valid_sql:
            valid_count += 1

        if result.execution_error or not result.execution_accuracy:
            errors.append({
                "index": i,
                "question": pred.get("question", ""),
                "db_id": pred["db_id"],
                "predicted_sql": pred["predicted_sql"],
                "gold_sql": pred["gold_sql"],
                "error": result.execution_error,
                "execution_match": result.execution_accuracy,
            })

    return DatasetEvaluationResult(
        total=total,
        execution_accuracy=ex_correct / total if total > 0 else 0.0,
        exact_match=em_correct / total if total > 0 else 0.0,
        valid_sql_rate=valid_count / total if total > 0 else 0.0,
        errors=errors,
    )


async def evaluate_dataset_iterative(
        predictions: AsyncIterator[dict],
        base_dir: str,
        max_concurrent: int = 10,
        dataset: str = "spider",
        split: str = "dev",
) -> DatasetEvaluationResult:
    """
    Evaluate predictions from an async iterator (memory efficient).

    Args:
        predictions: AsyncIterator of prediction dicts
        base_dir:   Path to dataset root directory
        max_concurrent: Maximum number of concurrent evaluations
        dataset:    Dataset type ("spider" or "bird")
        split:      Dataset split (used for BIRD to pick *_{split}_databases)

    Returns:
        DatasetEvaluationResult with aggregated metrics
    """
    base_dir = Path(base_dir)
    total = 0
    ex_correct = 0
    em_correct = 0
    valid_count = 0
    errors = []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_one(i: int, pred: dict) -> tuple:
        async with semaphore:
            db_path = _get_db_path(base_dir, pred, dataset, split)

            result = await evaluate_single(
                predicted_sql=pred["predicted_sql"],
                gold_sql=pred["gold_sql"],
                db_path=str(db_path),
            )

            return i, pred, result

    pending = set()
    i = 0

    async for pred in predictions:
        task = asyncio.create_task(evaluate_one(i, pred))
        pending.add(task)
        i += 1
        total += 1

        # Limit concurrent tasks
        if len(pending) >= max_concurrent:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                _, pred, result = task.result()

                if result.execution_accuracy:
                    ex_correct += 1
                if result.exact_match:
                    em_correct += 1
                if result.valid_sql:
                    valid_count += 1

                if result.execution_error or not result.execution_accuracy:
                    errors.append({
                        "index": i - len(pending) - 1,
                        "question": pred.get("question", ""),
                        "db_id": pred["db_id"],
                        "predicted_sql": pred["predicted_sql"],
                        "gold_sql": pred["gold_sql"],
                        "error": result.execution_error,
                        "execution_match": result.execution_accuracy,
                    })

    # Process remaining tasks
    if pending:
        done, _ = await asyncio.wait(pending)
        for task in done:
            i, pred, result = task.result()

            if result.execution_accuracy:
                ex_correct += 1
            if result.exact_match:
                em_correct += 1
            if result.valid_sql:
                valid_count += 1

            if result.execution_error or not result.execution_accuracy:
                errors.append({
                    "index": i,
                    "question": pred.get("question", ""),
                    "db_id": pred["db_id"],
                    "predicted_sql": pred["predicted_sql"],
                    "gold_sql": pred["gold_sql"],
                    "error": result.execution_error,
                    "execution_match": result.execution_accuracy,
                })

    return DatasetEvaluationResult(
        total=total,
        execution_accuracy=ex_correct / total if total > 0 else 0.0,
        exact_match=em_correct / total if total > 0 else 0.0,
        valid_sql_rate=valid_count / total if total > 0 else 0.0,
        errors=errors,
    )


async def main():
    """Example usage."""
    result = await evaluate_single(
        predicted_sql="SELECT COUNT(*) FROM singer;",
        gold_sql="SELECT count(*) FROM singer",
        db_path="databases/spider/database/concert_singer/concert_singer.sqlite",
    )

    logger.info(f"Execution Accuracy: {result.execution_accuracy}")
    logger.info(f"Exact Match: {result.exact_match}")
    logger.info(f"Valid SQL: {result.valid_sql}")
    if result.execution_error:
        logger.info(f"Error: {result.execution_error}")

if __name__ == "__main__":
    asyncio.run(main())