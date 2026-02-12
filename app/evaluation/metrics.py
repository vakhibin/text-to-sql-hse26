import re
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

from app.core.db import SQLiteDatabase


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


def execution_accuracy(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
) -> Tuple[bool, Optional[str], Optional[List], Optional[List]]:
    """
    Check if predicted SQL produces the same result as gold SQL.

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
        with SQLiteDatabase(db_path) as db:
            # Execute predicted SQL
            try:
                predicted_result = db.execute_query(predicted_sql)
            except Exception as e:
                error = f"Predicted SQL error: {str(e)}"
                return False, error, None, None

            # Execute gold SQL
            try:
                gold_result = db.execute_query(gold_sql)
            except Exception as e:
                error = f"Gold SQL error: {str(e)}"
                return False, error, predicted_result, None

            # Compare results (order-independent)
            match = compare_results(predicted_result, gold_result)
            return match, None, predicted_result, gold_result

    except Exception as e:
        return False, f"Database error: {str(e)}", None, None


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


def evaluate_single(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
) -> EvaluationResult:
    """
    Evaluate a single prediction.

    Args:
        predicted_sql: Predicted SQL query
        gold_sql: Gold SQL query
        db_path: Path to SQLite database

    Returns:
        EvaluationResult with all metrics
    """
    # Check exact match
    em = exact_match(predicted_sql, gold_sql)

    # Check execution accuracy
    ex, error, pred_result, gold_result = execution_accuracy(
        predicted_sql, gold_sql, db_path
    )

    # Check if SQL is valid (no execution error on predicted)
    valid = error is None or "Gold SQL error" in (error or "")

    return EvaluationResult(
        execution_accuracy=ex,
        exact_match=em,
        valid_sql=valid,
        execution_error=error,
        predicted_result=pred_result,
        gold_result=gold_result,
    )


def evaluate_dataset(
    predictions: List[dict],
    spider_dir: str,
) -> DatasetEvaluationResult:
    """
    Evaluate predictions on a dataset.

    Args:
        predictions: List of dicts with keys:
            - predicted_sql: str
            - gold_sql: str
            - db_id: str
            - question: str (optional)
        spider_dir: Path to Spider dataset directory

    Returns:
        DatasetEvaluationResult with aggregated metrics
    """
    from pathlib import Path

    spider_dir = Path(spider_dir)
    total = len(predictions)
    ex_correct = 0
    em_correct = 0
    valid_count = 0
    errors = []

    for i, pred in enumerate(predictions):
        db_path = spider_dir / "database" / pred["db_id"] / f"{pred['db_id']}.sqlite"

        result = evaluate_single(
            predicted_sql=pred["predicted_sql"],
            gold_sql=pred["gold_sql"],
            db_path=str(db_path),
        )

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


if __name__ == "__main__":
    # Example usage
    result = evaluate_single(
        predicted_sql="SELECT COUNT(*) FROM singer;",
        gold_sql="SELECT count(*) FROM singer",
        db_path="databases/spider/database/concert_singer/concert_singer.sqlite",
    )

    print(f"Execution Accuracy: {result.execution_accuracy}")
    print(f"Exact Match: {result.exact_match}")
    print(f"Valid SQL: {result.valid_sql}")
    if result.execution_error:
        print(f"Error: {result.execution_error}")
