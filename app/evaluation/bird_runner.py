import json
import asyncio
import aiofiles
from tqdm.asyncio import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.data.dataloaders.bird_dataloader import BirdDataLoader, BirdExample
from app.pipeline.text_to_sql import TextToSQLPipeline
from app.evaluation.metrics import evaluate_dataset, DatasetEvaluationResult
from app.core.logger import logger


@dataclass
class BirdRunConfig:
    """Configuration for BIRD evaluation run."""
    split: str = "dev"
    max_examples: Optional[int] = None
    use_schema_linking: bool = True
    top_k_tables: int = 5
    top_k_columns: int = 10
    save_predictions: bool = True
    output_dir: str = "outputs"
    max_concurrent: int = 10


@dataclass
class BirdRunResult:
    """Result of a BIRD evaluation run."""
    config: BirdRunConfig
    evaluation: DatasetEvaluationResult
    predictions: List[Dict[str, Any]]
    start_time: str
    end_time: str
    total_time_seconds: float
    avg_time_per_example: float


class BirdEvaluationRunner:
    """Runner for evaluating text-to-SQL pipeline on BIRD dataset."""

    def __init__(
            self,
            pipeline: TextToSQLPipeline,
            bird_dir: str | Path = "databases/bird",
    ):
        self._pipeline = pipeline
        self._bird_dir = Path(bird_dir)
        self._data_loader = BirdDataLoader(bird_dir)

    async def run(
            self,
            config: Optional[BirdRunConfig] = None,
            examples: Optional[List[BirdExample]] = None,
    ) -> BirdRunResult:
        config = config or BirdRunConfig()
        start_time = datetime.now()

        if examples is None:
            logger.info(f"Loading examples from BIRD split: {config.split}")
            examples = await self._data_loader.load_examples(config.split)

        if config.max_examples is not None:
            examples = examples[:config.max_examples]

        logger.info(f"Running BIRD evaluation on {len(examples)} examples")

        predictions = await self._run_pipeline_concurrent(examples, config)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        logger.info("Computing evaluation metrics...")
        evaluation = await evaluate_dataset(predictions, str(self._bird_dir))

        run_result = BirdRunResult(
            config=config,
            evaluation=evaluation,
            predictions=predictions,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_time_seconds=total_time,
            avg_time_per_example=total_time / len(examples) if examples else 0,
        )

        if config.save_predictions:
            await self._save_results(run_result, config)

        return run_result

    async def _run_pipeline_concurrent(
            self,
            examples: List[BirdExample],
            config: BirdRunConfig
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(config.max_concurrent)
        predictions = []

        async def process_one(example: BirdExample) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self._pipeline.run(
                        question=example.question,
                        db_id=example.db_id,
                        gold_sql=example.sql,
                        evidence=example.evidence,
                        execute=True,
                    )

                    return {
                        "question": example.question,
                        "evidence": example.evidence,
                        "db_id": example.db_id,
                        "difficulty": example.difficulty,
                        "predicted_sql": result.predicted_sql,
                        "gold_sql": example.sql,
                        "execution_match": result.execution_match,
                        "execution_error": result.execution_error,
                        "schema_linking_time": result.schema_linking_time,
                        "generation_time": result.generation_time,
                        "execution_time": result.execution_time,
                        "total_time": result.total_time,
                    }

                except Exception as e:
                    logger.error(f"Error processing example: {example.question[:50]}... - {e}")
                    return {
                        "question": example.question,
                        "evidence": example.evidence,
                        "db_id": example.db_id,
                        "difficulty": example.difficulty,
                        "predicted_sql": "",
                        "gold_sql": example.sql,
                        "execution_match": False,
                        "execution_error": str(e),
                        "schema_linking_time": 0,
                        "generation_time": 0,
                        "execution_time": 0,
                        "total_time": 0,
                    }

        tasks = [process_one(example) for example in examples]

        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating BIRD"):
            result = await coro
            predictions.append(result)

        return predictions

    async def _save_results(self, result: BirdRunResult, config: BirdRunConfig):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"bird_eval_{config.split}_{timestamp}.json"

        output_data = {
            "config": asdict(config),
            "metrics": {
                "total": result.evaluation.total,
                "execution_accuracy": result.evaluation.execution_accuracy,
                "exact_match": result.evaluation.exact_match,
                "valid_sql_rate": result.evaluation.valid_sql_rate,
            },
            "timing": {
                "start_time": result.start_time,
                "end_time": result.end_time,
                "total_time_seconds": result.total_time_seconds,
                "avg_time_per_example": result.avg_time_per_example,
            },
            "predictions": result.predictions,
            "errors": result.evaluation.errors,
        }

        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(output_data, indent=2, ensure_ascii=False))

        logger.info(f"Results saved to: {output_file}")