import json
import asyncio
import aiofiles
from tqdm.asyncio import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from tqdm import tqdm

from app.data.spider_v1.dataloaders import SpiderDataLoader, SpiderExample
from app.pipeline import TextToSQLPipeline
from app.evaluation.metrics import evaluate_dataset,  DatasetEvaluationResult
from app.core.logger import logger

@dataclass
class RunConfig:
    """Configuration for evaluation run."""
    split: str = "dev"
    max_examples: Optional[int] = None  # Limit for testing
    use_schema_linking: bool = True
    top_k_tables: int = 5
    top_k_columns: int = 10
    save_predictions: bool = True
    output_dir: str = "outputs"


@dataclass
class RunResult:
    """Result of an evaluation run."""
    config: RunConfig
    evaluation: DatasetEvaluationResult
    predictions: List[Dict[str, Any]]
    start_time: str
    end_time: str
    total_time_seconds: float
    avg_time_per_example: float


############################################
############################################
############# Async functions ##############
class EvaluationRunnerAsync:
    """Runner for evaluating text-to-SQL pipeline on Spider dataset."""

    def __init__(
            self,
            pipeline: TextToSQLPipeline,
            spider_dir: str | Path = "databases/spider",
    ):
        """
        Initialize evaluation runner.

        Args:
            pipeline: TextToSQLPipeline instance
            spider_dir: Path to Spider dataset directory
        """
        self._pipeline = pipeline
        self._spider_dir = Path(spider_dir)
        self._data_loader = SpiderDataLoader(spider_dir)

    async def run(
            self,
            config: Optional[RunConfig] = None,
            examples: Optional[List[SpiderExample]] = None,
    ) -> RunResult:
        """
        Run evaluation on Spider dataset asynchronously.

        Args:
            config: Run configuration
            examples: Optional list of examples (if not provided, loads from split)

        Returns:
            RunResult with evaluation metrics and predictions
        """
        config = config or RunConfig()
        start_time = datetime.now()

        # Load examples if not provided
        if examples is None:
            logger.info(f"Loading examples from split: {config.split}")
            examples = await self._data_loader.load_examples(config.split)

        # Limit examples if specified
        if config.max_examples is not None:
            examples = examples[:config.max_examples]

        logger.info(f"Running evaluation on {len(examples)} examples")

        # Run pipeline concurrently on examples
        predictions = await self._run_pipeline_concurrent(examples, config)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Evaluate predictions
        logger.info("Computing evaluation metrics...")
        evaluation = await evaluate_dataset(predictions, str(self._spider_dir))

        # Create result
        run_result = RunResult(
            config=config,
            evaluation=evaluation,
            predictions=predictions,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_time_seconds=total_time,
            avg_time_per_example=total_time / len(examples) if examples else 0,
        )

        # Save predictions if configured
        if config.save_predictions:
            await self._save_results(run_result, config)

        return run_result

    async def _run_pipeline_concurrent(
            self,
            examples: List[SpiderExample],
            config: RunConfig
    ) -> List[Dict[str, Any]]:
        """Run pipeline concurrently with semaphore control."""
        semaphore = asyncio.Semaphore(config.max_concurrent)
        predictions = []

        async def process_one(example: SpiderExample) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self._pipeline.run(
                        question=example.question,
                        db_id=example.db_id,
                        gold_sql=example.query,
                        execute=True,
                    )

                    return {
                        "question": example.question,
                        "db_id": example.db_id,
                        "predicted_sql": result.predicted_sql,
                        "gold_sql": example.query,
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
                        "db_id": example.db_id,
                        "predicted_sql": "",
                        "gold_sql": example.query,
                        "execution_match": False,
                        "execution_error": str(e),
                        "schema_linking_time": 0,
                        "generation_time": 0,
                        "execution_time": 0,
                        "total_time": 0,
                    }

        # Create tasks for all examples
        tasks = [process_one(example) for example in examples]

        # Process with progress bar
        for task in tqdm.as_completed(tasks, desc="Evaluating", total=len(tasks)):
            result = await task
            predictions.append(result)

        return predictions

    async def _save_results(self, result: RunResult, config: RunConfig):
        """Save evaluation results to file asynchronously."""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"eval_{config.split}_{timestamp}.json"

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

    async def run_quick_test(self, n_examples: int = 10) -> RunResult:
        """Run a quick test on a small number of examples asynchronously."""
        config = RunConfig(
            split="dev",
            max_examples=n_examples,
            save_predictions=False,
        )
        return await self.run(config)


async def run_evaluation(
        spider_dir: str = "databases/spider",
        split: str = "dev",
        max_examples: Optional[int] = None,
        use_schema_linking: bool = True,
        output_dir: str = "outputs",
        max_concurrent: int = 10,
) -> RunResult:
    """
    Convenience function to run evaluation asynchronously.

    Args:
        spider_dir: Path to Spider dataset
        split: Dataset split ("dev" or "train")
        max_examples: Limit number of examples (for testing)
        use_schema_linking: Whether to use schema linking
        output_dir: Directory to save results
        max_concurrent: Maximum concurrent pipeline executions

    Returns:
        RunResult with evaluation metrics
    """
    from app.settings import settings
    from app.core.llm import create_llm
    from app.core.embeddings import create_embeddings
    from app.sql_generator.sql_generator import GenerationMode

    # Initialize LLM asynchronously
    llm = await create_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    # Initialize embeddings asynchronously
    embeddings = None
    if use_schema_linking and settings.openrouter_api_key:
        embeddings = await create_embeddings(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
        )

    # Create pipeline
    pipeline = TextToSQLPipeline(
        llm=llm,
        embeddings=embeddings,
        spider_dir=spider_dir,
        generation_mode=GenerationMode.DIRECT,
        use_schema_linking=use_schema_linking,
    )

    # Create runner
    runner = EvaluationRunnerAsync(pipeline=pipeline, spider_dir=spider_dir)

    # Run evaluation
    config = RunConfig(
        split=split,
        max_examples=max_examples,
        use_schema_linking=use_schema_linking,
        save_predictions=True,
        output_dir=output_dir,
        max_concurrent=max_concurrent,
    )

    return await runner.run(config)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run text-to-SQL evaluation on Spider dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "train"],
        help="Dataset split to evaluate (default: dev)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--spider-dir",
        type=str,
        default="databases/spider",
        help="Path to Spider dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-schema-linking",
        action="store_true",
        help="Disable schema linking (use full schema)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent pipeline executions (default: 10)"
    )

    args = parser.parse_args()

    print(f"Running evaluation on Spider {args.split} set")
    print(f"Max examples: {args.max_examples or 'all'}")
    print(f"Schema linking: {'disabled' if args.no_schema_linking else 'enabled'}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()

    result = await run_evaluation(
        spider_dir=args.spider_dir,
        split=args.split,
        max_examples=args.max_examples,
        use_schema_linking=not args.no_schema_linking,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(result.evaluation)
    print(f"\nTotal time: {result.total_time_seconds:.2f}s")
    print(f"Avg time per example: {result.avg_time_per_example:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
