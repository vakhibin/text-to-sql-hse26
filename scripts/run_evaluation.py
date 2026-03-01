import asyncio
import sys
from pathlib import Path
from typing import Optional
import fire
from app.evaluation.runner import run_evaluation


class EvaluationCLI:
    """CLI for running text-to-SQL evaluation on Spider dataset."""

    def dev(
            self,
            spider_dir: str = "databases/spider",
            max_examples: Optional[int] = None,
            no_schema_linking: bool = False,
            output_dir: str = "outputs",
            max_concurrent: int = 10
    ):
        """Run evaluation on dev split."""
        result = asyncio.run(run_evaluation(
            spider_dir=spider_dir,
            split="dev",
            max_examples=max_examples,
            use_schema_linking=not no_schema_linking,
            output_dir=output_dir,
            max_concurrent=max_concurrent
        ))
        self._print_results(result)
        return result

    def train(
            self,
            spider_dir: str = "databases/spider",
            max_examples: Optional[int] = None,
            no_schema_linking: bool = False,
            output_dir: str = "outputs",
            max_concurrent: int = 10
    ):
        """Run evaluation on train split."""
        result = asyncio.run(run_evaluation(
            spider_dir=spider_dir,
            split="train",
            max_examples=max_examples,
            use_schema_linking=not no_schema_linking,
            output_dir=output_dir,
            max_concurrent=max_concurrent
        ))
        self._print_results(result)
        return result

    def _print_results(self, result):
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(result.evaluation)
        print(f"\nTotal time: {result.total_time_seconds:.2f}s")
        print(f"Avg time per example: {result.avg_time_per_example:.2f}s")


if __name__ == "__main__":
    fire.Fire(EvaluationCLI)