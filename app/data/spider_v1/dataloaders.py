import asyncio
import aiofiles
import aiofiles.os
import json
import pathlib

from dataclasses import dataclass
from typing import List, AsyncIterator
from app.data.spider_v1.downloader import DatasetDownloader

KAGGLE_DATASET_URL = "jeromeblanchet/yale-universitys-spider-10-nlp-dataset"
DEFAULT_DOWNLOAD_PATH = pathlib.Path("./databases/spider")

@dataclass
class SpiderExample:
    """Single example from Spider dataset."""
    db_id: str
    question: str
    query: str  # gold SQL
    question_toks: List[str] = None
    query_toks: List[str] = None
    query_toks_no_value: List[str] = None


class SpiderDataLoader:
    """Loader for Spider dataset examples."""

    def __init__(self, spider_dir: str | pathlib.Path = DEFAULT_DOWNLOAD_PATH):
        self.spider_dir = pathlib.Path(spider_dir)

    async def load_examples(self, split: str = "dev") -> List[SpiderExample]:
        """
        Load examples from a dataset split.

        Args:
            split: Dataset split - "dev", "train", or "train_spider"

        Returns:
            List of SpiderExample objects
        """
        if split == "dev":
            json_path = self.spider_dir / "dev.json"
        elif split in ("train", "train_spider"):
            json_path = self.spider_dir / "train_spider.json"
        else:
            raise ValueError(f"Unknown split: {split}. Use 'dev' or 'train'")

        if not await aiofiles.os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset file not found: {json_path}")

        async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)

        examples = []
        for item in data:
            examples.append(SpiderExample(
                db_id=item["db_id"],
                question=item["question"],
                query=item["query"],
                question_toks=item.get("question_toks"),
                query_toks=item.get("query_toks"),
                query_toks_no_value=item.get("query_toks_no_value"),
            ))

        return examples

    async def iter_examples(self, split: str = "dev") -> AsyncIterator[SpiderExample]:
        """Iterate over examples (memory efficient for large datasets)."""
        if split == "dev":
            json_path = self.spider_dir / "dev.json"
        elif split in ("train", "train_spider"):
            json_path = self.spider_dir / "train_spider.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)

        for item in data:
            yield SpiderExample(
                db_id=item["db_id"],
                question=item["question"],
                query=item["query"],
                question_toks=item.get("question_toks"),
                query_toks=item.get("query_toks"),
                query_toks_no_value=item.get("query_toks_no_value"),
            )

    def get_db_path(self, db_id: str) -> pathlib.Path:
        """Get path to SQLite database file."""
        return self.spider_dir / "database" / db_id / f"{db_id}.sqlite"

    def get_tables_json_path(self) -> pathlib.Path:
        """Get path to tables.json."""
        return self.spider_dir / "tables.json"

    async def get_unique_db_ids(self, split: str = "dev") -> List[str]:
        """Get list of unique database IDs in a split."""
        examples = await self.load_examples(split)
        return list(set(ex.db_id for ex in examples))


async def main():
    # Download dataset
    downloader = DatasetDownloader()
    await downloader.download_from_kaggle()

    # Load examples
    loader = SpiderDataLoader()
    examples = await loader.load_examples("dev")
    print(f"Loaded {len(examples)} dev examples")

    # Show first example
    if examples:
        ex = examples[0]
        print(f"\nFirst example:")
        print(f"  DB: {ex.db_id}")
        print(f"  Question: {ex.question}")
        print(f"  SQL: {ex.query}")


if __name__ == "__main__":
    asyncio.run(main())