import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import Optional, List, Iterator
import kagglehub
from app.core.logger import logger

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

    def load_examples(self, split: str = "dev") -> List[SpiderExample]:
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

        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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

    def iter_examples(self, split: str = "dev") -> Iterator[SpiderExample]:
        """Iterate over examples (memory efficient for large datasets)."""
        if split == "dev":
            json_path = self.spider_dir / "dev.json"
        elif split in ("train", "train_spider"):
            json_path = self.spider_dir / "train_spider.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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

    def get_unique_db_ids(self, split: str = "dev") -> List[str]:
        """Get list of unique database IDs in a split."""
        examples = self.load_examples(split)
        return list(set(ex.db_id for ex in examples))


class DatasetDownloader:
    def __init__(
            self,
            download_url: str = KAGGLE_DATASET_URL,
            download_path: Optional[str] = None
    ):
        # Use provided path or default
        self.download_path = pathlib.Path(download_path) if download_path else DEFAULT_DOWNLOAD_PATH

        # Create directory if it doesn't exist
        self.download_path.mkdir(parents=True, exist_ok=True)

        self._logger = logger
        self.download_url = download_url

    def download_from_kaggle(self, force_download: bool = False) -> pathlib.Path:
        try:
            self._logger.info(f"Starting Spider 1.0 dataset download from Kaggle...")
            self._logger.info(f"Source: {self.download_url}")
            self._logger.info(f"Target directory: {self.download_path.absolute()}")

            # Step 1: Download to kagglehub cache
            cached_path = kagglehub.dataset_download(self.download_url)
            self._logger.info(f"Downloaded to cache: {cached_path}")

            # Step 2: Copy to our desired location
            if self.download_path.exists() and any(self.download_path.iterdir()):
                if force_download:
                    shutil.rmtree(self.download_path)
                    self.download_path.mkdir(parents=True, exist_ok=True)
                    self._logger.info("Cleaned existing directory")
                else:
                    self._logger.warning(
                        f"Directory {self.download_path} already exists. Use force_download=True to overwrite")
                    return self.download_path

            # Copy all files from cache to target directory
            # kagglehub may nest files in a subdirectory, find the actual data
            cached_path = pathlib.Path(cached_path)

            # Check if data is in a nested 'spider' subdirectory
            nested_path = cached_path / "spider"
            if nested_path.exists() and (nested_path / "dev.json").exists():
                source_path = nested_path
            elif (cached_path / "dev.json").exists():
                source_path = cached_path
            else:
                # Search for dev.json in subdirectories
                for subdir in cached_path.iterdir():
                    if subdir.is_dir() and (subdir / "dev.json").exists():
                        source_path = subdir
                        break
                else:
                    source_path = cached_path

            # Copy contents
            for item in source_path.iterdir():
                dest = self.download_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            self._logger.info(f"Dataset successfully copied to: {self.download_path}")
            return self.download_path

        except Exception as e:
            error_msg = f"Error downloading dataset from Kaggle: {str(e)}"
            self._logger.error(f"{error_msg}")
            raise Exception(error_msg)


if __name__ == "__main__":
    # Download dataset
    downloader = DatasetDownloader()
    downloader.download_from_kaggle()

    # Load examples
    loader = SpiderDataLoader()
    examples = loader.load_examples("dev")
    print(f"Loaded {len(examples)} dev examples")

    # Show first example
    if examples:
        ex = examples[0]
        print(f"\nFirst example:")
        print(f"  DB: {ex.db_id}")
        print(f"  Question: {ex.question}")
        print(f"  SQL: {ex.query}")
