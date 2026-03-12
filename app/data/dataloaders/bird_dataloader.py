import asyncio
import aiofiles
import json
import pathlib
from dataclasses import dataclass
from typing import List, AsyncIterator, Optional

DEFAULT_BIRD_PATH = pathlib.Path("./databases/bird")

@dataclass
class BirdExample:
    """Single example from BIRD dataset."""
    db_id: str
    question: str
    evidence: Optional[str]  # BIRD has additional evidence field
    sql: str  # gold SQL
    difficulty: Optional[str] = None


class BirdDataLoader:
    """Loader for BIRD dataset examples."""

    def __init__(self, bird_dir: str | pathlib.Path = DEFAULT_BIRD_PATH):
        self.bird_dir = pathlib.Path(bird_dir)
        self.databases_dir = self.bird_dir / f"{self.bird_dir.name}_databases"

    async def load_examples(self, split: str = "dev") -> List[BirdExample]:
        """
        Load examples from a dataset split.
        
        Args:
            split: Dataset split - "dev" or "train"
        """
        json_path = self.bird_dir / split / f"{split}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_path}")

        async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)

        examples = []
        for item in data:
            examples.append(BirdExample(
                db_id=item["db_id"],
                question=item["question"],
                evidence=item.get("evidence", ""),
                sql=item.get("SQL", item.get("query", "")),  # BIRD uses "SQL" field
                difficulty=item.get("difficulty")
            ))

        return examples

    async def iter_examples(self, split: str = "dev") -> AsyncIterator[BirdExample]:
        """Iterate over examples (memory efficient for large datasets)."""
        json_path = self.bird_dir / split / f"{split}.json"
        
        async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)

        for item in data:
            yield BirdExample(
                db_id=item["db_id"],
                question=item["question"],
                evidence=item.get("evidence", ""),
                sql=item.get("SQL", item.get("query", "")),
                difficulty=item.get("difficulty")
            )

    def get_db_path(self, db_id: str) -> pathlib.Path:
        """Get path to SQLite database file."""
        return self.databases_dir / db_id / f"{db_id}.sqlite"

    def get_tables_json_path(self, split: str = "dev") -> pathlib.Path:
        """Get path to dev_tables.json."""
        return self.bird_dir / split / f"{split}_tables.json"

    async def get_unique_db_ids(self, split: str = "dev") -> List[str]:
        """Get list of unique database IDs in a split."""
        examples = await self.load_examples(split)
        return list(set(ex.db_id for ex in examples))


async def main():
    """Example usage."""
    loader = BirdDataLoader()
    
    try:
        examples = await loader.load_examples("dev")
        print(f"Loaded {len(examples)} dev examples from BIRD")
        
        if examples:
            ex = examples[0]
            print(f"\nFirst example:")
            print(f"  DB: {ex.db_id}")
            print(f"  Question: {ex.question}")
            print(f"  Evidence: {ex.evidence}")
            print(f"  SQL: {ex.sql[:100]}..." if len(ex.sql) > 100 else f"  SQL: {ex.sql}")
            print(f"  Difficulty: {ex.difficulty}")
            
            db_path = loader.get_db_path(ex.db_id)
            print(f"  DB path: {db_path}")
            print(f"  DB exists: {db_path.exists()}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure BIRD dataset is downloaded and extracted")


if __name__ == "__main__":
    asyncio.run(main())