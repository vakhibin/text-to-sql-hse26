import json
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.embeddings import Embeddings

from app.schema_linker.schema_linkers.base import BaseSchemaLinker, TableInfo


class BirdSchemaLinker(BaseSchemaLinker):
    """Schema linker for BIRD dataset.
    
    BIRD provides tables.json for each split (dev_tables.json, train_tables.json)
    with schema information including column types and descriptions.
    """

    def __init__(
            self,
            tables_json_path: str | Path,
            db_id: str,
            embeddings: Optional[Embeddings] = None,
            split: str = "dev"
    ):
        super().__init__(embeddings=embeddings)
        self._tables_json_path = Path(tables_json_path)
        self._db_id = db_id
        self._raw_schema: Optional[Dict] = None
        self._raw_schema_lock = asyncio.Lock()
        self._split = split

    async def _load_raw_schema(self) -> Dict:
        """Load raw schema from tables.json asynchronously."""
        if self._raw_schema is not None:
            return self._raw_schema

        async with self._raw_schema_lock:
            if self._raw_schema is not None:
                return self._raw_schema

            async with aiofiles.open(self._tables_json_path, "r", encoding="utf-8") as f:
                content = await f.read()
                all_tables = json.loads(content)

            for schema in all_tables:
                if schema.get("db_id") == self._db_id:
                    self._raw_schema = schema
                    return schema

            raise ValueError(f"Database '{self._db_id}' not found in {self._tables_json_path}")

    async def get_full_schema(self) -> List[TableInfo]:
        """Extract full schema from tables.json asynchronously."""
        if self._schema_cache is not None:
            return self._schema_cache

        async with self._cache_lock:
            if self._schema_cache is not None:
                return self._schema_cache

            raw = await self._load_raw_schema()

            # BIRD dev_tables.json uses same keys as Spider: table_names_original, column_names_original
            table_names = raw.get("table_names_original", raw.get("table_names", []))
            column_names = raw.get("column_names_original", raw.get("column_names", []))
            column_types = raw.get("column_types", [])

            tables_dict: Dict[str, TableInfo] = {}
            for table_idx, table_name in enumerate(table_names):
                tables_dict[table_name] = TableInfo(
                    name=table_name,
                    columns=[],
                    column_types={},
                    primary_keys=[],
                    foreign_keys=[],
                )

            # Assign columns to tables
            col_idx_to_info: Dict[int, tuple] = {}
            for col_idx, (table_idx, col_name) in enumerate(column_names):
                if table_idx == -1 or table_idx >= len(table_names):
                    continue
                table_name = table_names[table_idx]
                col_type = column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                
                tables_dict[table_name].columns.append(col_name)
                tables_dict[table_name].column_types[col_name] = col_type.upper()
                col_idx_to_info[col_idx] = (table_name, col_name)

            # BIRD may not have primary_keys/foreign_keys in same format as Spider
            # Handle them if present
            primary_keys = raw.get("primary_keys", [])
            for col_idx in primary_keys:
                if isinstance(col_idx, (int, str)) and int(col_idx) in col_idx_to_info:
                    table_name, col_name = col_idx_to_info[int(col_idx)]
                    tables_dict[table_name].primary_keys.append(col_name)

            foreign_keys = raw.get("foreign_keys", [])
            for fk_pair in foreign_keys:
                if len(fk_pair) == 2:
                    from_idx, to_idx = fk_pair
                    if from_idx in col_idx_to_info and to_idx in col_idx_to_info:
                        from_table, from_col = col_idx_to_info[from_idx]
                        to_table, to_col = col_idx_to_info[to_idx]
                        tables_dict[from_table].foreign_keys.append((from_col, to_table, to_col))

            self._schema_cache = list(tables_dict.values())
            return self._schema_cache

    @classmethod
    async def from_bird_dir(
            cls,
            bird_dir: str | Path,
            db_id: str,
            split: str = "dev",
            embeddings: Optional[Embeddings] = None,
    ) -> "BirdSchemaLinker":
        """
        Create linker from BIRD dataset directory.
        
        Args:
            bird_dir: Path to BIRD dataset root
            db_id: Database ID
            split: "dev" or "train"
            embeddings: Optional embeddings client
        """
        bird_dir = Path(bird_dir)
        tables_json = bird_dir / split / f"{split}_tables.json"

        if not await asyncio.to_thread(tables_json.exists):
            raise FileNotFoundError(f"tables.json not found: {tables_json}")

        return cls(tables_json_path=tables_json, db_id=db_id, embeddings=embeddings)

    def get_db_path(self, bird_dir: str | Path) -> Path:
        """Get path to SQLite database file."""
        bird_dir = Path(bird_dir)
        return bird_dir / f"{self._split}_databases" / self._db_id / f"{self._db_id}.sqlite"


async def main():
    """Example usage."""
    from app.settings import settings
    from app.core.embeddings import create_embeddings

    bird_dir = Path("databases/bird")
    
    if not bird_dir.exists():
        print(f"BIRD directory not found: {bird_dir}")
        return

    try:
        # Try with a known BIRD database
        db_id = "california_schools"  # Example from dev set
        
        embeddings = None
        if settings.openrouter_api_key:
            embeddings = await create_embeddings(
                api_key=settings.openrouter_api_key,
                model=settings.embedding_model,
            )

        linker = await BirdSchemaLinker.from_bird_dir(
            bird_dir=bird_dir,
            db_id=db_id,
            split="dev",
            embeddings=embeddings,
        )

        # Get full schema
        print(f"=== Full Schema for {db_id} ===")
        schema_string = await linker.get_schema_string()
        print(schema_string[:500] + "..." if len(schema_string) > 500 else schema_string)

        # Link to question
        print("\n=== Linked Schema ===")
        linked = await linker.link(
            question="What is the average enrollment in California schools?",
            top_k_tables=2,
            top_k_columns=5,
        )
        print(linked.to_schema_string())
        
        db_path = linker.get_db_path(bird_dir)
        print(f"\nDB path: {db_path}")
        print(f"DB exists: {db_path.exists()}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())