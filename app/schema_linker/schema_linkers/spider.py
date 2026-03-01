import json
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.embeddings import Embeddings

from app.schema_linker.schema_linkers.base import BaseSchemaLinker, TableInfo


class SpiderSchemaLinker(BaseSchemaLinker):
    """Schema linker for Spider dataset.

    Loads schema from tables.json file which contains:
    - table_names: list of table names
    - column_names: list of [table_idx, column_name] pairs
    - column_types: list of column types
    - primary_keys: list of column indices that are primary keys
    - foreign_keys: list of [from_col_idx, to_col_idx] pairs
    """

    def __init__(
            self,
            tables_json_path: str | Path,
            db_id: str,
            embeddings: Optional[Embeddings] = None,
    ):
        """
        Initialize Spider schema linker.

        Args:
            tables_json_path: Path to tables.json file
            db_id: Database ID (e.g., "concert_singer")
            embeddings: Optional embeddings client for semantic linking
        """
        super().__init__(embeddings=embeddings)
        self._tables_json_path = Path(tables_json_path)
        self._db_id = db_id
        self._raw_schema: Optional[Dict] = None
        self._raw_schema_lock = asyncio.Lock()

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
                if schema["db_id"] == self._db_id:
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

            # Parse table names (can be original or normalized)
            table_names = raw.get("table_names_original", raw.get("table_names", []))

            # Parse columns: column_names is list of [table_idx, column_name]
            column_names = raw.get("column_names_original", raw.get("column_names", []))
            column_types = raw.get("column_types", [])

            # Build table info
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
                if table_idx == -1:
                    continue
                table_name = table_names[table_idx]
                col_type = column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                tables_dict[table_name].columns.append(col_name)
                tables_dict[table_name].column_types[col_name] = col_type.upper()
                col_idx_to_info[col_idx] = (table_name, col_name)

            # Parse primary keys
            primary_keys = raw.get("primary_keys", [])
            for col_idx in primary_keys:
                if col_idx in col_idx_to_info:
                    table_name, col_name = col_idx_to_info[col_idx]
                    tables_dict[table_name].primary_keys.append(col_name)

            # Parse foreign keys
            foreign_keys = raw.get("foreign_keys", [])
            for from_idx, to_idx in foreign_keys:
                if from_idx in col_idx_to_info and to_idx in col_idx_to_info:
                    from_table, from_col = col_idx_to_info[from_idx]
                    to_table, to_col = col_idx_to_info[to_idx]
                    tables_dict[from_table].foreign_keys.append((from_col, to_table, to_col))

            self._schema_cache = list(tables_dict.values())
            return self._schema_cache

    @classmethod
    async def from_spider_dir(
            cls,
            spider_dir: str | Path,
            db_id: str,
            embeddings: Optional[Embeddings] = None,
    ) -> "SpiderSchemaLinker":
        """
        Create linker from Spider dataset directory asynchronously.

        Args:
            spider_dir: Path to Spider dataset root
            db_id: Database ID
            embeddings: Optional embeddings client

        Returns:
            SpiderSchemaLinker instance
        """
        spider_dir = Path(spider_dir)
        tables_json = spider_dir / "tables.json"

        if not await asyncio.to_thread(tables_json.exists):
            raise FileNotFoundError(f"tables.json not found in {spider_dir}")

        return cls(tables_json_path=tables_json, db_id=db_id, embeddings=embeddings)

    def get_db_path(self, spider_dir: str | Path) -> Path:
        """Get path to SQLite database file."""
        spider_dir = Path(spider_dir)
        return spider_dir / "database" / self._db_id / f"{self._db_id}.sqlite"


async def main():
    """Async example usage."""
    from app.settings import settings
    from app.core.embeddings import create_embeddings

    spider_dir = Path("databases/spider")

    if spider_dir.exists() and settings.openrouter_api_key:
        embeddings = await create_embeddings(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
        )

        linker = await SpiderSchemaLinker.from_spider_dir(
            spider_dir=spider_dir,
            db_id="concert_singer",
            embeddings=embeddings,
        )

        # Get full schema
        print("=== Full Schema ===")
        schema_string = await linker.get_schema_string()
        print(schema_string)

        # Link to question
        print("\n=== Linked Schema ===")
        linked = await linker.link(
            question="How many singers are there?",
            top_k_tables=2,
        )
        print(linked.to_schema_string())
        print("\n=== CREATE TABLE format ===")
        print(linked.to_create_table_string())


if __name__ == "__main__":
    asyncio.run(main())