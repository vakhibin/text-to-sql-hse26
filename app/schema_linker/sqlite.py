from typing import List, Optional
import asyncio

from langchain_core.embeddings import Embeddings

from app.core.db import SQLiteDatabase
from app.schema_linker.base import BaseSchemaLinker, TableInfo


class SQLiteSchemaLinker(BaseSchemaLinker):
    """Schema linker that reads schema directly from SQLite database."""

    def __init__(
        self,
        db_path: str,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Initialize SQLite schema linker.

        Args:
            db_path: Path to SQLite database file
            embeddings: Optional embeddings client for semantic linking
        """
        super().__init__(embeddings=embeddings)
        self.db_path = db_path

    async def get_full_schema(self) -> List[TableInfo]:
        """Extract full schema from SQLite database asynchronously."""
        if self._schema_cache is not None:
            return self._schema_cache

        async with self._cache_lock:
            if self._schema_cache is not None:
                return self._schema_cache

            tables = []
            async with SQLiteDatabase(self.db_path) as db:
                # Get table names
                table_rows = await db.execute_query(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
                )
                table_names = [row[0] for row in table_rows if not row[0].startswith("sqlite_")]

                for table_name in table_names:
                    # Get column info
                    column_info = await db.execute_query(f"PRAGMA table_info(`{table_name}`);")
                    columns = []
                    column_types = {}
                    primary_keys = []

                    for col in column_info:
                        col_name = col[1]
                        col_type = col[2].upper() if col[2] else "UNKNOWN"
                        is_pk = col[5] == 1

                        columns.append(col_name)
                        column_types[col_name] = col_type
                        if is_pk:
                            primary_keys.append(col_name)

                    # Get foreign keys
                    fk_info = await db.execute_query(f"PRAGMA foreign_key_list(`{table_name}`);")
                    foreign_keys = []
                    for fk in fk_info:
                        from_col = fk[3]
                        to_table = fk[2]
                        to_col = fk[4]
                        foreign_keys.append((from_col, to_table, to_col))

                    tables.append(TableInfo(
                        name=table_name,
                        columns=columns,
                        column_types=column_types,
                        primary_keys=primary_keys,
                        foreign_keys=foreign_keys,
                    ))

            self._schema_cache = tables
            return tables


async def main():
    """Async example usage."""
    from app.settings import settings
    from app.core.embeddings import create_embeddings

    db_path = "databases/spider/database/concert_singer/concert_singer.sqlite"

    if settings.openrouter_api_key:
        embeddings = await create_embeddings(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
        )

        linker = SQLiteSchemaLinker(db_path=db_path, embeddings=embeddings)

        print("=== Full Schema ===")
        schema_string = await linker.get_schema_string()
        print(schema_string)

        print("\n=== Linked Schema ===")
        linked = await linker.link(
            question="How many concerts are there?",
            top_k_tables=2,
        )
        print(linked.to_schema_string())


if __name__ == "__main__":
    asyncio.run(main())