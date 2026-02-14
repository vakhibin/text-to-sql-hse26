import asyncio
import sqlite3
import aiosqlite
from typing import Any, List, Tuple, Optional
from contextlib import asynccontextmanager


class SQLiteDatabase:
    """Asynchronous SQLite client using aiosqlite"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Establish asynchronous database connection"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            # Enable foreign key support
            await self._connection.execute("PRAGMA foreign_keys = ON;")

    async def disconnect(self) -> None:
        """Close database connection asynchronously"""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[Tuple]:
        """Execute SELECT query"""
        if self._connection is None:
            await self.connect()

        try:
            cursor = await self._connection.execute(query, params)
            result = await cursor.fetchall()
            await cursor.close()
            return result
        except aiosqlite.Error as e:
            raise e

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


if __name__ == "__main__":
    async def main():
        db_path = "databases/spider/database/concert_singer/concert_singer.sqlite"
        sql_query = "SELECT COUNT(*) FROM singer;"
        gold_sql_query = "SELECT count(*) FROM singer"
        async with SQLiteDatabase(db_path) as db:
            try:
                print("===="*10)
                predicted_result = await db.execute_query(sql_query)
                gold_result = await db.execute_query(gold_sql_query)
                print("Predicted result:", predicted_result)
                print("Gold result:", gold_result)
            except Exception as e:
                error = f"Predicted SQL error: {str(e)}"
                return False, error, None, None

    asyncio.run(main())