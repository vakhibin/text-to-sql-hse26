import sqlite3
from typing import Any, List, Tuple, Optional

class SQLiteDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)

    def disconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[Tuple]:
        if self._connection is None:
            self.connect()

        cursor = self._connection.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            raise e
        finally:
            cursor.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
