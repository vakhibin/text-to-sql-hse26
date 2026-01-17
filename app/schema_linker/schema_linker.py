from typing import List, Dict
from app.core.db import SQLiteDatabase


class SchemaLinker:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_schema_string(self) -> str:
        schema_parts: List[str] = []

        with SQLiteDatabase(self.db_path) as db:
            # Получаем список таблиц
            table_rows = db.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            table_names = [row[0] for row in table_rows]

            for table_name in table_names:
                # Пропускаем служебные таблицы SQLite
                if table_name.startswith("sqlite_"):
                    continue

                # Получаем информацию о столбцах
                column_info = db.execute_query(f"PRAGMA table_info(`{table_name}`);")
                columns = []
                for col in column_info:
                    # col[1] — имя столбца, col[2] — тип
                    col_name = col[1]
                    col_type = col[2].upper() if col[2] else "UNKNOWN"
                    columns.append(f"  - {col_name} ({col_type})")

                if columns:
                    schema_parts.append(f"Table: {table_name}")
                    schema_parts.extend(columns)
                    schema_parts.append("")  # пустая строка между таблицами

        return "\n".join(schema_parts).strip()