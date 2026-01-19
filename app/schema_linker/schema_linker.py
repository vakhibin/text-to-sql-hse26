from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.db import SQLiteDatabase
from app.core.embeddings import cosine_similarity


@dataclass
class TableInfo:
    name: str
    columns: List[str]
    column_types: Dict[str, str]


@dataclass
class LinkedSchema:
    """Result of schema linking."""
    tables: List[TableInfo]
    table_scores: Dict[str, float] = field(default_factory=dict)
    column_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_schema_string(self) -> str:
        """Convert linked schema to string format for prompt."""
        parts = []
        for table in self.tables:
            columns_str = ", ".join(
                f"{col} ({table.column_types.get(col, 'UNKNOWN')})"
                for col in table.columns
            )
            parts.append(f"Table: {table.name}\n  Columns: {columns_str}")
        return "\n\n".join(parts)

    def to_compact_string(self) -> str:
        """Compact format: table(col1, col2, ...)"""
        parts = []
        for table in self.tables:
            cols = ", ".join(table.columns)
            parts.append(f"{table.name}({cols})")
        return "\n".join(parts)


class SchemaLinker:
    def __init__(
        self,
        db_path: str,
        embeddings: Optional[Embeddings] = None,
    ):
        self.db_path = db_path
        self._embeddings = embeddings
        self._schema_cache: Optional[List[TableInfo]] = None

    def get_full_schema(self) -> List[TableInfo]:
        """Extract full schema from database."""
        if self._schema_cache is not None:
            return self._schema_cache

        tables = []
        with SQLiteDatabase(self.db_path) as db:
            table_rows = db.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            table_names = [row[0] for row in table_rows if not row[0].startswith("sqlite_")]

            for table_name in table_names:
                column_info = db.execute_query(f"PRAGMA table_info(`{table_name}`);")
                columns = []
                column_types = {}
                for col in column_info:
                    col_name = col[1]
                    col_type = col[2].upper() if col[2] else "UNKNOWN"
                    columns.append(col_name)
                    column_types[col_name] = col_type

                tables.append(TableInfo(
                    name=table_name,
                    columns=columns,
                    column_types=column_types,
                ))

        self._schema_cache = tables
        return tables

    def get_schema_string(self) -> str:
        """Get full schema as formatted string."""
        tables = self.get_full_schema()
        parts = []
        for table in tables:
            parts.append(f"Table: {table.name}")
            for col in table.columns:
                col_type = table.column_types.get(col, "UNKNOWN")
                parts.append(f"  - {col} ({col_type})")
            parts.append("")
        return "\n".join(parts).strip()

    def link(
        self,
        question: str,
        evidence: Optional[str] = None,
        top_k_tables: int = 5,
        top_k_columns: int = 10,
    ) -> LinkedSchema:
        """
        Link question to relevant schema elements using embeddings.

        Args:
            question: Natural language question
            evidence: Additional context (for BIRD dataset)
            top_k_tables: Max number of tables to return
            top_k_columns: Max number of columns per table

        Returns:
            LinkedSchema with relevant tables and columns
        """
        if self._embeddings is None:
            return LinkedSchema(tables=self.get_full_schema())

        tables = self.get_full_schema()
        if not tables:
            return LinkedSchema(tables=[])

        # Build query text
        query_text = question
        if evidence:
            query_text = f"{question} {evidence}"

        # Build schema element texts for embedding
        table_texts = [f"table {t.name}" for t in tables]
        column_texts = []
        column_to_table = []
        for table in tables:
            for col in table.columns:
                column_texts.append(f"column {col} in table {table.name}")
                column_to_table.append(table.name)

        # Get embeddings via LangChain
        query_emb = self._embeddings.embed_query(query_text)
        table_embs = self._embeddings.embed_documents(table_texts)
        column_embs = self._embeddings.embed_documents(column_texts)

        # Compute similarities
        table_scores = cosine_similarity(query_emb, table_embs)
        column_scores = cosine_similarity(query_emb, column_embs)

        # Select top-k tables
        top_table_indices = np.argsort(table_scores)[::-1][:top_k_tables]

        # Build result
        table_scores_dict = {tables[i].name: float(table_scores[i]) for i in top_table_indices}
        column_scores_dict: Dict[str, Dict[str, float]] = {}

        result_tables = []
        for table_idx in top_table_indices:
            table = tables[table_idx]

            # Get column scores for this table
            table_column_scores = []
            for col_idx, (col_table, score) in enumerate(zip(column_to_table, column_scores)):
                if col_table == table.name:
                    col_name = column_texts[col_idx].split()[1]  # extract column name
                    table_column_scores.append((col_name, float(score)))

            # Sort by score and take top-k
            table_column_scores.sort(key=lambda x: x[1], reverse=True)
            top_columns = table_column_scores[:top_k_columns]

            column_scores_dict[table.name] = {col: score for col, score in top_columns}

            # Build filtered table info
            selected_columns = [col for col, _ in top_columns]
            result_tables.append(TableInfo(
                name=table.name,
                columns=selected_columns,
                column_types={col: table.column_types[col] for col in selected_columns},
            ))

        return LinkedSchema(
            tables=result_tables,
            table_scores=table_scores_dict,
            column_scores=column_scores_dict,
        )


if __name__ == "__main__":
    from app.settings import settings
    from app.core.embeddings import create_embeddings

    if settings.openrouter_api_key:
        embeddings = create_embeddings(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_embedding_model,
        )
        linker = SchemaLinker(
            db_path="path/to/database.sqlite",
            embeddings=embeddings,
        )
        result = linker.link(
            question="Show all employees in Sales department",
            top_k_tables=3,
        )
        print(result.to_schema_string())
