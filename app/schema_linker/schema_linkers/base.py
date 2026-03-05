from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING
import asyncio

import numpy as np
from langchain_core.embeddings import Embeddings
from redisvl.extensions.cache.embeddings import EmbeddingsCache

from app.core.embeddings import cosine_similarity
from app.core.logger import logger
from app.settings import settings

if TYPE_CHECKING:
    from app.schema_linker.vector_storages.chromadb import ChromaSchemaStore

@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[str]
    column_types: Dict[str, str]
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[tuple] = field(default_factory=list)  # [(from_col, to_table, to_col), ...]


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
            table_str = f"Table: {table.name}\n  Columns: {columns_str}"

            if table.primary_keys:
                table_str += f"\n  Primary keys: {', '.join(table.primary_keys)}"

            if table.foreign_keys:
                fk_strs = [f"{fk[0]} -> {fk[1]}.{fk[2]}" for fk in table.foreign_keys]
                table_str += f"\n  Foreign keys: {', '.join(fk_strs)}"

            parts.append(table_str)
        return "\n\n".join(parts)

    def to_compact_string(self) -> str:
        """Compact format: table(col1, col2, ...)"""
        parts = []
        for table in self.tables:
            cols = ", ".join(table.columns)
            parts.append(f"{table.name}({cols})")
        return "\n".join(parts)

    def to_create_table_string(self) -> str:
        """Format as CREATE TABLE statements."""
        parts = []
        for table in self.tables:
            cols = []
            for col in table.columns:
                col_type = table.column_types.get(col, "TEXT")
                pk_marker = " PRIMARY KEY" if col in table.primary_keys else ""
                cols.append(f"  {col} {col_type}{pk_marker}")

            create_stmt = f"CREATE TABLE {table.name} (\n"
            create_stmt += ",\n".join(cols)

            # Add foreign key constraints
            for fk in table.foreign_keys:
                create_stmt += f",\n  FOREIGN KEY ({fk[0]}) REFERENCES {fk[1]}({fk[2]})"

            create_stmt += "\n);"
            parts.append(create_stmt)

        return "\n\n".join(parts)


class BaseSchemaLinker(ABC):
    """Abstract base class for schema linkers."""

    def __init__(
            self,
            embeddings: Optional[Embeddings] = None,
            redis_url: str = "redis://localhost:6379"
    ):
        self._embeddings = embeddings
        self._schema_cache: Optional[List[TableInfo]] = None
        self._cache_lock = asyncio.Lock()
        self._redis_cache = EmbeddingsCache(
            name="query_cache",  # name prefix for Redis keys
            redis_url=redis_url,  # Redis connection URL
            ttl=None  # Optional TTL in seconds (None means no expiration)
        )
        self._logger = logger

    @abstractmethod
    async def get_full_schema(self) -> List[TableInfo]:
        """Extract full schema asynchronously. Must be implemented by subclasses."""
        pass

    async def get_schema_string(self) -> str:
        """Get full schema as formatted string asynchronously."""
        tables = await self.get_full_schema()
        linked = LinkedSchema(tables=tables)
        return linked.to_schema_string()

    async def link(
        self,
        question: str,
        evidence: Optional[str] = None,
        top_k_tables: int = 5,
        top_k_columns: int = 10,
    ) -> LinkedSchema:
        """
        Link question to relevant schema elements using embeddings asynchronously.

        Args:
            question: Natural language question
            evidence: Additional context (for BIRD dataset)
            top_k_tables: Max number of tables to return
            top_k_columns: Max number of columns per table

        Returns:
            LinkedSchema with relevant tables and columns
        """
        tables = await self.get_full_schema()

        if self._embeddings is None or not tables:
            return LinkedSchema(tables=tables)

        # Build query text
        query_text = question
        if evidence:
            query_text = f"{question} {evidence}"

        # Build schema element texts for embedding
        table_texts = [f"table {t.name}" for t in tables]
        column_texts = []
        column_info = []  # (table_name, column_name)
        for table in tables:
            for col in table.columns:
                column_texts.append(f"column {col} in table {table.name}")
                column_info.append((table.name, col))

        # Get embeddings via LangChain asynchronously
        query_emb = await self._embeddings.aembed_query(query_text)
        table_embs = await self._embeddings.aembed_documents(table_texts)
        column_embs = await self._embeddings.aembed_documents(column_texts)

        # Compute similarities (CPU-bound, run in thread pool)
        table_scores = await asyncio.to_thread(cosine_similarity, query_emb, table_embs)
        column_scores = await asyncio.to_thread(cosine_similarity, query_emb, column_embs)

        # Select top-k tables (CPU-bound, run in thread pool)
        top_table_indices = await asyncio.to_thread(
            lambda: np.argsort(table_scores)[::-1][:top_k_tables]
        )

        # Build result directly (no extra helper methods)
        table_scores_dict = {tables[i].name: float(table_scores[i]) for i in top_table_indices}
        column_scores_dict: Dict[str, Dict[str, float]] = {}

        result_tables = []
        for table_idx in top_table_indices:
            table = tables[table_idx]

            # Get column scores for this table
            table_column_scores = []
            for col_idx, (col_table, col_name) in enumerate(column_info):
                if col_table == table.name:
                    table_column_scores.append((col_name, float(column_scores[col_idx])))

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
                primary_keys=[pk for pk in table.primary_keys if pk in selected_columns],
                foreign_keys=[fk for fk in table.foreign_keys if fk[0] in selected_columns],
            ))

        return LinkedSchema(
            tables=result_tables,
            table_scores=table_scores_dict,
            column_scores=column_scores_dict,
        )

    async def _get_query_embedding_with_cache(
            self,
            query: str,
            use_cache: bool = True
    ):
        """Get query embedding from cache or compute it."""
        # NOTE: EmbeddingsCache may be "falsy" even when initialized (e.g. __len__ == 0),
        # so we explicitly check for None instead of relying on truthiness.
        if use_cache and self._redis_cache is not None:
            try:
                # Try to get from cache
                cached = self._redis_cache.get(
                    content=query,
                      model_name=self._embeddings.model if hasattr(self._embeddings, 'model') else "unknown"
                )
                if cached and "embedding" in cached:
                    return np.array(cached["embedding"])
            except Exception as e:
                self._logger.error("Cache retrieval error:", repr(e))

        # Compute embedding if not in cache
        embedding = await self._embeddings.aembed_query(query)

        # Store in cache if enabled
        if use_cache and self._redis_cache is not None:
            await self._set_query_embedding_with_cache(query, embedding)

        return embedding

    async def _set_query_embedding_with_cache(
            self,
            query: str,
            embedding: List[float]
    ):
        """Store query embedding in cache."""
        try:
            # Optional metadata
            metadata = {"type": "query_embedding"}

            # Store in cache
            key = self._redis_cache.set(
                content=query,
                model_name=self._embeddings.model if hasattr(self._embeddings, 'model') else "unknown",
                embedding=embedding,
                metadata=metadata
            )
            self._logger.debug(f"Embedding for query {query} was stored with key: {key[:10]} ...")
        except Exception as e:
            self._logger.error("Cache storage error:", repr(e))


    async def link_with_vector_store(
        self,
        question: str,
        vector_store: "ChromaSchemaStore",
        db_id: str,
        evidence: Optional[str] = None,
        top_k_tables: int = 5,
        top_k_columns: int = 10,
        use_cache: bool = True
    ) -> LinkedSchema:
        """
        Link question to schema using vector store - идентично формату link()
        """
        tables = await self.get_full_schema()

        if self._embeddings is None or not tables:
            return LinkedSchema(tables=tables)

        query_text = question
        if evidence:
            query_text = f"{question} {evidence}"

        # Build schema element texts/ids exactly like in link()
        table_ids = [f"{db_id}_table_{t.name}" for t in tables]
        column_ids: List[str] = []
        column_info = []  # (table_name, column_name)
        for table in tables:
            for col in table.columns:
                column_ids.append(f"{db_id}_column_{table.name}_{col}")
                column_info.append((table.name, col))

        # 1. Query embedding (same as in link(), just cached)
        query_emb = await self._get_query_embedding_with_cache(query_text, use_cache)

        # 2. Fetch schema embeddings from vector store (not recomputing them)
        table_emb_map = await vector_store.get_embeddings_by_ids(table_ids)
        column_emb_map = await vector_store.get_embeddings_by_ids(column_ids)

        # Build embedding arrays in exactly the same order as `link()`
        missing_tables = [tid for tid in table_ids if tid not in table_emb_map]
        missing_columns = [cid for cid in column_ids if cid not in column_emb_map]
        if missing_tables or missing_columns:
            self._logger.warning(
                "Missing embeddings in vector store "
                f"(tables={len(missing_tables)}, columns={len(missing_columns)}) for db_id={db_id}. "
                "Falling back to computing missing embeddings to preserve behavior."
            )

        table_embs: List[List[float]] = []
        for i, tid in enumerate(table_ids):
            emb = table_emb_map.get(tid)
            if emb is None:
                emb = await self._embeddings.aembed_query(f"table {tables[i].name}")
            table_embs.append(list(emb))

        column_embs: List[List[float]] = []
        col_idx = 0
        for table in tables:
            for col in table.columns:
                cid = column_ids[col_idx]
                emb = column_emb_map.get(cid)
                if emb is None:
                    emb = await self._embeddings.aembed_query(f"column {col} in table {table.name}")
                column_embs.append(list(emb))
                col_idx += 1

        # 3. Compute cosine similarities exactly like in link()
        table_scores = await asyncio.to_thread(cosine_similarity, query_emb, table_embs)
        column_scores = await asyncio.to_thread(cosine_similarity, query_emb, column_embs)

        top_table_indices = await asyncio.to_thread(
            lambda: np.argsort(table_scores)[::-1][:top_k_tables]
        )

        table_scores_dict = {tables[i].name: float(table_scores[i]) for i in top_table_indices}
        column_scores_dict: Dict[str, Dict[str, float]] = {}

        result_tables = []
        for table_idx in top_table_indices:
            table = tables[table_idx]

            table_column_scores = []
            for cidx, (col_table, col_name) in enumerate(column_info):
                if col_table == table.name:
                    table_column_scores.append((col_name, float(column_scores[cidx])))

            table_column_scores.sort(key=lambda x: x[1], reverse=True)
            top_columns = table_column_scores[:top_k_columns]

            column_scores_dict[table.name] = {col: score for col, score in top_columns}

            selected_columns = [col for col, _ in top_columns]
            result_tables.append(TableInfo(
                name=table.name,
                columns=selected_columns,
                column_types={col: table.column_types[col] for col in selected_columns},
                primary_keys=[pk for pk in table.primary_keys if pk in selected_columns],
                foreign_keys=[fk for fk in table.foreign_keys if fk[0] in selected_columns],
            ))

        return LinkedSchema(
            tables=result_tables,
            table_scores=table_scores_dict,
            column_scores=column_scores_dict,
        )
