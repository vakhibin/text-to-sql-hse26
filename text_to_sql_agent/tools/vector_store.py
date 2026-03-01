"""Chroma vector store utilities for schema retrieval."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from text_to_sql_agent.config import settings


def _table_to_document(db_id: str, table: dict[str, Any]) -> Document:
    table_name = table.get("name", "unknown")
    col_parts: list[str] = []
    for col in table.get("columns", []):
        samples = col.get("sample_values") or []
        col_parts.append(f"{col.get('name')}:{col.get('type')} samples={samples}")
    fk_parts = [
        f"{fk['column']}->{fk['ref_table']}.{fk['ref_column']}"
        for fk in table.get("foreign_keys", [])
    ]
    page_content = (
        f"table={table_name}\n"
        f"columns={'; '.join(col_parts)}\n"
        f"primary_keys={table.get('primary_keys', [])}\n"
        f"foreign_keys={fk_parts}"
    )
    return Document(
        page_content=page_content,
        metadata={"db_id": db_id, "table_name": table_name},
    )


class VectorStoreClient:
    """Schema index/retrieval client backed by Chroma."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path = ".cache/chroma",
    ):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for vector embeddings")
        self._vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_directory,
            embedding_function=OpenAIEmbeddings(
                model=settings.embeddings_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            ),
        )

    def _index_schema_sync(self, db_id: str, schema: dict[str, Any]) -> None:
        docs = [_table_to_document(db_id, table) for table in schema.get("tables", [])]
        if not docs:
            return
        ids = [f"{db_id}:{doc.metadata['table_name']}" for doc in docs]
        self._vector_store.add_documents(documents=docs, ids=ids)

    async def index_schema(self, db_id: str, schema: dict[str, Any]) -> None:
        """Index table-level documents for a specific database schema."""
        await asyncio.to_thread(self._index_schema_sync, db_id, schema)

    def _query_tables_sync(self, query: str, db_id: str, top_k: int) -> list[dict[str, Any]]:
        docs = self._vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            filter={"db_id": db_id},
        )
        return [
            {
                "table_name": doc.metadata.get("table_name", ""),
                "db_id": doc.metadata.get("db_id", db_id),
                "score": float(score),
                "content": doc.page_content,
            }
            for doc, score in docs
        ]

    async def query_tables(self, query: str, db_id: str, top_k: int = 15) -> list[dict[str, Any]]:
        """Retrieve top-k table candidates from indexed schema."""
        return await asyncio.to_thread(self._query_tables_sync, query, db_id, top_k)


def build_vector_store(
    collection_name: str | None = None,
    *,
    persist_directory: str | Path | None = None,
) -> VectorStoreClient:
    """Build a Chroma-backed vector store client."""
    return VectorStoreClient(
        collection_name=collection_name or settings.chroma_collection_selector,
        persist_directory=persist_directory or settings.chroma_persist_directory,
    )

