"""Chroma vector store utilities for schema retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from text_to_sql_agent.config import settings
from text_to_sql_agent.tools.embedding_client import build_langchain_embeddings


def _format_sample_values(samples: Any, *, max_items: int = 5, max_item_chars: int = 80) -> str:
    if not samples:
        return "[]"
    items: list[str] = []
    for raw in list(samples)[:max_items]:
        s = str(raw).replace("\n", " ").strip()
        if len(s) > max_item_chars:
            s = s[: max_item_chars - 3] + "..."
        items.append(s)
    rest = len(samples) - max_items
    suffix = f" …(+{rest} more values)" if rest > 0 else ""
    return "[" + "; ".join(items) + suffix + "]"


def _table_to_document(db_id: str, table: dict[str, Any], *, schema_variant: str = "default") -> Document:
    table_name = table.get("name", "unknown")
    col_parts: list[str] = []
    for col in table.get("columns", []):
        samples = col.get("sample_values") or []
        sample_str = _format_sample_values(samples)
        col_parts.append(f"{col.get('name')}:{col.get('type')} samples={sample_str}")
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
    metadata: dict[str, str] = {"db_id": db_id, "table_name": table_name}
    if schema_variant != "default":
        metadata["schema_variant"] = schema_variant
    return Document(page_content=page_content, metadata=metadata)


def _embedding_namespace(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()[:48] or "embedding"
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _chroma_metadata_where(db_id: str, *, schema_variant: str = "default") -> dict[str, Any]:
    """Chroma ``where``: только одно «простое» условие на верхнем уровне; несколько полей — через ``$and``."""
    if schema_variant == "default":
        return {"db_id": db_id}
    return {"$and": [{"db_id": db_id}, {"schema_variant": schema_variant}]}


class VectorStoreClient:
    """Schema index/retrieval client backed by Chroma."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path = ".cache/chroma",
    ):
        self.collection_name = f"{collection_name}__{_embedding_namespace(settings.embeddings_model)}"
        self.persist_directory = str(persist_directory)
        self._namespace_key = (self.persist_directory, self.collection_name, settings.embeddings_model)
        self._indexed_db_ids: set[tuple[str, ...]] = set()
        self._index_lock = asyncio.Lock()
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for vector embeddings")
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=build_langchain_embeddings(),
        )

    def _index_schema_sync(self, db_id: str, schema: dict[str, Any], *, schema_variant: str = "default") -> None:
        docs = [
            _table_to_document(db_id, table, schema_variant=schema_variant) for table in schema.get("tables", [])
        ]
        if not docs:
            return
        if schema_variant == "default":
            ids = [f"{db_id}:{doc.metadata['table_name']}" for doc in docs]
        else:
            ids = [f"{schema_variant}:{db_id}:{doc.metadata['table_name']}" for doc in docs]
        try:
            self._vector_store.delete(ids=ids)
        except Exception:
            pass
        self._vector_store.add_documents(documents=docs, ids=ids)

    async def index_schema(self, db_id: str, schema: dict[str, Any], *, schema_variant: str = "default") -> None:
        """Index table-level documents for a specific database schema."""
        index_key = (*self._namespace_key, db_id, schema_variant)
        if index_key in self._indexed_db_ids:
            return
        async with self._index_lock:
            if index_key in self._indexed_db_ids:
                return
            await asyncio.to_thread(self._index_schema_sync, db_id, schema, schema_variant=schema_variant)
            self._indexed_db_ids.add(index_key)

    def _query_tables_sync(
        self, query: str, db_id: str, top_k: int, *, schema_variant: str = "default"
    ) -> list[dict[str, Any]]:
        docs = self._vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=_chroma_metadata_where(db_id, schema_variant=schema_variant),
        )
        return [
            {
                "table_name": doc.metadata.get("table_name", ""),
                "db_id": doc.metadata.get("db_id", db_id),
                "score": round(1.0 - i / max(len(docs), 1), 4),
                "content": doc.page_content,
            }
            for i, doc in enumerate(docs)
        ]

    async def query_tables(
        self, query: str, db_id: str, top_k: int = 15, *, schema_variant: str = "default"
    ) -> list[dict[str, Any]]:
        """Retrieve top-k table candidates from indexed schema."""
        return await asyncio.to_thread(self._query_tables_sync, query, db_id, top_k, schema_variant=schema_variant)


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

