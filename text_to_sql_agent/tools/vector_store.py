"""Chroma vector store placeholder."""

from typing import Optional


class VectorStoreClient:
    """Schema index/retrieval client (placeholder)."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    async def index_schema(self, db_id: str, schema: dict) -> None:
        _ = (db_id, schema)

    async def query_tables(self, query: str, db_id: str, top_k: int = 15) -> list[dict]:
        _ = (query, db_id, top_k)
        return []


def build_vector_store(collection_name: str) -> VectorStoreClient:
    """Build vector store client (placeholder)."""
    return VectorStoreClient(collection_name=collection_name)

