import asyncio
from typing import List, Optional, Tuple, Dict
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from app.schema_linker.schema_linkers.base import TableInfo
from app.core.logger import logger


class ChromaSchemaStore:
    """Vector store for database schemas using Chroma."""

    def __init__(
            self,
            embeddings: Embeddings,
            collection_name: str = "schema_store",
            persist_directory: Optional[str] = "./chroma_db"
    ):
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._vector_store = None
        self._logger = logger

    async def initialize(self):
        """Initialize connection to ChromaDB."""
        try:
            self._vector_store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_directory
            )
        except Exception as e:
            self._logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

    async def add_schema(
            self,
            db_id: str,
            tables: List[TableInfo],
            check_existing: bool = True,
            sleep_between: float = 0.0
    ):
        """Add schema elements to vector store."""
        if not self._vector_store:
            await self.initialize()

        existing_ids = set()
        if check_existing:
            try:
                existing = self._vector_store.get()
                existing_ids = set(existing['ids'])
            except:
                pass

        documents = []
        ids = []

        # Add tables
        for table in tables:
            doc_text = f"table {table.name}"
            table_id = f"{db_id}_table_{table.name}"

            if not check_existing or table_id not in existing_ids:
                documents.append(Document(
                    page_content=doc_text,
                    metadata={
                        "db_id": db_id,
                        "type": "table",
                        "table_name": table.name,
                        "columns": ",".join(table.columns)
                    }
                ))
                ids.append(table_id)

            # Column ID format must match base.link_with_vector_store (no col_idx;
            # re-add col_idx only if you re-populate all collections)
            for col_idx, col in enumerate(table.columns):
                col_id = f"{db_id}_column_{table.name}_{col}"
                if not check_existing or col_id not in existing_ids:
                    col_text = f"column {col} in table {table.name}"
                    documents.append(Document(
                        page_content=col_text,
                        metadata={
                            "db_id": db_id,
                            "type": "column",
                            "table_name": table.name,
                            "column_name": col,
                            "column_type": table.column_types.get(col, "UNKNOWN"),
                            "column_index": col_idx
                        }
                    ))
                    ids.append(col_id)

                    if sleep_between > 0:
                        await asyncio.sleep(sleep_between)

        # Add to vector store in batches
        if documents:
            await self._vector_store.aadd_documents(documents, ids=ids)
            self._logger.info(f"Added {len(documents)} schema elements for database {db_id}")
        else:
            self._logger.info(f"No new elements to add for database {db_id}")

    async def search_relevant(
            self,
            query: str,
            top_k_tables: int = 5,
            top_k_columns: int = 10,
            db_id: Optional[str] = None
    ) -> Tuple[List[Document], List[Document]]:
        """Search for relevant tables and columns."""
        if not self._vector_store:
            await self.initialize()

        # Build filter if db_id specified
        filter_dict = {}
        if db_id:
            filter_dict = {"db_id": db_id}

        # Search for tables
        table_results = await self._vector_store.asimilarity_search(
            query,
            k=top_k_tables * 2,  # Get more to account for filtering
            filter={"type": "table", **filter_dict}
        )

        # Search for columns
        column_results = await self._vector_store.asimilarity_search(
            query,
            k=top_k_columns * 2,
            filter={"type": "column", **filter_dict}
        )

        return table_results, column_results

    async def search_relevant_by_embedding(
        self,
        query_embedding: List[float],
        top_k_tables: int = 5,
        top_k_columns: int = 10,
        db_id: Optional[str] = None
    ) -> Tuple[List[Document], List[Document]]:
        """Search for relevant tables and columns using query embedding."""
        if not self._vector_store:
            await self.initialize()

        # 1. Сначала ищем таблицы (глобально)
        table_filter = {"$and": [{"type": {"$eq": "table"}}]}
        if db_id:
            table_filter["$and"].append({"db_id": {"$eq": db_id}})
        
        table_results = await self._vector_store.asimilarity_search_by_vector(
            embedding=query_embedding,
            k=top_k_tables,
            filter=table_filter
        )
        
        # 2. Для каждой найденной таблицы ищем её колонки
        column_results = []
        for table_doc in table_results:
            table_name = table_doc.metadata["table_name"]
            
            column_filter = {"$and": [
                {"type": {"$eq": "column"}},
                {"db_id": {"$eq": db_id}},
                {"table_name": {"$eq": table_name}}
            ]}
            
            cols = await self._vector_store.asimilarity_search_by_vector(
                embedding=query_embedding,
                k=top_k_columns,
                filter=column_filter
            )
            column_results.extend(cols)
        
        return table_results, column_results

    async def get_embeddings_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """Fetch stored embeddings for given Chroma document ids.

        Returns:
            Mapping from id -> embedding (list[float]). Missing ids are omitted.
        """
        if not ids:
            return {}

        if not self._vector_store:
            await self.initialize()

        try:
            data = self._vector_store.get(ids=ids, include=["embeddings"])
        except TypeError:
            # Some versions may not support include=...; fallback to default get()
            data = self._vector_store.get(ids=ids)

        out: Dict[str, List[float]] = {}
        if not data:
            return out

        got_ids = data.get("ids")
        got_embs = data.get("embeddings")

        if got_ids is None:
            got_ids = []
        if got_embs is None:
            got_embs = []

        for _id, emb in zip(got_ids, got_embs):
            if emb is None:
                continue
            out[_id] = list(emb)

        return out