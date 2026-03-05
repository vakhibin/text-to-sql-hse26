import json
import asyncio
from pathlib import Path
from typing import Optional, List

from fire import Fire

from app.settings import settings
from app.schema_linker.vector_storages.chromadb import ChromaSchemaStore
from app.core.embeddings import create_embeddings
from app.schema_linker.schema_linkers.spider import SpiderSchemaLinker
from app.core.logger import logger


class ChromaSpiderCLI:
    async def _populate_one(
            self,
            spider_dir: str,
            db_id: str,
            persist_dir: str,
            collection_name: str,
    ):
        embeddings = await create_embeddings(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
        )
        vector_storage = ChromaSchemaStore(
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        await vector_storage.initialize()

        linker = await SpiderSchemaLinker.from_spider_dir(
            spider_dir=spider_dir,
            db_id=db_id,
            embeddings=embeddings
        )

        tables = await linker.get_full_schema()
        await vector_storage.add_schema(db_id, tables)
        logger.info(
            f"Added {db_id} to {persist_dir} (collection: {collection_name})"
        )

    async def _populate_all(
            self,
            spider_dir: str,
            persist_dir: str,
            collection_name: str,
            db_ids: Optional[List[str]] = None,
            skip_existing: bool = False,
            sleep_between: float = 0.0
    ):
        """Populate all or selected databases."""
        embeddings = await create_embeddings(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
        )

        vector_store = ChromaSchemaStore(
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        await vector_store.initialize()

        # Load all schemas
        with open(Path(spider_dir) / "tables.json") as f:
            all_schemas = json.load(f)

        # Filter by db_ids if provided
        if db_ids:
            schemas = [s for s in all_schemas if s["db_id"] in db_ids]
        else:
            schemas = all_schemas

        for schema in schemas:
            db_id = schema["db_id"]
            print(f"Processing {db_id}...")

            linker = await SpiderSchemaLinker.from_spider_dir(
                spider_dir=spider_dir,
                db_id=db_id,
                embeddings=embeddings
            )

            tables = await linker.get_full_schema()
            await vector_store.add_schema(
                db_id,
                tables,
                check_existing=skip_existing,
                sleep_between=sleep_between
            )

        logger.info(
            f"Added {len(schemas)} databases to {persist_dir} "
            f"(collection: {collection_name})"
        )

    def populate_one(
            self,
            spider_dir: str,
            db_id: str,
            persist_dir: str = "./chroma_db",
            collection: str = "spider_schemas"
    ):
        """Populate single database."""
        asyncio.run(
            self._populate_one(spider_dir, db_id, persist_dir, collection)
        )

    def populate_all(
            self,
            spider_dir: str,
            persist_dir: str = "./chroma_db",
            collection: str = "spider_schemas",
            db_ids: Optional[List[str]] = None,
            skip_existing: bool = False,
            sleep_between: float = 0.0
    ):
        """Populate all Spider databases."""
        asyncio.run(self._populate_all(
            spider_dir,
            persist_dir,
            collection,
            db_ids,
            skip_existing=skip_existing,
            sleep_between=sleep_between
        ))

    def inspect(
            self,
            persist_dir: str = "./chroma_db",
            collection: str = "spider_schemas",
            limit: int = 10,
            db_id: Optional[str] = None
    ):
        """Inspect Chroma collection contents."""
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)

        try:
            collection = client.get_collection(collection)
        except Exception:
            print(f"Collection '{collection}' not found")
            return

        # Statistic
        count = collection.count()
        print(f"\n📊 Collection: {collection.name}")
        print(f"Total documents: {count}")

        # Get data
        where = {"db_id": db_id} if db_id else None
        results = collection.peek(limit) if not where else collection.query(
            query_texts=[""],
            n_results=min(limit, count),
            where=where
        )

        # Examples
        print(f"\n📄 Samples (first {limit}):")
        for i, (doc_id, metadata) in enumerate(
            zip(results["ids"][:limit], results["metadatas"][:limit])
        ):
            print(f"\n{i + 1}. ID: {doc_id}")
            print(f"   Metadata: {metadata}")


if __name__ == "__main__":
    Fire(ChromaSpiderCLI)