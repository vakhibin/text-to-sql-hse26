import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fire
from app.settings import settings
from app.core.embeddings import create_embeddings
from app.schema_linker.schema_linkers.bird import BirdSchemaLinker
from app.schema_linker.vector_storages.chromadb import ChromaSchemaStore
from app.core.logger import logger


class BirdChromaCLI:
    """CLI for populating ChromaDB with BIRD benchmark schemas."""
    
    async def _populate_split(
        self,
        bird_dir: str,
        split: str,
        persist_dir: str,
        collection: str,
        db_ids: Optional[List[str]] = None,
        check_existing: bool = True,
        sleep_between: float = 0.0
    ):
        """Populate vector store with schemas from a split."""
        embeddings = await create_embeddings(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
        )
        
        vector_store = ChromaSchemaStore(
            embeddings=embeddings,
            collection_name=collection,
            persist_directory=persist_dir
        )
        await vector_store.initialize()
        
        # Load tables.json for the split
        tables_path = Path(bird_dir) / split / f"{split}_tables.json"
        with open(tables_path) as f:
            all_schemas = json.load(f)
        
        # Filter by db_ids if provided
        if db_ids:
            schemas = [s for s in all_schemas if s["db_id"] in db_ids]
        else:
            schemas = all_schemas
        
        logger.info(f"Found {len(schemas)} databases in {split} split")
        
        for schema in schemas:
            db_id = schema["db_id"]
            logger.info(f"Processing {db_id}...")
            
            linker = await BirdSchemaLinker.from_bird_dir(
                bird_dir=bird_dir,
                db_id=db_id,
                split=split,
                embeddings=embeddings
            )
            
            tables = await linker.get_full_schema()
            await vector_store.add_schema(
                db_id, 
                tables,
                check_existing=check_existing,
                sleep_between=sleep_between
            )
        
        logger.info(f"Added {len(schemas)} databases to {persist_dir} (collection: {collection})")
    
    def populate_dev(
        self,
        bird_dir: str,
        persist_dir: str = "./chroma_db",
        collection: str = "bird_schemas",
        db_ids: Optional[List[str]] = None,
        check_existing: bool = True,
        sleep: float = 0.0
    ):
        """Populate vector store with dev split schemas."""
        asyncio.run(self._populate_split(
            bird_dir=bird_dir,
            split="dev",
            persist_dir=persist_dir,
            collection=collection,
            db_ids=db_ids,
            check_existing=check_existing,
            sleep_between=sleep
        ))
    
    def populate_train(
        self,
        bird_dir: str,
        persist_dir: str = "./chroma_db",
        collection: str = "bird_schemas",
        db_ids: Optional[List[str]] = None,
        check_existing: bool = True,
        sleep: float = 0.0
    ):
        """Populate vector store with train split schemas."""
        asyncio.run(self._populate_split(
            bird_dir=bird_dir,
            split="train",
            persist_dir=persist_dir,
            collection=collection,
            db_ids=db_ids,
            check_existing=check_existing,
            sleep_between=sleep
        ))
    
    def inspect(
        self,
        persist_dir: str = "./chroma_db",
        collection: str = "bird_schemas",
        limit: int = 10,
        db_id: Optional[str] = None
    ):
        """Inspect Chroma collection contents."""
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        
        try:
            collection = client.get_collection(collection)
        except Exception as e:
            print(f"Collection '{collection}' not found: {e}")
            return
        
        count = collection.count()
        print(f"\n📊 Collection: {collection.name}")
        print(f"Total documents: {count}")
        
        where = {"db_id": db_id} if db_id else None
        results = collection.peek(limit) if not where else collection.query(
            query_texts=[""],
            n_results=min(limit, count),
            where=where
        )
        
        print(f"\n📄 Samples (first {limit}):")
        for i, (doc_id, metadata) in enumerate(zip(
            results['ids'][:limit], 
            results['metadatas'][:limit]
        )):
            print(f"\n{i + 1}. ID: {doc_id}")
            print(f"   Metadata: {metadata}")
            if 'page_content' in results:
                print(f"   Content: {results['documents'][i][:100]}...")


if __name__ == "__main__":
    fire.Fire(BirdChromaCLI)