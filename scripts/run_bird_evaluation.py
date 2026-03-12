import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import fire
from app.settings import settings
from app.core.llm import create_llm
from app.core.embeddings import create_embeddings
from app.pipeline.text_to_sql import TextToSQLPipeline
from app.schema_linker.vector_storages.chromadb import ChromaSchemaStore
from app.evaluation.bird_runner import BirdEvaluationRunner, BirdRunConfig
from app.sql_generator.sql_generator import GenerationMode
from app.schema_linker.schema_linkers.bird import BirdSchemaLinker


class BirdEvaluationCLI:
    """CLI for running text-to-SQL evaluation on BIRD dataset."""
    
    async def _run(
        self,
        bird_dir: str,
        split: str,
        max_examples: Optional[int],
        no_schema_linking: bool,
        output_dir: str,
        max_concurrent: int,
        use_vector_store: bool,
        collection: str,
        top_k_tables: int,
        top_k_columns: int
    ):
        """Run evaluation."""
        llm = await create_llm(
            provider=settings.llm_provider,
            model=settings.llm_model,
            api_key=settings.openrouter_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        
        embeddings = None
        vector_store = None
        
        if not no_schema_linking and settings.openrouter_api_key:
            embeddings = await create_embeddings(
                api_key=settings.openrouter_api_key,
                model=settings.embedding_model,
            )
            
            if use_vector_store:
                vector_store = ChromaSchemaStore(
                    embeddings=embeddings,
                    collection_name=collection,
                    persist_directory="./chroma_db"
                )
                await vector_store.initialize()
                print(f"✅ Using vector store: {collection}")
        
        # Create pipeline configured directly for BIRD
        pipeline = TextToSQLPipeline(
            llm=llm,
            embeddings=embeddings,
            vector_store=vector_store,
            dataset_dir=bird_dir,
            dataset_type="bird",
            generation_mode=GenerationMode.DIRECT,
            use_schema_linking=not no_schema_linking,
            top_k_tables=top_k_tables,
            top_k_columns=top_k_columns,
        )
        
        runner = BirdEvaluationRunner(pipeline=pipeline, bird_dir=bird_dir)
        
        config = BirdRunConfig(
            split=split,
            max_examples=max_examples,
            use_schema_linking=not no_schema_linking,
            save_predictions=True,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            top_k_tables=top_k_tables,
            top_k_columns=top_k_columns,
        )
        
        return await runner.run(config)
    
    def dev(
        self,
        bird_dir: str = "databases/bird",
        max_examples: Optional[int] = None,
        no_schema_linking: bool = False,
        output_dir: str = "outputs",
        max_concurrent: int = 10,
        use_vector_store: bool = True,
        collection: str = "bird_schemas",
        top_k_tables: int = 5,
        top_k_columns: int = 10
    ):
        """Run evaluation on BIRD dev split."""
        result = asyncio.run(self._run(
            bird_dir=bird_dir,
            split="dev",
            max_examples=max_examples,
            no_schema_linking=no_schema_linking,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            use_vector_store=use_vector_store,
            collection=collection,
            top_k_tables=top_k_tables,
            top_k_columns=top_k_columns
        ))
        
        print("\n" + "=" * 50)
        print("BIRD DEV EVALUATION RESULTS")
        print("=" * 50)
        print(result.evaluation)
        print(f"\nTotal time: {result.total_time_seconds:.2f}s")
        print(f"Avg time per example: {result.avg_time_per_example:.2f}s")
        
        return result


if __name__ == "__main__":
    fire.Fire(BirdEvaluationCLI)