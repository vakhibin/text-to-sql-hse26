import argparse
import asyncio
from typing import Optional

from app.settings import settings
from app.core.llm import create_llm
from app.core.embeddings import create_embeddings
from app.schema_linker.schema_linkers.spider import SpiderSchemaLinker
from app.schema_linker.vector_storages.chromadb import ChromaSchemaStore
from app.pipeline.text_to_sql import TextToSQLPipeline
from app.sql_generator.sql_generator import GenerationMode


async def _compare_one(
    question: str,
    db_id: str,
    spider_dir: str = "databases/spider",
    evidence: Optional[str] = None,
    top_k_tables: int = 5,
    top_k_columns: int = 10,
) -> None:
    """Compare link() vs link_with_vector_store() for a single question/db."""
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    # LLM and embeddings (same as evaluation runner)
    llm = await create_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    embeddings = await create_embeddings(
        api_key=settings.openrouter_api_key,
        model=settings.embedding_model,
    )

    # Vector store (same config as pipeline/evaluation)
    vector_store = ChromaSchemaStore(
        embeddings=embeddings,
        collection_name="spider_schemas",
        persist_directory="./chroma_db",
    )
    await vector_store.initialize()

    # Schema linker
    linker = await SpiderSchemaLinker.from_spider_dir(
        spider_dir=spider_dir,
        db_id=db_id,
        embeddings=embeddings,
    )

    # 1) Schema linking with pure link()
    linked_direct = await linker.link(
        question=question,
        evidence=evidence,
        top_k_tables=top_k_tables,
        top_k_columns=top_k_columns,
    )

    # 2) Schema linking with link_with_vector_store()
    linked_vector = await linker.link_with_vector_store(
        question=question,
        vector_store=vector_store,
        db_id=db_id,
        evidence=evidence,
        top_k_tables=top_k_tables,
        top_k_columns=top_k_columns,
        use_cache=True,
    )

    print("\n=== QUESTION ===")
    print(question)
    print(f"\nDB ID: {db_id}")

    # Show selected tables/columns
    print("\n=== DIRECT LINK: TABLES ===")
    for t in linked_direct.tables:
        print(f"- {t.name}: {', '.join(t.columns)}")

    print("\n=== VECTOR LINK: TABLES ===")
    for t in linked_vector.tables:
        print(f"- {t.name}: {', '.join(t.columns)}")

    # Show table score differences
    print("\n=== TABLE SCORES (DIRECT) ===")
    for name, score in linked_direct.table_scores.items():
        print(f"{name}: {score:.4f}")

    print("\n=== TABLE SCORES (VECTOR) ===")
    for name, score in linked_vector.table_scores.items():
        print(f"{name}: {score:.4f}")

    # Prepare generator (reuse pipeline's default prompt/settings)
    pipeline = TextToSQLPipeline(
        llm=llm,
        embeddings=embeddings,
        spider_dir=spider_dir,
        generation_mode=GenerationMode.DIRECT,
        use_schema_linking=False,
        vector_store=None,
    )
    generator = pipeline._sql_generator

    schema_direct = linked_direct.to_create_table_string()
    schema_vector = linked_vector.to_create_table_string()

    # 3) SQL generation with direct-linked schema
    gen_direct = await generator.agenerate_sql(
        question=question,
        schema=schema_direct,
        db_id=db_id,
    )

    # 4) SQL generation with vector-linked schema
    gen_vector = await generator.agenerate_sql(
        question=question,
        schema=schema_vector,
        db_id=db_id,
    )

    print("\n=== GENERATED SQL (DIRECT LINK) ===")
    print(gen_direct.sql)

    print("\n=== GENERATED SQL (VECTOR LINK) ===")
    print(gen_vector.sql)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BaseSchemaLinker.link vs link_with_vector_store "
        "for a single Spider question/db.",
    )
    parser.add_argument("--question", type=str, required=True, help="NL question")
    parser.add_argument("--db-id", type=str, required=True, help="Spider DB id")
    parser.add_argument(
        "--spider-dir",
        type=str,
        default="databases/spider",
        help="Path to Spider dataset directory",
    )
    parser.add_argument(
        "--top-k-tables",
        type=int,
        default=5,
        help="Top-k tables for schema linking",
    )
    parser.add_argument(
        "--top-k-columns",
        type=int,
        default=10,
        help="Top-k columns per table for schema linking",
    )
    args = parser.parse_args()

    asyncio.run(
        _compare_one(
            question=args.question,
            db_id=args.db_id,
            spider_dir=args.spider_dir,
            top_k_tables=args.top_k_tables,
            top_k_columns=args.top_k_columns,
        )
    )


if __name__ == "__main__":
    main()

