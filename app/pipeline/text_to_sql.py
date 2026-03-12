import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.core.db import SQLiteDatabase
from app.schema_linker import SpiderSchemaLinker, LinkedSchema
from app.sql_generator.sql_generator import SQLGenerator, SQLGenerationResult, GenerationMode
from app.core.llm import NoTokenProvidedError
from app.core.logger import logger

@dataclass
class PipelineResult:
    """Result of text-to-SQL pipeline execution."""
    question: str
    db_id: str
    predicted_sql: str
    gold_sql: Optional[str] = None

    # Execution results
    predicted_result: Optional[List] = None
    gold_result: Optional[List] = None
    execution_match: Optional[bool] = None
    execution_error: Optional[str] = None

    # Intermediate results
    linked_schema: Optional[LinkedSchema] = None
    generation_result: Optional[SQLGenerationResult] = None

    # Timing
    schema_linking_time: float = 0.0
    generation_time: float = 0.0
    execution_time: float = 0.0
    total_time: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextToSQLPipeline:
    """Asynchronous pipeline for text-to-SQL generation on Spider dataset."""

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Optional[Embeddings] = None,
        vector_store: Optional['ChromaSchemaStore'] = None,
        dataset_dir: str | Path = "databases/spider",
        dataset_type: str = "spider",
        generation_mode: GenerationMode = GenerationMode.DIRECT,
        system_prompt: Optional[str] = None,
        use_schema_linking: bool = True,
        top_k_tables: int = 5,
        top_k_columns: int = 10,
    ):
        """
        Initialize asynchronous text-to-SQL pipeline.

        Args:
            llm: LangChain chat model for SQL generation
            embeddings: Optional embeddings for schema linking
            spider_dir: Path to Spider dataset directory
            generation_mode: SQL generation mode (direct, cot, few_shot)
            system_prompt: Optional system prompt for SQL generator
            use_schema_linking: Whether to use embedding-based schema linking
            top_k_tables: Number of tables to select in schema linking
            top_k_columns: Number of columns per table in schema linking
        """
        self._llm = llm
        self._embeddings = embeddings
        self._dataset_dir = Path(dataset_dir)
        self._dataset_type = dataset_type.lower()
        self._use_schema_linking = use_schema_linking and embeddings is not None
        self._top_k_tables = top_k_tables
        self._top_k_columns = top_k_columns

        # Initialize SQL generator
        self._sql_generator = SQLGenerator(
            llm=llm,
            system_prompt=system_prompt or self._default_system_prompt(),
            generation_mode=generation_mode,
        )

        # Cache for schema linkers (one per db_id)
        self._schema_linkers: Dict[str, Any] = {} 
        self._linker_lock = asyncio.Lock()
        self._vector_store = vector_store

    def _default_system_prompt(self) -> str:
        return """You are an expert SQL assistant specialized in generating accurate SQL queries from natural language questions.

RULES:
1. Output ONLY valid SQL code - no explanations, no markdown, no extra text
2. Use proper SQL syntax compatible with SQLite
3. Quote column/table names that contain spaces or special characters with backticks (e.g. `Column Name`)
4. Extract and use specific values from the question (numbers, names, dates, etc.)
5. Handle NULL values appropriately with IS NULL/IS NOT NULL
6. Use explicit JOIN syntax (INNER JOIN, LEFT JOIN) with ON clauses
7. Include all necessary WHERE conditions based on the question
8. End each statement with a semicolon

CRITICAL: Never include any reasoning, commentary, or explanations in your output. Only SQL code."""

    async def _get_schema_linker(self, db_id: str) -> SpiderSchemaLinker:
        """Get or create schema linker for a database asynchronously."""
        if db_id not in self._schema_linkers:
            async with self._linker_lock:
                if db_id not in self._schema_linkers:
                    if self._dataset_type == "spider":
                        from app.schema_linker.schema_linkers.spider import SpiderSchemaLinker
                        self._schema_linkers[db_id] = await SpiderSchemaLinker.from_spider_dir(
                            spider_dir=self._dataset_dir,
                            db_id=db_id,
                            embeddings=self._embeddings if self._use_schema_linking else None,
                        )
                    else:  # bird
                        from app.schema_linker.schema_linkers.bird import BirdSchemaLinker
                        # Для BIRD нужно знать split, возьмем из атрибута или передадим
                        split = getattr(self, "_bird_split", "dev")
                        self._schema_linkers[db_id] = await BirdSchemaLinker.from_bird_dir(
                            bird_dir=self._dataset_dir,
                            db_id=db_id,
                            split=split,
                            embeddings=self._embeddings if self._use_schema_linking else None,
                        )


        return self._schema_linkers[db_id]

    def _get_db_path(self, db_id: str) -> Path:
        """Get path to SQLite database."""
        if self._dataset_type == "spider":
            return self._dataset_dir / "database" / db_id / f"{db_id}.sqlite"
        else:  # bird
            # Для BIRD нужно знать split
            split = getattr(self, "_bird_split", "dev")
            return self._dataset_dir / f"{split}_databases" / db_id / f"{db_id}.sqlite"

    async def run(
        self,
        question: str,
        db_id: str,
        gold_sql: Optional[str] = None,
        evidence: Optional[str] = None,
        execute: bool = True,
    ) -> PipelineResult:
        """
        Run the text-to-SQL pipeline asynchronously.

        Args:
            question: Natural language question
            db_id: Database ID
            gold_sql: Optional gold SQL for evaluation
            evidence: Optional evidence/hint (for BIRD dataset)
            execute: Whether to execute the generated SQL

        Returns:
            PipelineResult with all outputs
        """
        total_start = time.time()
        result = PipelineResult(
            question=question,
            db_id=db_id,
            predicted_sql="",
            gold_sql=gold_sql,
        )

        # Step 1: Schema linking
        schema_start = time.time()
        schema_linker = await self._get_schema_linker(db_id)

        if self._use_schema_linking:
            linked_schema = await schema_linker.link_with_vector_store(
                question=question,
                evidence=evidence,
                vector_store=self._vector_store,
                top_k_tables=self._top_k_tables,
                top_k_columns=self._top_k_columns,
                db_id=db_id,
                use_cache=True
            )
        else:
            # Use full schema
            full_schema = await schema_linker.get_full_schema()  # убрать to_thread
            linked_schema = LinkedSchema(tables=full_schema)

        result.linked_schema = linked_schema
        result.schema_linking_time = time.time() - schema_start

        # Step 2: SQL generation (never pass empty schema)
        gen_start = time.time()
        if not linked_schema.tables:
            full_schema = await schema_linker.get_full_schema()
            linked_schema = LinkedSchema(tables=full_schema)
            result.linked_schema = linked_schema
        schema_str = linked_schema.to_create_table_string()

        generation_result = await self._sql_generator.agenerate_sql(
            question=question,
            schema=schema_str,
            db_id=db_id,
            evidence=evidence,
        )

        result.predicted_sql = generation_result.sql
        result.generation_result = generation_result
        result.generation_time = time.time() - gen_start

        if (result.predicted_sql or "").strip() in ("", ";"):
            logger.warning(
                "Empty or trivial SQL for db_id=%s; schema_len=%d; raw_response_len=%d; raw_preview=%s",
                db_id,
                len(schema_str),
                len(generation_result.raw_response or ""),
                (generation_result.raw_response or "")[:600],
            )

        # Step 3: Execution (optional)
        if execute:
            exec_start = time.time()
            db_path = self._get_db_path(db_id)

            try:
                async with SQLiteDatabase(str(db_path)) as db:
                    # Execute predicted SQL
                    try:
                        result.predicted_result = await db.execute_query(result.predicted_sql)
                    except Exception as e:
                        result.execution_error = f"Predicted SQL error: {str(e)}"

                    # Execute gold SQL if provided
                    if gold_sql:
                        try:
                            result.gold_result = await db.execute_query(gold_sql)
                        except Exception as e:
                            gold_error = f" Gold SQL error: {str(e)}"
                            result.execution_error = (result.execution_error or "") + gold_error

                    # Compare results
                    if result.predicted_result is not None and result.gold_result is not None:
                        result.execution_match = await asyncio.to_thread(
                            self._compare_results,
                            result.predicted_result,
                            result.gold_result,
                        )

            except Exception as e:
                result.execution_error = f"Database error: {str(e)}"

            result.execution_time = time.time() - exec_start

        result.total_time = time.time() - total_start
        return result

    def _compare_results(self, predicted: List, gold: List) -> bool:
        """Compare execution results for equality."""
        try:
            pred_set = set(
                tuple(row) if isinstance(row, (list, tuple)) else (row,)
                for row in predicted
            )
            gold_set = set(
                tuple(row) if isinstance(row, (list, tuple)) else (row,)
                for row in gold
            )
            return pred_set == gold_set
        except (TypeError, ValueError):
            return predicted == gold

    def set_generation_mode(self, mode: GenerationMode | str):
        """Set SQL generation mode."""
        self._sql_generator.set_generation_mode(mode)

    def add_few_shot_example(self, question: str, schema: str, sql: str):
        """Add a few-shot example."""
        self._sql_generator.add_few_shot_example(question, schema, sql)


async def main():
    """Async example usage."""
    from app.core.llm import create_llm
    from app.core.embeddings import create_embeddings
    from app.settings import settings
    from app.schema_linker.vector_storages.chromadb import ChromaSchemaStore

    if not settings.openrouter_api_key:
        raise NoTokenProvidedError("API key wasn't provided")

    # Initialize components asynchronously
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

    vector_store_spider = ChromaSchemaStore(
        embeddings=embeddings,
        collection_name="spider_schemas",
        persist_directory="./chroma_db",
    )
    await vector_store_spider.initialize()

    vector_store_bird = ChromaSchemaStore(
        embeddings=embeddings,
        collection_name="bird_schemas",
        persist_directory="./chroma_db",
    )
    await vector_store_bird.initialize()

    # Create pipeline
    pipeline_spider = TextToSQLPipeline(
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store_spider,
        dataset_dir="databases/spider",
        dataset_type="spider",
        generation_mode=GenerationMode.DIRECT,
        use_schema_linking=True,
        top_k_tables=5,
    )
    llm_bird = await create_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    pipeline_bird = TextToSQLPipeline(
        llm=llm_bird,
        embeddings=embeddings,
        vector_store=vector_store_bird,
        dataset_dir="databases/bird",
        dataset_type="bird",
        generation_mode=GenerationMode.DIRECT,
        use_schema_linking=True,
        top_k_tables=5,
    )

    # Run BIRD first so its LLM call is not "second" (avoids empty response from reuse)
    result_bird = await pipeline_bird.run(
        question="What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        db_id="california_schools",
        gold_sql="SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
        evidence="Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
        execute=True,
    )
    result_spider = await pipeline_spider.run(
        question="How many singers do we have?",
        db_id="concert_singer",
        gold_sql="SELECT count(*) FROM singer",
        execute=True,
    )

    print(f"Question bird: {result_bird.question}")
    print(f"Predicted SQL bird: {result_bird.predicted_sql}")
    print(f"Gold SQL bird: {result_bird.gold_sql}")
    print(f"Execution match bird: {result_bird.execution_match}")
    print(f"Total time bird: {result_bird.total_time:.2f}s")

    print(f"Question spider: {result_spider.question}")
    print(f"Predicted SQL spider: {result_spider.predicted_sql}")
    print(f"Gold SQL spider: {result_spider.gold_sql}")
    print(f"Execution match spider: {result_spider.execution_match}")
    print(f"Total time spider: {result_spider.total_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
