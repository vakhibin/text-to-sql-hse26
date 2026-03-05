import asyncio
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.llm import create_llm, LLMClientError


class SQLGeneratorError(Exception):
    pass


class GenerationMode(Enum):
    DIRECT = "direct"
    COT = "cot"
    FEW_SHOT = "few_shot"

@dataclass
class SQLGenerationResult:
    sql: str
    raw_response: str
    reasoning: Optional[str] = None
    confidence: float = 0.0
    generation_time: float = 0.0
    metadata: Optional[Dict] = None

TEST_QUERY = True

class SQLGenerator:
    def __init__(
            self,
            llm: BaseChatModel,
            system_prompt: Optional[str] = None,
            generation_mode: GenerationMode = GenerationMode.DIRECT,
    ):
        """
        Initialize SQL generator.

        Args:
            llm: LangChain chat model instance
            system_prompt: System prompt for the model
            generation_mode: Mode of SQL generation (direct, cot, few_shot)
        """
        self._llm = llm
        self._system_prompt = system_prompt
        self.generation_mode = generation_mode
        self.few_shot_examples: List[Dict] = []

    @property
    def system_prompt(self) -> Optional[str]:
        return self._system_prompt

    async def agenerate_sql(
            self,
            question: str,
            schema: Union[str, Dict],
            db_id: Optional[str] = None,
            **kwargs
    ) -> SQLGenerationResult:
        """Asynchronously generate SQL from natural language question."""
        start_time = time.time()

        try:
            schema_str = self._prepare_schema(schema)
            prompt = self._build_prompt(question, schema_str, db_id, **kwargs)
            messages = self._build_messages(prompt)

            response = await self._llm.ainvoke(messages)
            response_text = response.content

            sql = self._extract_sql(response_text)

            reasoning = None
            if self.generation_mode == GenerationMode.COT:
                reasoning, sql = self._extract_cot_response(response_text)

            generation_time = time.time() - start_time

            return SQLGenerationResult(
                sql=sql,
                raw_response=response_text,
                reasoning=reasoning,
                confidence=self._estimate_confidence(sql, response_text),
                generation_time=generation_time,
                metadata={
                    "question": question,
                    "db_id": db_id,
                    "schema": schema_str,
                    "mode": self.generation_mode.value,
                }
            )

        except LLMClientError as e:
            raise SQLGeneratorError(f"LLM client error: {e}")
        except Exception as e:
            raise SQLGeneratorError(f"Error generating SQL: {e}")

    def generate_sql(
            self,
            question: str,
            schema: Union[str, Dict],
            db_id: Optional[str] = None,
            **kwargs
    ) -> SQLGenerationResult:
        """Synchronous wrapper for agenerate_sql."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.agenerate_sql(question, schema, db_id, **kwargs)
        )

    def _prepare_schema(self, schema: Union[str, Dict]) -> str:
        """Prepare schema for prompt."""
        if isinstance(schema, dict):
            schema_lines = []
            for table_name, columns in schema.items():
                if isinstance(columns, list):
                    cols_str = ", ".join(columns)
                    schema_lines.append(f"{table_name}({cols_str})")
                elif isinstance(columns, dict):
                    cols_str = ", ".join([f"{col_name} ({col_type})"
                                          for col_name, col_type in columns.items()])
                    schema_lines.append(f"{table_name}({cols_str})")
                else:
                    schema_lines.append(f"{table_name}({columns})")
            return "\n".join(schema_lines)
        elif isinstance(schema, str):
            return schema
        else:
            raise SQLGeneratorError(f"Unsupported schema type: {type(schema)}")

    def _build_prompt(
            self,
            question: str,
            schema: str,
            db_id: Optional[str] = None,
            **kwargs
    ) -> str:
        """Build prompt based on generation mode."""
        if self.generation_mode == GenerationMode.DIRECT:
            return self._build_direct_prompt(question, schema, db_id)
        elif self.generation_mode == GenerationMode.COT:
            return self._build_cot_prompt(question, schema, db_id)
        elif self.generation_mode == GenerationMode.FEW_SHOT:
            return self._build_few_shot_prompt(question, schema, db_id)
        else:
            raise SQLGeneratorError(f"Unknown generation mode: {self.generation_mode}")

    def _build_direct_prompt(
            self,
            question: str,
            schema: str,
            db_id: Optional[str] = None
    ) -> str:
        """Direct prompt for SQL generation."""
        prompt = f"""Database Schema:
{schema}

Question: {question}"""
        if db_id:
            prompt += f"\nDatabase: {db_id}"
        prompt += "\n\nSQL Query:"
        return prompt

    def _build_cot_prompt(
            self,
            question: str,
            schema: str,
            db_id: Optional[str] = None
    ) -> str:
        """Chain-of-Thought prompt."""
        prompt = f"""You are an SQL expert. Think step by step to generate the correct SQL query.

Database Schema:
{schema}

Question: {question}"""
        if db_id:
            prompt += f"\nDatabase: {db_id}"

        prompt += """

Think step by step:
1. Understand what information is needed
2. Identify which tables and columns are relevant
3. Determine the relationships between tables
4. Plan the SQL structure (SELECT, FROM, WHERE, JOINs, etc.)
5. Write the SQL query

After your reasoning, output ONLY the SQL query."""
        return prompt

    def _build_few_shot_prompt(
            self,
            question: str,
            schema: str,
            db_id: Optional[str] = None
    ) -> str:
        """Few-shot prompt with examples."""
        prompt = "Examples of converting questions to SQL:\n\n"

        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Schema: {example['schema']}\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"SQL: {example['sql']}\n\n"

        prompt += f"Now generate SQL for this:\nSchema: {schema}\n"

        if db_id:
            prompt += f"Database: {db_id}\n"

        prompt += f"Question: {question}\nSQL:"
        return prompt

    def _build_messages(self, prompt: str) -> List[tuple]:
        """Build messages for LLM."""
        messages = []

        if self.system_prompt:
            messages.append(("system", self.system_prompt))

        messages.append(("user", prompt))
        return messages

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response."""
        response = response.strip()

        sql = re.sub(r'```sql\s*', '', response)
        sql = re.sub(r'```\s*', '', sql)
        sql = re.sub(r'`', '', sql)
        sql = ' '.join(sql.split())

        if not sql.endswith(';'):
            sql += ';'

        return sql

    def _extract_cot_response(self, response: str) -> tuple[str, str]:
        """Extract reasoning and SQL from CoT response."""
        sql_patterns = [
            r'SQL:\s*(.*?)(?:\n\n|$)',
            r'```sql\s*(.*?)\s*```',
            r'SELECT.*?;',
            r'select.*?;'
        ]

        sql = None
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip() if 'SELECT' not in pattern else match.group(0).strip()
                break

        if not sql:
            lines = response.strip().split('\n')
            sql = lines[-1].strip()

        sql = self._extract_sql(sql)

        reasoning = response
        if sql in reasoning:
            reasoning = reasoning.replace(sql, '').strip()

        return reasoning, sql

    def _estimate_confidence(self, sql: str, response: str) -> float:
        """Estimate confidence in generated SQL."""
        confidence = 0.5

        heuristics = {
            'contains_select': 0.2 if 'SELECT' in sql.upper() else -0.2,
            'contains_from': 0.15 if 'FROM' in sql.upper() else -0.15,
            'contains_where': 0.1 if 'WHERE' in sql.upper() else 0,
            'has_semicolon': 0.05 if sql.endswith(';') else -0.05,
            'response_length': min(len(response) / 1000, 0.1),
            'no_explanation': 0.1 if 'explain' not in response.lower() else 0,
        }

        for _, value in heuristics.items():
            confidence += value

        return round(max(0.0, min(1.0, confidence)), 2)

    def set_generation_mode(self, mode: Union[GenerationMode, str]):
        """Set generation mode."""
        if isinstance(mode, str):
            mode = GenerationMode(mode.lower())
        self.generation_mode = mode

    def add_few_shot_example(self, question: str, schema: str, sql: str):
        """Add a few-shot example."""
        self.few_shot_examples.append({
            "question": question,
            "schema": schema,
            "sql": sql
        })

    def clear_few_shot_examples(self):
        """Clear few-shot examples."""
        self.few_shot_examples.clear()


async def main():
    """Async test function."""
    if TEST_QUERY:
        from app.settings import settings
        print("Model:", settings.llm_model)
        llm = await create_llm(
            provider=settings.llm_provider,
            model=settings.llm_model,
            api_key=settings.openrouter_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=60
        )

        generator = SQLGenerator(
            llm=llm,
            generation_mode=GenerationMode.COT,
            system_prompt="You are an expert SQL assistant. Output only SQL."
        )

        schema = \
        """
        Table: airlines
        Columns: Abbreviation (TEXT), uid (NUMBER), Airline (TEXT), Country (TEXT)
        Primary keys: uid

        Table: flights
        Columns: SourceAirport (TEXT), DestAirport (TEXT), Airline (NUMBER), FlightNo (NUMBER)
        Primary keys: Airline
        Foreign keys: DestAirport -> airports.AirportCode, SourceAirport -> airports.AirportCode
        """

        question = "Which airline has most number of flights?"

        try:
            result = await generator.agenerate_sql(
                question=question,
                schema=schema,
                db_id="flight_2"
            )

            print("SQL generated successfully!")
            print(f"SQL: {result.sql}")
            print(f"Reasoning: {result.reasoning[:100]}..." if result.reasoning else "No reasoning")
            print(f"Confidence: {result.confidence}")
            print(f"Generation time: {result.generation_time:.2f}s")

        except SQLGeneratorError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if TEST_QUERY:
        asyncio.run(main())