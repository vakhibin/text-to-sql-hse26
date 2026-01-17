import re
from typing import Dict, List, Optional, Union

from app.core.llm import OllamaLLMClient, OpenRouterLLMClient, LLMClientError
from app.sql_generator.sql_generator import SQLGeneratorError, SQLGenerationResult

class SQLGenerator:
    def __init__(
            self,
            llm_client: Union[OllamaLLMClient, OpenRouterLLMClient],
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,
    ):
        """
        Initilizing SQL generator

        Args:
            llm_client: client for working woth llm
            model: LLM model for SQL generation
            system_prompt: system prompt
        """
        self._llm_client = llm_client

        # Устанавливаем модель если передана
        if model:
            self._llm_client.set_model(model)
        elif not hasattr(self._llm_client, '_model') or not self._llm_client._model:
            raise SQLGeneratorError("Model must be specified either in client or generator")

        # Системный промпт по умолчанию
        self._system_prompt = system_prompt


    def generate_sql(
            self,
            question: str,
            schema: Union[str, Dict],
            db_id: Optional[str] = None,
            temperature: float = 0.1,
            max_tokens: int = 512,
            **kwargs
    ) -> SQLGenerationResult:
        try:
            # Подготавливаем схему
            schema_str = self._prepare_schema(schema)

            # Формируем промпт в зависимости от режима
            prompt = self._build_prompt(question, schema_str, db_id, **kwargs)

            # Формируем сообщения для LLM
            messages = self._build_messages(prompt)

            # Вызываем LLM
            response = self._llm_client.call_llm(messages)

            # Извлекаем SQL из ответа
            sql = self._extract_sql(response)

            # Извлекаем reasoning если используется CoT
            reasoning = None
            if self.generation_mode == GenerationMode.COT:
                reasoning, sql = self._extract_cot_response(response)

            generation_time = time.time() - start_time

            return SQLGenerationResult(
                sql=sql,
                raw_response=response,
                reasoning=reasoning,
                confidence=self._estimate_confidence(sql, response),
                generation_time=generation_time,
                metadata={
                    "question": question,
                    "db_id": db_id,
                    "schema": schema_str,
                    "mode": self.generation_mode.value,
                    "model": self._llm_client._model,
                    "temperature": temperature
                }
            )

        except LLMClientError as e:
            raise SQLGeneratorError(f"LLM client error: {e}")
        except Exception as e:
            raise SQLGeneratorError(f"Error generating SQL: {e}")


    def _prepare_schema(self, schema: Union[str, Dict]) -> str:
        """Подготовка схемы для промпта."""
        if isinstance(schema, dict):
            # Преобразуем словарь в читаемый формат
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
        """Построение промпта в зависимости от режима генерации."""

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
        """Прямой промпт для генерации SQL."""
        prompt = f"""
Database Schema:
{schema}

Question: {question}
"""
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
        """Chain-of-Thought промпт."""
        prompt = f"""
You are an SQL expert. Think step by step to generate the correct SQL query.

Database Schema:
{schema}

Question: {question}
"""
        if db_id:
            prompt += f"\nDatabase: {db_id}"

        prompt += """
Think step by step:
1. Understand what information is needed
2. Identify which tables and columns are relevant
3. Determine the relationships between tables
4. Plan the SQL structure (SELECT, FROM, WHERE, JOINs, etc.)
5. Write the SQL query

After your reasoning, output ONLY the SQL query.
"""
        return prompt

    def _build_few_shot_prompt(
            self,
            question: str,
            schema: str,
            db_id: Optional[str] = None
    ) -> str:
        """Few-shot промпт с примерами."""
        prompt = "Examples of converting questions to SQL:\n\n"

        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Schema: {example['schema']}\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"SQL: {example['sql']}\n\n"

        prompt += f"Now generate SQL for this:\n"
        prompt += f"Schema: {schema}\n"

        if db_id:
            prompt += f"Database: {db_id}\n"

        prompt += f"Question: {question}\n"
        prompt += "SQL:"

        return prompt

    def _build_messages(self, prompt: str) -> List[Dict]:
        """Формирование сообщений для LLM."""
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages

    def _extract_sql(self, response: str) -> str:
        """Извлечение SQL из ответа LLM."""
        # Удаляем лишние пробелы
        response = response.strip()

        # Удаляем markdown code blocks если есть
        sql = re.sub(r'```sql\s*', '', response)
        sql = re.sub(r'```\s*', '', sql)
        sql = re.sub(r'`', '', sql)

        # Удаляем лишние переносы и пробелы
        sql = ' '.join(sql.split())

        # Гарантируем, что SQL заканчивается точкой с запятой
        if not sql.endswith(';'):
            sql += ';'

        return sql

    def _extract_cot_response(self, response: str) -> tuple[str, str]:
        """Извлечение reasoning и SQL из CoT ответа."""
        # Ищем SQL в ответе
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
            # Если не нашли SQL, берем последнюю строку
            lines = response.strip().split('\n')
            sql = lines[-1].strip()

        # Очищаем SQL
        sql = self._extract_sql(sql)

        # Reasoning - все что не SQL
        reasoning = response
        if sql in reasoning:
            reasoning = reasoning.replace(sql, '').strip()

        return reasoning, sql

    def _estimate_confidence(self, sql: str, response: str) -> float:
        """Оценка уверенности в сгенерированном SQL."""
        confidence = 0.5  # Базовая уверенность

        # Эвристики для оценки уверенности
        heuristics = {
            'contains_select': 0.2 if 'SELECT' in sql.upper() else -0.2,
            'contains_from': 0.15 if 'FROM' in sql.upper() else -0.15,
            'contains_where': 0.1,
            'has_semicolon': 0.05 if sql.endswith(';') else -0.05,
            'response_length': min(len(response) / 1000, 0.1),  # Длинный ответ может быть лучше
            'no_explanation': 0.1 if 'explain' not in response.lower() else 0,
        }

        # Применяем эвристики
        for _, value in heuristics.items():
            confidence += value

        # Ограничиваем от 0 до 1
        confidence = max(0.0, min(1.0, confidence))

        return round(confidence, 2)

    def set_generation_mode(self, mode: Union[GenerationMode, str]):
        """Установка режима генерации."""
        if isinstance(mode, str):
            mode = GenerationMode(mode.lower())
        self.generation_mode = mode

    def add_few_shot_example(self, question: str, schema: str, sql: str):
        """Добавление few-shot примера."""
        self.few_shot_examples.append({
            "question": question,
            "schema": schema,
            "sql": sql
        })

    def clear_few_shot_examples(self):
        """Очистка few-shot примеров."""
        self.few_shot_examples.clear()

# Пример использования
if __name__ == "__main__":
    # Инициализация клиента
    ollama_client = OllamaLLMClient(api_key=None, model="codellama:7b-instruct")

    # Инициализация генератора
    generator = SQLGenerator(
        llm_client=ollama_client,
        generation_mode=GenerationMode.COT,
        system_prompt="You are an expert SQL assistant. Output only SQL."
    )

    # Пример схемы
    schema = """
    employees (id, name, department_id, hire_date, salary)
    departments (id, name, manager_id)
    projects (id, name, department_id, budget, start_date, end_date)
    """

    # Вопрос
    question = "Show all employees in the Sales department with salary above 50000"

    # Генерация SQL
    try:
        result = generator.generate_sql(
            question=question,
            schema=schema,
            db_id="company_db"
        )

        print("✅ SQL generated successfully!")
        print(f"SQL: {result.sql}")
        print(f"Reasoning: {result.reasoning[:100]}..." if result.reasoning else "No reasoning")
        print(f"Confidence: {result.confidence}")
        print(f"Generation time: {result.generation_time:.2f}s")

    except SQLGeneratorError as e:
        print(f"❌ Error: {e}")

    # Пример с OpenRouter
    print("\n--- Testing with OpenRouter ---")

    # Инициализация OpenRouter клиента (замените API_KEY на реальный)
    # openrouter_client = OpenRouterLLMClient(api_key="your-api-key", model="meta-llama/llama-3.1-70b-instruct")

    # openrouter_generator = SQLGenerator(
    #     llm_client=openrouter_client,
    #     generation_mode=GenerationMode.FEW_SHOT
    # )

    # # Добавляем custom few-shot пример
    # openrouter_generator.add_few_shot_example(
    #     question="Count projects by department",
    #     schema="departments(id, name), projects(id, name, department_id)",
    #     sql="SELECT d.name, COUNT(p.id) as project_count FROM departments d LEFT JOIN projects p ON d.id = p.department_id GROUP BY d.id, d.name;"
    # )

    # # Генерация с повторными попытками
    # try:
    #     result = openrouter_generator.generate_sql_with_retry(
    #         question="Show total budget by department",
    #         schema=schema,
    #         max_attempts=2
    #     )
    #     print(f"Generated SQL: {result.sql}")
    # except SQLGeneratorError as e:
    #     print(f"Error: {e}")