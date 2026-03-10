# Архитектурный обзор text-to-sql агента

## 1. Общая схема

Агент реализован как ориентированный ациклический граф (DAG) с одним циклическим ребром (refiner -> refiner). Граф компилируется LangGraph в runtime и исполняется асинхронно.

```
                START
                  |
            [1. Selector]
                  |
            (failed?) --yes--> END
                  |no
            [2. Decomposer]
                  |
            [3. Generator]
                  |
            (пустые кандидаты?) --yes--> END
                  |no
            [4. Execution Filter]
                  |
            (ни одного валидного?) --yes--> END
                  |no
            [5. Judge]
                  |
            (нет best_sql?) --yes--> END
                  |no
            [6. Refiner] <----+
                  |           |
            (ошибка исполнения |
             И attempts < 3?) |
                  |yes--------+
                  |no
                 END
```

Каждый узел -- async-функция, принимающая и возвращающая `SQLAgentState` (TypedDict). Состояние иммутабельно: каждый узел возвращает новый словарь через `{**state, ...}`.

---

## 2. Общее состояние (`SQLAgentState`)

Все стадии работают с единым типизированным словарём. Ключевые группы полей:

| Группа | Поля | Кто пишет |
|---|---|---|
| Вход | `question`, `db_id`, `evidence` | `make_initial_state()` |
| Selector | `full_schema`, `filtered_schema` | Selector |
| Decomposer | `complexity`, `sub_questions` | Decomposer |
| Generator | `candidates` | Generator |
| Exec Filter | `valid_candidates` | Execution Filter |
| Judge | `best_sql`, `judge_reasoning` | Judge |
| Refiner | `final_sql`, `execution_result`, `refine_attempts`, `error_message` | Refiner |
| Оркестрация | `stage_status`, `stage_timings`, `trace_id`, `warnings` | Все стадии |

`stage_status` -- словарь `{stage_name: "pending" | "running" | "success" | "failed"}`. Используется для условной маршрутизации: например, `_route_after_selector` проверяет `stage_status["selector"] == "failed"` и выходит в END.

`warnings` -- накопительный список строк. Каждая стадия **дописывает**, не перезаписывая. Собираются в итоговый JSON результатов.

---

## 3. Стадия 1: Selector (Schema Linking)

**Файлы**: `agents/selector.py`, `prompts/selector.py`, `tools/vector_store.py`, `tools/schema_loader.py`

### 3.1. Загрузка схемы

`load_schema(db_id)` парсит `tables.json` из Spider-датасета и строит структурированный словарь:

```python
{
  "db_id": "concert_singer",
  "tables": [
    {
      "name": "singer",
      "columns": [{"name": "Singer_ID", "type": "NUMBER", "sample_values": ["1", "2", "3"]}, ...],
      "primary_keys": ["Singer_ID"],
      "foreign_keys": [{"column": "...", "ref_table": "...", "ref_column": "..."}]
    },
    ...
  ],
  "db_path": "databases/spider/database/concert_singer/concert_singer.sqlite"
}
```

Для каждой колонки выполняется `SELECT col FROM table WHERE col IS NOT NULL LIMIT 3` в SQLite -- это даёт примеры реальных значений (`sample_values`). Примеры помогают LLM понять типы данных и формат значений (например, что `Year` -- строка `"2014"`, а не число).

### 3.2. Векторная индексация

`VectorStoreClient` индексирует каждую таблицу как отдельный документ в ChromaDB. Формат документа:

```
table=singer
columns=Singer_ID:NUMBER samples=['1', '2', '3']; Name:TEXT samples=['Joe Sharp', ...]
primary_keys=['Singer_ID']
foreign_keys=[]
```

Индексация происходит **один раз на db_id** (защита через `asyncio.Lock` + `_indexed_db_ids` set). Повторные вызовы для того же `db_id` -- no-op.

Эмбеддинги: `openai/text-embedding-3-large` через OpenRouter.

### 3.3. Retrieval

`similarity_search(query=question, k=15, filter={"db_id": db_id})` -- берём Top-15 таблиц по косинусной близости к вопросу. На базах с < 15 таблицами возвращаются все.

Score заменён на позиционный ранг (`1.0 - i/len(docs)`) вместо raw cosine similarity -- убирает предупреждения LangChain о score вне [0,1].

### 3.4. LLM-реранкинг

Top-15 кандидатов передаются LLM (primary generator, temperature=0.0) с промптом:

```
Select 3 to 5 table names that are most relevant to answer the question.
Return STRICT JSON: {"selected_tables": [...], "reasoning": "..."}
```

Ответ парсится через многоуровневую стратегию (`_safe_parse_selected_tables`):
1. Strict JSON
2. JSON в code fence (````json ... ````)
3. Первый JSON-объект в тексте
4. Plain list `[table1, table2]`
5. Построчный парсинг

### 3.5. Padding-политика

После парсинга -- фильтрация: оставляем только имена, которые реально есть среди candidate_names. Далее:

- Если реранкер вернул 0 валидных имён -> **fallback**: берём top-`min` из vector candidates.
- Если вернул меньше `min` (по умолчанию 3) -> **padding**: дополняем из vector candidates, сохраняя выбор реранкера.
- Если вернул >= `min` -> обрезаем до `max` (по умолчанию 5).

### 3.6. mSchema

Отфильтрованные таблицы форматируются в компактный mSchema:

```
singer(Singer_ID:NUMBER sample=['1', '2', '3'], Name:TEXT sample=['Joe Sharp', ...]) pk=['Singer_ID']
singer_in_concert(concert_ID:NUMBER, Singer_ID:TEXT) pk=['concert_ID'] fk=[Singer_ID->singer.Singer_ID; concert_ID->concert.concert_ID]
```

Этот формат передаётся во все последующие стадии, которым нужна информация о схеме.

---

## 4. Стадия 2: Decomposer

**Файлы**: `agents/decomposer.py`, `prompts/decomposer.py`

### 4.1. Что делает

Принимает `question` и опциональный `evidence`, возвращает:
- `complexity`: `"simple"` | `"moderate"` | `"complex"` | `"unknown"`
- `sub_questions`: список подвопросов (может быть пустым)

### 4.2. Как complexity используется дальше

`complexity` и `sub_questions` передаются **в промпт генератора** как подсказки:

```
Complexity:
moderate

Decomposition hints:
- Which table stores singer information?
- How to filter by country France?
- Which aggregate functions needed for avg, min, max?
```

Это помогает генератору структурировать рассуждение, особенно для complex-запросов с вложенными подзапросами, GROUP BY + HAVING, INTERSECT/EXCEPT.

Для `simple` -- подвопросы не генерируются (пустой список), в промпте будет `- (none)`. Генератор получает сигнал, что запрос простой и не требует сложной декомпозиции.

### 4.3. Парсинг ответа

LLM вызывается с temperature=0.0 и системным сообщением "Return strict JSON only." Ожидаемый формат:

```json
{
  "complexity": "moderate",
  "sub_questions": ["Which table has age info?", "How to compute average?"],
  "reasoning": "Question requires aggregation with filter"
}
```

Парсинг: `_extract_json_blob` (снятие code fence, поиск `{...}`) -> `json.loads` -> валидация. Если complexity не из `{"simple", "moderate", "complex"}` -- ставится `"unknown"`.

### 4.4. Fallback-поведение

- JSON не распарсился -> `complexity="unknown"`, `sub_questions=[]`, warning в лог.
- Complexity `"complex"` но подвопросов нет -> warning, но pipeline продолжает.
- Complexity `"simple"` -> подвопросы принудительно `[]` (даже если LLM вернул что-то).
- Полный exception (сеть, timeout) -> `complexity="unknown"`, `error_message` заполняется.

Ни один из этих fallback'ов не останавливает pipeline -- decomposer всегда возвращает `stage_status="success"` или `"failed"`, но даже при `"failed"` граф продолжает к generator (нет условного ребра после decomposer).

---

## 5. Стадия 3: Generator (Ensemble)

**Файлы**: `agents/generator.py`, `prompts/generator.py`, `tools/few_shot.py`, `tools/llm_router.py`

### 5.1. Ансамбль

Генерируется `NUM_CANDIDATES` (по умолчанию 8) SQL-кандидатов **параллельно** через `asyncio.gather`. Кандидаты распределяются между моделями:

- Первые `PRIMARY_CALLS` (5) -> `google/gemini-2.5-pro`, temperature=0.2
- Следующие `SECONDARY_CALLS` (3) -> `deepseek/deepseek-chat-v3`, temperature=0.6

Распределение через `LLMRouter.generator_roles()` -- возвращает список `[PRIMARY, PRIMARY, PRIMARY, PRIMARY, PRIMARY, SECONDARY, SECONDARY, SECONDARY]`. Каждый кандидат получает `roles[idx % len(roles)]`.

### 5.2. Разнообразие через few-shot

Каждый кандидат получает **уникальный** набор few-shot примеров:

```python
rng = random.Random(seed + candidate_index)
examples = rng.sample(source, k=2)
```

`seed=42` (конфигурируемый) + `candidate_index` -- детерминированное, но разное сэмплирование для каждого кандидата. Приоритет отдаётся примерам из **той же БД** (`target_db_id`): если в пуле >= k примеров из этой БД, берём их. Иначе -- из общего пула.

Пул загружается из `train_spider.json` (до 3000 примеров).

### 5.3. Промпт генератора

Каждый кандидат получает полный промпт с:
- Вопрос
- `complexity` и `sub_questions` от decomposer
- `filtered_schema` (mSchema) от selector
- Уникальные few-shot примеры
- Правила: "Use ONLY the tables and columns listed in the mSchema above"

System message: `"Output only SQL."` -- минимизирует "болтовню" и reasoning в ответе.

### 5.4. Извлечение SQL

`_extract_sql()` -- очистка ответа LLM:
1. Снятие markdown code fences (` ```sql ... ``` `)
2. Коллапс whitespace в одну строку
3. Добавление `;` если отсутствует

Пустые ответы отфильтровываются. Если все 8 кандидатов пустые -- warning, pipeline продолжает к execution filter с пустым `candidates=[]`.

### 5.5. Температурная стратегия

- Primary (0.2) -- низкая температура, более "каноничные" запросы
- Secondary (0.6) -- высокая температура, более креативные/альтернативные подходы

Разные модели + разные температуры + разные few-shot = максимальное разнообразие при фиксированном числе вызовов.

---

## 6. Стадия 4: Execution Filter

**Файлы**: `agents/execution_filter.py`, `tools/sql_executor.py`

### 6.1. Что делает

Исполняет **каждый** SQL-кандидат на реальной SQLite-базе и отсеивает те, что упали с ошибкой.

### 6.2. Механика исполнения

```python
engine = create_async_engine("sqlite+aiosqlite:///path/to/db.sqlite")
async with engine.connect() as conn:
    result = await asyncio.wait_for(conn.execute(text(sql)), timeout=20)
```

- Используется `aiosqlite` + `SQLAlchemy async` -- неблокирующее исполнение.
- Каждый запрос изолирован в отдельном `engine` (создаётся и dispose'ится для каждого вызова).
- Timeout: `execution_timeout_seconds=20` -- защита от зависших запросов (бесконечные JOIN'ы, CROSS JOIN).

### 6.3. Результат

```python
@dataclass
class SQLExecutionResult:
    success: bool
    rows: Optional[list[tuple[Any, ...]]]
    error: Optional[str]
```

Кандидаты с `success=True` попадают в `valid_candidates`. Ошибки каждого невалидного кандидата логируются в `warnings`.

### 6.4. Fallback при нуле валидных

Если ни один кандидат не прошёл -- `valid_candidates=[]`, но pipeline **продолжает**: Judge получит `candidates` (нефильтрованные) как fallback-пул через логику `preferred = valid_candidates if valid_candidates else candidates`.

---

## 7. Стадия 5: Judge (LLM-as-Judge)

**Файлы**: `agents/judge.py`, `prompts/judge.py`

### 7.1. Пул кандидатов

Judge работает с `valid_candidates` (приоритет) или `candidates` (fallback):

```python
preferred = state.get("valid_candidates", [])
fallback = state.get("candidates", [])
pool = preferred if preferred else fallback
```

### 7.2. Промпт

Judge получает:
- Вопрос
- mSchema
- Пронумерованный список кандидатов: `0: SELECT ...\n1: SELECT ...\n2: SELECT ...`

Должен вернуть: `{"best_index": 0, "reasoning": "..."}`.

### 7.3. Модель и температура

Используется **отдельная модель** -- `openai/gpt-4.1` (temperature=0.0). Это принципиально: judge не должен совпадать с generator, чтобы избежать bias'а "нравится свой собственный стиль".

### 7.4. Fallback-политика

- JSON не распарсился -> берём кандидат `[0]` (первый), warning.
- `best_index` не int или вне диапазона -> берём кандидат `[0]`, warning.
- Exception при вызове LLM -> берём кандидат `[0]`, `stage_status="success"` (pipeline не падает).

Judge **никогда** не ставит `stage_status="failed"` при наличии кандидатов -- это защита от потери уже сгенерированного SQL из-за сетевой ошибки на стадии judge.

---

## 8. Стадия 6: Refiner (Итеративная самокоррекция)

**Файлы**: `agents/refiner.py`, `prompts/refiner.py`

### 8.1. Цикл работы

1. Берёт `final_sql` (если уже было уточнение) или `best_sql` (от judge).
2. Исполняет на реальной БД.
3. Если `success=True` -> записывает `final_sql`, обнуляет `error_message`, выходит.
4. Если execution failed -> инкрементирует `refine_attempts`, вызывает LLM для исправления.

### 8.2. Промпт рефайнера

```
Failed SQL:
SELECT ... ;

Execution error:
(sqlite3.OperationalError) no such table: song

Output only corrected SQL ending with semicolon.
```

LLM получает полный контекст: вопрос, mSchema, падающий SQL и точный текст ошибки. Модель: `openai/gpt-4.1` (temperature=0.0) -- та же, что у judge, для максимальной точности при исправлении.

### 8.3. Условная маршрутизация цикла

```python
def _route_after_refiner(state):
    has_error = bool(state.get("error_message"))
    attempts = state.get("refine_attempts", 0)
    if has_error and attempts < MAX_REFINE_ATTEMPTS:
        return "retry_refiner"   # -> ребро обратно к refiner
    return "finish"              # -> END
```

Максимум 3 итерации (конфигурируемо). Если после 3 попыток SQL всё ещё падает -- pipeline завершается с `error_message` и последней версией `final_sql`.

### 8.4. Что происходит при каждой итерации

- `refine_attempts` инкрементируется.
- `final_sql` обновляется на исправленный вариант от LLM.
- `error_message` содержит ошибку **текущей** итерации.
- На следующей итерации refiner берёт **обновлённый** `final_sql` и пытается исполнить снова.

---

## 9. LLM Router

**Файл**: `tools/llm_router.py`

### 9.1. Маршрутизация моделей

`ModelRole` -> model id:

| Role | Model | Temperature |
|---|---|---|
| `GENERATOR_PRIMARY` | `google/gemini-2.5-pro` | 0.2 |
| `GENERATOR_SECONDARY` | `deepseek/deepseek-chat-v3` | 0.6 |
| `JUDGE` | `openai/gpt-4.1` | 0.0 |
| `REFINER` | `openai/gpt-4.1` | 0.0 |

Refiner использует ту же модель, что и Judge (`gpt-4.1`), а не generator -- для более точного исправления.

### 9.2. Кэширование клиентов

`ChatOpenAI` инстансы кэшируются по ключу `(model, temperature)`. Это позволяет переиспользовать HTTP-соединения внутри одного прогона.

### 9.3. Retry-политика

Все вызовы `ainvoke` обёрнуты `@retry` от tenacity:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
```

3 попытки, экспоненциальный backoff 1-8 сек. После 3 неудач -- exception пробрасывается выше (в агент), где обрабатывается fallback'ом.

---

## 10. Конфигурация

**Файл**: `config.py`

Все параметры управляются через `.env` + Pydantic-settings с дефолтами:

| Параметр | Default | Описание |
|---|---|---|
| `NUM_CANDIDATES` | 8 | Кандидатов в ансамбле |
| `PRIMARY_CALLS` | 5 | Из них primary-моделью |
| `SECONDARY_CALLS` | 3 | Из них secondary-моделью |
| `SELECTOR_TOP_K_TABLES` | 15 | Top-K из vector search |
| `SELECTOR_TARGET_TABLES_MIN` | 3 | Минимум таблиц после реранкинга |
| `SELECTOR_TARGET_TABLES_MAX` | 5 | Максимум таблиц после реранкинга |
| `LLM_TEMPERATURE_PRIMARY` | 0.2 | Температура primary-генератора |
| `LLM_TEMPERATURE_SECONDARY` | 0.6 | Температура secondary-генератора |
| `LLM_TEMPERATURE_JUDGE` | 0.0 | Температура judge и refiner |
| `LLM_MAX_TOKENS` | 1024 | Лимит токенов ответа LLM |
| `RETRY_ATTEMPTS` | 3 | Retry при сбое LLM |
| `MAX_REFINE_ATTEMPTS` | 3 | Макс. итераций refiner |
| `EXECUTION_TIMEOUT_SECONDS` | 20 | Timeout исполнения SQL |
| `FEW_SHOT_EXAMPLES_PER_CANDIDATE` | 2 | Few-shot примеров на кандидата |
| `FEW_SHOT_SEED` | 42 | Seed для детерминированного сэмплирования |

---

## 11. Поток данных (Data Flow)

Полная цепочка трансформаций для одного вопроса:

```
Вход: question="How many singers from France?", db_id="concert_singer"

1. Selector:
   tables.json -> parse -> schema{tables, columns, fk, pk, samples}
   schema -> ChromaDB index (one-time per db_id)
   question -> ChromaDB query -> top-15 candidates
   candidates -> LLM rerank -> 3-5 selected tables
   selected tables -> filter schema -> to_mschema() -> filtered_schema

2. Decomposer:
   question -> LLM -> {"complexity": "simple", "sub_questions": []}

3. Generator (x8 параллельно):
   question + complexity + sub_questions + filtered_schema + few_shot[i]
   -> LLM[role_i] -> raw_sql -> _extract_sql() -> candidate_i

4. Execution Filter:
   candidate_i -> execute on SQLite -> success/fail
   -> valid_candidates (только успешные)

5. Judge:
   question + filtered_schema + valid_candidates
   -> LLM(gpt-4.1) -> {"best_index": 2, "reasoning": "..."}
   -> best_sql = valid_candidates[2]

6. Refiner:
   best_sql -> execute on SQLite
   -> success? -> final_sql = best_sql, done
   -> fail? -> LLM fix -> new_sql -> retry (до 3 раз)

Выход: final_sql="SELECT COUNT(*) FROM singer WHERE Country = 'France';"
```

---

## 12. Обработка ошибок и устойчивость

Каждая стадия реализует паттерн "controlled degradation":

| Стадия | При ошибке | Pipeline продолжает? |
|---|---|---|
| Selector | LLM rerank не распарсился | Да, fallback к top vector candidates |
| Selector | Vector search пустой | Да, используется полная схема |
| Selector | Критический exception | Нет, early exit в END |
| Decomposer | JSON не распарсился | Да, complexity="unknown", sub_questions=[] |
| Decomposer | Exception | Да, pipeline продолжает (нет conditional edge) |
| Generator | Часть кандидатов пустые | Да, фильтрация пустых, остальные идут дальше |
| Generator | Все кандидаты пустые | Нет, early exit в END |
| Exec Filter | Часть кандидатов невалидны | Да, valid_candidates = только успешные |
| Exec Filter | Все невалидны | Да, judge получит raw candidates как fallback |
| Judge | JSON не распарсился | Да, fallback к первому кандидату |
| Judge | LLM exception | Да, fallback к первому кандидату |
| Refiner | SQL исполнение упало | Да, LLM-коррекция + retry (до 3 раз) |
| Refiner | LLM-коррекция exception | Да, сохраняет предыдущий SQL, retry |
