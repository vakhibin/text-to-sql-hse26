# Архитектурный обзор text-to-sql агента

## 1. Общая схема

Агент реализован как ориентированный граф с одним циклическим ребром (refiner → refiner). Граф компилируется LangGraph и исполняется асинхронно.

```
                START
                  │
            [1. Selector]         — schema linking: vector + lexical retrieval → LLM reranking
                  │
            (failed?) ──yes──→ END
                  │no
            [2. Decomposer]       — complexity classification + sub-questions
                  │
            [3. Query Sketcher]   — structured query plan (tables, joins, filters, aggregations)
                  │
            [4. Generator]        — ensemble SQL generation (multi-model, multi-temperature)
                  │
            (пустые кандидаты?) ──yes──→ END
                  │no
            [5. Execution Filter] — SQL execution + schema validation + refusal guardrail
                  │
            ┌─── (simple cheap path?) ──yes──→ END (best_sql = first valid)
            │no
            [6. Judge]            — LLM-as-judge selection
                  │
            (нет best_sql?) ──yes──→ END
                  │no
            [7. Refiner] ←────┐  — iterative self-correction
                  │            │
            (ошибка исполнения │
             И attempts < 3?) │
                  │yes─────────┘
                  │no
                 END
```

Каждый узел — async-функция, принимающая и возвращающая `SQLAgentState` (TypedDict).

---

## 2. Общее состояние (`SQLAgentState`)

| Группа | Поля | Кто пишет |
|---|---|---|
| Вход | `question`, `db_id`, `evidence`, `schema_root` | `make_initial_state()` |
| Selector | `full_schema`, `filtered_schema`, `retrieved_schema_context` | Selector |
| Decomposer | `complexity`, `sub_questions` | Decomposer |
| Sketcher | `query_sketch`, `query_sketch_text` | Query Sketcher |
| Generator | `candidates` | Generator |
| Exec Filter | `valid_candidates`, `candidate_diagnostics` | Execution Filter |
| Judge | `best_sql`, `judge_reasoning`, `judge_confidence` | Judge |
| Refiner | `final_sql`, `execution_result`, `refine_attempts`, `error_message` | Refiner |
| Observability | `stage_status`, `stage_timings`, `trace_id`, `warnings`, `llm_usage`, `total_cost_usd` | Все стадии |

`warnings` — накопительный список строк. Каждая стадия дописывает, не перезаписывая.

`llm_usage` — список dict'ов с информацией о каждом LLM-вызове (модель, токены, стоимость).

---

## 3. Стадия 1: Selector (Schema Linking)

**Файлы**: `agents/selector.py`, `prompts/selector.py`, `tools/vector_store.py`, `tools/schema_loader.py`

### 3.1. Загрузка схемы

`load_schema(db_id)` парсит `tables.json` и строит структурированный словарь с таблицами, колонками (с типами данных), primary/foreign keys и sample values. Для каждой колонки выполняется `SELECT col FROM table WHERE col IS NOT NULL LIMIT 3` — примеры реальных значений помогают LLM понять типы и формат.

Результаты кэшируются по ключу `(schema_root, db_id, with_sample_values, sample_limit)`.

### 3.2. Гибридный retrieval (vector + lexical)

Два параллельных канала:

1. **Векторный**: ChromaDB, одна запись на таблицу. Формат документа включает имя таблицы, колонки с типами, sample values, PK/FK. Эмбеддинги: `openai/text-embedding-3-large`. Top-K=15 по косинусной близости к вопросу.

2. **Лексический**: `selector_top_k_lexical_tables=8` таблиц, отобранных по совпадению токенов вопроса с именами таблиц и колонок. Ловит случаи, где семантический поиск промахивается по редким именам.

Результаты объединяются: vector-кандидаты + lexical-кандидаты (deduplicated).

### 3.3. LLM-реранкинг

Объединённые кандидаты передаются LLM (primary generator, temperature=0.0):

```
Select 3 to 5 table names that are most relevant to answer the question.
Return STRICT JSON: {"selected_tables": [...], "reasoning": "..."}
```

Парсинг многоуровневый: strict JSON → JSON в code fence → первый JSON-объект → plain list → построчный.

### 3.4. Padding и fallback

- Реранкер вернул 0 валидных имён → fallback к top vector candidates.
- Вернул < min (3) → padding из vector candidates.
- Вернул >= min → обрезка до max (5).

### 3.5. mSchema

Отфильтрованные таблицы форматируются в компактный mSchema:

```
singer(Singer_ID:NUMBER sample=['1','2'], Name:TEXT sample=['Joe Sharp']) pk=['Singer_ID']
concert(concert_ID:NUMBER, Year:TEXT) pk=['concert_ID'] fk=[Singer_ID->singer.Singer_ID]
```

Этот формат передаётся во все последующие стадии.

### 3.6. Prewarm

`prewarm_selector_cache()` предзагружает схемы и vector index для всех БД до начала benchmark-прогона. Устраняет cold-start на первых примерах.

---

## 4. Стадия 2: Decomposer

**Файлы**: `agents/decomposer.py`, `prompts/decomposer.py`

Возвращает:
- `complexity`: `"simple"` | `"moderate"` | `"complex"` | `"unknown"`
- `sub_questions`: список подвопросов (может быть пустым)

Complexity управляет маршрутизацией:
- `simple`: сокращённый бюджет генерации, eligible для cheap path (skip judge)
- `moderate`: сокращённый ансамбль (`MODERATE_NUM_CANDIDATES=2`, `MODERATE_PRIMARY_CALLS=2`)
- `complex` / `unknown`: полный ансамбль

Fallback: JSON не распарсился → `complexity="unknown"`, pipeline продолжает.

---

## 5. Стадия 3: Query Sketcher

**Файлы**: `agents/query_sketcher.py`, `prompts/query_sketcher.py`

### 5.1. Назначение

Генерирует структурированный план запроса **до** генерации SQL. Скетч указывает генератору: какие таблицы использовать, как соединять, какие фильтры и агрегации применять, нужны ли подзапросы.

### 5.2. Формат скетча

```json
{
  "tables": ["singer", "concert", "singer_in_concert"],
  "join_path": "singer → singer_in_concert → concert",
  "filters": ["singer.Country = 'France'"],
  "aggregations": ["COUNT(*)"],
  "grouping": [],
  "ordering": [],
  "needs_subquery": false,
  "risk_flags": []
}
```

### 5.3. Модель

Настраивается отдельно: `QUERY_SKETCHER_MODEL` (по умолчанию fallback на primary generator). Можно ставить мощную модель на скетчер и дешёвые на генерацию — скетчер берёт на себя "мышление", генератору остаётся кодирование.

Может быть отключен: `QUERY_SKETCHER_ENABLED=false` (для ablation study).

### 5.4. Трёхступенчатая надёжность

1. **Толерантный JSON-парсинг** raw-ответа (code fence removal, поиск `{...}`)
2. **Structured output repair**: если JSON сломан, переотправка через `with_structured_output()` с Pydantic-схемой
3. **Детерминистический fallback**: минимальный скетч на основе state (filtered_schema → извлечение таблиц и колонок)

Каждый шаг логируется в `warnings`.

---

## 6. Стадия 4: Generator (Ensemble)

**Файлы**: `agents/generator.py`, `prompts/generator.py`, `tools/few_shot.py`, `tools/llm_router.py`

### 6.1. Адаптивный ансамбль

Бюджет кандидатов зависит от `complexity`:

| Complexity | Кандидатов | Primary | Secondary |
|---|---|---|---|
| `complex` / `unknown` | 5 | 3 (gemini-2.5-pro) | 2 (gpt-oss-120b) |
| `moderate` | 2 | 2 (gemini-2.5-pro) | 0 |
| `simple` | 5 | 3 | 2 (но может выйти на cheap path) |

Кандидаты генерируются **параллельно** через `asyncio.gather`.

### 6.2. Промпт генератора

Каждый кандидат получает:
- Вопрос + evidence
- `complexity` и `sub_questions` от decomposer
- **`query_sketch_text`** от sketcher — структурированный план
- `filtered_schema` (mSchema)
- Уникальные few-shot примеры
- Правила генерации (13 правил):
  - Rule 9: порядок колонок в SELECT по порядку упоминания в вопросе
  - Rule 11: не добавлять JOIN если все колонки в одной таблице
  - Rule 13: "all information about X" → `SELECT *`

System message: `"Output only SQL."` — минимизирует reasoning в ответе.

### 6.3. Few-shot примеры

Каждый кандидат получает уникальный набор (seed + candidate_index). Приоритет — примеры из той же БД. Пул загружается из `train_spider.json`.

Опционально: семантический retrieval few-shot из ChromaDB (`FEW_SHOT_SEMANTIC_RETRIEVAL=true`).

### 6.4. Температурная стратегия

- Primary (0.2) — каноничные запросы
- Secondary (0.6) — альтернативные подходы

Разные модели + разные температуры + разные few-shot = максимальное разнообразие.

---

## 7. Стадия 5: Execution Filter

**Файлы**: `agents/execution_filter.py`, `tools/sql_executor.py`, `tools/sql_schema_validator.py`, `tools/sql_candidate_analysis.py`

### 7.1. Валидация

Для каждого кандидата выполняются три проверки:

1. **Refusal guardrail** (`_is_refusal_sql`): детектирует LLM-отказы вида `SELECT 'I cannot answer...'` по regex-паттернам. Отказы отсеиваются до исполнения.

2. **SQL schema validation** (`sqlglot`): парсит SQL в AST, проверяет существование таблиц и колонок относительно загруженной схемы. Ловит hallucinated identifiers до обращения к SQLite.

3. **SQL execution**: `aiosqlite` + SQLAlchemy async, timeout 20 секунд. Кандидаты с `success=True` попадают в `valid_candidates`.

### 7.2. Структурный анализ

`sql_candidate_analysis` через `sqlglot` собирает диагностику:
- `join_count`, `has_subquery`, `set_operation`
- `projected_columns`, `table_count`
- Передаётся в `candidate_diagnostics` для judge и cheap path.

### 7.3. Simple cheap path

Для `simple` запросов, если первый валидный кандидат проходит структурные проверки (0 join'ов, нет подзапросов, нет set operations, нет schema ошибок):
- `best_sql` = первый валидный кандидат
- **Judge пропускается**
- Экономия одного LLM-вызова + ускорение ~2-3 секунды

### 7.4. Fallback

Все невалидны → `valid_candidates=[]`, judge получит raw candidates как fallback.

---

## 8. Стадия 6: Judge (LLM-as-Judge)

**Файлы**: `agents/judge.py`, `prompts/judge.py`

### 8.1. Lean-промпт

Урок: перегрузка judge контекстом (sketch, risk flags, sub_questions, candidate diffs, rejected candidates) **ухудшает** качество selection. Текущий промпт — lean:
- Вопрос + evidence
- Filtered schema (mSchema)
- Пронумерованные кандидаты с краткими structural summaries

### 8.2. Structured output

Judge возвращает:
```json
{
  "best_index": 2,
  "confidence": "high",
  "needs_refine": false,
  "issues": [],
  "reasoning": "..."
}
```

`confidence`: `"high"` | `"medium"` | `"low"`.
`issues`: из фиксированного словаря (`"projection_mismatch"`, `"unnecessary_join"` и т.д.).

### 8.3. Модель

`openai/gpt-4.1`, temperature=0.0. Отдельная от генераторов — избегает bias.

### 8.4. Fallback

JSON не распарсился / exception → берём кандидат `[0]`. Judge никогда не ставит `stage_status="failed"` при наличии кандидатов.

---

## 9. Стадия 7: Refiner (Итеративная самокоррекция)

**Файлы**: `agents/refiner.py`, `prompts/refiner.py`

### 9.1. Цикл

1. Берёт `best_sql` от judge
2. Исполняет на реальной БД
3. `success=True` → `final_sql`, выход
4. Execution failed → LLM-коррекция, `refine_attempts++`, retry

Максимум 3 итерации. Модель: `openai/gpt-4.1` (temperature=0.0).

### 9.2. Schema-reference validation

Перед исполнением на SQLite запускается легковесная проверка через `sql_schema_validator` — ловит очевидные ошибки (несуществующие таблицы/колонки) и передаёт их в промпт рефайнера для точечного исправления.

### 9.3. Защита от деструкции

Refiner **никогда не уничтожает** последний рабочий SQL. Если LLM-коррекция вернула пустоту — сохраняется предыдущая версия.

---

## 10. LLM Router

**Файл**: `tools/llm_router.py`

### 10.1. Маршрутизация моделей

| Role | Model | Temperature |
|---|---|---|
| `GENERATOR_PRIMARY` | `google/gemini-2.5-pro` | 0.2 |
| `GENERATOR_SECONDARY` | `openai/gpt-oss-120b` | 0.6 |
| `QUERY_SKETCHER` | `google/gemini-2.5-pro` (настраиваемо) | 0.0 |
| `JUDGE` | `openai/gpt-4.1` | 0.0 |
| `REFINER` | `openai/gpt-4.1` | 0.0 |

### 10.2. Retry-политика

Все вызовы обёрнуты `@retry` от tenacity: 3 попытки, экспоненциальный backoff 1–8 сек.

### 10.3. Cost tracking

Каждый LLM-вызов записывает usage (prompt/completion tokens, cost_usd) в `llm_usage` через `_extract_usage()`. Агрегация: per-example и per-run.

---

## 11. Конфигурация

**Файл**: `config.py`

Все параметры через `.env` + Pydantic-settings:

| Параметр | Текущее значение | Описание |
|---|---|---|
| `NUM_CANDIDATES` | 5 | Кандидатов в ансамбле (complex/unknown) |
| `PRIMARY_CALLS` | 3 | Из них primary-моделью |
| `SECONDARY_CALLS` | 2 | Из них secondary-моделью |
| `MODERATE_NUM_CANDIDATES` | 2 | Кандидатов для moderate |
| `MODERATE_PRIMARY_CALLS` | 2 | Primary для moderate |
| `MODERATE_SECONDARY_CALLS` | 0 | Secondary для moderate |
| `SIMPLE_SKIP_JUDGE_WHEN_VALID` | true | Cheap path для simple |
| `QUERY_SKETCHER_ENABLED` | true | Включить/выключить скетчер |
| `SELECTOR_TOP_K_TABLES` | 15 | Top-K из vector search |
| `SELECTOR_TOP_K_LEXICAL_TABLES` | 8 | Top-K из lexical search |
| `SELECTOR_TARGET_TABLES_MIN` | 3 | Минимум таблиц после реранкинга |
| `SELECTOR_TARGET_TABLES_MAX` | 5 | Максимум таблиц |
| `LLM_TEMPERATURE_PRIMARY` | 0.2 | Температура primary-генератора |
| `LLM_TEMPERATURE_SECONDARY` | 0.6 | Температура secondary-генератора |
| `LLM_MAX_TOKENS` | 2048 | Лимит токенов ответа LLM |
| `LLM_TIMEOUT_SECONDS` | 180 | Timeout LLM-вызова |
| `MAX_REFINE_ATTEMPTS` | 3 | Макс. итераций refiner |
| `EXECUTION_TIMEOUT_SECONDS` | 20 | Timeout исполнения SQL |
| `FEW_SHOT_EXAMPLES_PER_CANDIDATE` | 2 | Few-shot примеров на кандидата |
| `FEW_SHOT_SEMANTIC_RETRIEVAL` | false | Семантический retrieval few-shot |

---

## 12. Метрики оценки

**Файл**: `evaluation/metrics.py`

### 12.1. Execution Accuracy (EX)

Официальный алгоритм Spider `result_eq` (порт из `taoyds/test-suite-sql-eval`):
- Поиск допустимой перестановки колонок между predicted и gold результатами
- Multiset (bag) семантика для строк (если gold SQL не содержит ORDER BY)
- Строгий порядок строк только при наличии ORDER BY в gold SQL

### 12.2. Exact Match (EM)

AST-based нормализация через `sqlglot`:
- Парсинг в AST (SQLite dialect)
- Normalize identifiers (lowercase)
- Resolve алиасов (`T1.Name` → `teacher.name`)
- Strip qualifier для single-table запросов
- Fallback на строковое сравнение при ошибке парсинга

Наш EM строже официального Spider EM (который игнорирует значения и сравнивает компонентно).

---

## 13. Обработка ошибок и устойчивость

Каждая стадия реализует "controlled degradation":

| Стадия | При ошибке | Pipeline продолжает? |
|---|---|---|
| Selector | LLM rerank не распарсился | Да, fallback к vector candidates |
| Selector | Vector search пустой | Да, полная схема |
| Selector | Критический exception | Нет, early exit |
| Decomposer | JSON не распарсился | Да, complexity="unknown" |
| Sketcher | JSON не распарсился | Да, structured output repair → fallback sketch |
| Sketcher | Полный exception | Да, minimal fallback sketch |
| Sketcher | Disabled | Да, пустой sketch, generator работает без плана |
| Generator | Часть кандидатов пустые | Да, фильтрация пустых |
| Generator | Все кандидаты пустые | Нет, early exit |
| Exec Filter | Refusal SQL обнаружен | Да, кандидат отсеивается |
| Exec Filter | Schema validation failed | Да, кандидат отсеивается |
| Exec Filter | Все невалидны | Да, judge получит raw candidates |
| Judge | JSON не распарсился | Да, fallback к первому кандидату |
| Judge | LLM exception | Да, fallback к первому кандидату |
| Refiner | SQL execution failed | Да, LLM-коррекция + retry (до 3 раз) |
| Refiner | LLM-коррекция exception | Да, сохраняет предыдущий SQL |

---

## 14. Benchmark Runner

**Файлы**: `evaluation/run_spider.py`, `evaluation/run_bird.py`

Возможности:
- `--concurrency N` — параллельное исполнение примеров
- `--prewarm` — предзагрузка schema cache и vector index
- `--subset-manifest` — запуск на фиксированном debug subset
- Per-example hard timeout (default: `llm_timeout * retry_attempts + 120s`)
- Atomic partial JSON writes каждые 25 примеров
- Timestamped output files

Ablation runner: `scripts/run_ablation.sh` — подставляет env-переменные из конфиг-файла и запускает на debug subset.

---

## 15. Текущие результаты

### Spider v1 dev (полный набор, 1034 примера)

| Метрика | Значение |
|---|---|
| **Execution Accuracy (EX)** | **72.92%** |
| **Exact Match (EM)** | **29.11%** |
| Errors | 24 (2.3%) |
| Valid SQL Rate | ~97.7% |
| Avg time/example | 4.49s |
| Total cost | $67.78 |
| Total time | 1:17:24 |
| Concurrency | 12 |

### Сравнение с baseline

| | Baseline (single-model) | Multi-agent pipeline | Дельта |
|---|---|---|---|
| **EX** | 64.22% | **72.92%** | **+8.7 ppt** |
| **EM** | ~21% (naive) | **29.11%** (AST) | **+8 ppt** |
| **Errors** | 45 (4.4%) | 24 (2.3%) | **−47%** |
| **Speed** | 15.27s/q | 4.49s/q | **3.4× faster** |

### Model stack

| Роль | Модель | Вызовов/пример |
|---|---|---|
| Sketcher | gemini-2.5-pro | 1 (+1 repair при необходимости) |
| Primary generator | gemini-2.5-pro | 3 |
| Secondary generator | gpt-oss-120b | 2 |
| Judge | gpt-4.1 | 1 |
| Refiner | gpt-4.1 | 0–3 |
| Embeddings | text-embedding-3-large | ~3–5 |
