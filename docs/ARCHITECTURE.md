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
            [2. Query Sketcher]   — structured query plan (tables, joins, filters, aggregations)
                  │
            [3. Generator]        — ensemble SQL generation (multi-model, multi-temperature)
                  │
            (пустые кандидаты?) ──yes──→ END
                  │no
            [4. Execution Filter] — SQL execution + schema validation + refusal guardrail
                  │
            (нет кандидатов?) ──yes──→ END
                  │no
            [5. Voting]           — self-consistency majority voting по результатам исполнения
                  │
            (нет best_sql?) ──yes──→ END
                  │no
            [6. Refiner] ←────┐  — iterative self-correction
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
| Sketcher | `query_sketch`, `query_sketch_text` | Query Sketcher |
| Generator | `candidates` | Generator |
| Exec Filter | `valid_candidates`, `candidate_diagnostics` (incl. `execution_rows`) | Execution Filter |
| Voting | `best_sql`, `selection_reasoning`, `selection_confidence`, `selection_method` | Voting |
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

## 4. Стадия 2: Query Sketcher

**Файлы**: `agents/query_sketcher.py`, `prompts/query_sketcher.py`

### 4.1. Назначение

Генерирует структурированный план запроса **до** генерации SQL. Скетч указывает генератору: какие таблицы использовать, как соединять, какие фильтры и агрегации применять, нужны ли подзапросы.

### 4.2. Формат скетча

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

### 4.3. Модель

Настраивается отдельно: `QUERY_SKETCHER_MODEL` (по умолчанию fallback на primary generator). Можно ставить мощную модель на скетчер и дешёвые на генерацию — скетчер берёт на себя "мышление", генератору остаётся кодирование.

Может быть отключен: `QUERY_SKETCHER_ENABLED=false` (для ablation study).

### 4.4. Трёхступенчатая надёжность

1. **Толерантный JSON-парсинг** raw-ответа (code fence removal, поиск `{...}`)
2. **Structured output repair**: если JSON сломан, переотправка через `with_structured_output()` с Pydantic-схемой
3. **Детерминистический fallback**: минимальный скетч на основе state (filtered_schema → извлечение таблиц и колонок)

Каждый шаг логируется в `warnings`.

---

## 5. Стадия 3: Generator (Ensemble)

**Файлы**: `agents/generator.py`, `prompts/generator.py`, `tools/few_shot.py`, `tools/llm_router.py`

### 5.1. Фиксированный ансамбль

Бюджет кандидатов одинаковый для всех запросов (complexity routing убран):

| Кандидатов | Primary | Secondary |
|---|---|---|
| 8 (default) | 5 (gemini-2.5-pro) | 3 (deepseek-chat-v3) |

Кандидаты генерируются **параллельно** через `asyncio.gather`.

### 5.2. Промпт генератора

Каждый кандидат получает:
- Вопрос + evidence
- **`query_sketch_text`** от sketcher — структурированный план
- `filtered_schema` (mSchema)
- Уникальные few-shot примеры
- Правила генерации:
  - Порядок колонок в SELECT по порядку упоминания в вопросе
  - Не добавлять JOIN если все колонки в одной таблице
  - "all information about X" → `SELECT *`

System message: `"Output only SQL."` — минимизирует reasoning в ответе.

### 5.3. Few-shot примеры

Каждый кандидат получает уникальный набор (seed + candidate_index). Приоритет — примеры из той же БД. Пул загружается из `train_spider.json`.

Опционально: семантический retrieval few-shot из ChromaDB (`FEW_SHOT_SEMANTIC_RETRIEVAL=true`).

### 5.4. Температурная стратегия

- Primary (0.2) — каноничные запросы
- Secondary (0.6) — альтернативные подходы

Разные модели + разные температуры + разные few-shot = максимальное разнообразие.

---

## 6. Стадия 4: Execution Filter

**Файлы**: `agents/execution_filter.py`, `tools/sql_executor.py`, `tools/sql_schema_validator.py`, `tools/sql_candidate_analysis.py`

### 6.1. Валидация

Для каждого кандидата выполняются три проверки:

1. **Refusal guardrail** (`_is_refusal_sql`): детектирует LLM-отказы вида `SELECT 'I cannot answer...'` по regex-паттернам. Отказы отсеиваются до исполнения.

2. **SQL schema validation** (`sqlglot`): парсит SQL в AST, проверяет существование таблиц и колонок относительно загруженной схемы. Ловит hallucinated identifiers до обращения к SQLite.

3. **SQL execution**: `aiosqlite` + SQLAlchemy async, timeout 20 секунд. Кандидаты с `success=True` попадают в `valid_candidates`.

### 6.2. Структурный анализ и результаты исполнения

`sql_candidate_analysis` через `sqlglot` собирает диагностику:
- `join_count`, `has_subquery`, `set_operation`
- `projected_columns`, `table_count`

Для каждого валидного кандидата сохраняются `execution_rows` — фактические строки результата из SQLite. Эти данные используются voting-стадией для сравнения результатов.

Всё передаётся в `candidate_diagnostics`.

### 6.3. Fallback

Все невалидны → `valid_candidates=[]`, voting получит raw candidates как fallback.

---

## 7. Стадия 5: Voting (Self-Consistency Majority Voting)

**Файл**: `agents/voting.py`

### 7.1. Принцип

Вместо LLM-as-judge выбор лучшего SQL кандидата происходит детерминистически: кандидаты группируются по результатам исполнения, побеждает SQL из самой большой группы.

Ключевое преимущество: **zero LLM calls** — чисто вычислительный этап.

### 7.2. Каноникализация результатов

Для группировки каждый результат приводится к каноническому виду:
- Каждая строка → tuple строковых представлений значений
- Множество строк сортируется → bag equality (порядок строк не важен)
- `None` (failed execution) → специальный sentinel, никогда не совпадающий с реальным результатом

### 7.3. Tie-breaking

При равном размере групп предпочитается более простой SQL:
1. Меньше `JOIN`-ов
2. Короче по длине

### 7.4. Уровень уверенности

| Доля согласия | Confidence |
|---|---|
| >= 60% | `high` |
| >= 40% | `medium` |
| < 40% | `low` |

При `low` confidence устанавливается `selection_needs_refine=True`.

### 7.5. Пример

8 кандидатов: 5 возвращают `[(42,)]`, 2 возвращают `[(41,)]`, 1 возвращает `[(42, 'x')]`.

→ Побеждает группа из 5 кандидатов (62.5%, confidence=high). Из них выбирается SQL с наименьшим числом JOIN.

### 7.6. Fallback

Нет кандидатов → `best_sql=""`, pipeline завершается. Нет `execution_rows` в diagnostics → каждый кандидат образует свою группу, выбирается первый (деградация к fallback-у первого кандидата).

---

## 8. Стадия 6: Refiner (Итеративная самокоррекция)

**Файлы**: `agents/refiner.py`, `prompts/refiner.py`

### 8.1. Цикл

1. Берёт `best_sql` от voting
2. Исполняет на реальной БД
3. `success=True` → `final_sql`, выход
4. Execution failed → LLM-коррекция, `refine_attempts++`, retry

Максимум 3 итерации. Модель: `openai/gpt-4.1` (temperature=0.0).

### 8.2. Контекст для LLM-коррекции

Промпт рефайнера включает:
- Вопрос + evidence + query sketch
- mSchema + retrieved schema context
- Текущий (сломанный) SQL
- Selection context (confidence + reasoning от voting)
- Schema validation errors/warnings
- Execution error
- Structural summary выбранного кандидата
- Summaries отвергнутых кандидатов (до 3)

### 8.3. Schema-reference validation

Перед исполнением на SQLite запускается легковесная проверка через `sql_schema_validator` — ловит очевидные ошибки (несуществующие таблицы/колонки) и передаёт их в промпт рефайнера для точечного исправления.

### 8.4. Защита от деструкции

Refiner **никогда не уничтожает** последний рабочий SQL. Если LLM-коррекция вернула пустоту — сохраняется предыдущая версия.

---

## 9. LLM Router

**Файл**: `tools/llm_router.py`

### 9.1. Маршрутизация моделей

| Role | Model | Temperature |
|---|---|---|
| `GENERATOR_PRIMARY` | `google/gemini-2.5-pro` | 0.2 |
| `GENERATOR_SECONDARY` | `deepseek/deepseek-chat-v3` | 0.6 |
| `QUERY_SKETCHER` | `google/gemini-2.5-pro` (настраиваемо) | 0.0 |
| `REFINER` | `openai/gpt-4.1` | 0.0 |

Voting не использует LLM — чисто вычислительный этап.

### 9.2. Retry-политика

Все вызовы обёрнуты `@retry` от tenacity: 3 попытки, экспоненциальный backoff 1–8 сек.

### 9.3. Cost tracking

Каждый LLM-вызов записывает usage (prompt/completion tokens, cost_usd) в `llm_usage` через `_extract_usage()`. Агрегация: per-example и per-run.

---

## 10. Конфигурация

**Файл**: `config.py`

Все параметры через `.env` + Pydantic-settings:

| Параметр | Текущее значение | Описание |
|---|---|---|
| `NUM_CANDIDATES` | 8 | Кандидатов в ансамбле |
| `PRIMARY_CALLS` | 5 | Из них primary-моделью |
| `SECONDARY_CALLS` | 3 | Из них secondary-моделью |
| `QUERY_SKETCHER_ENABLED` | true | Включить/выключить скетчер |
| `SELECTOR_TOP_K_TABLES` | 15 | Top-K из vector search |
| `SELECTOR_TOP_K_LEXICAL_TABLES` | 8 | Top-K из lexical search |
| `SELECTOR_TARGET_TABLES_MIN` | 3 | Минимум таблиц после реранкинга |
| `SELECTOR_TARGET_TABLES_MAX` | 5 | Максимум таблиц |
| `LLM_TEMPERATURE_PRIMARY` | 0.2 | Температура primary-генератора |
| `LLM_TEMPERATURE_SECONDARY` | 0.6 | Температура secondary-генератора |
| `LLM_MAX_TOKENS` | 1024 | Лимит токенов ответа LLM |
| `LLM_TIMEOUT_SECONDS` | 90 | Timeout LLM-вызова |
| `MAX_REFINE_ATTEMPTS` | 3 | Макс. итераций refiner |
| `EXECUTION_TIMEOUT_SECONDS` | 20 | Timeout исполнения SQL |
| `FEW_SHOT_EXAMPLES_PER_CANDIDATE` | 2 | Few-shot примеров на кандидата |
| `FEW_SHOT_SEMANTIC_RETRIEVAL` | true | Семантический retrieval few-shot |

---

## 11. Метрики оценки

**Файл**: `evaluation/metrics.py`

### 11.1. Execution Accuracy (EX)

Официальный алгоритм Spider `result_eq` (порт из `taoyds/test-suite-sql-eval`):
- Поиск допустимой перестановки колонок между predicted и gold результатами
- Multiset (bag) семантика для строк (если gold SQL не содержит ORDER BY)
- Строгий порядок строк только при наличии ORDER BY в gold SQL

### 11.2. Exact Match (EM)

AST-based нормализация через `sqlglot`:
- Парсинг в AST (SQLite dialect)
- Normalize identifiers (lowercase)
- Resolve алиасов (`T1.Name` → `teacher.name`)
- Strip qualifier для single-table запросов
- Fallback на строковое сравнение при ошибке парсинга

Наш EM строже официального Spider EM (который игнорирует значения и сравнивает компонентно).

---

## 12. Обработка ошибок и устойчивость

Каждая стадия реализует "controlled degradation":

| Стадия | При ошибке | Pipeline продолжает? |
|---|---|---|
| Selector | LLM rerank не распарсился | Да, fallback к vector candidates |
| Selector | Vector search пустой | Да, полная схема |
| Selector | Критический exception | Нет, early exit |
| Sketcher | JSON не распарсился | Да, structured output repair → fallback sketch |
| Sketcher | Полный exception | Да, minimal fallback sketch |
| Sketcher | Disabled | Да, пустой sketch, generator работает без плана |
| Generator | Часть кандидатов пустые | Да, фильтрация пустых |
| Generator | Все кандидаты пустые | Нет, early exit |
| Exec Filter | Refusal SQL обнаружен | Да, кандидат отсеивается |
| Exec Filter | Schema validation failed | Да, кандидат отсеивается |
| Exec Filter | Все невалидны | Да, voting получит raw candidates |
| Voting | Нет кандидатов | Нет, early exit |
| Voting | Нет execution_rows | Да, деградация к первому кандидату |
| Refiner | SQL execution failed | Да, LLM-коррекция + retry (до 3 раз) |
| Refiner | LLM-коррекция exception | Да, сохраняет предыдущий SQL |

---

## 13. Benchmark Runner

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

## 14. Текущие результаты

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

*Результаты на предыдущей архитектуре (с decomposer + LLM judge). Бенчмарк на новой архитектуре (voting) ещё не проведён.*

### Сравнение с baseline

| | Baseline (single-model) | Multi-agent pipeline | Дельта |
|---|---|---|---|
| **EX** | 64.22% | **72.92%** | **+8.7 ppt** |
| **EM** | ~21% (naive) | **29.11%** (AST) | **+8 ppt** |
| **Errors** | 45 (4.4%) | 24 (2.3%) | **−47%** |
| **Speed** | 15.27s/q | 4.49s/q | **3.4× faster** |

### Model stack (текущий)

| Роль | Модель | Вызовов/пример |
|---|---|---|
| Sketcher | gemini-2.5-pro | 1 (+1 repair при необходимости) |
| Primary generator | gemini-2.5-pro | 5 |
| Secondary generator | deepseek-chat-v3 | 3 |
| Voting | — (детерминистический) | 0 |
| Refiner | gpt-4.1 | 0–3 |
| Embeddings | text-embedding-3-large | ~3–5 |

### Архитектурные изменения (текущая ветка)

1. **Decomposer удалён** — ablation E4 показал, что complexity classification не даёт значимого прироста EX; основная ценность была в cost routing, который больше не нужен при фиксированном ансамбле.

2. **LLM Judge заменён на majority voting** — группировка кандидатов по результатам исполнения вместо LLM-оценки. Экономит один LLM-вызов на запрос, устраняет субъективность judge.
