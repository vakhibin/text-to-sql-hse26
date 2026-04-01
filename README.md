# text-to-sql-hse26
 Master's thesis that tries to solve text-to-sql problem by applying modern approaches

## Documentation

- [baseline.md](baseline.md) — архитектура агента (schema linking, SQL generator, pipeline) и описание компонентов системы
- [results.md](results.md) — ход экспериментов, результаты оценки на Spider dev set, анализ ошибок и выводы 

## Installation
1. Clone repository. For example
```commandline
git clone git@github.com:vakhibin/text-to-sql-hse26.git
```

2.  We manage environments and dependencies with UV. That's why one need to install UV first.

2. Install virtual environment and dependencies.
```commandline
uv venv 
uv sync
```

### Optional observability

If you want Langfuse tracing during benchmark runs, add these env vars:

```commandline
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Benchmark notes

- `run_spider.py` now supports `--prewarm` to preload schema cache and Chroma index before timed evaluation.
- `run_spider.py` now also supports `--subset-manifest` for cheap reproducible debug runs on a fixed Spider subset.
- Benchmark outputs include timing and approximate cost summary:
  - `prewarm_time_s`
  - `eval_time_s`
  - `avg_time_per_example_s`
  - `total_cost_usd`
  - `avg_cost_per_example_usd`

### Spider debug subset workflow

- Use `data/debug/spider_dev_subset_v1.json` for most architecture iterations.
- The current subset is fixed and balanced as `50 simple / 50 moderate / 50 complex`.
- Run the cheap debug benchmark first, and only promote promising changes to the full Spider dev run.

Build or refresh the subset manifest:

```commandline
./.venv/bin/python scripts/build_spider_debug_subset.py --output data/debug/spider_dev_subset_v1.json
```

Run the debug subset benchmark:

```commandline
./.venv/bin/python -m text_to_sql_agent.evaluation.run_spider --subset-manifest data/debug/spider_dev_subset_v1.json --concurrency 12 --prewarm
```

If you want an even cheaper smoke check inside the fixed subset:

```commandline
./.venv/bin/python -m text_to_sql_agent.evaluation.run_spider --subset-manifest data/debug/spider_dev_subset_v1.json --smoke --smoke-size 20
```


## Архитектура текущего агента

### Обзор

Агент представляет собой **многоэтапный text-to-SQL пайплайн**, оркестрируемый [LangGraph](https://github.com/langchain-ai/langgraph). На вход подаётся вопрос на естественном языке и идентификатор базы данных, на выходе -- исполняемый SQL-запрос, полученный через семь последовательных стадий с условной маршрутизацией и ранним выходом при критических ошибках.

Все вызовы LLM проходят через [OpenRouter](https://openrouter.ai) -- единый API-шлюз к нескольким провайдерам моделей.

### Обоснование архитектурных решений

В основу архитектуры легли три работы:

1. **MAC-SQL** (COLING 2025, [arXiv 2312.11242](https://arxiv.org/abs/2312.11242)) -- отсюда взята трёхкомпонентная структура агента: Selector -> Decomposer -> Refiner. MAC-SQL даёт 59.59% EX на BIRD и служит базовым скелетом нашего пайплайна. Мы сохранили идею декомпозиции сложных вопросов на подвопросы с классификацией сложности (simple/moderate/complex) и итеративной самокоррекции SQL по тексту ошибки исполнения (до 3 попыток).

2. **CHASE-SQL** (Google, ICLR 2025, [arXiv 2410.01943](https://arxiv.org/abs/2410.01943)) -- отсюда идея генерации нескольких SQL-кандидатов и использования LLM-as-Judge для выбора лучшего. Вместо одной попытки генерации мы создаём ансамбль из N кандидатов, фильтруем невалидные через исполнение, а затем независимая LLM (Judge) выбирает лучший с обоснованием.

3. **XiYan-SQL** (Alibaba, [arXiv 2411.08599](https://arxiv.org/abs/2411.08599)) -- отсюда идея мультимодельного ансамбля: кандидаты генерируются не одной моделью, а несколькими (primary + secondary) с разными температурами и few-shot примерами, что увеличивает разнообразие и покрытие. XiYan-SQL даёт 72.23% EX на BIRD dev.

**Ключевая идея**: взять структуру декомпозиции и самокоррекции из MAC-SQL и усилить её ансамблевой генерацией в духе CHASE-SQL/XiYan-SQL, запустив генерацию асинхронно на нескольких моделях через единый API-шлюз (OpenRouter). Между декомпозицией и генерацией теперь добавлен `query-sketcher`, который сначала строит компактный schema-grounded план запроса, а уже потом генератор пишет SQL по этому каркасу. Между генерацией и судейством добавлена стадия фильтрации по исполнению (Execution Filter), которая отсеивает синтаксически и семантически ошибочные кандидаты до того, как Judge их увидит.

### Стадии пайплайна

![Pipeline Graph](docs/pipeline_graph.png)

| Стадия | Описание |
|---|---|
| **Selector** | Векторный поиск (ChromaDB) + LLM-реранкер -> 3-5 релевантных таблиц |
| **Decomposer** | Классификация сложности + генерация подвопросов (CoT) |
| **Query Sketcher** | Компактный schema-grounded план: таблицы, join intent, фильтры, агрегации, grouping, ordering, subquery need |
| **Generator** | Ансамбль: N кандидатов асинхронно (primary + secondary модели, разнообразные few-shot) |
| **Exec Filter** | Валидация через SQLAlchemy -> отсев невалидных кандидатов |
| **Judge** | LLM-as-Judge -> выбор лучшего кандидата с обоснованием |
| **Refiner** | Исполнение -> при ошибке самокоррекция (макс. 3 итерации) |

**Условная маршрутизация**: каждая стадия может инициировать ранний выход при критическом сбое (например, схема не найдена, кандидаты не сгенерированы). Refiner зацикливается на себя до `MAX_REFINE_ATTEMPTS` раз.

`Query Sketcher` старается быть отказоустойчивым: сначала парсит сырой JSON-ответ, затем при необходимости пытается структурно восстановить его, а в худшем случае строит детерминированный fallback-plan из вопроса и выбранной схемы, чтобы генератор не оставался без planning scaffold.

### Политика моделей

| Роль | Модель | Назначение |
|---|---|---|
| Основной генератор | `google/gemini-2.5-pro` | Генерация SQL (5 из 8 кандидатов) |
| Дополнительный генератор | `deepseek/deepseek-chat-v3` | Разнообразие в ансамбле (3 из 8 кандидатов) |
| Query Sketcher | `QUERY_SKETCHER_MODEL` или primary generator | Построение schema-grounded плана до генерации SQL |
| Судья | `openai/gpt-4.1` | Выбор лучшего кандидата |
| Эмбеддинги | `openai/text-embedding-3-large` | Векторный поиск по схеме |
| Реранкер селектора | Основной генератор | Переранжирование таблиц |
| Декомпозер | Основной генератор | Анализ вопроса |
| Рефайнер | Основной генератор | Исправление SQL по ошибке |

### Ключевые технические решения

- **LangGraph для оркестрации** -- обеспечивает типизированное состояние, условные рёбра и компилируемый граф, удобный для визуализации и отладки. Предпочтён raw asyncio-цепочкам ради поддерживаемости.
- **ChromaDB для индексации схемы** -- легковесное встраиваемое векторное хранилище. Каждая таблица индексируется как документ с колонками, примерами значений, первичными/внешними ключами. Персистентное хранение на диске (`.cache/chroma`).
- **Формат mSchema** -- компактное текстовое представление схемы, передаваемое генераторам. Включает имена таблиц, имена/типы колонок, примеры значений и связи по внешним ключам.
- **Few-shot из train-сплита Spider** -- примеры сэмплируются для каждого кандидата с детерминированными seed'ами для воспроизводимости. Приоритет отдаётся примерам из той же БД.
- **Retry-политика через Tenacity** -- все вызовы LLM обёрнуты экспоненциальным backoff'ом (3 попытки, ожидание 1-8 сек) для обработки сбоев API.
- **Асинхронность повсюду** -- вызовы LLM, исполнение SQL, загрузка схемы -- всё через `asyncio`. Скрипт оценки поддерживает `--concurrency N` для параллельной обработки примеров.

### Структура проекта

```
text_to_sql_agent/
  agents/           # Реализация стадий пайплайна
    selector.py       Schema linking (векторный поиск + LLM-реранкинг)
    decomposer.py     Классификация сложности вопроса + декомпозиция
    generator.py      Ансамблевая генерация SQL (N кандидатов, async)
    execution_filter.py  Валидация SQL через SQLAlchemy
    judge.py          LLM-as-Judge отбор кандидатов
    refiner.py        Итеративная самокоррекция
  graph/            # Оркестрация LangGraph
    state.py          Контракт общего состояния (SQLAgentState TypedDict)
    pipeline.py       Связывание графа и условная маршрутизация
  tools/            # Общие утилиты
    llm_router.py     Централизованный LLM-клиент с маршрутизацией моделей и retry
    sql_executor.py   Асинхронное исполнение SQL (SQLAlchemy + aiosqlite)
    schema_loader.py  Парсинг схемы Spider и форматирование в mSchema
    vector_store.py   Клиент индексации/поиска ChromaDB
    few_shot.py       Пул few-shot примеров и сэмплирование
  prompts/          # Шаблоны промптов для каждой стадии
  evaluation/       # Скрипты бенчмарков (Spider v1, BIRD, Spider v2)
  config.py         # Конфигурация через Pydantic-settings (.env)
  main.py           # CLI точка входа для одиночного запроса
```

# Текущие результаты

## Spider dev: multi-agent run

Статус: `practically completed`. Прогон дошел до `1033/1034`, после чего последний хвост завис до финальной записи JSON. Зафиксированные метрики ниже используются как фактический итог этого запуска.

### Конфигурация запуска

- Command: `./.venv/bin/python -m text_to_sql_agent.evaluation.run_spider --concurrency 12 --prewarm`
- Primary generator: `google/gemini-2.5-pro`
- Secondary generator: `openai/gpt-oss-120b`
- Judge: `openai/gpt-4.1`
- Embeddings: `openai/text-embedding-3-large`
- Candidate budget:
  - `complex/unknown`: `5` (`3` primary + `2` secondary)
  - `moderate`: `2` (`2` primary)
  - `simple`: skip judge when a valid candidate already exists

### Зафиксированные метрики на момент остановки

- Progress: `1033/1034`
- Execution Accuracy (EX): `69%`
- Exact Match (EM): `21%`
- Error count: `45`
- Throughput snapshot: `1:24:20<00:15, 15.27s/q`
- Summary file: `outputs/spider_v1_summary_practical_latest.json`

### Интерпретация

- По текущему тренду этот запуск выглядел лучше baseline по `EX`.
- Последний хвост был испорчен нехваткой кредитов, но на итоговые метрики это уже вряд ли влияло существенно.
- Для строгой повторяемости все равно полезно позже сделать еще один полный чистый прогон без `402` ошибок.

### Ближайший план экспериментов

- Основные архитектурные итерации сначала гонять на фиксированном Spider debug subset (`150` запросов, `50/50/50` по сложности).
- Полный Spider запускать только для изменений, которые уже выглядят сильными на subset.
- Продолжать model ablation на Spider до выхода на устойчивую конфигурацию генераторов.
- Если метрики застрянут около текущего уровня, следующим архитектурным шагом добавить semantic retrieval для few-shot examples из `train_spider.json`.
- Делать это как отдельный retrieval-контур для train-примеров, а не смешивать его с текущей schema-vector retrieval логикой.
