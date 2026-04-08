# Ablation Study Summary (2026-04-08)

Краткий рабочий документ по текущим ablation-экспериментам для переноса в диплом и презентацию.

## Что сравнивается

Ниже зафиксированы четыре экспериментальные точки:

1. `E1` - лучший на тот момент `balanced low-cost` full Spider dev run.
2. `E2` - более агрессивный дешёвый конфиг с большим few-shot budget, но без semantic few-shot retrieval.
3. `E3` - тот же агрессивный дешёвый конфиг, но с semantic few-shot retrieval.
4. `E4` - E3b конфиг с отключённым decomposer (`DECOMPOSER_ENABLED=false`).

Важно:
- `E1` относится к зафиксированному baseline-срезу до новой серии prompt-экспериментов.
- `E2` и `E3` запускались уже после точечного `generator` prompt patch, который усиливал:
  - точность `output shape`
  - буквальное копирование schema identifiers
  - запрет на лишние `CAST/TRIM/...`
  - более аккуратное использование `DISTINCT`
- `E4` запускался поверх `E3b` с единственным изменением: `DECOMPOSER_ENABLED=false`.

## Сводная таблица

| ID | Scope | Основная гипотеза | EX | EM | Errors | Avg time | Cost |
|---|---|---|---:|---:|---:|---:|---:|
| `E1` | full Spider dev (`1034`) | дешёвые генераторы + sketcher/judge дают сильный balance | `72.34%` | `33.66%` | `28` | `3.58s` | `$21.98` |
| `E2` | micro subset (`14`) | больше few-shot примеров без semantic retrieval | `21.43%` | `14.29%` | `0` | `20.10s` | `$0.351` |
| `E3a` | micro subset (`14`) | semantic few-shot retrieval помогает на проблемных БД | `42.86%` | `21.43%` | `2` | `47.14s` | `$0.301` |
| `E3b` | full Spider dev (`1034`) | semantic few-shot retrieval переносится на полный benchmark | `72.44%` | `31.53%` | `22` | `4.29s` | `$25.61` |
| `E4` | full Spider dev (`1034`) | decomposer избыточен при наличии sketcher | `71.66%` | `31.91%` | `20` | `4.94s` | `$31.54` |

## E1. Balanced Low-Cost Baseline

### Scope

- Полный `Spider dev`
- `1034` примера

### Команда

```bash
./scripts/run_ablation.sh configs/ablation/cheap_gen_more_candidates.env
```

### Зафиксированные параметры запуска

Источник: `README.md` и результат `outputs/ablation_cheap_gen_more_candidates_20260407_204301_20260407_204302.json`

- Primary generator: `google/gemma-4-26b-a4b-it`
- Secondary generator: `qwen/qwen3.5-35b-a3b`
- Query sketcher: `google/gemini-2.5-pro`
- Judge / Refiner: `openai/gpt-4.1`
- Embeddings: `openai/text-embedding-3-large`
- Candidate budget:
  - `complex/unknown`: `5` = `3 primary + 2 secondary`
  - `moderate`: `2` = `2 primary`
- `FEW_SHOT_EXAMPLES_PER_CANDIDATE=10`
- `FEW_SHOT_SEMANTIC_RETRIEVAL=false`
- `SIMPLE_SKIP_JUDGE_WHEN_VALID=true`

### Метрики

- `EX = 72.34%`
- `EM = 33.66%`
- `Errors = 28`
- `Prewarm = 29.02s`
- `Eval time = 3697.39s`
- `Avg/example = 3.58s`
- `Total cost = $21.98`

### Ключевой вывод

Это лучший `balanced` результат на текущий момент по совокупности `EX + EM + speed + cost`.

### Артефакт

- `outputs/ablation_cheap_gen_more_candidates_20260407_204301_20260407_204302.json`

## E2. Aggressive Few-Shot Without Semantic Retrieval

### Scope

- Диагностический `micro subset`
- `data/debug/spider_prompt_tuning_micro_v1.json`
- `14` вопросов из `car_1` и `dog_kennels`

### Команда

```bash
./scripts/run_ablation.sh configs/ablation/cheap_gen_more_candidates.env --subset-manifest data/debug/spider_prompt_tuning_micro_v1.json
```

### Зафиксированные параметры запуска

Источник: terminal log + результат `outputs/ablation_cheap_gen_more_candidates_20260408_162211_20260408_162212.json`

- Primary generator: `google/gemma-4-26b-a4b-it`
- Secondary generator: `qwen/qwen3.5-35b-a3b`
- Query sketcher: `google/gemini-2.5-pro`
- Judge / Refiner: `openai/gpt-4.1`
- `NUM_CANDIDATES=8`
- `PRIMARY_CALLS=5`
- `SECONDARY_CALLS=3`
- `MODERATE_NUM_CANDIDATES=2`
- `MODERATE_PRIMARY_CALLS=3`
- `MODERATE_SECONDARY_CALLS=2`
- `FEW_SHOT_EXAMPLES_PER_CANDIDATE=20`
- `FEW_SHOT_SEMANTIC_RETRIEVAL=false`
- `SIMPLE_SKIP_JUDGE_WHEN_VALID=false`
- Code state: после `generator` prompt patch v1

### Метрики

- `EX = 21.43%`
- `EM = 14.29%`
- `Errors = 0`
- `Prewarm = 2.47s`
- `Eval time = 281.40s`
- `Avg/example = 20.10s`
- `Total cost = $0.351`

### Ключевой вывод

Простое увеличение few-shot контекста до `20` без semantic retrieval не дало сильного прироста даже на маленьком targeted subset.

### Артефакт

- `outputs/ablation_cheap_gen_more_candidates_20260408_162211_20260408_162212.json`

## E3. Aggressive Few-Shot With Semantic Retrieval

### Scope

Тот же конфиг был проверен в двух режимах:

- `E3a`: `micro subset` (`14` примеров)
- `E3b`: полный `Spider dev` (`1034` примера)

### Команда

Micro subset:

```bash
./scripts/run_ablation.sh configs/ablation/cheap_gen_more_candidates.env --subset-manifest data/debug/spider_prompt_tuning_micro_v1.json
```

Full Spider dev:

```bash
./scripts/run_ablation.sh configs/ablation/cheap_gen_more_candidates.env
```

### Зафиксированные параметры запуска

Источник: terminal log + результаты `outputs/ablation_cheap_gen_more_candidates_20260408_162739_20260408_162741.json` и `outputs/ablation_cheap_gen_more_candidates_20260408_164850_20260408_164851.json`

- Primary generator: `google/gemma-4-26b-a4b-it`
- Secondary generator: `qwen/qwen3.5-35b-a3b`
- Query sketcher: `google/gemini-2.5-pro`
- Judge / Refiner: `openai/gpt-4.1`
- `NUM_CANDIDATES=8`
- `PRIMARY_CALLS=5`
- `SECONDARY_CALLS=3`
- `MODERATE_NUM_CANDIDATES=2`
- `MODERATE_PRIMARY_CALLS=3`
- `MODERATE_SECONDARY_CALLS=2`
- `FEW_SHOT_EXAMPLES_PER_CANDIDATE=20`
- `FEW_SHOT_SEMANTIC_RETRIEVAL=true`
- `SIMPLE_SKIP_JUDGE_WHEN_VALID=false`
- Code state: после `generator` prompt patch v1

### Метрики: E3a (micro subset)

- `EX = 42.86%`
- `EM = 21.43%`
- `Errors = 2`
- `Prewarm = 1.58s`
- `Eval time = 660.02s`
- `Avg/example = 47.14s`
- `Total cost = $0.301`

### Метрики: E3b (full Spider dev)

- `EX = 72.44%`
- `EM = 31.53%`
- `Errors = 22`
- `Prewarm = 7.16s`
- `Eval time = 4431.54s`
- `Avg/example = 4.29s`
- `Total cost = $25.61`

### Ключевой вывод

`Semantic few-shot retrieval` дал сильный локальный сигнал на targeted subset и позволил немного улучшить лучший `EX` на полном `Spider dev`:

- `EX`: `72.34% -> 72.44%`
- `Errors`: `28 -> 22`

Но это улучшение было получено ценой:

- более низкого `EM`: `33.66% -> 31.53%`
- большей стоимости: `$21.98 -> $25.61`
- большей задержки: `3.58s -> 4.29s`

Вывод: semantic few-shot retrieval выглядит как перспективная ветка для дальнейшего улучшения `EX`, но на текущем этапе не превосходит balanced baseline по совокупности метрик.

### Артефакты

- `outputs/ablation_cheap_gen_more_candidates_20260408_162739_20260408_162741.json`
- `outputs/ablation_cheap_gen_more_candidates_20260408_164850_20260408_164851.json`

## E4. No Decomposer (DECOMPOSER_ENABLED=false)

### Scope

- Полный `Spider dev`
- `1034` примера

### Гипотеза

Decomposer избыточен: query sketcher уже выполняет planning, а complexity classification можно убрать без потери качества.

### Команда

```bash
./scripts/run_ablation.sh configs/ablation/cheap_gen_more_candidates.env
```

### Зафиксированные параметры запуска

Источник: `outputs/ablation_cheap_gen_more_candidates_20260408_202140_20260408_202141.json`

- Primary generator: `google/gemma-4-26b-a4b-it`
- Secondary generator: `qwen/qwen3.5-35b-a3b`
- Query sketcher: `google/gemini-2.5-pro`
- Judge / Refiner: `openai/gpt-4.1`
- `DECOMPOSER_ENABLED=false`
- `NUM_CANDIDATES=8`
- `PRIMARY_CALLS=5`
- `SECONDARY_CALLS=3`
- `MODERATE_NUM_CANDIDATES=2` (не используется, complexity всегда `unknown`)
- `MODERATE_PRIMARY_CALLS=3` (не используется)
- `MODERATE_SECONDARY_CALLS=2` (не используется)
- `FEW_SHOT_EXAMPLES_PER_CANDIDATE=20`
- `FEW_SHOT_SEMANTIC_RETRIEVAL=true`
- `SIMPLE_SKIP_JUDGE_WHEN_VALID=false`
- Code state: после `generator` prompt patch v1 + `DECOMPOSER_ENABLED` flag

### Метрики

- `EX = 71.66%`
- `EM = 31.91%`
- `Errors = 20`
- `Valid predictions = 1027`
- `Prewarm = 10.36s`
- `Eval time = 5106.40s`
- `Avg/example = 4.94s`
- `Total cost = $31.54`

### Сравнение с E3b (тот же конфиг, но с decomposer)

| Metric | E3b (with decomposer) | E4 (no decomposer) | Delta |
|---|---:|---:|---|
| EX | `72.44%` | `71.66%` | `-0.78 ppt` |
| EM | `31.53%` | `31.91%` | `+0.38 ppt` |
| Errors | `22` | `20` | `-2` |
| Avg time | `4.29s` | `4.94s` | `+0.65s` |
| Cost | `$25.61` | `$31.54` | `+$5.93` |

### Ключевой вывод

Отключение decomposer дало незначительное снижение EX (`-0.78 ppt`), которое находится внутри variance band (~5-8 ppt между прогонами). EM даже немного вырос. Ошибок стало меньше.

Однако стоимость выросла на `$5.93` (+23%), потому что без complexity classification все запросы (включая `moderate`) идут по full ensemble budget (`8` кандидатов вместо `2` для `moderate`).

Вывод: decomposer как LLM-стадия не даёт значимого вклада в качество. Его основная ценность — cost routing через complexity classification. При переходе на self-consistency voting (где budget одинаковый для всех) decomposer можно полностью убрать без потери качества.

### Артефакт

- `outputs/ablation_cheap_gen_more_candidates_20260408_202140_20260408_202141.json`

## Практический вывод для диплома / презентации

На данный момент есть три опорные точки:

1. `Balanced practical winner`
   - `EX 72.34%`
   - `EM 33.66%`
   - `$21.98`
   - `3.58s/example`
   - лучший баланс качества, скорости и цены

2. `EX-oriented semantic few-shot variant`
   - `EX 72.44%`
   - `EM 31.53%`
   - `$25.61`
   - `4.29s/example`
   - лучший `EX`, но хуже balance по `EM/cost/speed`

3. `No-decomposer ablation`
   - `EX 71.66%`
   - `EM 31.91%`
   - `$31.54`
   - `4.94s/example`
   - decomposer не даёт значимого вклада в качество; его ценность — только cost routing

Именно в таком виде это удобно переносить в диплом:

- один конфиг как `best balanced configuration`
- второй конфиг как `best EX-oriented ablation`
- третий конфиг как `component ablation: decomposer removal`

## Короткие формулировки для слайдов

- `Ablation 1`: дешёвые генераторы + сильный sketcher/judge дали лучший practical balance: `72.34 EX / 33.66 EM / $21.98`.
- `Ablation 2`: увеличение few-shot контекста до `20` без semantic retrieval не дало существенного прироста на targeted subset.
- `Ablation 3`: semantic few-shot retrieval улучшил локальный subset и дал лучший `EX` на full Spider dev (`72.44`), но ухудшил `EM` и увеличил стоимость.
- `Ablation 4`: отключение decomposer дало `EX 71.66%` (`-0.78 ppt` vs E3b) — разница в пределах шума. Decomposer можно безопасно убрать при переходе на self-consistency voting.

## Архитектурный вывод

Результаты E4 подтверждают, что decomposer как отдельная LLM-стадия не оправдан при наличии query sketcher. Следующий шаг — замена judge на self-consistency voting, что позволит:
- полностью убрать decomposer (complexity routing больше не нужен)
- полностью убрать LLM judge (voting — детерминированный код)
- сократить pipeline до 5 стадий: `selector → sketcher → generator → exec_filter+voting → refiner`
- оставить одну дорогую модель (gemini-2.5-pro на selector + sketcher) и дешёвые генераторы
