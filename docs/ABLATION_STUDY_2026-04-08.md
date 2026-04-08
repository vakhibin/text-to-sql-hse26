# Ablation Study Summary (2026-04-08)

Краткий рабочий документ по текущим ablation-экспериментам для переноса в диплом и презентацию.

## Что сравнивается

Ниже зафиксированы три экспериментальные точки:

1. `E1` - лучший на тот момент `balanced low-cost` full Spider dev run.
2. `E2` - более агрессивный дешёвый конфиг с большим few-shot budget, но без semantic few-shot retrieval.
3. `E3` - тот же агрессивный дешёвый конфиг, но с semantic few-shot retrieval.

Важно:
- `E1` относится к зафиксированному baseline-срезу до новой серии prompt-экспериментов.
- `E2` и `E3` запускались уже после точечного `generator` prompt patch, который усиливал:
  - точность `output shape`
  - буквальное копирование schema identifiers
  - запрет на лишние `CAST/TRIM/...`
  - более аккуратное использование `DISTINCT`

## Сводная таблица

| ID | Scope | Основная гипотеза | EX | EM | Errors | Avg time | Cost |
|---|---|---|---:|---:|---:|---:|---:|
| `E1` | full Spider dev (`1034`) | дешёвые генераторы + sketcher/judge дают сильный balance | `72.34%` | `33.66%` | `28` | `3.58s` | `$21.98` |
| `E2` | micro subset (`14`) | больше few-shot примеров без semantic retrieval | `21.43%` | `14.29%` | `0` | `20.10s` | `$0.351` |
| `E3a` | micro subset (`14`) | semantic few-shot retrieval помогает на проблемных БД | `42.86%` | `21.43%` | `2` | `47.14s` | `$0.301` |
| `E3b` | full Spider dev (`1034`) | semantic few-shot retrieval переносится на полный benchmark | `72.44%` | `31.53%` | `22` | `4.29s` | `$25.61` |

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

## Практический вывод для диплома / презентации

На данный момент есть две сильные опорные точки:

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

Именно в таком виде это удобно переносить в диплом:

- один конфиг как `best balanced configuration`
- второй конфиг как `best EX-oriented ablation`

## Короткие формулировки для слайдов

- `Ablation 1`: дешёвые генераторы + сильный sketcher/judge дали лучший practical balance: `72.34 EX / 33.66 EM / $21.98`.
- `Ablation 2`: увеличение few-shot контекста до `20` без semantic retrieval не дало существенного прироста на targeted subset.
- `Ablation 3`: semantic few-shot retrieval улучшил локальный subset и дал лучший `EX` на full Spider dev (`72.44`), но ухудшил `EM` и увеличил стоимость.
