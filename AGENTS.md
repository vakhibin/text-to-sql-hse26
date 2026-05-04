# AGENTS.md

## Project Purpose

This repository contains a multi-stage `text-to-sql` agent for a master's thesis.
The system is built around `LangGraph`, uses `OpenRouter` for LLM access, and targets benchmark evaluation on Spider and BIRD.

## Core Pipeline

The main pipeline lives in `text_to_sql_agent/graph/pipeline.py` and is:

1. `selector`
2. `decomposer`
3. `sketcher`
4. `generator`
5. `execution_filter`
6. `judge`
7. `refiner`

The graph has conditional exits and one retry loop on `refiner`.

## Shared State

The shared LangGraph state is defined in `text_to_sql_agent/graph/state.py`.

Important fields:
- Input: `question`, `db_id`, `evidence`, `schema_root`
- Schema: `full_schema`, `filtered_schema`
- Decomposition/planning: `complexity`, `sub_questions`, `query_sketch`, `query_sketch_text`
- Generation: `candidates`, `valid_candidates`
- Selection/refinement: `best_sql`, `final_sql`, `refine_attempts`, `error_message`
- Observability: `trace_id`, `warnings`, `stage_status`, `stage_timings`, `llm_usage`, `total_cost_usd`

When adding new agent behavior, prefer extending state explicitly instead of passing hidden globals.

## Model Policy

Configured in `text_to_sql_agent/config.py`.

Current intended roles:
- Primary generator: `google/gemini-2.5-pro`
- Secondary generator: `deepseek/deepseek-chat-v3`
- Query sketcher: configurable separately via `QUERY_SKETCHER_MODEL` (fallback: primary generator)
- Judge: `openai/gpt-4.1`
- Embeddings: `openai/text-embedding-3-large`

Do not mix up generator/judge model changes with embedding-model changes:
- Changing generator/judge models does not invalidate schema cache.
- Changing embedding model must use a different Chroma namespace.

## Schema Loading And mSchema

Implemented in `text_to_sql_agent/tools/schema_loader.py`.

Important behavior:
- Loads schema from `tables.json` or `dev_tables.json`
- Resolves SQLite DB paths across Spider/BIRD-style layouts
- Caches:
  - tables JSON by `schema_root`
  - schema by `(schema_root, db_id, with_sample_values, sample_limit)`
  - mSchema lines by the same key

`schema_to_mschema(...)` is the preferred cached renderer.

## Vector Retrieval

Implemented in `text_to_sql_agent/tools/vector_store.py`.

Current design:
- Chroma stores one document per table
- Document content includes table name, columns, sample values, PK/FK info
- Chroma collection name is embedding-aware
- In-process indexing guard is tied to:
  - persist directory
  - collection name
  - embedding model
  - `db_id`

This is intentional. Do not revert to a model-agnostic Chroma collection.

## Selector Rules

`text_to_sql_agent/agents/selector.py` does:
- schema load
- Chroma top-k retrieval
- LLM reranking
- fallback/padding when reranker output is incomplete

If reranker output is malformed:
- keep pipeline alive
- prefer fallback to vector candidates over hard failure

## Complexity Classification

`text_to_sql_agent/agents/decomposer.py` classifies into:
- `simple`
- `moderate`
- `complex`
- `unknown`

This currently feeds generator prompting.
It now also feeds routing policy:
- `simple`: eligible for a cheap path after execution validation
- `moderate`: uses a reduced generator ensemble budget
- `complex` and `unknown`: stay on the fuller default ensemble

The decomposer can be bypassed with `DECOMPOSER_ENABLED=false`.
In that mode, downstream stages should treat decomposition outputs conservatively:
- `complexity = unknown`
- `sub_questions = []`
- default empty risk flags

Current runtime knobs live in `.env` / `config.py`:
- `DECOMPOSER_ENABLED`
- `NUM_CANDIDATES`, `PRIMARY_CALLS`, `SECONDARY_CALLS`
- `MODERATE_NUM_CANDIDATES`, `MODERATE_PRIMARY_CALLS`, `MODERATE_SECONDARY_CALLS`
- `SIMPLE_SKIP_JUDGE_WHEN_VALID`

Any optimization based on complexity should still be benchmarked against EX/EM before becoming default.

## Generation And Validation

`text_to_sql_agent/agents/generator.py`:
- generates an ensemble of candidates asynchronously
- uses model-role routing
- includes few-shot when available
- adapts candidate budget by `complexity`
- now consumes `query-sketcher` output as a planning scaffold before writing SQL

`text_to_sql_agent/agents/query_sketcher.py`:
- runs after `decomposer` and before `generator`
- produces a compact schema-grounded query plan instead of SQL
- should identify likely tables, join path, filters, aggregations, grouping, ordering, and subquery need
- must stay conservative: ambiguity should become an explicit risk, not a hallucinated identifier
- now uses a three-step reliability path:
  - tolerant JSON parsing of the raw response
  - structured-output repair when the raw JSON is malformed
  - deterministic fallback sketch so `generator` still receives a planning scaffold

Current few-shot status:
- few-shot examples are loaded from `train_spider.json`
- examples are sampled deterministically, with preference for the same `db_id` when possible
- semantic retrieval over train examples now exists as an optional experiment path
- if Spider metrics plateau, treat few-shot retrieval as a tunable lever rather than a guaranteed improvement

`text_to_sql_agent/agents/execution_filter.py`:
- executes generated SQL
- drops invalid candidates before judging
- includes refusal-SQL guardrail (`_is_refusal_sql`) to detect LLM refusals wrapped as `SELECT 'I cannot...'`
- collects structural diagnostics per candidate via `sql_candidate_analysis`
- can promote the best valid candidate directly to `best_sql` for `simple` queries (soft cheap-path based on SQL structure, not LLM risk flags)

`text_to_sql_agent/agents/judge.py`:
- selects best candidate from a lean prompt: question + evidence + schema + compact candidate SQL with structural summaries
- should degrade gracefully on parse/provider failures
- may be skipped on the simple-query cheap path
- outputs structured signals: `confidence`, `needs_refine`, `issues` (from a fixed vocabulary)
- lesson learned: overloading judge with sketch/risk-flags/diffs/rejected candidates hurts selection quality — keep it lean

`text_to_sql_agent/agents/refiner.py`:
- retries SQL correction using execution feedback
- must never destroy the last usable SQL candidate
- now runs a lightweight schema-reference validation before DB execution
- is a natural place for future deterministic repair tools before or alongside the LLM fix step

`text_to_sql_agent/tools/sql_schema_validator.py`:
- uses `sqlglot` to parse SQLite SQL into an AST
- validates referenced tables against the loaded schema
- validates qualified and unqualified column references against available sources
- can surface deterministic schema errors to the refiner before the query reaches SQLite

Evaluation metrics (`text_to_sql_agent/evaluation/metrics.py`):
- EX uses official Spider `result_eq`: column permutation search + multiset bag semantics + ORDER BY awareness
- EM uses AST-based normalization via `sqlglot`: alias resolution → table-name substitution → single-table qualifier stripping → lowercase canonical SQL comparison, with string fallback

Planned architectural follow-ups:
- `ast-repair` tool: use SQL AST-based deterministic repair inside `refiner` for obvious table/column/qualification fixes
- `self-consistency voting`: pick candidate by majority execution result instead of LLM judge
- `value linking`: look up actual DB values before generation to fix literal casing/spelling

## Cost And Observability

Centralized in:
- `text_to_sql_agent/tools/llm_router.py`
- `text_to_sql_agent/tools/observability.py`

Rules:
- Capture usage/cost at the router layer, not separately in each agent
- Preserve `trace_id` across all LLM calls
- Keep benchmark output compact: aggregate run-level cost, do not dump all per-call usage into prediction rows

Langfuse is optional and controlled by env:
- `LANGFUSE_ENABLED`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`

When Langfuse is enabled, each `/run` produces a hierarchical trace:
- root span `text_to_sql_run` (chain), opened by `pipeline_adapter.run_pipeline`
- one child span per LangGraph stage (`selector`, `value_linker`, `sketcher`,
  `generator`, `execution_filter`, `voting`, `refiner`), opened by
  `text_to_sql_agent.graph.tracing.trace_pipeline_stage`
- LLM generations from `LLMRouter.ainvoke_with_metadata` automatically attach
  as children of the current stage span via OTEL parent context

Stage span input/output is verbose-but-bounded: nested dicts are kept whole,
strings are truncated to ~4KB and lists to ~50 items by
`safe_state_snapshot`. Tracing is best-effort — when Langfuse is disabled or
the SDK fails, the wrappers fall through to no-op contexts and the pipeline
runs unchanged.

## Benchmark Runners

Spider runner:
- `text_to_sql_agent/evaluation/run_spider.py`

BIRD runner:
- `text_to_sql_agent/evaluation/run_bird.py`

Current runner expectations:
- support `--prewarm`
- support `--subset-manifest` for stable cheap Spider debug runs
- emit timing summary:
  - `prewarm_time_s`
  - `eval_time_s`
  - `avg_time_per_example_s`
- emit cost summary:
  - `total_cost_usd`
  - `avg_cost_per_example_usd`

When editing runners:
- keep timestamped outputs
- preserve progress bars and live error logging
- do not silently remove benchmark-level metadata
- keep subset metadata in payloads when running from a fixed manifest

## Current Results (baseline for future changes)

### Full Spider v1 dev (1034 examples)

| Date | EX | EM | Errors | Avg time | Cost | Notes |
|------|----|----|--------|----------|------|-------|
| 2026-04-05 | **72.92%** | **29.11%** | 24 (2.3%) | 4.49s/q | $67.78 | official EX + AST EM, full pipeline |
| baseline | 64.22% | ~21% | 45 (4.4%) | 15.27s/q | — | gpt-oss-120b single generator |

Improvement over baseline: **+8.7 ppt EX, +8 ppt EM, 2x fewer errors, 3.4x faster**.

### Debug subset v1 (150 examples)

| Date | EX | EM | Eval method | Generator primary | Generator secondary | Notes |
|------|----|----|-------------|-------------------|---------------------|-------|
| 2026-04-05 | **80.0%** | **40.7%** | official EX + AST EM | gemini-2.5-pro | gpt-oss-120b | lean judge + prompt tuning + official EX + AST EM |
| 2026-04-05 | 73.3% | — | official EX (reeval of old run) | gemini-2.5-pro | gpt-oss-120b | same model, before prompt tuning |
| 2026-04-05 | 67.3% | 19.3% | naive `==` | gemini-2.5-pro | gpt-oss-120b | old metrics, same pipeline |

Key changes that drove improvement:
- **+6 ppt EX**: switched to official Spider `result_eq` (column permutation, multiset row comparison)
- **+3 ppt EX**: lean judge prompt (removed sketch/risk-flags/sub-questions/diffs/rejected from judge context)
- **+2 ppt EX**: generator prompt hardening (SELECT *, no unnecessary JOIN, projection order)
- **+2 ppt EX**: soft cheap-path + refusal SQL guardrail
- **+8 ppt EM**: AST-based `_canonical_sql()` via `sqlglot` (alias resolution, single-table qualifier stripping, lowercase canonicalization)

Config: `NUM_CANDIDATES=5`, `PRIMARY_CALLS=3`, `SECONDARY_CALLS=2`, judge=`gpt-4.1`, sketcher=`gemini-2.5-pro`.
Cost: ~$9.4 per 150-example debug run, ~$68 per full 1034-example run. Avg 4.5s/example at concurrency 12.

## Pipeline Stability Assessment

Analysis based on 4+ healthy debug-v1 runs (150 examples each, error rate < 5%):

| Metric | Value | Notes |
|--------|-------|-------|
| Deterministic core (always pass) | 95/150 (63%) | These examples pass in every run |
| Deterministic fail (always fail) | 27/150 (18%) | These examples fail in every run |
| Flaky (nondeterministic) | 28/150 (19%) | Pass in some runs, fail in others |
| EX floor | ~63% | Only deterministic passes |
| EX ceiling | ~82% | Deterministic + all flaky pass |
| SQL prediction stability | ~50% | Only half of predictions are identical across back-to-back runs |
| Error rate (latest) | 0.7% (1/150) | Very stable infrastructure |

Nondeterminism is inherent in the LLM ensemble (temperature > 0, multiple generators).
Variance band: **~5-8 ppt** between runs from LLM randomness alone.

Full Spider (1034 examples) actual:
- Time: 1:17:24 at concurrency 12
- Cost: $67.78
- Avg: 4.49s/example
- Errors: 24 (2.3%)

## Iterative Improvement Workflow

1. Run full v1 subset (150 examples) → establish baseline EX
2. Build failure subset from results: `data/debug/spider_v1_failures.json`
3. Apply targeted fixes (prompt tuning, routing, guardrails)
4. Test on failure subset (~50 examples, ~$1-1.5, ~3 min)
5. When failure subset improves, re-run full v1 to check for regressions
6. When v1 EX is stable, run full Spider (1034) for final number

Do not optimize to the failure subset — use it as a diagnostic tool only.

## Current Experiment Plan

Near-term tuning priority:
- use the fixed Spider debug subset at `data/debug/spider_dev_subset_v1.json` for most architecture iterations
- use `data/debug/spider_v1_failures.json` (49 examples) for cheap targeted iteration
- reserve full Spider dev runs for changes that already look promising on the subset
- next priorities: semantic few-shot retrieval, self-consistency voting, value linking
- then continue model-stack ablations (Gemma 4 on secondary, etc.) and EM hardening
- promote changes to BIRD only after Spider EX is stable

Spider debug subset policy:
- current manifest target is `150` examples with `50 simple / 50 moderate / 50 complex`
- build or refresh via `scripts/build_spider_debug_subset.py`
- keep the subset fixed and versioned; create `v2` only as a deliberate benchmark change

Few-shot retrieval direction to preserve:
- index train examples separately from schema-table retrieval
- compare dev questions against train questions semantically
- prefer validated or otherwise strong examples when building the retrieval pool
- keep this retrieval path distinct from schema linking in Chroma

Query-sketcher direction to preserve:
- output a compact structured plan rather than full SQL
- capture tables, join intent, filters, grouping, ordering, and whether subqueries are needed
- feed the sketch into `generator` as a grounding scaffold, not as an end-user artifact
- keep the prompt strict about schema grounding and explicit about uncertainty reporting

Judge hardening direction to preserve:
- consider a stronger judge model as an experiment axis, not as a silent default
- prefer richer comparison context over longer free-form reasoning
- explicitly penalize unnecessary joins, wrong projection shape, bag-semantics mistakes, and schema drift

Generator prompt direction to preserve:
- preserve output column order to follow the question wording unless the question explicitly asks for another order
- prefer the simplest valid query shape; avoid joins when a single-table query is sufficient
- keep output shape faithful to the requested projection before optimizing for stylistic SQL preferences

Evaluation analysis direction to preserve:
- add analytic labels for near-miss failures instead of treating all mismatches as one bucket
- first useful labels:
  - `projection-order mismatch`
  - `projection-width mismatch`
  - `duplicate-row mismatch`
  - `unnecessary-join mismatch`
  - `aggregation-shape mismatch`

AST-repair direction to preserve:
- keep deterministic repairs narrow and reversible
- target obvious hallucinated identifiers, missing qualifications, and simple syntax/structure cleanups
- fall back to the LLM refiner for semantic fixes that code cannot safely infer

## Safety Notes

- Do not cache model outputs such as `best_sql`, `selected_tables`, `sub_questions`, or `final_sql`.
- Do not assume Spider and BIRD have the same schema layout.
- Prefer graceful degradation to hard failure in selector/decomposer/judge.
- If changing routing or fast-path logic, compare against `docs/baseline_results.md`.
- Keep `AGENTS.md`, `.env.example`, and `.env` aligned when changing generator budgets or cheap-path flags.
- Keep schema validation lightweight and non-destructive: it should catch obvious table/column mistakes, not over-constrain valid SQL patterns.

## Good Next Places To Look

- Architecture overview: `docs/ARCHITECTURE.md`
- Benchmark baseline reference: `docs/baseline_results.md`
- Project roadmap: `PROJECT_PLAN.md`
- Pipeline graph export: `docs/pipeline_graph.png`
