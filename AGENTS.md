# AGENTS.md

## Project Purpose

This repository contains a multi-stage `text-to-sql` agent for a master's thesis.
The system is built around `LangGraph`, uses `OpenRouter` for LLM access, and targets benchmark evaluation on Spider and BIRD.

## Core Pipeline

The main pipeline lives in `text_to_sql_agent/graph/pipeline.py` and is:

1. `selector`
2. `decomposer`
3. `generator`
4. `execution_filter`
5. `judge`
6. `refiner`

The graph has conditional exits and one retry loop on `refiner`.

## Shared State

The shared LangGraph state is defined in `text_to_sql_agent/graph/state.py`.

Important fields:
- Input: `question`, `db_id`, `evidence`, `schema_root`
- Schema: `full_schema`, `filtered_schema`
- Decomposition: `complexity`, `sub_questions`
- Generation: `candidates`, `valid_candidates`
- Selection/refinement: `best_sql`, `final_sql`, `refine_attempts`, `error_message`
- Observability: `trace_id`, `warnings`, `stage_status`, `stage_timings`, `llm_usage`, `total_cost_usd`

When adding new agent behavior, prefer extending state explicitly instead of passing hidden globals.

## Model Policy

Configured in `text_to_sql_agent/config.py`.

Current intended roles:
- Primary generator: `google/gemini-2.5-pro`
- Secondary generator: `deepseek/deepseek-chat-v3`
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

Current runtime knobs live in `.env` / `config.py`:
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

`text_to_sql_agent/agents/execution_filter.py`:
- executes generated SQL
- drops invalid candidates before judging
- can promote the first valid candidate directly to `best_sql` for `simple` queries

`text_to_sql_agent/agents/judge.py`:
- selects best candidate
- should degrade gracefully on parse/provider failures
- may be skipped on the simple-query cheap path

`text_to_sql_agent/agents/refiner.py`:
- retries SQL correction using execution feedback
- must never destroy the last usable SQL candidate
- now runs a lightweight schema-reference validation before DB execution

`text_to_sql_agent/tools/sql_schema_validator.py`:
- uses `sqlglot` to parse SQLite SQL into an AST
- validates referenced tables against the loaded schema
- validates qualified and unqualified column references against available sources
- can surface deterministic schema errors to the refiner before the query reaches SQLite

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

## Benchmark Runners

Spider runner:
- `text_to_sql_agent/evaluation/run_spider.py`

BIRD runner:
- `text_to_sql_agent/evaluation/run_bird.py`

Current runner expectations:
- support `--prewarm`
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
