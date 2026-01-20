# Baseline Text-to-SQL Solution

## Overview

This baseline implements a text-to-SQL system evaluated on the Spider dataset. The system uses a simple architecture: **schema linking** → **SQL generation** → **execution**.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Question +    │────▶│  Schema Linking  │────▶│  SQL Generator  │
│     DB ID       │     │   (Embeddings)   │     │      (LLM)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                         │
                                ▼                         ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Relevant Schema  │     │  Generated SQL  │
                        │ (top-k tables)   │     │                 │
                        └──────────────────┘     └─────────────────┘
```

## Components

### 1. Schema Linking (`app/schema_linker/`)

Schema linking selects the most relevant tables and columns for a given question, reducing the schema size passed to the LLM.

#### How it works:

1. **Load schema metadata** from Spider's `tables.json`:
   - Table names
   - Column names and types
   - Primary keys
   - Foreign key relationships

2. **Generate embeddings** for schema elements:
   - Each table is represented as: `"{table_name}: {column1}, {column2}, ..."`
   - Embeddings are computed using `qwen/qwen3-embedding-8b` via OpenRouter API

3. **Compute similarity** between question and schema elements:
   - Question is embedded using the same model
   - Cosine similarity is calculated between question embedding and each table embedding

4. **Select top-k tables** (default: k=5):
   - Tables are ranked by similarity score
   - Top-k tables are included in the prompt

5. **Generate CREATE TABLE statements**:
   - Selected tables are formatted as SQL DDL
   - Primary/foreign key constraints are included

#### Example:

**Question:** "How many singers do we have?"

**Full schema (4 tables):**
- stadium (7 columns)
- singer (7 columns)
- concert (5 columns)
- singer_in_concert (2 columns)

**After schema linking (top-1 relevant):**
```sql
CREATE TABLE singer (
  Singer_ID NUMBER PRIMARY KEY,
  Name TEXT,
  Country TEXT,
  Song_Name TEXT,
  Song_release_year TEXT,
  Age NUMBER,
  Is_male OTHERS
);
```

### 2. SQL Generator (`app/sql_generator/`)

The SQL generator uses an LLM to convert natural language questions to SQL queries.

#### Configuration:
- **Model:** `openai/gpt-oss-20b` via OpenRouter
- **Temperature:** 0.0 (deterministic)
- **Max tokens:** 512
- **Generation mode:** DIRECT (no chain-of-thought)

#### System Prompt:
```
You are an expert SQL assistant specialized in generating accurate SQL queries
from natural language questions.

RULES:
1. Output ONLY valid SQL code - no explanations, no markdown, no extra text
2. Use proper SQL syntax compatible with SQLite
3. Extract and use specific values from the question (numbers, names, dates, etc.)
4. Handle NULL values appropriately with IS NULL/IS NOT NULL
5. Use explicit JOIN syntax (INNER JOIN, LEFT JOIN) with ON clauses
6. Include all necessary WHERE conditions based on the question
7. End each statement with a semicolon

CRITICAL: Never include any reasoning, commentary, or explanations in your output.
Only SQL code.
```

#### User Prompt Format:
```
Database Schema:
{CREATE TABLE statements}

Question: {natural language question}
Database: {db_id}

SQL Query:
```

### 3. Pipeline (`app/pipeline/`)

The `TextToSQLPipeline` orchestrates the full workflow:

1. **Schema linking** (if enabled) — selects relevant tables
2. **SQL generation** — LLM generates SQL from question + schema
3. **Execution** (optional) — runs SQL against SQLite database
4. **Comparison** — compares predicted vs gold results

### 4. Evaluation (`app/evaluation/`)

#### Metrics:
- **Execution Accuracy (EX):** Predicted SQL returns same results as gold SQL
- **Exact Match (EM):** Normalized SQL strings are identical
- **Valid SQL Rate:** Percentage of syntactically valid SQL queries

#### Runner:
```bash
# Run on Spider dev set
uv run python -m app.evaluation.runner --split dev

# Run on subset
uv run python -m app.evaluation.runner --split dev --max-examples 100

# Without schema linking
uv run python -m app.evaluation.runner --split dev --no-schema-linking
```

## Results

### Spider Dev Set (1034 examples)

| Metric | Value |
|--------|-------|
| Execution Accuracy | 64.1% |
| Exact Match | 15.4% |
| Valid SQL Rate | 94.1% |
| Avg time/example | ~9.9 sec |

### Comparison with Other Systems

| System | EX |
|--------|-----|
| Simple GPT-3.5 baseline | ~55-60% |
| **This baseline** | **64%** |
| With few-shot / RAG | 70-75% |
| DAIL-SQL, DIN-SQL | 80-85% |
| SOTA (CHESS, MAC-SQL) | 87-90% |

## Configuration

Environment variables (`.env`):

```bash
# OpenRouter API
OPENROUTER_API_KEY=your-api-key

# LLM settings
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-oss-20b
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=512

# Embeddings (for schema linking)
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
```

## Limitations

1. **No few-shot examples** — LLM sees only the current question
2. **No error correction** — if SQL fails, no retry mechanism
3. **Simple schema linking** — only cosine similarity, no graph-based reasoning
4. **No query decomposition** — complex queries handled as single generation

## Potential Improvements

1. **Few-shot prompting** — add similar examples from training set
2. **Error correction** — retry with error message if SQL fails
3. **Self-consistency** — generate multiple SQLs, vote on best
4. **Better schema linking** — use question keywords, foreign keys graph
5. **Query decomposition** — break complex questions into sub-queries
