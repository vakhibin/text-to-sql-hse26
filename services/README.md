# Services

Microservices for the conversational text-to-SQL agent platform. Each service
runs independently and communicates over HTTP.

The two agent runtimes live at repo root as reusable libraries, and each
FastAPI service is a thin HTTP wrapper over its corresponding runtime:

- `text_to_sql_agent/` → `services/text_to_sql_api/`
- `orchestrator_agent/` → `services/orchestrator_api/`

This mirrors the split so either runtime can be driven from notebooks, CLI
tools, or tests without pulling in FastAPI.

## Layout

- `text_to_sql_api/` — FastAPI wrapper around the core LangGraph text-to-SQL
  pipeline. Endpoints:
  - `GET  /health`
  - `GET  /databases` — list available databases under the configured schema root
  - `GET  /databases/{db_id}/schema` — full schema dump (tables, columns, PK/FK, sample values)
  - `POST /run` — run the full LangGraph pipeline: question → SQL → execute → rows
  - `POST /execute` — execute a given SQL (read-only, guardrail-enforced)
  - `POST /refine` — AST repair + tool-augmented LLM fix on a given SQL
  - `POST /modify` — natural-language edit to a SQL query (single LLM call)
  - `POST /explain` — natural-language explanation of a given SQL
- `orchestrator_api/` — FastAPI + LangGraph conversational agent with
  tool-calling, session memory (SQLite checkpointer by default; Postgres
  in Phase 10), and the user-confirmed write-SQL flow (Phase 9). Endpoints:
  - `GET    /health`
  - `POST   /chat` — one conversational turn per call
  - `GET    /sessions/{session_id}` — inspect current state (messages,
    active db, last SQL, etc.)
  - `DELETE /sessions/{session_id}` — best-effort session reset

  Tools available to the LLM:
  - Core (Phase 4):
    - `run_text_to_sql(question, db_id?, evidence?)` — full pipeline
    - `execute_sql(sql, db_id?)` — read-only execution
    - `explain_sql(sql, db_id?)` — plain-English explanation
  - Discovery (Phase 5):
    - `list_databases()` — catalog overview (db_ids + table counts)
    - `describe_database(db_id?)` — schema dump for one database
    - `switch_database(db_id)` — set the active database for the rest of
      the session; validates existence and clears stale `last_sql` / rows
    - `sample_table(table_name, db_id?, limit=5)` — preview first rows of
      a table (read-only, identifier-validated)
    - `search_table_values(table_name, column_name, search_term, db_id?)`
      — look up real literal values in a column so the LLM can use correct
      casing/spelling in `WHERE` clauses
  - SQL manipulation + history (Phase 6):
    - `fix_sql(sql?, db_id?, error_hint?)` — delegate to `/refine`; defaults
      to `last_sql` / `active_db_id`
    - `modify_sql(instruction, sql?, db_id?)` — delegate to `/modify` for
      user-driven edits (distinct from error repair)
    - `list_recent(limit=10)` — dump session `sql_history` (newest first)
    - `rerun(index=1)` — re-execute a past query by 1-based history index;
      uses the db_id stored on the entry. Falls back to `last_sql` when
      history is empty.
  - Result UX (Phase 7):
    - `summarize_results(limit=5)` — summarize the latest row preview stored
      in session state (zero network / LLM calls)
    - `export_results(format=markdown|csv|json)` — export the latest stored
      preview and save it as `last_result_export`
  - Write SQL guardrails (Phase 9):
    - `propose_write_sql(sql, db_id?, rationale?)` — store write/DDL SQL in
      `pending_confirmation` without executing it
    - `confirm_write_sql()` — execute the pending write only after explicit
      user confirmation via `/execute-confirmed`
    - `cancel_pending_confirmation()` — discard the pending write request

  Each tool that touches SQL appends an entry to `sql_history` (capped at
  20) so `list_recent` and `rerun` stay consistent across turns.

  Env knobs:
  - `ORCH_CHECKPOINTER_BACKEND` — `sqlite` (default) or `postgres` (Phase 10)
  - `ORCH_SQLITE_PATH` — override default `.cache/orchestrator/sessions.sqlite`
  - `TEXT_TO_SQL_API_URL` — base URL of `text_to_sql_api` (default `http://localhost:8001`)
  - `TEXT_TO_SQL_API_TIMEOUT_S` — per-request timeout (default `120`)
  - `ORCHESTRATOR_MAX_TOOL_STEPS` — max tool-calling iterations per turn (default `6`)
- `ui/` — Streamlit chat interface that talks to `orchestrator_api`.
  Phase 8+ UI features:
  - API URL, session id, user id, and active db controls in the sidebar
  - chat loop over `POST /chat`
  - reload/reset session controls over `/sessions/{id}`
  - display of `last_sql`, latest result metadata, `last_result_export`,
    pending write confirmations, and recent SQL history from the session snapshot

## Run locally (dev)

```bash
# Text-to-SQL service
uv run uvicorn services.text_to_sql_api.main:app --port 8001 --reload

# Orchestrator service
uv run uvicorn services.orchestrator_api.main:app --port 8002 --reload

# UI
uv run streamlit run services/ui/app.py
```

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## Tests

```bash
uv run pytest tests/services -v
```

## Run with Docker Compose

Phase 10 adds a three-service compose stack:

- `text-to-sql-api` on `http://localhost:8001`
- `orchestrator-api` on `http://localhost:8002`
- `ui` (Streamlit) on `http://localhost:8501`

Prepare env and data:

```bash
cp .env.compose.example .env
# edit OPENROUTER_API_KEY and model settings if needed
# ensure Spider data exists under ./databases/spider
```

Start the stack:

```bash
docker compose up --build
# or: make compose-up
```

Useful checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
open http://localhost:8501
```

Stop:

```bash
docker compose down
# or: make compose-down
```
