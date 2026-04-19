# Services

Microservices for the conversational text-to-SQL agent platform. Each service
runs independently and communicates over HTTP.

## Layout

- `text_to_sql_api/` — FastAPI wrapper around the core LangGraph text-to-SQL
  pipeline. Endpoints:
  - `GET  /health`
  - `GET  /databases` — list available databases under the configured schema root
  - `GET  /databases/{db_id}/schema` — full schema dump (tables, columns, PK/FK, sample values)
  - `POST /run` — run the full LangGraph pipeline: question → SQL → execute → rows
  - `POST /execute` — execute a given SQL (read-only, guardrail-enforced)
  - `POST /refine` — AST repair + tool-augmented LLM fix on a given SQL
  - `POST /explain` — natural-language explanation of a given SQL
- `orchestrator_api/` — FastAPI + LangGraph conversational agent with
  tool-calling, session memory (SQLite checkpointer by default; Postgres
  in Phase 10), and the user-confirmed write-SQL flow (Phase 9). Endpoints:
  - `GET    /health`
  - `POST   /chat` — one conversational turn per call
  - `GET    /sessions/{session_id}` — inspect current state (messages,
    active db, last SQL, etc.)
  - `DELETE /sessions/{session_id}` — best-effort session reset

  Tools available to the LLM (Phase 4):
  - `run_text_to_sql(question, db_id?, evidence?)` — full pipeline
  - `execute_sql(sql, db_id?)` — read-only execution
  - `explain_sql(sql, db_id?)` — plain-English explanation

  Env knobs:
  - `ORCH_CHECKPOINTER_BACKEND` — `sqlite` (default) or `postgres` (Phase 10)
  - `ORCH_SQLITE_PATH` — override default `.cache/orchestrator/sessions.sqlite`
  - `TEXT_TO_SQL_API_URL` — base URL of `text_to_sql_api` (default `http://localhost:8001`)
  - `TEXT_TO_SQL_API_TIMEOUT_S` — per-request timeout (default `120`)
  - `ORCHESTRATOR_MAX_TOOL_STEPS` — max tool-calling iterations per turn (default `6`)
- `ui/` — Streamlit chat interface that talks to `orchestrator_api`.

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

Docker/compose packaging lands in Phase 10.
