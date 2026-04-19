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
  tool-calling, session memory, and user-confirmed write-SQL flow.
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
