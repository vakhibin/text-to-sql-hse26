"""FastAPI app: OpenAI-compatible `/v1/chat/completions` and `/v1/models`."""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from text_to_sql_agent.config import settings
from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import SchemaLayout, make_initial_state
from text_to_sql_agent.tools.schema_loader import list_db_ids

# table_name -> list of (db_id, layout, schema_root); order: spider DBs first, then bird (later = preferred on duplicate names)
TABLE_MAP: dict[str, list[tuple[str, str, str]]] = {}

_AUTO_MODELS = frozenset({"auto", "auto:auto"})


def _is_auto_model(model: str) -> bool:
    """DBeaver may send `auto`, `auto:auto`, or whitespace variants."""
    m = model.strip().lower()
    return m in _AUTO_MODELS


def _openai_error(message: str, type_: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    stream: bool = False


def _parse_model_field(model: str) -> tuple[str, SchemaLayout]:
    """Return (db_id, layout). `dataset:db_id`, plain `db_id` (Spider), or empty db_id for auto mode."""
    raw = model.strip()
    if not raw:
        raise HTTPException(status_code=400, detail=_openai_error("model must be non-empty"))
    # Must not treat `auto:auto` as db_id — ":" branch would otherwise return literal "auto:auto".
    if _is_auto_model(raw):
        return "", "spider"
    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        p = prefix.strip().lower()
        db_id = rest.strip()
        if p in ("spider", "bird") and db_id:
            return db_id, p  # type: ignore[return-value]
    return raw, "spider"


def _message_text(msg: ChatMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _last_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        return _message_text(msg).strip()
    raise HTTPException(
        status_code=400,
        detail=_openai_error("No user message found in messages"),
    )


def _optional_api_token(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = settings.api_token
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=_openai_error("Missing or invalid Authorization", "invalid_request_error"))
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail=_openai_error("Incorrect API token", "invalid_request_error"))


@lru_cache(maxsize=1)
def _compiled_graph():
    return build_graph()


def _sql_content(final_sql: str | None, best_sql: str | None) -> str:
    sql = (final_sql or best_sql or "").strip()
    if not sql:
        return "-- No SQL produced"
    return sql


def _collect_model_ids() -> list[str]:
    out: list[str] = []
    for layout, root in (("spider", settings.spider_root), ("bird", settings.bird_root)):
        path = Path(root)
        if not path.is_dir():
            continue
        try:
            for db_id in list_db_ids(path):
                out.append(f"{layout}:{db_id}")
        except OSError:
            continue
        except FileNotFoundError:
            continue
    if not out:
        out = ["spider:concert_singer"]
    return sorted(out)


def _find_dataset_root_for_sqlite(db_file: Path) -> Path | None:
    """Walk parents until a directory contains tables.json or dev_tables.json."""
    cur = db_file.parent
    for _ in range(12):
        if (cur / "tables.json").exists() or (cur / "dev_tables.json").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _table_map_add(table_low: str, entry: tuple[str, str, str]) -> None:
    if table_low not in TABLE_MAP:
        TABLE_MAP[table_low] = []
    if entry not in TABLE_MAP[table_low]:
        TABLE_MAP[table_low].append(entry)


def init_table_map() -> None:
    configs = [
        ("spider", settings.spider_root),
        ("bird", settings.bird_root),
    ]
    for layout, root_dir in configs:
        path = Path(root_dir)
        if not path.is_dir():
            continue
        for db_file in path.glob("**/*.sqlite"):
            db_id = db_file.stem
            dataset_root = _find_dataset_root_for_sqlite(db_file)
            if dataset_root is None:
                dataset_root = path
            schema_root_str = str(dataset_root.resolve())
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                raw_names = [row[0] for row in cursor.fetchall()]
                conn.close()
            except (OSError, sqlite3.DatabaseError):
                continue
            for name in raw_names:
                tlow = name.lower()
                if tlow.startswith("sqlite_"):
                    continue
                _table_map_add(tlow, (db_id, layout, schema_root_str))


# DDL: CREATE TABLE [IF NOT EXISTS] `name` | "name" | 'name' | [name] | bare_identifier
_DDL_TABLE_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|\[([^\]]+)\]|([a-zA-Z_][a-zA-Z0-9_]*))",
    re.IGNORECASE,
)


def _extract_table_names_from_ddl(context_text: str) -> list[str]:
    """Names appear in source order (first CREATE TABLE first)."""
    out: list[str] = []
    for m in _DDL_TABLE_PATTERN.finditer(context_text):
        raw = next((g for g in m.groups() if g is not None and str(g).strip()), None)
        if raw is None:
            continue
        name = raw.strip()
        if name.lower().startswith("sqlite_"):
            continue
        out.append(name)
    return out


def _all_messages_text(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(_message_text(m))
    return " ".join(parts)


init_table_map()

app = FastAPI(title="Text-to-SQL OpenAI-compatible API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def list_models(_: Annotated[None, Depends(_optional_api_token)]) -> dict[str, Any]:
    ids = _collect_model_ids()
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": created, "owned_by": "local"}
            for mid in ids
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    _: Annotated[None, Depends(_optional_api_token)],
) -> JSONResponse:
    if body.stream:
        return JSONResponse(
            status_code=501,
            content=_openai_error("Streaming is not supported; set stream to false", "not_implemented_error"),
        )

    db_id, layout = _parse_model_field(body.model)
    schema_root = settings.spider_root if layout == "spider" else settings.bird_root

    if _is_auto_model(body.model):
        context_text = _all_messages_text(body.messages)
        found_tables = _extract_table_names_from_ddl(context_text)
        for table in found_tables:
            table_low = table.lower()
            if table_low in TABLE_MAP:
                candidates = TABLE_MAP[table_low]
                db_id, layout, schema_root = candidates[-1]
                break
        if not db_id:
            return JSONResponse(
                status_code=400,
                content=_openai_error(
                    "Auto mode could not resolve a database: no CREATE TABLE in the request matched a "
                    "known table (index built from local Spider/BIRD sqlite files). "
                    "Include DDL with CREATE TABLE … for at least one real table, or set model to "
                    "`bird:<db_id>` / `spider:<db_id>`.",
                    "invalid_request_error",
                ),
            )

    question = _last_user_text(body.messages)
    if not question:
        return JSONResponse(status_code=400, content=_openai_error("Empty user message"))

    trace_id = str(uuid.uuid4())
    cmpl_id = f"chatcmpl-{trace_id[:8]}"

    initial = make_initial_state(
        question=question,
        db_id=db_id,
        evidence=None,
        trace_id=trace_id,
        schema_root=schema_root,
        schema_layout=layout,
    )

    try:
        result = await _compiled_graph().ainvoke(initial)
    except Exception as exc:  # pragma: no cover - defensive
        return JSONResponse(
            status_code=500,
            content=_openai_error(str(exc), "server_error", "internal_error"),
        )

    err = result.get("error_message")
    if err and not (result.get("final_sql") or result.get("best_sql")):
        return JSONResponse(
            status_code=502,
            content=_openai_error(f"Pipeline failed: {err}", "server_error"),
        )

    content = _sql_content(result.get("final_sql"), result.get("best_sql"))
    payload = {
        "id": cmpl_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return JSONResponse(content=payload)
