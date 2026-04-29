#!/usr/bin/env python3
"""End-to-end smoke test for the conversational orchestrator API.

This script talks to the already-running ``orchestrator_api`` over HTTP. It is
meant for manual/debug runs after ``docker compose up``:

  uv run python scripts/run_orchestrator_smoke.py

Default scenario:
1. select a Spider dev database,
2. ask a real text-to-SQL question,
3. ask the agent to modify the last SQL and rerun it,
4. ask for a result summary/export.

The script checks session artifacts after each step and exits non-zero when the
orchestrator fails to produce SQL/history/results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import uuid4


DEFAULT_BASE_URL = "http://localhost:8002"
DEFAULT_DB_ID = "concert_singer"
DEFAULT_QUESTION = "Show the stadium names without any concert."
DEFAULT_MODIFICATION = "Modify the last SQL to order the stadium names alphabetically."
DEFAULT_RERUN_REQUEST = "Run the modified SQL."
DEFAULT_SUMMARY_REQUEST = "Summarize the latest result and export it as markdown."


class SmokeFailure(RuntimeError):
    """Raised when the smoke scenario does not reach an expected state."""


def _post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SmokeFailure(f"POST {url} failed: HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise SmokeFailure(f"POST {url} failed: {exc}") from exc


def _get_json(url: str, *, timeout: int) -> dict[str, Any]:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SmokeFailure(f"GET {url} failed: HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise SmokeFailure(f"GET {url} failed: {exc}") from exc


def _chat(
    *,
    base_url: str,
    session_id: str,
    user_id: str,
    message: str,
    active_db_id: str | None,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
    }
    if active_db_id:
        payload["active_db_id"] = active_db_id
    started = time.perf_counter()
    response = _post_json(f"{base_url}/chat", payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    print(f"\nUSER: {message}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"reply: {response.get('reply') or '(empty)'}")
    tool_names = [
        msg.get("name")
        for msg in response.get("messages_delta", [])
        if msg.get("role") == "tool" and msg.get("name")
    ]
    if tool_names:
        print(f"tools: {', '.join(tool_names)}")
    if response.get("last_sql"):
        print(f"last_sql: {response['last_sql']}")
    return response


def _session(base_url: str, session_id: str, *, timeout: int) -> dict[str, Any]:
    return _get_json(f"{base_url}/sessions/{session_id}", timeout=timeout)


def _history_sources(session: dict[str, Any]) -> list[str]:
    history = (session.get("extra") or {}).get("sql_history") or []
    return [str(entry.get("source") or "") for entry in history]


def _write_output(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote debug payload: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    session_id = args.session_id or f"smoke-{uuid4().hex[:8]}"
    user_id = args.user_id

    health = _get_json(f"{base_url}/health", timeout=args.timeout)
    print(f"orchestrator health: {health}")
    print(f"session_id: {session_id}")

    responses: list[dict[str, Any]] = []

    responses.append(
        _chat(
            base_url=base_url,
            session_id=session_id,
            user_id=user_id,
            message=f"Use {args.db_id} database.",
            active_db_id=args.db_id,
            timeout=args.timeout,
        )
    )
    if responses[-1].get("active_db_id") != args.db_id:
        raise SmokeFailure(f"active_db_id was not set to {args.db_id!r}")

    responses.append(
        _chat(
            base_url=base_url,
            session_id=session_id,
            user_id=user_id,
            message=args.question,
            active_db_id=args.db_id,
            timeout=args.timeout,
        )
    )
    if not responses[-1].get("last_sql"):
        raise SmokeFailure("initial text-to-SQL request did not produce last_sql")
    if not any(
        msg.get("role") == "tool" and msg.get("name") == "run_text_to_sql"
        for msg in responses[-1].get("messages_delta", [])
    ):
        raise SmokeFailure("initial request did not call run_text_to_sql")

    responses.append(
        _chat(
            base_url=base_url,
            session_id=session_id,
            user_id=user_id,
            message=args.modification,
            active_db_id=None,
            timeout=args.timeout,
        )
    )
    session_after_modify = _session(base_url, session_id, timeout=args.timeout)
    sources = _history_sources(session_after_modify)
    if "modify" not in sources:
        raise SmokeFailure(f"modify_sql did not appear in sql_history; sources={sources}")
    if not (session_after_modify.get("last_sql") or "").strip():
        raise SmokeFailure("session last_sql is empty after modification")

    responses.append(
        _chat(
            base_url=base_url,
            session_id=session_id,
            user_id=user_id,
            message=args.rerun_request,
            active_db_id=None,
            timeout=args.timeout,
        )
    )
    session_after_rerun = _session(base_url, session_id, timeout=args.timeout)
    sources = _history_sources(session_after_rerun)
    if not any(source in sources for source in ("rerun", "execute")):
        raise SmokeFailure(f"modified SQL was not executed; sources={sources}")

    responses.append(
        _chat(
            base_url=base_url,
            session_id=session_id,
            user_id=user_id,
            message=args.summary_request,
            active_db_id=None,
            timeout=args.timeout,
        )
    )
    final_session = _session(base_url, session_id, timeout=args.timeout)
    extra = final_session.get("extra") or {}
    if not extra.get("last_result_export"):
        raise SmokeFailure("summary/export step did not populate last_result_export")

    result = {
        "base_url": base_url,
        "session_id": session_id,
        "db_id": args.db_id,
        "question": args.question,
        "modification": args.modification,
        "rerun_request": args.rerun_request,
        "responses": responses,
        "final_session": final_session,
    }
    _write_output(args.output, result)
    print("\nSMOKE OK")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--user-id", default="smoke")
    parser.add_argument("--db-id", default=DEFAULT_DB_ID)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--modification", default=DEFAULT_MODIFICATION)
    parser.add_argument("--rerun-request", default=DEFAULT_RERUN_REQUEST)
    parser.add_argument("--summary-request", default=DEFAULT_SUMMARY_REQUEST)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/orchestrator_smoke_last.json"),
        help="Where to write full responses/session debug payload. Use '-' to disable.",
    )
    args = parser.parse_args()
    if str(args.output) == "-":
        args.output = None
    return args


def main() -> int:
    try:
        run(parse_args())
    except SmokeFailure as exc:
        print(f"\nSMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
