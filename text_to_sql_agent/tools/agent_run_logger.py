"""File logger for single-shot agent runs: one log file per invocation with per-node payloads."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keys to record per graph node (nodes return full **state; we avoid echoing redundant bulk).
NODE_LOG_KEYS: dict[str, frozenset[str]] = {
    "selector": frozenset(
        {
            "full_schema",
            "filtered_schema",
            "retrieved_schema_context",
            "warnings",
            "stage_status",
            "stage_timings",
            "llm_usage",
            "total_cost_usd",
            "error_message",
        }
    ),
    "value_linker": frozenset(
        {
            "value_hints",
            "column_hints",
            "warnings",
            "stage_timings",
        }
    ),
    "sketcher": frozenset(
        {
            "query_sketch",
            "query_sketch_text",
            "warnings",
            "stage_status",
            "stage_timings",
            "llm_usage",
            "total_cost_usd",
        }
    ),
    "generator": frozenset(
        {
            "candidates",
            "warnings",
            "stage_status",
            "stage_timings",
            "llm_usage",
            "total_cost_usd",
            "error_message",
        }
    ),
    "execution_filter": frozenset(
        {
            "valid_candidates",
            "candidate_diagnostics",
            "warnings",
            "stage_status",
            "stage_timings",
            "error_message",
        }
    ),
    "voting": frozenset(
        {
            "best_sql",
            "selection_reasoning",
            "selection_confidence",
            "selection_method",
            "selection_needs_refine",
            "selected_candidate_diagnostic",
            "warnings",
            "stage_status",
            "stage_timings",
            "error_message",
            "llm_usage",
            "total_cost_usd",
        }
    ),
    "refiner": frozenset(
        {
            "final_sql",
            "execution_result",
            "refine_attempts",
            "warnings",
            "stage_status",
            "stage_timings",
            "error_message",
            "llm_usage",
            "total_cost_usd",
            "selection_needs_refine",
        }
    ),
}

NODE_SECTION_TITLE: dict[str, str] = {
    "selector": "selector — retrieval & schema linking (Chroma + rerank → filtered mSchema)",
    "value_linker": "value_linker — column/value hints (pre-generation grounding)",
    "sketcher": "sketcher — query plan (structured sketch, no SQL)",
    "generator": "generator — SQL candidates (ensemble)",
    "execution_filter": "execution_filter — execute & validate candidates",
    "voting": "voting — majority / self-consistency selection",
    "refiner": "refiner — AST repair, execute, optional LLM fix",
}


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Best-effort conversion for log JSON (tuples, dataclasses, nested dicts)."""
    if depth > 30:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if is_dataclass(value):
        return _to_jsonable(asdict(value), depth=depth + 1)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v, depth=depth + 1) for v in value]
    if hasattr(value, "__dict__") and type(value).__module__ != "builtins":
        try:
            return _to_jsonable(vars(value), depth=depth + 1)
        except Exception:
            return repr(value)
    return str(value)


def filter_node_update(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Keep only whitelisted keys present in this node's return payload."""
    allowed = NODE_LOG_KEYS.get(node)
    if allowed is None:
        return {
            "_note": "unknown node — logging key list only to avoid huge **state echo",
            "node": node,
            "keys": sorted(update),
            "warnings": update.get("warnings"),
            "error_message": update.get("error_message"),
        }
    return {k: update[k] for k in sorted(allowed) if k in update}


def format_executive_summary_lines(fields: dict[str, Any]) -> str:
    """Human-readable block for the top of the log (after INPUT)."""

    def _line(label: str, value: Any) -> str:
        if value is None:
            return f"{label}: (not available)\n"
        if isinstance(value, bool):
            return f"{label}: {value}\n"
        if isinstance(value, str) and not value.strip():
            return f"{label}: (empty)\n"
        return f"{label}: {value}\n"

    lines: list[str] = []
    lines.append(_line("predicted_sql", fields.get("predicted_sql")))
    lines.append(_line("gold_sql", fields.get("gold_sql")))
    if fields.get("gold_sql_source"):
        lines.append(_line("gold_sql_source", fields.get("gold_sql_source")))
    if fields.get("gold_lookup_note"):
        lines.append(_line("gold_lookup_note", fields.get("gold_lookup_note")))
    ex = fields.get("execution_match")
    if ex is None:
        lines.append("execution_match (EX): N/A (no gold SQL or could not compare)\n")
    else:
        lines.append(_line("execution_match (EX)", ex))
    em = fields.get("exact_match")
    if em is None:
        lines.append("exact_match (EM): N/A (no gold SQL)\n")
    else:
        lines.append(_line("exact_match (EM)", em))
    lines.append(_line("pipeline error_message", fields.get("error_message")))
    pe = fields.get("predicted_execute_error")
    ge = fields.get("gold_execute_error")
    if pe:
        lines.append(_line("predicted_sql execute error", pe))
    if ge:
        lines.append(_line("gold_sql execute error", ge))
    lines.append(_line("trace_id", fields.get("trace_id")))
    lines.append(_line("total_cost_usd", fields.get("total_cost_usd")))
    warnings = fields.get("warnings") or []
    if warnings:
        lines.append(f"warnings ({len(warnings)} total):\n")
        for i, w in enumerate(warnings[:50], 1):
            lines.append(f"  [{i}] {w}\n")
        if len(warnings) > 50:
            lines.append(f"  ... and {len(warnings) - 50} more (see extended JSON below)\n")
    else:
        lines.append("warnings: (none)\n")
    st = fields.get("stage_status") or {}
    if st:
        lines.append("stage_status:\n")
        for k, v in sorted(st.items()):
            lines.append(f"  {k}: {v}\n")
    return "".join(lines)


class AgentRunLogger:
    """Buffer pipeline steps, then write one log file: preamble → input → result → steps → appendix."""

    def __init__(self, path: Path, *, agent_run_id: str) -> None:
        self.path = path
        self.agent_run_id = agent_run_id
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._input_block = ""
        self._step_blocks: list[str] = []
        self._written = False

    def log_run_header(
        self,
        *,
        trace_id: str,
        question: str,
        db_id: str,
        evidence: str | None,
        schema_root: str | None,
    ) -> None:
        self._input_block = (
            "\n"
            + "=" * 80
            + "\nINPUT\n"
            + "=" * 80
            + f"\ntrace_id={trace_id}\n"
            f"db_id={db_id}\n"
            f"schema_root={schema_root!r}\n"
            f"question={question!r}\n"
            f"evidence={evidence!r}\n"
        )

    def log_step(self, step: int, node: str, raw_update: dict[str, Any]) -> None:
        title = NODE_SECTION_TITLE.get(node, node)
        payload = _to_jsonable(filter_node_update(node, raw_update))
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        self._step_blocks.append(
            f"\n{'-' * 80}\n"
            f"STEP {step} | node={node}\n"
            f"{title}\n"
            f"{'-' * 80}\n"
            f"{body}\n"
        )

    def write(
        self,
        *,
        executive_summary: dict[str, Any],
        extended_summary: dict[str, Any] | None = None,
    ) -> None:
        """Write the full log file. Call once after graph run and optional gold/EX evaluation."""
        if self._written:
            raise RuntimeError("AgentRunLogger.write() already called for this run")
        self._written = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        finished = datetime.now(timezone.utc).isoformat()
        parts: list[str] = [
            f"agent_run_id={self.agent_run_id}\n",
            f"started_utc={self._started_utc}\n",
            self._input_block,
            "\n",
            "=" * 80,
            "\nRESULT SUMMARY (predicted vs gold, EX/EM, errors)\n",
            "=" * 80,
            "\n",
            format_executive_summary_lines(executive_summary),
            "\n",
            "=" * 80,
            "\nDETAILED PIPELINE TRACE (per-node outputs)\n",
            "=" * 80,
            "\n",
            "".join(self._step_blocks),
        ]
        if extended_summary:
            parts.extend(
                [
                    "\n",
                    "=" * 80,
                    "\nEXTENDED ARTIFACTS (full state / benchmark-shaped JSON)\n",
                    "=" * 80,
                    "\n",
                    json.dumps(_to_jsonable(extended_summary), ensure_ascii=False, indent=2),
                    "\n",
                ]
            )
        parts.append(f"\nfinished_utc={finished}\n")
        self.path.write_text("".join(parts), encoding="utf-8")


def build_prediction_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Subset of state aligned with `predictions[]` rows in benchmark JSON."""
    predicted = (state.get("final_sql") or state.get("best_sql") or "").strip()
    return {
        "db_id": state.get("db_id"),
        "question": state.get("question"),
        "predicted_sql": predicted,
        "error_message": state.get("error_message"),
        "warnings": list(state.get("warnings") or []),
        "trace_id": state.get("trace_id"),
        "total_cost_usd": float(state.get("total_cost_usd") or 0.0),
        "stage_status": dict(state.get("stage_status") or {}),
        "stage_timings": dict(state.get("stage_timings") or {}),
        "best_sql": state.get("best_sql") or "",
        "final_sql": state.get("final_sql") or "",
        "execution_result": state.get("execution_result"),
        "refine_attempts": int(state.get("refine_attempts") or 0),
        "candidates_count": len(state.get("candidates") or []),
        "valid_candidates_count": len(state.get("valid_candidates") or []),
        "selection_method": state.get("selection_method") or "",
        "selection_confidence": state.get("selection_confidence") or "",
        "selection_reasoning": state.get("selection_reasoning") or "",
    }
