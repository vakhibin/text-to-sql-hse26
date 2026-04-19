"""Optional observability helpers (Langfuse v4 + benchmark summaries).

Design: uses Langfuse v4 OTel-based context managers for hierarchical tracing.
When Langfuse is disabled or not installed, all helpers return no-op sentinels.
"""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from text_to_sql_agent.config import settings

_MAX_PAYLOAD_CHARS = 48_000


@dataclass
class LLMUsageRecord:
    stage: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


def _truncate(value: Any, limit: int = _MAX_PAYLOAD_CHARS) -> Any:
    """Truncate large strings to stay within Langfuse payload limits."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"\n... [truncated, {len(value)} chars total]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value]
    return value


class _NullObservation:
    """Sentinel for when Langfuse is disabled. Silently absorbs .update() calls."""

    def update(self, **kwargs: Any) -> "_NullObservation":
        return self


_NULL = _NullObservation()


def _trace_id_to_hex(trace_id: str) -> str:
    """Convert an arbitrary trace_id string to a 32-char lowercase hex string (OTel format)."""
    return hashlib.md5(trace_id.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_langfuse_client():
    """Return Langfuse client if configured and installed, otherwise None."""
    if not (
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    ):
        return None
    try:
        from langfuse import Langfuse
    except Exception:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


# ---------------------------------------------------------------------------
# Root trace observation (one per pipeline question)
# ---------------------------------------------------------------------------

def start_langfuse_trace(
    *,
    trace_id: str,
    session_id: str | None = None,
    name: str = "text-to-sql",
    input_payload: Any = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> AbstractContextManager[Any]:
    """Return a context manager that creates a root Langfuse trace + span.

    Usage::

        with start_langfuse_trace(trace_id=tid, session_id=sid) as root:
            ...  # agent spans auto-nest under root
            root.update(output={...})

    Returns ``nullcontext(_NullObservation())`` when disabled.
    """
    client = get_langfuse_client()
    if client is None:
        return nullcontext(_NULL)
    try:
        from langfuse.types import TraceContext

        tc: dict[str, Any] = {"trace_id": _trace_id_to_hex(trace_id)}
        if session_id:
            tc["session_id"] = session_id
        if tags:
            tc["tags"] = tags
        if name:
            tc["trace_name"] = name

        params: dict[str, Any] = {
            "trace_context": TraceContext(**tc),
            "name": name,
            "as_type": "span",
        }
        if input_payload is not None:
            params["input"] = _truncate(input_payload)
        if metadata:
            params["metadata"] = metadata
        return client.start_as_current_observation(**params)
    except Exception:
        return nullcontext(_NULL)


# ---------------------------------------------------------------------------
# Agent span (one per pipeline stage)
# ---------------------------------------------------------------------------

def start_langfuse_span(
    *,
    name: str,
    input_payload: Any = None,
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """Return a context manager for a pipeline-stage span. Auto-nests under the active trace."""
    client = get_langfuse_client()
    if client is None:
        return nullcontext(_NULL)
    try:
        params: dict[str, Any] = {"name": name, "as_type": "span"}
        if input_payload is not None:
            params["input"] = _truncate(input_payload)
        if metadata:
            params["metadata"] = metadata
        return client.start_as_current_observation(**params)
    except Exception:
        return nullcontext(_NULL)


# ---------------------------------------------------------------------------
# Generation (LLM call — nests under active span)
# ---------------------------------------------------------------------------

def start_langfuse_generation(
    *,
    name: str,
    trace_id: str | None,
    model: str,
    input_payload: Any,
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """Return a context manager for a Langfuse generation observation.

    Usage::

        with start_langfuse_generation(name=..., model=..., ...) as gen:
            response = await llm.ainvoke(...)
            gen.update(output=..., usage_details=...)

    The ``gen`` object is a ``LangfuseGeneration`` (or ``_NullObservation``
    when disabled).  Call ``gen.update(...)`` **inside** the ``with`` block
    so the data is captured before the OTel span is finalized.
    """
    client = get_langfuse_client()
    if client is None:
        return nullcontext(_NULL)
    try:
        params: dict[str, Any] = {
            "name": name,
            "as_type": "generation",
            "model": model,
            "input": _truncate(input_payload),
        }
        if metadata:
            params["metadata"] = metadata
        return client.start_as_current_observation(**params)
    except Exception:
        return nullcontext(_NULL)


def update_langfuse_generation(
    generation: Any,
    *,
    output: Any | None = None,
    usage: LLMUsageRecord | None = None,
    level: str | None = None,
    status_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort generation update, never raising to caller."""
    if generation is None or isinstance(generation, _NullObservation):
        return
    payload: dict[str, Any] = {}
    if output is not None:
        payload["output"] = _truncate(output)
    if usage is not None:
        payload["usage_details"] = {
            "input": usage.prompt_tokens,
            "output": usage.completion_tokens,
            "total": usage.total_tokens,
        }
        if usage.cost_usd:
            payload["cost_details"] = {"total": usage.cost_usd}
    if level is not None:
        payload["level"] = level
    if status_message is not None:
        payload["status_message"] = status_message
    if metadata:
        payload["metadata"] = metadata
    try:
        if payload:
            generation.update(**payload)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------

def flush_langfuse() -> None:
    """Flush pending Langfuse events when enabled."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        return
