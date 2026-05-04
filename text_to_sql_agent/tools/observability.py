"""Optional observability helpers (Langfuse and benchmark summaries)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from text_to_sql_agent.config import settings

# Default truncation budgets when serializing pipeline state for Langfuse.
# Stage payloads are kept verbose but bounded so a single trace cannot blow up
# the Langfuse ingestion request size.
_DEFAULT_MAX_STR = 4000
_DEFAULT_MAX_LIST = 50
_DEFAULT_MAX_DEPTH = 6


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


def start_langfuse_generation(
    *,
    name: str,
    trace_id: str | None,
    model: str,
    input_payload: Any,
    metadata: dict[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """Return a context manager that opens a Langfuse generation observation.

    Usage::

        with start_langfuse_generation(name=..., model=...):
            # call the model
            update_langfuse_generation(output=..., usage=...)

    When Langfuse is disabled or fails to initialize, this returns a
    ``nullcontext`` so the caller can keep the same shape unconditionally.
    The yielded value is the LangfuseGeneration handle (or ``None`` when
    disabled), but in practice updates should go through
    :func:`update_langfuse_generation` which uses
    ``client.update_current_generation(...)`` and therefore does not need the
    handle.
    """
    client = get_langfuse_client()
    if client is None:
        return nullcontext()

    params: dict[str, Any] = {"name": name, "model": model, "input": input_payload}
    if trace_id:
        params["session_id"] = trace_id
    if metadata:
        params["metadata"] = metadata

    try:
        return client.start_as_current_observation(as_type="generation", **params)
    except Exception:
        return nullcontext()


def update_langfuse_generation(
    *,
    output: Any | None = None,
    usage: LLMUsageRecord | None = None,
    level: str | None = None,
    status_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort update of the *current* Langfuse generation observation.

    Must be called from inside ``with start_langfuse_generation(...)`` (or any
    other span/generation context started via the Langfuse SDK) — the SDK
    routes the update to whatever generation is the current OTEL span.
    Calling outside any context is a safe no-op.
    """
    client = get_langfuse_client()
    if client is None:
        return
    payload: dict[str, Any] = {}
    if output is not None:
        payload["output"] = output
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
    if not payload:
        return
    try:
        client.update_current_generation(**payload)
    except Exception:
        return


def flush_langfuse() -> None:
    """Flush pending Langfuse events when enabled."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Pipeline / orchestrator span helpers
# ---------------------------------------------------------------------------


def start_langfuse_span(
    *,
    name: str,
    trace_id: str | None,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
    as_type: str = "span",
) -> AbstractContextManager[Any]:
    """Return a context manager that opens a Langfuse observation as the *current* OTEL span.

    Used to wrap pipeline stages and orchestrator tool calls so that LLM
    generations created inside become children of this span automatically
    via OTEL context propagation.

    Usage::

        with start_langfuse_span(name="selector", trace_id=...):
            # do work
            update_langfuse_span(output=...)

    When Langfuse is disabled or fails to initialize, returns ``nullcontext()``
    so callers can keep the same shape unconditionally.
    """
    client = get_langfuse_client()
    if client is None:
        return nullcontext()

    merged_metadata: dict[str, Any] = {}
    if trace_id:
        merged_metadata["session_id"] = trace_id
    if metadata:
        merged_metadata.update({k: v for k, v in metadata.items() if v is not None})

    params: dict[str, Any] = {"name": name, "as_type": as_type}
    if input_payload is not None:
        params["input"] = input_payload
    if merged_metadata:
        params["metadata"] = merged_metadata

    try:
        return client.start_as_current_observation(**params)
    except Exception:
        return nullcontext()


def update_langfuse_span(
    *,
    output: Any | None = None,
    level: str | None = None,
    status_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort update of the *current* Langfuse span.

    Must be called from inside ``with start_langfuse_span(...)`` — the SDK
    routes the update to whatever span is the current OTEL span. Calling
    outside any active span is a safe no-op.

    Note: ``client.update_current_span(...)`` does not accept ``cost_details``
    or ``usage_details`` (those belong to generations). For LLM-call updates
    use :func:`update_langfuse_generation` instead.
    """
    client = get_langfuse_client()
    if client is None:
        return
    payload: dict[str, Any] = {}
    if output is not None:
        payload["output"] = output
    if level is not None:
        payload["level"] = level
    if status_message is not None:
        payload["status_message"] = status_message
    if metadata:
        payload["metadata"] = metadata
    if not payload:
        return
    try:
        client.update_current_span(**payload)
    except Exception:
        return


def safe_state_snapshot(
    payload: Any,
    *,
    max_str: int = _DEFAULT_MAX_STR,
    max_list: int = _DEFAULT_MAX_LIST,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> Any:
    """Serialize an arbitrary value for Langfuse with bounded depth/length.

    Verbose by design: dictionaries and lists are kept whole, but each scalar
    string is truncated to ``max_str`` characters and each list is truncated
    to ``max_list`` items so a single trace can never carry an arbitrarily
    large schema dump or candidate list. Non-serializable objects fall back to
    ``str(obj)`` (also truncated).
    """
    return _safe_serialize(payload, max_str=max_str, max_list=max_list, depth=max_depth)


def _safe_serialize(value: Any, *, max_str: int, max_list: int, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_str(value, max_str)
    if depth <= 0:
        return _truncate_str(str(value), max_str)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = _safe_serialize(
                item, max_str=max_str, max_list=max_list, depth=depth - 1
            )
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items: list[Any] = list(value)
        truncated = False
        if len(items) > max_list:
            items = items[:max_list]
            truncated = True
        rendered = [
            _safe_serialize(item, max_str=max_str, max_list=max_list, depth=depth - 1)
            for item in items
        ]
        if truncated:
            rendered.append(f"... [+{len(value) - max_list} more]")
        return rendered
    if isinstance(value, Iterable):
        try:
            materialized = list(value)
        except Exception:
            return _truncate_str(str(value), max_str)
        return _safe_serialize(
            materialized, max_str=max_str, max_list=max_list, depth=depth
        )
    return _truncate_str(str(value), max_str)


def _truncate_str(text: str, max_str: int) -> str:
    if len(text) <= max_str:
        return text
    return text[:max_str] + f"... [+{len(text) - max_str} chars]"

