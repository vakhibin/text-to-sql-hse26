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
) -> tuple[Any, AbstractContextManager[Any]]:
    """Start a Langfuse generation context if enabled."""
    client = get_langfuse_client()
    if client is None:
        return None, nullcontext()

    params: dict[str, Any] = {"name": name, "model": model, "input": input_payload}
    if trace_id:
        params["session_id"] = trace_id
    if metadata:
        params["metadata"] = metadata

    try:
        generation = client.start_as_current_observation(as_type="generation", **params)
        return generation, generation
    except Exception:
        return None, nullcontext()


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
    if generation is None:
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
    try:
        if payload:
            generation.update(**payload)
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
) -> tuple[Any, AbstractContextManager[Any]]:
    """Start a Langfuse observation as the *current* OTEL span.

    Used to wrap pipeline stages and orchestrator tool calls so that LLM
    generations created inside become children of the stage span automatically.

    The returned context manager is always safe to ``with``; when Langfuse is
    disabled or fails to initialize it falls back to ``nullcontext()`` and the
    span handle is ``None``. Callers should pass the handle to
    :func:`update_langfuse_span` regardless.
    """
    client = get_langfuse_client()
    if client is None:
        return None, nullcontext()

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
        observation_ctx = client.start_as_current_observation(**params)
        return observation_ctx, observation_ctx
    except Exception:
        return None, nullcontext()


def update_langfuse_span(
    span: Any,
    *,
    output: Any | None = None,
    level: str | None = None,
    status_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort span update, never raising to caller."""
    if span is None:
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
        span.update(**payload)
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

