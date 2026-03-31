"""Optional observability helpers (Langfuse and benchmark summaries)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from text_to_sql_agent.config import settings


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

