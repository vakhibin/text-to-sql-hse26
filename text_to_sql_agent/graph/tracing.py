"""Helpers to wrap LangGraph pipeline nodes with Langfuse stage spans.

The wrapper opens a Langfuse observation around each node so that LLM
generations created inside the node automatically become child spans of the
stage span (Langfuse SDK uses OpenTelemetry under the hood, so OTEL parent
context propagates through ``await`` boundaries).

Tracing is best-effort: when Langfuse is disabled or fails to start, the node
runs unchanged and the wrapper short-circuits to a no-op context manager.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.tools.observability import (
    safe_state_snapshot,
    start_langfuse_span,
    update_langfuse_span,
)

NodeFn = Callable[[SQLAgentState], Awaitable[Mapping[str, Any] | SQLAgentState]]


def _stage_input_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot the state right before a stage runs (verbose, bounded)."""
    return safe_state_snapshot(dict(state))


def _stage_output_payload(
    update: Mapping[str, Any] | SQLAgentState | None,
) -> dict[str, Any]:
    """Snapshot the patch returned by a node (LangGraph nodes return updates)."""
    if update is None:
        return {}
    return safe_state_snapshot(dict(update))


def trace_pipeline_stage(
    name: str,
    fn: NodeFn,
    *,
    as_type: str = "chain",
) -> NodeFn:
    """Wrap a LangGraph node with a Langfuse span.

    The wrapped node:
    - opens an observation named ``name`` as the current OTEL span
    - records a verbose-but-bounded snapshot of the incoming state as ``input``
    - records the node's returned patch as ``output``
    - marks the span ``ERROR`` and re-raises if the node raises
    """

    async def wrapped(state: SQLAgentState) -> Any:
        trace_id = state.get("trace_id")
        db_id = state.get("db_id")
        question = state.get("question")
        metadata: dict[str, Any] = {
            "stage": name,
            "db_id": db_id,
            "question_preview": (str(question)[:200] if question else None),
        }
        span, ctx = start_langfuse_span(
            name=name,
            trace_id=trace_id,
            input_payload=_stage_input_payload(state),
            metadata=metadata,
            as_type=as_type,
        )
        try:
            with ctx:
                result = await fn(state)
        except Exception as exc:
            update_langfuse_span(
                span,
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
                metadata={"stage": name, "db_id": db_id},
            )
            raise

        post_status = None
        try:
            if isinstance(result, Mapping):
                stage_status = result.get("stage_status") or {}
                if isinstance(stage_status, Mapping):
                    post_status = stage_status.get(name)
        except Exception:
            post_status = None

        update_langfuse_span(
            span,
            output=_stage_output_payload(result),
            metadata={"stage": name, "db_id": db_id, "stage_status": post_status},
        )
        return result

    wrapped.__name__ = f"traced_{name}"
    wrapped.__qualname__ = f"trace_pipeline_stage.<{name}>"
    return wrapped
