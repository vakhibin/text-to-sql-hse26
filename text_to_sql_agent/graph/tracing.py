"""Helpers to wrap LangGraph pipeline nodes with Langfuse stage spans.

The wrapper opens a Langfuse observation around each node so that LLM
generations created inside the node automatically become child spans of the
stage span (Langfuse SDK uses OpenTelemetry under the hood, so OTEL parent
context propagates through ``await`` boundaries).

Tracing is best-effort: when Langfuse is disabled or fails to start, the node
runs unchanged and the wrapper short-circuits to a no-op context manager.

Each stage can declare ``input_keys`` and ``output_keys`` so the Langfuse
``input`` / ``output`` panes show only what is meaningful for that stage.
Service fields that the node also writes (``stage_status``, ``stage_timings``,
``warnings``, ``llm_usage``, ``total_cost_usd``, ...) are routed to
``metadata`` instead, which keeps the trace readable in the UI.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, Mapping

from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.tools.observability import (
    safe_state_snapshot,
    start_langfuse_span,
    update_langfuse_span,
)

NodeFn = Callable[[SQLAgentState], Awaitable[Mapping[str, Any] | SQLAgentState]]


def _filter_keys(
    payload: Mapping[str, Any] | None, keys: tuple[str, ...] | None
) -> dict[str, Any]:
    """Return a snapshot of ``payload`` restricted to ``keys`` when provided."""
    if payload is None:
        return {}
    if keys is None:
        return safe_state_snapshot(dict(payload))
    selected = {key: payload.get(key) for key in keys if key in payload}
    return safe_state_snapshot(selected)


def _patch_extras(
    patch: Mapping[str, Any] | None, output_keys: tuple[str, ...] | None
) -> dict[str, Any]:
    """Service fields the node added but that are not the stage's main output."""
    if patch is None or output_keys is None:
        return {}
    output_set = set(output_keys)
    extras = {key: value for key, value in patch.items() if key not in output_set}
    return safe_state_snapshot(extras)


def trace_pipeline_stage(
    name: str,
    fn: NodeFn,
    *,
    as_type: str = "chain",
    input_keys: Iterable[str] | None = None,
    output_keys: Iterable[str] | None = None,
) -> NodeFn:
    """Wrap a LangGraph node with a Langfuse span.

    Args:
        name: stage name shown in Langfuse.
        fn: the original LangGraph node coroutine.
        as_type: Langfuse observation type (``chain`` by default).
        input_keys: when given, only those state fields are recorded as
            Langfuse ``input``. The remaining state is dropped from the trace.
        output_keys: when given, only those patch fields go into Langfuse
            ``output``. Other patch fields move into ``metadata.patch_extras``
            so they remain inspectable without polluting the output panel.

    Behavior:
        - opens an observation named ``name`` as the current OTEL span
        - records the (filtered) state snapshot as ``input``
        - records the (filtered) patch as ``output``
        - puts service fields and after-run ``stage_status`` into ``metadata``
        - marks the span ``ERROR`` and re-raises if the node raises
    """

    input_filter = tuple(input_keys) if input_keys is not None else None
    output_filter = tuple(output_keys) if output_keys is not None else None

    async def wrapped(state: SQLAgentState) -> Any:
        trace_id = state.get("trace_id")
        db_id = state.get("db_id")
        question = state.get("question")
        metadata: dict[str, Any] = {
            "stage": name,
            "db_id": db_id,
            "question_preview": (str(question)[:200] if question else None),
        }
        # IMPORTANT: every Langfuse span update must happen INSIDE the ``with``
        # block. ``update_langfuse_span()`` writes to the *current* OTEL span,
        # so it only finds our span between ``__enter__`` and ``__exit__``.
        with start_langfuse_span(
            name=name,
            trace_id=trace_id,
            input_payload=_filter_keys(state, input_filter),
            metadata=metadata,
            as_type=as_type,
        ):
            try:
                result = await fn(state)
            except Exception as exc:
                update_langfuse_span(
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

            patch = result if isinstance(result, Mapping) else None
            update_metadata: dict[str, Any] = {
                "stage": name,
                "db_id": db_id,
                "stage_status": post_status,
            }
            extras = _patch_extras(patch, output_filter)
            if extras:
                update_metadata["patch_extras"] = extras

            # If none of the declared ``output_keys`` are present in the patch,
            # pass ``output=None`` so Langfuse keeps the "no output" placeholder
            # instead of rendering an empty ``{}`` blob.
            output_payload = _filter_keys(patch, output_filter)
            update_langfuse_span(
                output=output_payload if output_payload else None,
                metadata=update_metadata,
            )
        return result

    wrapped.__name__ = f"traced_{name}"
    wrapped.__qualname__ = f"trace_pipeline_stage.<{name}>"
    return wrapped
