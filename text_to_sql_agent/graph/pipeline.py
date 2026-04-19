"""LangGraph StateGraph scaffold for text-to-sql pipeline."""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from langgraph.graph import END, START, StateGraph

from text_to_sql_agent.config import settings
from text_to_sql_agent.agents.execution_filter import run_execution_filter
from text_to_sql_agent.agents.generator import run_generator
from text_to_sql_agent.agents.value_linker import run_value_linker
from text_to_sql_agent.agents.voting import run_voting
from text_to_sql_agent.agents.query_sketcher import run_query_sketcher
from text_to_sql_agent.agents.refiner import run_refiner
from text_to_sql_agent.agents.selector import run_selector
from text_to_sql_agent.graph.state import SQLAgentState
from text_to_sql_agent.tools.observability import start_langfuse_span


_STAGE_IO: dict[str, dict[str, list[str]]] = {
    "selector": {
        "input": ["question", "db_id"],
        "output": ["filtered_schema"],
    },
    "value_linker": {
        "input": ["question"],
        "output": ["value_hints", "column_hints"],
    },
    "sketcher": {
        "input": ["question", "filtered_schema"],
        "output": ["query_sketch_text"],
    },
    "generator": {
        "input": ["query_sketch_text"],
        "output": ["candidates"],
    },
    "execution_filter": {
        "input": ["candidates"],
        "output": ["valid_candidates"],
    },
    "voting": {
        "input": ["valid_candidates"],
        "output": ["best_sql", "selection_confidence", "selection_method", "selection_reasoning"],
    },
    "refiner": {
        "input": ["best_sql", "error_message"],
        "output": ["final_sql", "refine_attempts"],
    },
}


def _span_summary(keys: list[str], data: dict[str, Any]) -> dict[str, Any]:
    """Build a compact payload for Langfuse span input/output."""
    out: dict[str, Any] = {}
    for k in keys:
        v = data.get(k)
        if v is None:
            continue
        if isinstance(v, list) and len(v) > 3:
            out[k] = {"count": len(v), "preview": v[:2]}
        else:
            out[k] = v
    return out


def _traced(
    stage_name: str,
    agent_fn: Callable[[SQLAgentState], Awaitable[SQLAgentState]],
) -> Callable[[SQLAgentState], Awaitable[SQLAgentState]]:
    """Wrap an agent node with Langfuse span tracking (best-effort, no-op when disabled)."""
    io = _STAGE_IO.get(stage_name, {})
    input_keys = io.get("input", [])
    output_keys = io.get("output", [])

    async def wrapper(state: SQLAgentState) -> SQLAgentState:
        with start_langfuse_span(
            name=stage_name,
            input_payload=_span_summary(input_keys, state),
        ) as span:
            try:
                result = await agent_fn(state)
                stage_failed = result.get("stage_status", {}).get(stage_name) == "failed"
                span.update(
                    output=_span_summary(output_keys, result),
                    **({"level": "ERROR", "status_message": result.get("error_message", "")} if stage_failed else {}),
                )
                return result
            except Exception as exc:
                span.update(level="ERROR", status_message=str(exc))
                raise

    wrapper.__name__ = agent_fn.__name__
    wrapper.__qualname__ = agent_fn.__qualname__
    return wrapper


def _route_after_selector(state: SQLAgentState) -> str:
    """Stop early when schema selection cannot proceed."""
    if state.get("stage_status", {}).get("selector") == "failed":
        return "finish"
    return "value_linker"


def _route_after_generator(state: SQLAgentState) -> str:
    """Proceed only if generation produced at least one candidate."""
    if state.get("candidates"):
        return "execution_filter"
    return "finish"


def _route_after_execution_filter(state: SQLAgentState) -> str:
    """Route to voting when candidates survive, otherwise stop."""
    if state.get("valid_candidates") or state.get("candidates"):
        return "voting"
    return "finish"


def _route_after_voting(state: SQLAgentState) -> str:
    """Refiner needs a selected SQL candidate."""
    if state.get("best_sql"):
        return "refiner"
    return "finish"


def _route_after_refiner(state: SQLAgentState) -> str:
    """Route graph based on refiner status and max attempts policy."""
    has_error = bool(state.get("error_message"))
    attempts = state.get("refine_attempts", 0)
    if has_error and attempts < settings.max_refine_attempts:
        return "retry_refiner"
    return "finish"


def build_graph():
    """Build StateGraph wiring: selector → value_linker → sketcher → generator → exec_filter → voting → refiner."""
    graph = StateGraph(SQLAgentState)

    graph.add_node("selector", _traced("selector", run_selector))
    graph.add_node("value_linker", _traced("value_linker", run_value_linker))
    graph.add_node("sketcher", _traced("sketcher", run_query_sketcher))
    graph.add_node("generator", _traced("generator", run_generator))
    graph.add_node("execution_filter", _traced("execution_filter", run_execution_filter))
    graph.add_node("voting", _traced("voting", run_voting))
    graph.add_node("refiner", _traced("refiner", run_refiner))

    graph.add_edge(START, "selector")
    graph.add_conditional_edges(
        "selector",
        _route_after_selector,
        {"value_linker": "value_linker", "finish": END},
    )
    graph.add_edge("value_linker", "sketcher")
    graph.add_edge("sketcher", "generator")
    graph.add_conditional_edges(
        "generator",
        _route_after_generator,
        {"execution_filter": "execution_filter", "finish": END},
    )
    graph.add_conditional_edges(
        "execution_filter",
        _route_after_execution_filter,
        {"voting": "voting", "finish": END},
    )
    graph.add_conditional_edges(
        "voting",
        _route_after_voting,
        {"refiner": "refiner", "finish": END},
    )
    graph.add_conditional_edges(
        "refiner",
        _route_after_refiner,
        {"retry_refiner": "refiner", "finish": END},
    )

    return graph.compile()
