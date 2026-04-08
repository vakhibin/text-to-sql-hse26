"""LangGraph StateGraph scaffold for text-to-sql pipeline."""

from langgraph.graph import END, START, StateGraph

from text_to_sql_agent.config import settings
from text_to_sql_agent.agents.execution_filter import run_execution_filter
from text_to_sql_agent.agents.generator import run_generator
from text_to_sql_agent.agents.voting import run_voting
from text_to_sql_agent.agents.query_sketcher import run_query_sketcher
from text_to_sql_agent.agents.refiner import run_refiner
from text_to_sql_agent.agents.selector import run_selector
from text_to_sql_agent.graph.state import SQLAgentState


def _route_after_selector(state: SQLAgentState) -> str:
    """Stop early when schema selection cannot proceed."""
    if state.get("stage_status", {}).get("selector") == "failed":
        return "finish"
    return "sketcher"


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
    """Build StateGraph wiring: selector → sketcher → generator → exec_filter → voting → refiner."""
    graph = StateGraph(SQLAgentState)

    graph.add_node("selector", run_selector)
    graph.add_node("sketcher", run_query_sketcher)
    graph.add_node("generator", run_generator)
    graph.add_node("execution_filter", run_execution_filter)
    graph.add_node("voting", run_voting)
    graph.add_node("refiner", run_refiner)

    graph.add_edge(START, "selector")
    graph.add_conditional_edges(
        "selector",
        _route_after_selector,
        {"sketcher": "sketcher", "finish": END},
    )
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
