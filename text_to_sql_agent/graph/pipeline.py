"""LangGraph StateGraph scaffold for text-to-sql pipeline."""

from langgraph.graph import END, START, StateGraph

from text_to_sql_agent.config import settings
from text_to_sql_agent.agents.decomposer import run_decomposer
from text_to_sql_agent.agents.generator import run_generator
from text_to_sql_agent.agents.judge import run_judge
from text_to_sql_agent.agents.refiner import run_refiner
from text_to_sql_agent.agents.selector import run_selector
from text_to_sql_agent.graph.state import SQLAgentState


def _route_after_refiner(state: SQLAgentState) -> str:
    """Route graph based on refiner status and max attempts policy."""
    has_error = bool(state.get("error_message"))
    attempts = state.get("refine_attempts", 0)
    if has_error and attempts < settings.max_refine_attempts:
        return "retry_refiner"
    return "finish"


def build_graph():
    """Build initial StateGraph wiring for six-stage pipeline."""
    graph = StateGraph(SQLAgentState)

    graph.add_node("selector", run_selector)
    graph.add_node("decomposer", run_decomposer)
    graph.add_node("generator", run_generator)
    graph.add_node("judge", run_judge)
    graph.add_node("refiner", run_refiner)

    graph.add_edge(START, "selector")
    graph.add_edge("selector", "decomposer")
    graph.add_edge("decomposer", "generator")
    graph.add_edge("generator", "judge")
    graph.add_edge("judge", "refiner")
    graph.add_conditional_edges(
        "refiner",
        _route_after_refiner,
        {"retry_refiner": "refiner", "finish": END},
    )

    return graph.compile()

