"""Compile the orchestrator LangGraph with a given checkpointer.

Phase 3 graph shape::

    START → agent → END

Phase 4 will insert a ``tools`` node and a conditional edge ``agent → tools``
when the LLM emits ``tool_calls``; the core wiring stays the same.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from services.orchestrator_api.agent import agent_node
from services.orchestrator_api.state import OrchestratorState


def build_orchestrator_graph(checkpointer: BaseCheckpointSaver):
    """Build and compile the orchestrator graph with the given checkpointer."""
    graph = StateGraph(OrchestratorState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)
