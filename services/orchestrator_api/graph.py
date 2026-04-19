"""Compile the orchestrator LangGraph with a given checkpointer and tool set.

Graph shape::

    START → agent → [tool_calls?] → tools → agent → ... → END

- ``agent`` calls the LLM with bound tools; when the model emits tool calls,
  the conditional edge routes to ``tools``; otherwise the turn ends.
- ``tools`` executes the requested tool(s) via LangGraph's prebuilt
  ``ToolNode``. Tools return ``Command`` updates that append ``ToolMessage``s
  and update conversation artifacts (``active_db_id``, ``last_sql``, ...).
- After ``tools``, control flows back to ``agent`` so the LLM can read the
  tool output and either call more tools or produce the final reply.

The recursion limit defaults to ``ORCHESTRATOR_MAX_TOOL_STEPS * 2 + 2`` to
bound total graph steps per ``/chat`` turn; each tool hop costs one agent
step and one tool step. Callers can override by passing ``max_tool_steps``.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from services.orchestrator_api.agent import make_agent_node
from services.orchestrator_api.state import OrchestratorState


def build_orchestrator_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    tools: list | None = None,
):
    """Build and compile the orchestrator graph with the given checkpointer.

    ``tools`` may be empty in tests that only need the conversational skeleton;
    in that case the graph reduces to ``START → agent → END`` (no tools node).
    """
    tools = tools or []
    graph = StateGraph(OrchestratorState)
    graph.add_node("agent", make_agent_node(tools))
    graph.add_edge(START, "agent")

    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    return graph.compile(checkpointer=checkpointer)
