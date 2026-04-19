"""LLM-backed agent node for the orchestrator LangGraph.

The agent node:

1. Resolves a chat model (real LLMRouter in prod; injected fake in tests).
2. Binds the provided tools to the model so the LLM can emit ``tool_calls``.
3. Builds a dynamic system prompt that surfaces small pieces of state the LLM
   needs for good decisions (active database, last SQL it produced). Keeping
   this prompt in one place avoids leaking UI-level concerns into the graph.

Tests inject a fake model via ``set_chat_model(...)``; the bound-tool wrapping
is applied on top of whatever model is returned.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from orchestrator_agent.state import OrchestratorState

_SYSTEM_PROMPT_HEADER = (
    "You are a conversational assistant for a text-to-SQL system. Users ask "
    "questions about their databases in natural language. You keep a "
    "multi-turn dialogue and remember the chosen database and the last SQL "
    "you produced.\n\n"
    "You have tools to:\n"
    "- run the full text-to-SQL pipeline on a natural-language question,\n"
    "- execute a SQL SELECT directly (read-only),\n"
    "- explain a given SQL statement in plain English.\n\n"
    "Rules:\n"
    "- Prefer calling tools over guessing. If the user asks anything that "
    "requires data, call a tool.\n"
    "- Write queries (INSERT/UPDATE/DELETE/DDL) are NOT executed automatically. "
    "If the user asks for one, reply that you can produce it but the user must "
    "explicitly confirm before it runs.\n"
    "- Keep answers concise and grounded in the tool output. If a tool fails, "
    "summarise the error and suggest a next step.\n"
    "- Never invent column or table names: rely on tool responses."
)


def _build_context_suffix(state: OrchestratorState) -> str:
    """Turn small state artifacts into a short context block for the LLM."""
    bits: list[str] = []
    active_db = state.get("active_db_id")
    last_sql = state.get("last_sql")
    if active_db:
        bits.append(f"Active database: {active_db}")
    if last_sql:
        bits.append(f"Last SQL:\n{last_sql}")
    if not bits:
        return ""
    return "\n\nSession context:\n" + "\n\n".join(bits)


_LLM: BaseChatModel | None = None


def set_chat_model(model: BaseChatModel | None) -> None:
    """Override (or reset) the chat model used by the agent node.

    Tests call this with a fake model so the graph never touches OpenRouter.
    Pass ``None`` to restore the default lazy resolution.
    """
    global _LLM
    _LLM = model


def get_chat_model() -> BaseChatModel:
    """Return the cached chat model, building a default one on first use."""
    global _LLM
    if _LLM is None:
        from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole

        router = LLMRouter()
        _LLM = router.get_chat_model(ModelRole.REFINER)
    return _LLM


def make_agent_node(tools: list):
    """Build an agent node bound to the given tool set.

    The returned coroutine captures ``tools`` in closure so tests can build a
    node against mocked tools without touching the module-level chat-model
    cache for tool selection.
    """

    async def agent_node(state: OrchestratorState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        llm = get_chat_model()
        bound = llm.bind_tools(tools) if tools else llm
        system_text = _SYSTEM_PROMPT_HEADER + _build_context_suffix(state)
        prompt = [SystemMessage(content=system_text), *messages]
        response = await bound.ainvoke(prompt)
        return {"messages": [response]}

    return agent_node
