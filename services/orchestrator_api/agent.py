"""LLM-backed agent node for the orchestrator LangGraph.

Phase 3: no tools yet. The node simply prepends a system prompt to the
accumulated conversation and asks the LLM to respond. Tool binding is added
in Phase 4 without changing the node signature.

The chat model is resolved lazily so tests can inject a fake model via
``set_chat_model(fake)`` without touching the real LLMRouter (which
requires ``OPENROUTER_API_KEY``).
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from services.orchestrator_api.state import OrchestratorState

_SYSTEM_PROMPT = (
    "You are a helpful conversational assistant for a text-to-SQL system. "
    "Users ask questions about their databases in natural language. "
    "You hold a multi-turn dialogue and remember prior context in this session "
    "(e.g. which database is active and what SQL you last produced).\n\n"
    "Phase 3 limitations (will be lifted in later phases):\n"
    "- You do not yet have tools for querying databases or running SQL.\n"
    "- If the user asks a question that would require executing SQL, explain "
    "that tools are not yet wired up and that you can still help clarify the "
    "request or plan the SQL in words.\n\n"
    "Keep responses concise and plain-spoken."
)

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


async def agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Call the LLM with a system prompt + accumulated conversation."""
    messages = list(state.get("messages") or [])
    llm = get_chat_model()
    prompt = [SystemMessage(content=_SYSTEM_PROMPT), *messages]
    response = await llm.ainvoke(prompt)
    return {"messages": [response]}
