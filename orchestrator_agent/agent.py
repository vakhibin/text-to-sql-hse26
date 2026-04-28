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
    "- explain a given SQL statement in plain English,\n"
    "- list the available databases and describe any one of them,\n"
    "- switch the active database for the rest of the conversation,\n"
    "- preview the first rows of a table, and look up real literal values "
    "in a column (use this before writing a WHERE clause with an exact "
    "literal if you are unsure about spelling or casing),\n"
    "- fix a broken or wrong SQL (use for error repair),\n"
    "- modify a SQL by a natural-language instruction "
    "(use for user-driven edits like \"add a WHERE clause for 2023\"),\n"
    "- list the recent SQL queries in this session, and re-run one by index,\n"
    "- summarize or export the latest query result preview as Markdown, CSV, "
    "or JSON,\n"
    "- propose write SQL for explicit user confirmation, then execute it only "
    "after the user clearly confirms.\n\n"
    "Rules:\n"
    "- Prefer calling tools over guessing. If the user asks anything that "
    "requires data, call a tool.\n"
    "- If no database is active yet, call list_databases first and either "
    "pick one that clearly matches the user's intent (then call "
    "switch_database) or ask the user to choose.\n"
    "- Write queries (INSERT/UPDATE/DELETE/DDL) are NOT executed automatically. "
    "If the user asks for one, call propose_write_sql and ask for explicit "
    "confirmation. Only call confirm_write_sql after a clear yes/confirm from "
    "the user. If they decline, call cancel_pending_confirmation.\n"
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
    if state.get("last_rows_preview") is not None:
        row_count = state.get("last_row_count")
        columns = state.get("last_rows_columns") or []
        row_text = "unknown rows" if row_count is None else f"{row_count} rows"
        bits.append(
            "Latest result available: "
            f"{row_text}; columns: {', '.join(columns) if columns else '(unknown)'}"
        )
    pending = state.get("pending_confirmation")
    if pending:
        bits.append(
            "Pending confirmation: "
            f"{pending.get('type')} on db={pending.get('db_id')}\n"
            f"SQL:\n{pending.get('sql')}"
        )
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
