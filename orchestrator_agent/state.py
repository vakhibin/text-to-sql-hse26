"""LangGraph state for the conversational orchestrator.

Fields live here and not in the FastAPI layer so the graph can be reused
independently (e.g. in notebooks or integration tests).

Design notes:

- ``messages`` uses the LangGraph ``add_messages`` reducer, so each graph
  invocation appends new messages rather than overwriting them.
- ``session_id`` / ``user_id`` are replicated into state so agent nodes
  and tool nodes can read them without fishing through ``config``.
- ``active_db_id``, ``last_sql`` and the ``last_*`` result fields are the
  "conversation artifacts" that tools populate and subsequent turns consume.
  They are intentionally small so the checkpointer persists them cheaply.
- ``sql_history`` is an append-only, capped list of executed / produced SQL
  entries. Tools replace the whole list on update (there is no reducer) —
  each tool reads the current list from injected state, appends, and caps
  before writing. This keeps the value checkpointer-friendly.
- ``pending_confirmation`` is reserved for the Phase 9 write-SQL flow.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict, total=False):
    """State passed through the orchestrator LangGraph."""

    messages: Annotated[list, add_messages]
    session_id: str
    user_id: str

    active_db_id: Optional[str]
    last_sql: Optional[str]
    last_rows_preview: Optional[list[list[Any]]]
    last_rows_columns: Optional[list[str]]
    last_row_count: Optional[int]
    last_result_export: Optional[str]

    sql_history: list[dict[str, Any]]

    pending_confirmation: Optional[dict[str, Any]]
