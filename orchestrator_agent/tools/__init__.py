"""Agent tools the orchestrator LLM can invoke.

Tools are split by concern:

- ``core``: text-to-SQL pipeline, raw SQL execution, SQL explanation
- ``discovery``: catalog exploration (list/describe databases, sample table,
  search values, switch active database)
- ``history``: SQL manipulation (fix, modify) and session history (list
  recent, rerun)

``make_all_tools`` is the composition used at graph-build time. Individual
``make_*_tools`` factories stay exposed so tests can exercise a single tool
family in isolation.
"""

from orchestrator_agent.clients.text_to_sql import TextToSQLClient
from orchestrator_agent.tools.core import make_core_tools
from orchestrator_agent.tools.discovery import make_discovery_tools
from orchestrator_agent.tools.history import make_history_tools


def make_all_tools(client: TextToSQLClient) -> list:
    """Return the full agent tool set bound to a concrete HTTP client."""
    return [
        *make_core_tools(client),
        *make_discovery_tools(client),
        *make_history_tools(client),
    ]


__all__ = [
    "make_core_tools",
    "make_discovery_tools",
    "make_history_tools",
    "make_all_tools",
]
