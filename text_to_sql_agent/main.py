"""Entry point for running the v2 text-to-sql agent."""

from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state


def main() -> None:
    """Build graph and initialize deterministic state placeholder."""
    graph = build_graph()
    initial_state = make_initial_state(
        question="How many singers do we have?",
        db_id="concert_singer",
    )
    print(f"Graph initialized: {type(graph).__name__}")
    print(f"Initial state trace_id: {initial_state['trace_id']}")


if __name__ == "__main__":
    main()

