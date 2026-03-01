"""Entry point for running the v2 text-to-sql agent."""

from text_to_sql_agent.graph.pipeline import build_graph


def main() -> None:
    """Build and print graph metadata placeholder."""
    graph = build_graph()
    print(f"Graph initialized: {type(graph).__name__}")


if __name__ == "__main__":
    main()

