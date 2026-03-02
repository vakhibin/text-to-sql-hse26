"""Entry point for running the v2 text-to-sql agent."""

import argparse
import asyncio

from text_to_sql_agent.graph.pipeline import build_graph
from text_to_sql_agent.graph.state import make_initial_state


async def _run_once(question: str, db_id: str, evidence: str | None = None) -> None:
    """Run one end-to-end graph invocation and print compact summary."""
    graph = build_graph()
    initial_state = make_initial_state(question=question, db_id=db_id, evidence=evidence)
    result = await graph.ainvoke(initial_state)

    print(f"Graph initialized: {type(graph).__name__}")
    print(f"Trace ID: {result.get('trace_id')}")
    print(f"Stage status: {result.get('stage_status')}")
    print(f"Warnings: {len(result.get('warnings', []))}")
    print(f"Final SQL: {result.get('final_sql') or result.get('best_sql') or ''}")
    print(f"Execution result: {result.get('execution_result')}")
    print(f"Error message: {result.get('error_message')}")


def main() -> None:
    """CLI entrypoint for one-shot pipeline run."""
    parser = argparse.ArgumentParser(description="Run text_to_sql_agent pipeline once")
    parser.add_argument("--question", type=str, default="How many singers do we have?")
    parser.add_argument("--db-id", type=str, default="concert_singer")
    parser.add_argument("--evidence", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(_run_once(question=args.question, db_id=args.db_id, evidence=args.evidence))


if __name__ == "__main__":
    main()

