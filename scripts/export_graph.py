"""Export the compiled LangGraph pipeline as a PNG image."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pipeline graph to PNG")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="docs/pipeline_graph.png",
        help="Output PNG file path",
    )
    parser.add_argument(
        "--method",
        choices=["mermaid", "graphviz"],
        default="mermaid",
        help="Rendering backend: mermaid (no deps) or graphviz (needs pygraphviz)",
    )
    args = parser.parse_args()

    from text_to_sql_agent.graph.pipeline import build_graph

    graph = build_graph()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "mermaid":
        png_bytes = graph.get_graph().draw_mermaid_png()
        output_path.write_bytes(png_bytes)
    else:
        try:
            png_bytes = graph.get_graph().draw_png()
            output_path.write_bytes(png_bytes)
        except Exception as exc:
            print(
                f"graphviz export failed: {exc}\n"
                "Install graphviz: brew install graphviz && pip install pygraphviz\n"
                "Or use --method mermaid instead.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Graph exported to {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
