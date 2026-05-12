"""Download BIRD datasets from Hugging Face (unified layout with ``scripts/download_spider.py``).

Usage:
    python scripts/download_bird.py              # mini-dev (default)
    python scripts/download_bird.py --mini       # mini-dev (500 examples, small)
    python scripts/download_bird.py --full       # full dev (1534 examples, ~33 GB)
    python scripts/download_bird.py --both       # both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_to_sql_agent.datasets.bird_assets import download_bird_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BIRD datasets from Hugging Face")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mini", action="store_true", help="Download Mini-Dev (default)")
    group.add_argument("--full", action="store_true", help="Download full Dev set (~33 GB)")
    group.add_argument("--both", action="store_true", help="Download both mini and full")
    parser.add_argument(
        "--bird-root",
        type=str,
        default=None,
        help="Override install directory for the selected variant (defaults from settings)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when required files already exist",
    )
    args = parser.parse_args()

    root: Path | None = Path(args.bird_root) if args.bird_root else None
    if args.both and root is not None:
        print(
            "--bird-root is ignored with --both (mini and full use BIRD_MINI_ROOT / BIRD_ROOT from settings).",
            file=sys.stderr,
        )
        root = None

    if args.both:
        download_bird_assets("mini", bird_root=root, force=args.force)
        download_bird_assets("full", bird_root=root, force=args.force)
    elif args.full:
        download_bird_assets("full", bird_root=root, force=args.force)
    else:
        download_bird_assets("mini", bird_root=root, force=args.force)


if __name__ == "__main__":
    main()
