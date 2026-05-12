"""Download Spider dataset artifacts (unified layout with ``scripts/download_bird.py``).

Usage:
    python scripts/download_spider.py              # core: tables.json, train/dev, database/ (Kaggle)
    python scripts/download_spider.py --test       # test.json, test_tables.json, test_gold.sql, test_database/
    python scripts/download_spider.py --both       # core then test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_to_sql_agent.datasets.spider_assets import download_spider_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Spider v1 artifacts into SPIDER_ROOT (see text_to_sql_agent.datasets.spider_assets)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--core",
        action="store_true",
        help="Download core bundle only (train_spider.json, dev.json, tables.json, database/) — default",
    )
    group.add_argument(
        "--test",
        action="store_true",
        help="Download test split only (Google Drive zip → test.json, test_tables.json, test_gold.sql, test_database/)",
    )
    group.add_argument("--both", action="store_true", help="Download core then test")
    parser.add_argument(
        "--spider-root",
        type=str,
        default=None,
        help="Target directory (defaults to SPIDER_ROOT / settings.spider_root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when required files already exist",
    )
    args = parser.parse_args()

    root = Path(args.spider_root) if args.spider_root else None

    if args.both:
        download_spider_assets("core", spider_root=root, force=args.force)
        download_spider_assets("test", spider_root=root, force=args.force)
    elif args.test:
        download_spider_assets("test", spider_root=root, force=args.force)
    else:
        download_spider_assets("core", spider_root=root, force=args.force)


if __name__ == "__main__":
    main()
