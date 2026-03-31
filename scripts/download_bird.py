"""Download BIRD datasets from Hugging Face.

Usage:
    python scripts/download_bird.py              # download mini-dev (default)
    python scripts/download_bird.py --mini       # download mini-dev (500 examples, small)
    python scripts/download_bird.py --full       # download full dev (1534 examples, ~33 GB)
    python scripts/download_bird.py --both       # download both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_to_sql_agent.config import settings

DATASETS = {
    "mini": {
        "repo_id": "birdsql/bird_mini_dev",
        "target_key": "bird_mini_root",
        "default_path": "databases/bird_mini",
        "check_files": ("mini_dev_sqlite.json", "mini_dev_databases"),
        "label": "BIRD Mini-Dev (500 examples)",
    },
    "full": {
        "repo_id": "birdsql/bird_sql_dev_20251106",
        "target_key": "bird_root",
        "default_path": "databases/bird",
        "check_files": ("dev.json", "dev_databases"),
        "label": "BIRD Full Dev (1534 examples, ~33 GB)",
    },
}


def _download_one(variant: str) -> None:
    info = DATASETS[variant]
    target = Path(getattr(settings, info["target_key"], info["default_path"]))

    json_file, db_dir = info["check_files"]
    if (target / json_file).exists() and (target / db_dir).exists():
        print(f"{info['label']} already exists at {target}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {info['label']} to {target} ...")
    snapshot_download(
        repo_id=info["repo_id"],
        repo_type="dataset",
        local_dir=str(target),
    )

    if (target / json_file).exists():
        print(f"Done. {info['label']} ready at {target}")
    elif (target / db_dir).exists():
        print(f"Done. Databases found at {target}/{db_dir} (JSON may have a different name, check directory)")
    elif (target / "data" / "mini_dev_sqlite-00000-of-00001.json").exists():
        print(
            "Downloaded Mini-Dev JSON shards, but SQLite databases are missing.\n"
            "For runnable evaluation, download the complete package from:\n"
            "https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view?usp=sharing",
            file=sys.stderr,
        )
    else:
        print(
            f"Download completed but expected files not found at {target}.\n"
            f"Expected: {json_file} and {db_dir}/\n"
            "Check the directory structure and move files if needed.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BIRD datasets from Hugging Face")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mini", action="store_true", help="Download Mini-Dev (default)")
    group.add_argument("--full", action="store_true", help="Download full Dev set (~33 GB)")
    group.add_argument("--both", action="store_true", help="Download both mini and full")
    args = parser.parse_args()

    if args.both:
        _download_one("mini")
        _download_one("full")
    elif args.full:
        _download_one("full")
    else:
        _download_one("mini")


if __name__ == "__main__":
    main()
