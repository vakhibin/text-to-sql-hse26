"""BIRD dataset artifacts (Hugging Face snapshots).

Structure mirrors ``text_to_sql_agent.datasets.spider_assets`` (VARIANTS + download_assets).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from text_to_sql_agent.config import settings
from text_to_sql_agent.datasets._common import paths_ready, resolve_target_root

BirdVariant = Literal["mini", "full"]

BIRD_VARIANTS: dict[str, dict[str, Any]] = {
    "mini": {
        "label": "BIRD Mini-Dev (500 examples)",
        "target_key": "bird_mini_root",
        "default_path": "databases/bird_mini",
        "check_paths": ("mini_dev_sqlite.json", "mini_dev_databases"),
        "backend": "huggingface",
        "repo_id": "birdsql/bird_mini_dev",
    },
    "full": {
        "label": "BIRD Full Dev (1534 examples, ~33 GB)",
        "target_key": "bird_root",
        "default_path": "databases/bird",
        "check_paths": ("dev.json", "dev_databases"),
        "backend": "huggingface",
        "repo_id": "birdsql/bird_sql_dev_20251106",
    },
}


def download_bird_assets(
    variant: BirdVariant,
    *,
    bird_root: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download one BIRD bundle into the configured root (or ``bird_root`` override).

    Mirrors ``download_spider_assets`` naming for scripts/runners.
    """
    info = BIRD_VARIANTS[variant]
    target = resolve_target_root(
        target_key=str(info["target_key"]),
        default_path=str(info["default_path"]),
        override=bird_root,
        settings_module=settings,
    )
    json_file, db_dir = info["check_paths"][0], info["check_paths"][1]

    if not force and (target / json_file).exists() and (target / db_dir).exists():
        print(f"{info['label']} already exists at {target}")
        return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first: uv add huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"Downloading {info['label']} to {target} ...")
    snapshot_download(
        repo_id=str(info["repo_id"]),
        repo_type="dataset",
        local_dir=str(target),
    )

    if (target / json_file).exists():
        print(f"Done. {info['label']} ready at {target}")
    elif (target / db_dir).exists():
        print(
            f"Done. Databases found at {target}/{db_dir} "
            "(JSON may use another filename; inspect the directory)."
        )
    elif (target / "data" / "mini_dev_sqlite-00000-of-00001.json").exists():
        print(
            "Downloaded Mini-Dev JSON shards, but SQLite databases may be missing.\n"
            "See HF dataset page or BIRD instructions for the complete SQLite package.",
            file=sys.stderr,
        )
    else:
        print(
            f"Download completed but expected files not found at {target}.\n"
            f"Expected: {json_file} and {db_dir}/",
            file=sys.stderr,
        )
    return target


def bird_bundle_ready(variant: BirdVariant, root: Path | None = None) -> bool:
    """Whether required JSON + DB directory exist for a variant."""
    info = BIRD_VARIANTS[variant]
    target = resolve_target_root(
        target_key=str(info["target_key"]),
        default_path=str(info["default_path"]),
        override=root,
        settings_module=settings,
    )
    return paths_ready(target, tuple(str(x) for x in info["check_paths"]))
