"""Spider dataset artifacts (core train/dev via Kaggle, test split via Google Drive zip).

Test bundle source (official Yale Spider release mirror):
https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Literal

import kagglehub

from text_to_sql_agent.config import settings
from text_to_sql_agent.datasets._common import paths_ready, resolve_target_root

SpiderVariant = Literal["core", "test"]

SPIDER_VARIANTS: dict[str, dict[str, Any]] = {
    "core": {
        "label": "Spider v1 (tables.json, train_spider.json, dev.json, database/)",
        "target_key": "spider_root",
        "default_path": "databases/spider",
        "check_paths": ("tables.json", "dev.json", "train_spider.json", "database"),
        "backend": "kaggle",
        "kaggle_slug": "jeromeblanchet/yale-universitys-spider-10-nlp-dataset",
        "copy_names": ("tables.json", "dev.json", "train_spider.json", "database"),
    },
    "test": {
        "label": "Spider test (test.json, test_tables.json, test_gold.sql, test_database/)",
        "target_key": "spider_root",
        "default_path": "databases/spider",
        "check_paths": ("test.json", "test_tables.json", "test_gold.sql", "test_database"),
        "backend": "gdrive_zip",
        "gdrive_file_id": "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J",
        "zip_cache_name": "spider_test_drive.zip",
    },
}


def _required_spider_core_paths(root: Path) -> list[Path]:
    meta = SPIDER_VARIANTS["core"]
    return [root / rel for rel in meta["check_paths"]]


def _is_spider_core_ready(root: Path) -> bool:
    return paths_ready(root, tuple(SPIDER_VARIANTS["core"]["check_paths"]))


def _find_spider_core_root(downloaded_dir: Path) -> Path | None:
    if _is_spider_core_ready(downloaded_dir):
        return downloaded_dir
    for candidate in downloaded_dir.rglob("*"):
        if candidate.is_dir() and _is_spider_core_ready(candidate):
            return candidate
    return None


def _copy_named_tree(source_root: Path, target_root: Path, names: tuple[str, ...]) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = source_root / name
        dst = target_root / name
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _normalize_zip_entry(name: str) -> str:
    return name.replace("\\", "/").strip("/")


def _find_zip_member_with_basename(names: list[str], basename: str) -> str | None:
    hits = [
        n
        for n in names
        if not _is_zip_junk_or_directory(n)
        and not n.endswith("/")
        and Path(_normalize_zip_entry(n)).name == basename
    ]
    if not hits:
        return None
    return min(hits, key=len)


def _find_test_database_prefix(names: list[str]) -> str | None:
    """Return zip-internal prefix ending at .../test_database (no trailing slash)."""
    marker = "test_database"
    best: str | None = None
    for raw in names:
        norm = _normalize_zip_entry(raw).rstrip("/")
        if marker not in norm.split("/"):
            continue
        parts = norm.split("/")
        try:
            idx = parts.index(marker)
        except ValueError:
            continue
        prefix = "/".join(parts[: idx + 1])
        if best is None or len(prefix) < len(best):
            best = prefix
    return best


def _ensure_dir_under_anchor(dir_path: Path, anchor: Path) -> None:
    """Ensure dir_path exists as a directory; unlink a regular file at the same path if present."""
    if dir_path == anchor:
        return
    if anchor not in dir_path.parents:
        raise ValueError(f"Path {dir_path} is not under {anchor}")
    _ensure_dir_under_anchor(dir_path.parent, anchor)
    if dir_path.exists():
        if dir_path.is_dir():
            return
        dir_path.unlink()
    dir_path.mkdir(exist_ok=False)


def _is_zip_junk_or_directory(raw: str) -> bool:
    raw_fwd = raw.replace("\\", "/")
    if raw_fwd.endswith("/"):
        return True
    if "__MACOSX/" in raw_fwd:
        return True
    if "/._" in raw_fwd or Path(raw_fwd).name.startswith("._"):
        return True
    return False


def _extract_spider_test_zip(zip_path: Path, spider_root: Path) -> None:
    """Copy test artifacts from zip root into spider_root flat layout."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        for basename in ("test_tables.json", "test.json", "test_gold.sql"):
            member = _find_zip_member_with_basename(names, basename)
            if member is None:
                raise FileNotFoundError(
                    f"Zip {zip_path} does not contain file ending with {basename!r}. "
                    "Expected Yale Spider test bundle layout."
                )
            dest = spider_root / basename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)

        prefix = _find_test_database_prefix(names)
        if prefix is None:
            raise FileNotFoundError(
                f"Zip {zip_path} does not contain a test_database/ directory tree."
            )

        test_db_dest = spider_root / "test_database"
        if test_db_dest.exists():
            shutil.rmtree(test_db_dest)
        test_db_dest.mkdir(parents=True, exist_ok=True)

        prefix_norm = _normalize_zip_entry(prefix).rstrip("/")
        file_jobs: list[tuple[str, str]] = []
        for raw in names:
            if _is_zip_junk_or_directory(raw):
                continue
            norm = _normalize_zip_entry(raw)
            if norm == prefix_norm or norm.startswith(prefix_norm + "/"):
                inner = norm[len(prefix_norm) :].lstrip("/")
                if not inner:
                    continue
                file_jobs.append((raw, inner))

        # Deeper paths first so we never create e.g. …/browser_web as a file before …/browser_web/x.sqlite.
        file_jobs.sort(key=lambda job: (-job[1].count("/"), job[1]))

        for raw, inner in file_jobs:
            target = test_db_dest / inner
            _ensure_dir_under_anchor(target.parent, test_db_dest)
            if target.exists() and target.is_dir():
                shutil.rmtree(target)
            with zf.open(raw) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _download_gdrive_zip(file_id: str, dest_zip: Path) -> None:
    import gdown

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest_zip), quiet=False)


def download_spider_assets(
    variant: SpiderVariant,
    *,
    spider_root: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download one Spider artifact bundle into ``SPIDER_ROOT`` (or ``spider_root``).

    Variants mirror ``SPIDER_VARIANTS`` keys: ``core`` (Kaggle), ``test`` (Drive zip).
    """
    meta = SPIDER_VARIANTS[variant]
    root = resolve_target_root(
        target_key=str(meta["target_key"]),
        default_path=str(meta["default_path"]),
        override=spider_root,
        settings_module=settings,
    )
    checks = tuple(str(x) for x in meta["check_paths"])

    if not force and paths_ready(root, checks):
        print(f"{meta['label']} already present under {root}")
        return root

    backend = meta["backend"]
    if backend == "kaggle":
        downloaded_path = Path(kagglehub.dataset_download(str(meta["kaggle_slug"])))
        source_root = _find_spider_core_root(downloaded_path)
        if source_root is None:
            raise FileNotFoundError(
                f"Downloaded Kaggle dataset at {downloaded_path}, but Spider core files were not detected."
            )
        _copy_named_tree(source_root, root, tuple(str(x) for x in meta["copy_names"]))
        if not _is_spider_core_ready(root):
            raise FileNotFoundError(
                "Spider core copy finished, but tables.json / dev.json / train_spider.json / database/ "
                f"are still missing under {root}."
            )
        print(f"{meta['label']} ready at {root}")
        return root

    if backend == "gdrive_zip":
        cache_dir = root / ".download_cache"
        zip_path = cache_dir / str(meta["zip_cache_name"])
        if force or not zip_path.exists():
            _download_gdrive_zip(str(meta["gdrive_file_id"]), zip_path)
        _extract_spider_test_zip(zip_path, root)
        if not paths_ready(root, checks):
            raise FileNotFoundError(
                f"Spider test extraction finished, but expected files are missing under {root}. "
                f"Expected: {checks}"
            )
        print(f"{meta['label']} ready at {root}")
        return root

    raise ValueError(f"Unknown backend: {backend!r}")


def ensure_spider_core(spider_root: Path, *, allow_download: bool) -> Path:
    """Ensure Yale Spider train/dev bundle exists; optionally download via Kaggle."""
    if _is_spider_core_ready(spider_root):
        return spider_root
    if not allow_download:
        raise FileNotFoundError(
            f"Spider dataset not found at {spider_root}. "
            "Use --download or run: python scripts/download_spider.py --core"
        )
    download_spider_assets("core", spider_root=spider_root, force=False)
    return spider_root


def _test_database_root_has_only_directories(spider_root: Path) -> bool:
    """Yale test layout uses ``test_database/<db_id>/<db_id>.sqlite`` — no regular files at the root."""
    td = spider_root / "test_database"
    if not td.is_dir():
        return False
    for child in td.iterdir():
        if child.is_file():
            return False
    return True


def ensure_spider_for_eval(spider_root: Path, split: str, *, allow_download: bool) -> Path:
    """Ensure files needed for a Spider split exist (core always; test assets when ``split=='test'``)."""
    root = Path(spider_root).resolve()
    root = ensure_spider_core(root, allow_download=allow_download)
    if split != "test":
        return root
    checks = tuple(str(x) for x in SPIDER_VARIANTS["test"]["check_paths"])
    complete = paths_ready(root, checks)
    layout_ok = _test_database_root_has_only_directories(root)
    if complete and layout_ok:
        return root
    if not allow_download:
        hint = (
            f"Spider test files missing or corrupted under {root}. Expected: {checks}. "
            "Loose files directly under test_database/ usually mean a bad unzip — delete that folder "
            "or run: python scripts/download_spider.py --test --force"
        )
        raise FileNotFoundError(hint)
    download_spider_assets("test", spider_root=root, force=True)
    if not paths_ready(root, checks) or not _test_database_root_has_only_directories(root):
        raise FileNotFoundError(
            f"Spider test download finished, but required files are still missing or mis-layout under {root}."
        )
    return root
