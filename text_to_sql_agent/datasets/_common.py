"""Shared helpers for dataset download modules."""

from __future__ import annotations

from pathlib import Path


def paths_ready(root: Path, relatives: tuple[str, ...]) -> bool:
    """Return True if each relative path exists under root (file or directory)."""
    return all((root / rel).exists() for rel in relatives)


def resolve_target_root(
    *,
    target_key: str,
    default_path: str,
    override: Path | None,
    settings_module,
) -> Path:
    """Resolve dataset root from explicit override, settings field, or default path string."""
    if override is not None:
        return override.expanduser().resolve()
    configured = getattr(settings_module, target_key, None)
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return Path(default_path).expanduser().resolve()
