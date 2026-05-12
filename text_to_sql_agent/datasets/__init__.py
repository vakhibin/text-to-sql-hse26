"""Dataset acquisition helpers (Spider, BIRD) shared by runners and scripts."""

from text_to_sql_agent.datasets.bird_assets import (
    BIRD_VARIANTS,
    bird_bundle_ready,
    download_bird_assets,
)
from text_to_sql_agent.datasets.spider_assets import (
    SPIDER_VARIANTS,
    download_spider_assets,
    ensure_spider_core,
    ensure_spider_for_eval,
)

__all__ = [
    "BIRD_VARIANTS",
    "SPIDER_VARIANTS",
    "bird_bundle_ready",
    "download_bird_assets",
    "download_spider_assets",
    "ensure_spider_core",
    "ensure_spider_for_eval",
]
