"""Central configuration for text_to_sql_agent."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Runtime settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Fixed model policy (can be overridden by env if needed)
    generator_model_primary: str = "google/gemini-2.5-pro"
    generator_model_secondary: str = "deepseek/deepseek-chat-v3"
    judge_model: str = "openai/gpt-4.1"
    embeddings_model: str = "openai/text-embedding-3-small"

    # Data and retrieval paths
    spider_root: str = "databases/spider"
    chroma_persist_directory: str = ".cache/chroma"
    chroma_collection_selector: str = "schema_selector"

    # Ensemble defaults
    num_candidates: int = 8
    primary_calls: int = 5
    secondary_calls: int = 3

    # Selector defaults
    selector_top_k_tables: int = 15
    selector_target_tables_min: int = 3
    selector_target_tables_max: int = 5

    # LLM generation defaults
    llm_temperature_primary: float = 0.2
    llm_temperature_secondary: float = 0.6
    llm_temperature_judge: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 90

    # Retry policy (tenacity)
    retry_attempts: int = 3
    retry_wait_min_seconds: int = 1
    retry_wait_max_seconds: int = 8

    # Refiner defaults
    max_refine_attempts: int = 3


settings = AgentSettings()

