from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter API
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Ollama
    ollama_host: str = "http://localhost:11434"

    # LLM settings
    llm_provider: str = "openrouter"  # "openrouter" or "ollama"
    llm_model: str = "deepseek/deepseek-r1-0528:free"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    # Embeddings
    embedding_model: str = "qwen/qwen3-embedding-8b"


settings = Settings()
