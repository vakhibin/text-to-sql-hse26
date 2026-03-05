from pydantic_settings import BaseSettings, SettingsConfigDict
from app.config.secrets import (OPENROUTER_API_KEY,
                                LLM_PROVIDER,
                                LLM_MODEL,
                                LLM_TEMPERATURE,
                                LLM_MAX_TOKENS,
                                EMBEDDING_MODEL)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
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
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    # Embeddings
    embedding_model: str = "qwen/qwen3-embedding-8b"


settings = Settings()
