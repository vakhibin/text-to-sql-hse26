from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str | None = None
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_embedding_model: str = "qwen/qwen3-embedding-8b"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct"

    # LLM defaults
    llm_provider: str = "openrouter"
    llm_model: str = "anthropic/claude-3.5-sonnet"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512


settings = Settings()
