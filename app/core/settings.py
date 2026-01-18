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

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct"

    # LLM defaults
    llm_provider: str = "ollama"  # "ollama" or "openrouter"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512


settings = Settings()
