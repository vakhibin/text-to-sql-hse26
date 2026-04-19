"""Runtime settings for the orchestrator service.

Kept separate from the core ``text_to_sql_agent.config`` so the two services
can evolve their env contracts independently.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    text_to_sql_api_url: str = Field(
        default="http://localhost:8001",
        alias="TEXT_TO_SQL_API_URL",
        description="Base URL of the text_to_sql_api service.",
    )
    text_to_sql_api_timeout_s: float = Field(
        default=120.0,
        alias="TEXT_TO_SQL_API_TIMEOUT_S",
        description="Per-request timeout for calls from the orchestrator to text_to_sql_api.",
    )
    orchestrator_max_tool_steps: int = Field(
        default=6,
        alias="ORCHESTRATOR_MAX_TOOL_STEPS",
        description="Maximum number of tool-calling iterations inside a single /chat turn.",
    )


settings = OrchestratorSettings()
