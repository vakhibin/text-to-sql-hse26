"""Unified OpenRouter model router built on LangChain."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Iterable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from text_to_sql_agent.config import settings


class ModelRole(StrEnum):
    """Supported model roles in the pipeline."""

    GENERATOR_PRIMARY = "generator_primary"
    GENERATOR_SECONDARY = "generator_secondary"
    JUDGE = "judge"
    REFINER = "refiner"


class LLMRouterError(Exception):
    """Base error for model routing/invocation."""


class MissingOpenRouterKeyError(LLMRouterError):
    """Raised when OPENROUTER_API_KEY is not configured."""


class LLMInvocationError(LLMRouterError):
    """Raised when invocation fails after retries."""


class LLMRouter:
    """Create and route LangChain chat models via OpenRouter."""

    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            raise MissingOpenRouterKeyError("OPENROUTER_API_KEY is required for LLM calls")
        self._cache: dict[tuple[str, float], BaseChatModel] = {}

    def model_for_role(self, role: ModelRole) -> str:
        """Resolve model id for a role."""
        if role == ModelRole.GENERATOR_PRIMARY:
            return settings.generator_model_primary
        if role == ModelRole.GENERATOR_SECONDARY:
            return settings.generator_model_secondary
        if role == ModelRole.JUDGE:
            return settings.judge_model
        if role == ModelRole.REFINER:
            # Refiner defaults to judge-grade model for stronger correction.
            return settings.judge_model
        raise LLMRouterError(f"Unknown role: {role}")

    def temperature_for_role(self, role: ModelRole) -> float:
        """Default temperature by role."""
        if role == ModelRole.GENERATOR_PRIMARY:
            return settings.llm_temperature_primary
        if role == ModelRole.GENERATOR_SECONDARY:
            return settings.llm_temperature_secondary
        return settings.llm_temperature_judge

    def get_chat_model(
        self,
        role: ModelRole,
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
    ) -> BaseChatModel:
        """Return cached ChatOpenAI client configured for OpenRouter."""
        model = model_override or self.model_for_role(role)
        temperature = (
            temperature_override if temperature_override is not None else self.temperature_for_role(role)
        )
        cache_key = (model, temperature)
        if cache_key not in self._cache:
            self._cache[cache_key] = ChatOpenAI(
                model=model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                temperature=temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_seconds,
            )
        return self._cache[cache_key]

    @retry(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(
            min=settings.retry_wait_min_seconds,
            max=settings.retry_wait_max_seconds,
        ),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def ainvoke(
        self,
        role: ModelRole,
        messages: Sequence[tuple[str, str]],
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
    ) -> str:
        """Invoke model for role and return normalized text content."""
        llm = self.get_chat_model(
            role,
            model_override=model_override,
            temperature_override=temperature_override,
        )
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:  # pragma: no cover - runtime/network path
            raise LLMInvocationError(str(exc)) from exc
        return str(response.content).strip()

    def generator_roles(self) -> list[ModelRole]:
        """Return fixed role allocation for N=8 ensemble."""
        primary = [ModelRole.GENERATOR_PRIMARY] * settings.primary_calls
        secondary = [ModelRole.GENERATOR_SECONDARY] * settings.secondary_calls
        return [*primary, *secondary]

    async def abatch_generate(
        self,
        messages_batch: Iterable[Sequence[tuple[str, str]]],
    ) -> list[str]:
        """Generate with fixed 5/3 role routing for ensemble stage."""
        roles = self.generator_roles()
        tasks = []
        for idx, messages in enumerate(messages_batch):
            role = roles[idx % len(roles)]
            tasks.append(self.ainvoke(role, messages))
        return await asyncio.gather(*tasks)

