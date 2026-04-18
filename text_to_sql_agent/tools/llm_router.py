"""Unified OpenRouter model router built on LangChain."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from text_to_sql_agent.config import settings
from text_to_sql_agent.tools.observability import (
    LLMUsageRecord,
    start_langfuse_generation,
    update_langfuse_generation,
)


class ModelRole(StrEnum):
    """Supported model roles in the pipeline."""

    GENERATOR_PRIMARY = "generator_primary"
    GENERATOR_SECONDARY = "generator_secondary"
    QUERY_SKETCHER = "query_sketcher"
    REFINER = "refiner"


class LLMRouterError(Exception):
    """Base error for model routing/invocation."""


class MissingOpenRouterKeyError(LLMRouterError):
    """Raised when OPENROUTER_API_KEY is not configured."""


class LLMInvocationError(LLMRouterError):
    """Raised when invocation fails after retries."""


@dataclass
class LLMInvokeResult:
    """Normalized LLM response with usage metadata."""

    text: str
    usage: dict[str, Any]
    response_metadata: dict[str, Any]


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
        if role == ModelRole.QUERY_SKETCHER:
            return settings.query_sketcher_model or settings.generator_model_primary
        if role == ModelRole.REFINER:
            return settings.refiner_model
        raise LLMRouterError(f"Unknown role: {role}")

    def temperature_for_role(self, role: ModelRole) -> float:
        """Default temperature by role."""
        if role == ModelRole.GENERATOR_PRIMARY:
            return settings.llm_temperature_primary
        if role == ModelRole.GENERATOR_SECONDARY:
            return settings.llm_temperature_secondary
        if role == ModelRole.QUERY_SKETCHER:
            return settings.llm_temperature_refiner
        return settings.llm_temperature_refiner

    def max_tokens_for_role(self, role: ModelRole) -> int:
        """Max completion tokens; sketcher may use a higher budget than other stages."""
        if role == ModelRole.QUERY_SKETCHER:
            return settings.llm_max_tokens_query_sketcher
        return settings.llm_max_tokens

    def get_chat_model(
        self,
        role: ModelRole,
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
    ) -> BaseChatModel:
        """Return cached ChatOpenAI client configured for OpenRouter."""
        model = model_override or self.model_for_role(role)
        temperature = (
            temperature_override if temperature_override is not None else self.temperature_for_role(role)
        )
        max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens_for_role(role)
        cache_key = (model, temperature, max_tokens)
        if cache_key not in self._cache:
            self._cache[cache_key] = ChatOpenAI(
                model=model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.llm_timeout_seconds,
            )
        return self._cache[cache_key]

    def _normalize_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return " ".join(part.strip() for part in parts if part).strip()
        return str(content).strip()

    def _extract_usage(
        self,
        *,
        response: Any,
        model_name: str,
        stage: str,
    ) -> LLMUsageRecord:
        usage_metadata = getattr(response, "usage_metadata", {}) or {}
        response_metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage", {}) or response_metadata.get("usage", {}) or {}

        prompt_tokens = int(
            usage_metadata.get("input_tokens")
            or usage_metadata.get("prompt_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0
        )
        completion_tokens = int(
            usage_metadata.get("output_tokens")
            or usage_metadata.get("completion_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        )
        total_tokens = int(
            usage_metadata.get("total_tokens")
            or token_usage.get("total_tokens")
            or (prompt_tokens + completion_tokens)
        )

        cost_usd = 0.0
        for source in (
            response_metadata,
            response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {},
            response_metadata.get("cost_details", {}) if isinstance(response_metadata, dict) else {},
            getattr(response, "additional_kwargs", {}) or {},
        ):
            if not isinstance(source, dict):
                continue
            raw = source.get("cost") or source.get("total_cost") or source.get("upstream_inference_cost")
            if raw is not None:
                try:
                    cost_usd = float(raw)
                    break
                except (TypeError, ValueError):
                    continue

        return LLMUsageRecord(
            stage=stage,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )

    @retry(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(
            min=settings.retry_wait_min_seconds,
            max=settings.retry_wait_max_seconds,
        ),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def ainvoke_with_metadata(
        self,
        role: ModelRole,
        messages: Sequence[tuple[str, str]],
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
        trace_id: str | None = None,
        db_id: str | None = None,
        stage: str | None = None,
    ) -> LLMInvokeResult:
        """Invoke model for role and return normalized text plus usage metadata."""
        model_name = model_override or self.model_for_role(role)
        stage_name = stage or role.value
        llm = self.get_chat_model(
            role,
            model_override=model_name,
            temperature_override=temperature_override,
        )
        generation, generation_ctx = start_langfuse_generation(
            name=stage_name,
            trace_id=trace_id,
            model=model_name,
            input_payload=list(messages),
            metadata={"db_id": db_id, "stage": stage_name, "role": role.value},
        )
        try:
            with generation_ctx:
                response = await llm.ainvoke(messages)
        except Exception as exc:  # pragma: no cover - runtime/network path
            update_langfuse_generation(
                generation,
                level="ERROR",
                status_message=str(exc),
                metadata={"db_id": db_id, "stage": stage_name, "role": role.value},
            )
            raise LLMInvocationError(str(exc)) from exc

        normalized_text = self._normalize_text(response.content)
        usage = self._extract_usage(response=response, model_name=model_name, stage=stage_name)
        update_langfuse_generation(
            generation,
            output=normalized_text,
            usage=usage,
            metadata={"db_id": db_id, "stage": stage_name, "role": role.value},
        )
        return LLMInvokeResult(
            text=normalized_text,
            usage=usage.as_dict(),
            response_metadata=getattr(response, "response_metadata", {}) or {},
        )

    async def ainvoke(
        self,
        role: ModelRole,
        messages: Sequence[tuple[str, str]],
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
    ) -> str:
        """Backward-compatible text-only invoke wrapper."""
        result = await self.ainvoke_with_metadata(
            role,
            messages,
            model_override=model_override,
            temperature_override=temperature_override,
        )
        return result.text

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

