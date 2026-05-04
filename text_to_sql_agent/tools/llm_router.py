"""Unified OpenRouter model router built on LangChain."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable, Iterable, Sequence, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
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
    JUDGE = "judge"


class LLMRouterError(Exception):
    """Base error for model routing/invocation."""


class MissingOpenRouterKeyError(LLMRouterError):
    """Raised when OPENROUTER_API_KEY is not configured."""


class LLMInvocationError(LLMRouterError):
    """Raised when invocation fails after retries."""


T = TypeVar("T")


def _model_for_langfuse(model_name: str) -> str:
    """Strip the provider prefix so Langfuse can match its default model registry.

    Langfuse ships a built-in cost table keyed by short model names
    (``gpt-4.1``, ``gemini-2.5-pro``, ...). OpenRouter exposes the same models
    as ``provider/model`` (``openai/gpt-4.1``, ``google/gemini-2.5-pro``).
    Keeping the slash hides our generations from the cost calculator, so we
    drop the prefix when reporting the model to Langfuse. The full id is
    still recorded in span metadata via ``model_full_id``.
    """
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _is_gateway_timeout(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 504:
        return True
    code = getattr(exc, "code", None)
    if code in (504, "504"):
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 504:
        return True
    lowered = str(exc).lower()
    return (
        ("504" in lowered and ("gateway" in lowered or "timeout" in lowered))
        or "gateway timeout" in lowered
    )


async def _ainvoke_with_gateway_retry(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run async LLM call; on gateway timeout, sleep once and retry (single extra attempt)."""
    last_exc: BaseException | None = None
    for attempt in range(2):
        try:
            return await coro_factory()
        except BaseException as exc:
            last_exc = exc
            if attempt == 0 and _is_gateway_timeout(exc):
                await asyncio.sleep(float(settings.retry_wait_min_seconds))
                continue
            raise
    assert last_exc is not None
    raise last_exc


@dataclass
class LLMInvokeResult:
    """Normalized LLM response with usage metadata."""

    text: str
    usage: dict[str, Any]
    response_metadata: dict[str, Any]
    structured: BaseModel | None = None


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
        if role == ModelRole.JUDGE:
            return settings.judge_model
        raise LLMRouterError(f"Unknown role: {role}")

    def temperature_for_role(self, role: ModelRole) -> float:
        """Default temperature by role."""
        if role == ModelRole.GENERATOR_PRIMARY:
            return settings.llm_temperature_primary
        if role == ModelRole.GENERATOR_SECONDARY:
            return settings.llm_temperature_secondary
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
                # Ask OpenRouter to include the actual upstream cost in the
                # response. Without this flag the response carries only token
                # counts and ``cost`` stays at 0, which makes Langfuse fall
                # back to its default-model price table.
                extra_body={"usage": {"include": True}},
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
        structured_output: type[BaseModel] | None = None,
    ) -> LLMInvokeResult:
        """Invoke model for role and return normalized text plus usage metadata."""
        model_name = model_override or self.model_for_role(role)
        stage_name = stage or role.value
        llm = self.get_chat_model(
            role,
            model_override=model_name,
            temperature_override=temperature_override,
        )
        langfuse_model = _model_for_langfuse(model_name)
        base_generation_metadata: dict[str, Any] = {
            "db_id": db_id,
            "stage": stage_name,
            "role": role.value,
            "model_full_id": model_name,
        }
        # Generation updates write to the *current* OTEL span, so they only
        # land on the right generation while we're inside the ``with`` block.
        with start_langfuse_generation(
            name=stage_name,
            trace_id=trace_id,
            model=langfuse_model,
            input_payload=list(messages),
            metadata=base_generation_metadata,
        ):
            try:
                if structured_output is not None:
                    structured_llm = llm.with_structured_output(structured_output, include_raw=True)
                    packed = await _ainvoke_with_gateway_retry(lambda: structured_llm.ainvoke(messages))
                    if not isinstance(packed, dict):
                        raise LLMInvocationError("structured_output: unexpected invoke payload")
                    raw_response = packed.get("raw")
                    parsed = packed.get("parsed")
                    parsing_error = packed.get("parsing_error")
                    structured_obj: BaseModel | None = None
                    if isinstance(parsed, BaseModel):
                        structured_obj = parsed
                    elif isinstance(parsed, dict):
                        try:
                            structured_obj = structured_output.model_validate(parsed)
                        except Exception:
                            structured_obj = None
                    if raw_response is None:
                        raise LLMInvocationError("structured_output: missing raw response")
                    normalized_text = self._normalize_text(raw_response.content)
                    usage = self._extract_usage(
                        response=raw_response,
                        model_name=model_name,
                        stage=stage_name,
                    )
                    if parsing_error is not None:
                        update_langfuse_generation(
                            output=normalized_text,
                            usage=usage,
                            metadata={
                                **base_generation_metadata,
                                "structured_parsing_error": str(parsing_error),
                            },
                        )
                        return LLMInvokeResult(
                            text=normalized_text,
                            usage=usage.as_dict(),
                            response_metadata=getattr(raw_response, "response_metadata", {}) or {},
                            structured=None,
                        )
                    update_langfuse_generation(
                        output=normalized_text,
                        usage=usage,
                        metadata=base_generation_metadata,
                    )
                    return LLMInvokeResult(
                        text=normalized_text,
                        usage=usage.as_dict(),
                        response_metadata=getattr(raw_response, "response_metadata", {}) or {},
                        structured=structured_obj,
                    )

                response = await _ainvoke_with_gateway_retry(lambda: llm.ainvoke(messages))
                normalized_text = self._normalize_text(response.content)
                usage = self._extract_usage(response=response, model_name=model_name, stage=stage_name)
                update_langfuse_generation(
                    output=normalized_text,
                    usage=usage,
                    metadata=base_generation_metadata,
                )
                return LLMInvokeResult(
                    text=normalized_text,
                    usage=usage.as_dict(),
                    response_metadata=getattr(response, "response_metadata", {}) or {},
                    structured=None,
                )
            except Exception as exc:  # pragma: no cover - runtime/network path
                update_langfuse_generation(
                    level="ERROR",
                    status_message=str(exc),
                    metadata=base_generation_metadata,
                )
                raise LLMInvocationError(str(exc)) from exc

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

