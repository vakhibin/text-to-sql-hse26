import asyncio
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel


class LLMClientError(Exception):
    pass

class NoTokenProvidedError(Exception):
    pass


async def create_llm(
    provider: Literal["openrouter", "ollama"],
    model: str,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 60,
) -> BaseChatModel:
    """
    Create a LangChain LLM client for the specified provider asynchronously.

    Args:
        provider: LLM provider ("openrouter" or "ollama")
        model: Model name (e.g., "anthropic/claude-3.5-sonnet" for OpenRouter,
               "mistral:7b-instruct" for Ollama)
        api_key: API key (required for OpenRouter)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        LangChain chat model instance
    """
    if provider == "openrouter":
        if not api_key:
            raise LLMClientError("API key is required for OpenRouter")
        return await asyncio.to_thread(
            lambda: ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        )
    elif provider == "ollama":
        return await asyncio.to_thread(
            lambda: ChatOllama(
                model=model,
                temperature=temperature,
                num_predict=max_tokens,
                timeout=timeout,
            )
        )
    else:
        raise LLMClientError(f"Unknown provider: {provider}")

if __name__ == "__main__":
    async def main():
        llm = await create_llm(provider="openrouter", model="mistral:7b-instruct")

        messages = [
            ("system", "Just have some funny conversation with me"),
            ("user", "Hello! How are you today?"),
        ]

        response = await llm.ainvoke(messages)
        print(response.content)

    asyncio.run(main())