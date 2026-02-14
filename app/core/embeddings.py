import asyncio
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from langchain_openai import OpenAIEmbeddings


async def create_embeddings(
    provider: Literal["openrouter"] = "openrouter",
    model: str = "qwen/qwen3-embedding-8b",
    api_key: str | None = None,
) -> OpenAIEmbeddings:
    """
    Create a LangChain embeddings client asynchronously.

    Args:
        provider: Embeddings provider (currently only "openrouter")
        model: Model name
        api_key: API key (required for OpenRouter)

    Returns:
        LangChain embeddings instance
    """
    if provider == "openrouter":
        if not api_key:
            raise ValueError("API key is required for OpenRouter")
        return await asyncio.to_thread(
            lambda: OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def cosine_similarity(
    query_embedding: list[float] | NDArray[np.float32],
    corpus_embeddings: list[list[float]] | NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute cosine similarity between query and corpus embeddings.

    Args:
        query_embedding: Single query embedding
        corpus_embeddings: Corpus embeddings

    Returns:
        Similarity scores
    """
    query = np.array(query_embedding, dtype=np.float32)
    corpus = np.array(corpus_embeddings, dtype=np.float32)

    query_norm = query / np.linalg.norm(query)
    corpus_norms = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    return np.dot(corpus_norms, query_norm)


async def compute_cosine_similarity(
    query_embedding: list[float] | NDArray[np.float32],
    corpus_embeddings: list[list[float]] | NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute cosine similarity between query and corpus embeddings asynchronously.

    Args:
        query_embedding: Single query embedding
        corpus_embeddings: Corpus embeddings

    Returns:
        Similarity scores
    """
    return await asyncio.to_thread(
        cosine_similarity,
        query_embedding,
        corpus_embeddings
    )
