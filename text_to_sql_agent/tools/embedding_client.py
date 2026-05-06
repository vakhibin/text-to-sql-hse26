"""LangChain-compatible embeddings via OpenAI-compatible HTTP API (OpenRouter)."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.embeddings import Embeddings

from text_to_sql_agent.config import settings

# OpenAI / OpenRouter embedding models enforce a per-input token cap (typically 8192).
_EMBED_MAX_TOKENS = 7500


def _truncate_for_embedding(text: str, *, max_tokens: int = _EMBED_MAX_TOKENS) -> str:
    if not text:
        return text
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens]) + "\n...[truncated-for-embedding]"
    except Exception:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated-for-embedding]"


class OpenRouterHttpEmbeddings(Embeddings):
    """POST /v1/embeddings — avoids langchain-openai parsers that fail on some OpenRouter responses."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        batch_size: int = 16,
        timeout_s: float = 120.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/embeddings"
        self._batch_size = max(1, batch_size)
        self._timeout_s = timeout_s

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clipped = [_truncate_for_embedding(t) for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(clipped), self._batch_size):
            batch = clipped[i : i + self._batch_size]
            out.extend(self._embed_batch(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self._model, "input": texts}
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/text-to-sql-hse26",
                "X-Title": "text-to-sql-hse26",
            },
        )
        try:
            with urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embeddings HTTP {e.code}: {detail}") from e
        except URLError as e:
            raise RuntimeError(f"Embeddings request failed: {e}") from e

        data = json.loads(raw)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Embeddings API error: {data.get('error')}")
        items = data.get("data") if isinstance(data, dict) else None
        if not items:
            raise ValueError(
                "No embedding data received; "
                f"keys={list(data.keys()) if isinstance(data, dict) else None}"
            )

        indexed: list[tuple[int, list[float]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("index", len(indexed)))
            emb = item.get("embedding")
            if not isinstance(emb, list):
                continue
            indexed.append((idx, [float(x) for x in emb]))
        indexed.sort(key=lambda x: x[0])
        vectors = [v for _, v in indexed]
        if len(vectors) != len(texts):
            raise ValueError(f"Embedding count mismatch: {len(vectors)} vs {len(texts)} inputs")
        return vectors


def build_langchain_embeddings() -> Embeddings:
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for vector embeddings")
    return OpenRouterHttpEmbeddings(
        model=settings.embeddings_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
