"""Few-shot example loading and semantic retrieval utilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from text_to_sql_agent.config import settings
from text_to_sql_agent.tools.embedding_client import build_langchain_embeddings


def _read_train_examples_sync(spider_root: str | Path, max_pool_size: int) -> list[dict[str, str]]:
    root = Path(spider_root)
    train_path = root / "train_spider.json"
    if not train_path.exists():
        return []
    with train_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pool: list[dict[str, str]] = []
    for item in raw:
        q = str(item.get("question", "")).strip()
        sql = str(item.get("query", "")).strip()
        db_id = str(item.get("db_id", "")).strip()
        if not q or not sql:
            continue
        pool.append({"question": q, "sql": sql, "db_id": db_id})
        if len(pool) >= max_pool_size:
            break
    return pool


async def load_few_shot_pool(
    *,
    spider_root: str | Path | None = None,
    max_pool_size: int | None = None,
) -> list[dict[str, str]]:
    """Load candidate few-shot examples from Spider train split."""
    root = spider_root or settings.spider_root
    pool_limit = max_pool_size or settings.few_shot_max_pool_size
    return await asyncio.to_thread(_read_train_examples_sync, root, pool_limit)


def _embedding_namespace(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()[:48] or "embedding"
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _example_to_document(example_id: str, example: dict[str, str]) -> Document:
    question = str(example.get("question", "")).strip()
    sql = str(example.get("sql", "")).strip()
    db_id = str(example.get("db_id", "")).strip()
    content = f"question={question}\nsql={sql}"
    return Document(
        page_content=content,
        metadata={
            "example_id": example_id,
            "question": question,
            "sql": sql,
            "db_id": db_id,
        },
    )


class FewShotRetriever:
    """Semantic retriever for Spider train few-shot examples."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path = ".cache/chroma",
    ) -> None:
        self.collection_name = f"{collection_name}__{_embedding_namespace(settings.embeddings_model)}"
        self.persist_directory = str(persist_directory)
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=build_langchain_embeddings(),
        )
        self._index_lock = asyncio.Lock()
        self._indexed_size = 0

    def _collection_count(self) -> int:
        collection = getattr(self._vector_store, "_collection", None)
        if collection is None:
            return 0
        try:
            return int(collection.count())
        except Exception:
            return 0

    def _index_examples_sync(self, pool: list[dict[str, str]]) -> None:
        docs = []
        ids = []
        for idx, example in enumerate(pool):
            example_id = f"train_example:{idx}"
            docs.append(_example_to_document(example_id, example))
            ids.append(example_id)
        if not docs:
            return
        try:
            self._vector_store.delete(ids=ids)
        except Exception:
            pass
        self._vector_store.add_documents(documents=docs, ids=ids)

    async def ensure_indexed(self, pool: list[dict[str, str]]) -> None:
        if not settings.few_shot_semantic_retrieval or not pool:
            return
        expected = len(pool)
        if self._indexed_size == expected:
            return
        async with self._index_lock:
            if self._indexed_size == expected:
                return
            if self._collection_count() < expected:
                await asyncio.to_thread(self._index_examples_sync, pool)
            self._indexed_size = expected

    def _query_sync(self, query: str, *, db_id: str | None, top_k: int) -> list[dict[str, str]]:
        docs = self._vector_store.similarity_search(
            query=query,
            k=top_k,
            filter={"db_id": db_id} if db_id else None,
        )
        return [
            {
                "question": str(doc.metadata.get("question", "")).strip(),
                "sql": str(doc.metadata.get("sql", "")).strip(),
                "db_id": str(doc.metadata.get("db_id", "")).strip(),
                "example_id": str(doc.metadata.get("example_id", "")).strip(),
            }
            for doc in docs
            if str(doc.metadata.get("question", "")).strip() and str(doc.metadata.get("sql", "")).strip()
        ]

    async def query_examples(
        self,
        *,
        query: str,
        target_db_id: str | None,
    ) -> list[dict[str, str]]:
        same_db = []
        if target_db_id:
            same_db = await asyncio.to_thread(
                self._query_sync,
                query,
                db_id=target_db_id,
                top_k=settings.few_shot_same_db_top_k,
            )
        global_examples = await asyncio.to_thread(
            self._query_sync,
            query,
            db_id=None,
            top_k=settings.few_shot_retrieval_top_k,
        )
        merged: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for item in [*same_db, *global_examples]:
            example_id = item.get("example_id", "")
            if not example_id or example_id in seen_ids:
                continue
            merged.append(item)
            seen_ids.add(example_id)
        return merged


_retriever: FewShotRetriever | None = None


def _get_few_shot_retriever() -> FewShotRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FewShotRetriever(
            collection_name=settings.chroma_collection_few_shot,
            persist_directory=settings.chroma_persist_directory,
        )
    return _retriever


async def prewarm_few_shot_cache(*, spider_root: str | Path | None = None) -> None:
    """Load Spider train pool and build Chroma few-shot index once (matches per-example ``ensure_indexed``)."""
    import sys

    pool = await load_few_shot_pool(spider_root=spider_root)
    if not pool:
        print(
            "few_shot: train_spider.json missing or empty under SPIDER_ROOT — generator uses zero-shot",
            file=sys.stderr,
        )
        return
    if not settings.few_shot_semantic_retrieval:
        return
    try:
        retriever = _get_few_shot_retriever()
        await retriever.ensure_indexed(pool)
    except Exception as exc:
        print(f"few_shot: prewarm failed ({exc}); semantic retrieval may fall back per-example", file=sys.stderr)


async def retrieve_examples_for_candidate(
    *,
    pool: list[dict[str, str]],
    question: str,
    candidate_index: int,
    k: int,
    seed: int,
    target_db_id: str | None = None,
) -> list[dict[str, str]]:
    """Retrieve few-shot examples semantically with deterministic diversity."""
    if not pool or k <= 0:
        return []
    if not settings.few_shot_semantic_retrieval:
        return sample_examples_for_candidate(
            pool=pool,
            candidate_index=candidate_index,
            k=k,
            seed=seed,
            target_db_id=target_db_id,
        )

    try:
        retriever = _get_few_shot_retriever()
        await retriever.ensure_indexed(pool)
        ranked = await retriever.query_examples(query=question, target_db_id=target_db_id)
    except Exception:
        ranked = []

    if not ranked:
        return sample_examples_for_candidate(
            pool=pool,
            candidate_index=candidate_index,
            k=k,
            seed=seed,
            target_db_id=target_db_id,
        )

    window_size = min(
        len(ranked),
        max(k, settings.few_shot_retrieval_window),
    )
    window = ranked[:window_size]
    if len(window) <= k:
        return window
    rng = random.Random(seed + candidate_index)
    return rng.sample(window, k=k)


def sample_examples_for_candidate(
    *,
    pool: list[dict[str, str]],
    candidate_index: int,
    k: int,
    seed: int,
    target_db_id: str | None = None,
) -> list[dict[str, str]]:
    """Sample deterministic few-shot subset for one candidate."""
    if not pool or k <= 0:
        return []

    same_db = [item for item in pool if target_db_id and item.get("db_id") == target_db_id]
    source = same_db if len(same_db) >= k else pool

    rng = random.Random(seed + candidate_index)
    if len(source) <= k:
        return source
    return rng.sample(source, k=k)

