"""Async HTTP client for the text-to-SQL microservice.

Thin wrapper around ``httpx.AsyncClient`` that the orchestrator's tools call
to invoke ``text_to_sql_api`` endpoints. Responses are validated against the
same Pydantic schemas the FastAPI service exposes, so the contract is shared.

The client is owned by the FastAPI lifespan; tests can pass a client built on
``httpx.MockTransport`` to avoid any real network I/O.
"""

from __future__ import annotations

from typing import Any

import httpx

from services.text_to_sql_api.schemas import (
    DatabasesResponse,
    ExecuteResponse,
    ExplainResponse,
    RefineResponse,
    RunResponse,
    SchemaResponse,
)


class TextToSQLAPIError(RuntimeError):
    """Raised when the text-to-SQL service returns an unexpected HTTP status.

    Successful pipeline runs that end in validation/DB errors are NOT raised:
    they come back as 200 responses with ``success=False`` so the orchestrator
    LLM can see the error text and decide what to do next.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"text_to_sql_api HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class TextToSQLClient:
    """Narrow async client for ``text_to_sql_api`` endpoints used by tools."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(path, json=payload)
        if response.status_code >= 400:
            raise TextToSQLAPIError(response.status_code, response.text)
        return response.json()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._client.get(path, params=params)
        if response.status_code >= 400:
            raise TextToSQLAPIError(response.status_code, response.text)
        return response.json()

    async def health(self) -> dict[str, Any]:
        return await self._get("/health")

    async def list_databases(self, *, schema_root: str | None = None) -> DatabasesResponse:
        params = {"schema_root": schema_root} if schema_root else None
        data = await self._get("/databases", params=params)
        return DatabasesResponse.model_validate(data)

    async def get_schema(self, db_id: str, *, schema_root: str | None = None) -> SchemaResponse:
        params = {"schema_root": schema_root} if schema_root else None
        data = await self._get(f"/databases/{db_id}/schema", params=params)
        return SchemaResponse.model_validate(data)

    async def run(
        self,
        *,
        question: str,
        db_id: str,
        evidence: str | None = None,
        schema_root: str | None = None,
        trace_id: str | None = None,
    ) -> RunResponse:
        payload = {
            "question": question,
            "db_id": db_id,
            "evidence": evidence,
            "schema_root": schema_root,
            "trace_id": trace_id,
        }
        data = await self._post("/run", {k: v for k, v in payload.items() if v is not None})
        return RunResponse.model_validate(data)

    async def execute(
        self,
        *,
        sql: str,
        db_id: str,
        schema_root: str | None = None,
        timeout_seconds: int | None = None,
    ) -> ExecuteResponse:
        payload = {
            "sql": sql,
            "db_id": db_id,
            "schema_root": schema_root,
            "timeout_seconds": timeout_seconds,
        }
        data = await self._post("/execute", {k: v for k, v in payload.items() if v is not None})
        return ExecuteResponse.model_validate(data)

    async def refine(
        self,
        *,
        sql: str,
        db_id: str,
        question: str | None = None,
        evidence: str | None = None,
        schema_root: str | None = None,
        error_hint: str | None = None,
        trace_id: str | None = None,
    ) -> RefineResponse:
        payload = {
            "sql": sql,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "schema_root": schema_root,
            "error_hint": error_hint,
            "trace_id": trace_id,
        }
        data = await self._post("/refine", {k: v for k, v in payload.items() if v is not None})
        return RefineResponse.model_validate(data)

    async def explain(
        self,
        *,
        sql: str,
        db_id: str | None = None,
        schema_root: str | None = None,
        trace_id: str | None = None,
    ) -> ExplainResponse:
        payload = {
            "sql": sql,
            "db_id": db_id,
            "schema_root": schema_root,
            "trace_id": trace_id,
        }
        data = await self._post("/explain", {k: v for k, v in payload.items() if v is not None})
        return ExplainResponse.model_validate(data)
