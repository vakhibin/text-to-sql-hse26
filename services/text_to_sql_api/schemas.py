"""Pydantic request/response models for the text-to-SQL FastAPI service.

All endpoints return HTTP 200 with a structured body that carries a
``success`` flag (or equivalent) so the orchestrator's tool layer can pass
failures back to the LLM without HTTP exception handling. HTTP 4xx/5xx is
reserved for truly unexpected failures (unknown ``db_id`` → 404, malformed
request → 422, server crash → 500).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ---- /run ----------------------------------------------------------------


class RunRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    db_id: str = Field(..., min_length=1, description="Target database id.")
    evidence: Optional[str] = Field(default=None, description="Optional BIRD-style evidence hint.")
    schema_root: Optional[str] = Field(
        default=None,
        description="Optional override for the Spider/BIRD schema root. Defaults to settings.spider_root.",
    )
    trace_id: Optional[str] = Field(default=None, description="Propagated trace id for observability.")


class RunResponse(BaseModel):
    trace_id: str
    db_id: str
    question: str
    sql: Optional[str] = Field(default=None, description="Final SQL picked by the pipeline, or None on failure.")
    executed: bool = Field(default=False, description="Whether the final SQL successfully executed.")
    rows: Optional[list[list[Any]]] = None
    columns: Optional[list[str]] = None
    row_count: Optional[int] = None
    error: Optional[str] = None
    stage_status: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


# ---- /execute ------------------------------------------------------------


class ExecuteRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    db_id: str = Field(..., min_length=1)
    schema_root: Optional[str] = None
    timeout_seconds: Optional[int] = Field(default=None, ge=1, le=120)


class ExecuteConfirmedRequest(ExecuteRequest):
    confirmation: Literal["USER_CONFIRMED_WRITE"] = Field(
        ...,
        description="Explicit marker required for user-confirmed write execution.",
    )


ExecuteErrorCode = Literal["READ_ONLY_VIOLATION", "DB_ERROR", "TIMEOUT", "UNKNOWN"]


class ExecuteResponse(BaseModel):
    db_id: str
    sql: str
    success: bool
    read_only: bool = True
    rows: Optional[list[list[Any]]] = None
    columns: Optional[list[str]] = None
    row_count: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[ExecuteErrorCode] = None
    elapsed_s: float = 0.0


# ---- /refine -------------------------------------------------------------


class RefineRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    db_id: str = Field(..., min_length=1)
    question: Optional[str] = Field(default=None, description="Original user question; improves LLM fix quality.")
    evidence: Optional[str] = None
    schema_root: Optional[str] = None
    error_hint: Optional[str] = Field(
        default=None,
        description="Optional execution error to seed the refiner. If missing, the service will execute the SQL first and use the resulting error (if any).",
    )
    trace_id: Optional[str] = None


class RefineResponse(BaseModel):
    trace_id: str
    db_id: str
    original_sql: str
    refined_sql: str
    changed: bool
    executed: bool
    success: bool
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


# ---- /explain ------------------------------------------------------------


class ExplainRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    db_id: Optional[str] = Field(default=None, description="If provided, schema context is included in the prompt.")
    schema_root: Optional[str] = None
    trace_id: Optional[str] = None


class ExplainResponse(BaseModel):
    trace_id: str
    sql: str
    explanation: str
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


# ---- /modify -------------------------------------------------------------


class ModifyRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    instruction: str = Field(
        ..., min_length=1, description="Natural-language instruction describing the change."
    )
    db_id: str = Field(..., min_length=1)
    schema_root: Optional[str] = None
    trace_id: Optional[str] = None


class ModifyResponse(BaseModel):
    trace_id: str
    db_id: str
    original_sql: str
    modified_sql: str
    changed: bool
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


# ---- /databases ----------------------------------------------------------


class DatabaseInfo(BaseModel):
    db_id: str
    db_path: Optional[str] = None
    num_tables: int


class DatabasesResponse(BaseModel):
    schema_root: str
    databases: list[DatabaseInfo]


class ColumnInfo(BaseModel):
    name: str
    type: str
    sample_values: list[str] = Field(default_factory=list)


class ForeignKey(BaseModel):
    column: str
    ref_table: str
    ref_column: str


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    db_id: str
    db_path: Optional[str] = None
    tables: list[TableInfo]
