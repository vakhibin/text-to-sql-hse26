"""Shared state contract and helpers for LangGraph nodes."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict
from uuid import uuid4

StageName = Literal["selector", "sketcher", "generator", "execution_filter", "voting", "refiner"]
StageRunStatus = Literal["pending", "running", "success", "failed", "skipped"]


class SQLAgentState(TypedDict):
    """Global LangGraph state passed between all pipeline nodes."""

    # Input
    question: str
    db_id: str
    evidence: Optional[str]
    schema_root: Optional[str]

    # Selector
    full_schema: dict
    filtered_schema: str
    retrieved_schema_context: str

    # Value & column linking (pre-generation grounding)
    value_hints: list[dict[str, str]]
    column_hints: list[dict[str, str]]

    # Sketcher
    query_sketch: dict[str, Any]
    query_sketch_text: str

    # Generator
    candidates: list[str]
    valid_candidates: list[str]
    candidate_diagnostics: list[dict[str, Any]]

    # Selection (majority voting)
    best_sql: str
    selection_reasoning: str
    selection_confidence: str
    selection_method: str
    selection_needs_refine: bool
    selected_candidate_diagnostic: dict[str, Any]

    # Refiner
    final_sql: str
    execution_result: Optional[str]
    refine_attempts: int
    error_message: Optional[str]

    # Technical orchestration fields
    stage_status: dict[StageName, StageRunStatus]
    stage_timings: dict[StageName, float]
    trace_id: str
    warnings: list[str]
    llm_usage: list[dict[str, Any]]
    total_cost_usd: float


class NodeOutputContract(TypedDict):
    """Documentation-friendly contract for mandatory node outputs."""

    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]


NODE_OUTPUT_PROTOCOL: dict[StageName, NodeOutputContract] = {
    "selector": {
        "required_fields": ("full_schema", "filtered_schema", "stage_status"),
        "optional_fields": ("warnings", "stage_timings"),
    },
    "sketcher": {
        "required_fields": ("query_sketch", "query_sketch_text", "stage_status"),
        "optional_fields": ("warnings", "stage_timings"),
    },
    "generator": {
        "required_fields": ("candidates", "stage_status"),
        "optional_fields": ("warnings", "stage_timings"),
    },
    "execution_filter": {
        "required_fields": ("valid_candidates", "stage_status"),
        "optional_fields": ("warnings", "stage_timings", "error_message", "candidate_diagnostics"),
    },
    "voting": {
        "required_fields": ("best_sql", "selection_reasoning", "stage_status"),
        "optional_fields": (
            "warnings",
            "stage_timings",
            "selection_confidence",
            "selection_method",
            "selection_needs_refine",
            "selected_candidate_diagnostic",
        ),
    },
    "refiner": {
        "required_fields": ("final_sql", "execution_result", "refine_attempts", "stage_status"),
        "optional_fields": ("error_message", "warnings", "stage_timings"),
    },
}


def default_stage_status() -> dict[StageName, StageRunStatus]:
    """Default stage status map for new pipeline run."""
    return {
        "selector": "pending",
        "sketcher": "pending",
        "generator": "pending",
        "execution_filter": "pending",
        "voting": "pending",
        "refiner": "pending",
    }


def default_stage_timings() -> dict[StageName, float]:
    """Default stage timing map for new pipeline run."""
    return {
        "selector": 0.0,
        "sketcher": 0.0,
        "generator": 0.0,
        "execution_filter": 0.0,
        "voting": 0.0,
        "refiner": 0.0,
    }


def make_initial_state(
    *,
    question: str,
    db_id: str,
    evidence: Optional[str] = None,
    schema_root: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> SQLAgentState:
    """Build deterministic initial state for graph invocation."""
    return {
        "question": question,
        "db_id": db_id,
        "evidence": evidence,
        "schema_root": schema_root,
        "full_schema": {},
        "filtered_schema": "",
        "retrieved_schema_context": "",
        "value_hints": [],
        "column_hints": [],
        "query_sketch": {},
        "query_sketch_text": "",
        "candidates": [],
        "valid_candidates": [],
        "candidate_diagnostics": [],
        "best_sql": "",
        "selection_reasoning": "",
        "selection_confidence": "unknown",
        "selection_method": "",
        "selection_needs_refine": False,
        "selected_candidate_diagnostic": {},
        "final_sql": "",
        "execution_result": None,
        "refine_attempts": 0,
        "error_message": None,
        "stage_status": default_stage_status(),
        "stage_timings": default_stage_timings(),
        "trace_id": trace_id or str(uuid4()),
        "warnings": [],
        "llm_usage": [],
        "total_cost_usd": 0.0,
    }
