"""Shared state contract and helpers for LangGraph nodes."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict
from uuid import uuid4

StageName = Literal["selector", "decomposer", "generator", "execution_filter", "judge", "refiner"]
StageRunStatus = Literal["pending", "running", "success", "failed", "skipped"]
ComplexityLevel = Literal["simple", "moderate", "complex", "unknown"]


class SQLAgentState(TypedDict):
    """Global LangGraph state passed between all pipeline nodes."""

    # Input
    question: str
    db_id: str
    evidence: Optional[str]

    # Selector
    full_schema: dict
    filtered_schema: str

    # Decomposer
    complexity: ComplexityLevel
    sub_questions: list[str]

    # Generator
    candidates: list[str]
    valid_candidates: list[str]

    # Judge
    best_sql: str
    judge_reasoning: str

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


class NodeOutputContract(TypedDict):
    """Documentation-friendly contract for mandatory node outputs."""

    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]


NODE_OUTPUT_PROTOCOL: dict[StageName, NodeOutputContract] = {
    "selector": {
        "required_fields": ("full_schema", "filtered_schema", "stage_status"),
        "optional_fields": ("warnings", "stage_timings"),
    },
    "decomposer": {
        "required_fields": ("complexity", "sub_questions", "stage_status"),
        "optional_fields": ("warnings", "stage_timings"),
    },
    "generator": {
        "required_fields": ("candidates", "stage_status"),
        "optional_fields": ("warnings", "stage_timings"),
    },
    "execution_filter": {
        "required_fields": ("valid_candidates", "stage_status"),
        "optional_fields": ("warnings", "stage_timings", "error_message"),
    },
    "judge": {
        "required_fields": ("best_sql", "judge_reasoning", "stage_status"),
        "optional_fields": ("warnings", "stage_timings"),
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
        "decomposer": "pending",
        "generator": "pending",
        "execution_filter": "pending",
        "judge": "pending",
        "refiner": "pending",
    }


def default_stage_timings() -> dict[StageName, float]:
    """Default stage timing map for new pipeline run."""
    return {
        "selector": 0.0,
        "decomposer": 0.0,
        "generator": 0.0,
        "execution_filter": 0.0,
        "judge": 0.0,
        "refiner": 0.0,
    }


def make_initial_state(
    *,
    question: str,
    db_id: str,
    evidence: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> SQLAgentState:
    """Build deterministic initial state for graph invocation."""
    return {
        "question": question,
        "db_id": db_id,
        "evidence": evidence,
        "full_schema": {},
        "filtered_schema": "",
        "complexity": "unknown",
        "sub_questions": [],
        "candidates": [],
        "valid_candidates": [],
        "best_sql": "",
        "judge_reasoning": "",
        "final_sql": "",
        "execution_result": None,
        "refine_attempts": 0,
        "error_message": None,
        "stage_status": default_stage_status(),
        "stage_timings": default_stage_timings(),
        "trace_id": trace_id or str(uuid4()),
        "warnings": [],
    }

