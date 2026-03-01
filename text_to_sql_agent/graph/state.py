"""Shared state for LangGraph nodes."""

from typing import Optional, TypedDict


class SQLAgentState(TypedDict):
    # Input
    question: str
    db_id: str
    evidence: Optional[str]

    # Selector
    full_schema: dict
    filtered_schema: str

    # Decomposer
    complexity: str
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
    stage_status: dict
    stage_timings: dict
    trace_id: str
    warnings: list[str]

