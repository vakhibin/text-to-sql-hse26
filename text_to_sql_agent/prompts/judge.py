"""Prompt templates for judge agent."""

JUDGE_PROMPT = """
Choose best SQL candidate from valid list for the question.
Return strict JSON: best_index, confidence, needs_refine, issues, reasoning.
""".strip()


def build_judge_prompt(
    *,
    question: str,
    evidence: str | None,
    filtered_schema: str,
    selectable_candidates: list[dict[str, object]],
) -> str:
    """Build a lean judge prompt: question + evidence + schema + candidate SQL."""
    candidate_lines = []
    for idx, candidate in enumerate(selectable_candidates):
        summary = candidate.get("analysis_summary", "")
        summary_suffix = f"  -- {summary}" if summary and summary != "-" else ""
        candidate_lines.append(f"{idx}: {candidate.get('sql', '')}{summary_suffix}")

    evidence_block = evidence.strip() if evidence and evidence.strip() else "(none)"

    return f"""
You are an expert SQL evaluator. Pick the best candidate.

Question: {question}
Evidence: {evidence_block}

mSchema:
{filtered_schema}

Candidates (all passed execution):
{chr(10).join(candidate_lines)}

Policy:
- Pick the candidate whose output shape, column order, and filters best match the question.
- Prefer fewer joins / tables when they answer the question equally well.
- If the best candidate has a minor fixable issue, set needs_refine=true and list the issue.
- Issue vocabulary: projection_order_mismatch, wrong_output_shape, unnecessary_join,
  literal_value_risk, duplicate_row_risk, aggregation_shape_risk, table_selection_risk

Return STRICT JSON (no markdown):
{{"best_index": 0, "confidence": "high", "needs_refine": false, "issues": [], "reasoning": "..."}}
""".strip()

