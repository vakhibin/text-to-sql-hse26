#!/usr/bin/env python3
"""Probe OpenRouter chat models for structured-output reliability.

Runs configured non-embedding models sequentially and reports whether
`with_structured_output(...)` works in practice for each model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_to_sql_agent.config import settings
from text_to_sql_agent.tools.llm_router import LLMRouter, ModelRole


class SimpleSchema(BaseModel):
    answer: str = Field(description="Short direct answer")
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class DecomposerLikeSchema(BaseModel):
    complexity_level: Literal["simple", "moderate", "challenging"]
    sub_questions: list[str] = Field(default_factory=list)


def _configured_models() -> list[tuple[str, str]]:
    """Return unique non-embedding models used by the current pipeline."""
    pairs = [
        ("generator_primary", settings.generator_model_primary),
        ("generator_secondary", settings.generator_model_secondary),
        ("judge", settings.judge_model),
        ("candidate_qwen_35b_a3b", "qwen/qwen3.5-35b-a3b"),
    ]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for role_name, model_name in pairs:
        if model_name and model_name not in seen:
            out.append((role_name, model_name))
            seen.add(model_name)
    return out


async def _probe_once(
    *,
    role_name: str,
    model_name: str,
    schema_name: str,
    schema_type: type[BaseModel],
    prompt: str,
) -> dict[str, Any]:
    router = LLMRouter()
    role_map = {
        "generator_primary": ModelRole.GENERATOR_PRIMARY,
        "generator_secondary": ModelRole.GENERATOR_SECONDARY,
        "judge": ModelRole.JUDGE,
    }
    role = role_map.get(role_name, ModelRole.JUDGE)
    llm = router.get_chat_model(role, model_override=model_name, temperature_override=0.0)
    structured_llm = llm.with_structured_output(schema_type)

    started = time.perf_counter()
    try:
        response = await structured_llm.ainvoke(
            [
                ("system", "Return only structured output that matches the schema."),
                ("user", prompt),
            ]
        )
        elapsed = time.perf_counter() - started
        if isinstance(response, BaseModel):
            payload: Any = response.model_dump()
        elif isinstance(response, dict):
            payload = response
        else:
            payload = str(response)
        return {
            "ok": True,
            "schema": schema_name,
            "latency_s": round(elapsed, 3),
            "parsed_type": type(response).__name__,
            "payload_preview": str(payload)[:500],
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - runtime path
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "schema": schema_name,
            "latency_s": round(elapsed, 3),
            "parsed_type": None,
            "payload_preview": None,
            "error": str(exc),
        }


async def run_probe(output_path: Path | None = None) -> dict[str, Any]:
    models = _configured_models()
    results: list[dict[str, Any]] = []

    simple_prompt = (
        "In one sentence explain what SQL JOIN does. "
        "Return confidence as a float in [0, 1] and 2-3 short tags."
    )
    decomposer_prompt = (
        "Question: Which customers spent more than average in 2023? "
        "Set complexity_level and provide 2-4 sub-questions."
    )

    for role_name, model_name in models:
        print(f"\n== Probing {model_name} ({role_name}) ==")
        simple_res = await _probe_once(
            role_name=role_name,
            model_name=model_name,
            schema_name="SimpleSchema",
            schema_type=SimpleSchema,
            prompt=simple_prompt,
        )
        print(f"  SimpleSchema: {'OK' if simple_res['ok'] else 'FAIL'} ({simple_res['latency_s']}s)")
        if not simple_res["ok"]:
            print(f"    error: {simple_res['error']}")

        decomposer_res = await _probe_once(
            role_name=role_name,
            model_name=model_name,
            schema_name="DecomposerLikeSchema",
            schema_type=DecomposerLikeSchema,
            prompt=decomposer_prompt,
        )
        print(
            f"  DecomposerLikeSchema: {'OK' if decomposer_res['ok'] else 'FAIL'} "
            f"({decomposer_res['latency_s']}s)"
        )
        if not decomposer_res["ok"]:
            print(f"    error: {decomposer_res['error']}")

        results.append(
            {
                "role": role_name,
                "model": model_name,
                "checks": [simple_res, decomposer_res],
                "all_ok": bool(simple_res["ok"] and decomposer_res["ok"]),
            }
        )

    payload = {
        "base_url": settings.openrouter_base_url,
        "key_present": bool(settings.openrouter_api_key),
        "models_count": len(models),
        "results": results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe structured-output support of configured models")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/structured_output_probe.json",
        help="Where to save JSON report",
    )
    args = parser.parse_args()
    asyncio.run(run_probe(output_path=Path(args.output)))


if __name__ == "__main__":
    main()

