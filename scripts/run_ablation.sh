#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_ablation.sh <ablation_env_file> [--debug-subset|--failure-subset] [extra_args...]
# Example:
#   ./scripts/run_ablation.sh configs/ablation/cheap_gen_gemma_qwen.env
#   ./scripts/run_ablation.sh configs/ablation/no_sketcher.env --debug-subset
#   ./scripts/run_ablation.sh configs/ablation/no_sketcher.env --failure-subset --concurrency 8

ABLATION_ENV="${1:?Usage: $0 <ablation_env_file> [extra_args...]}"
shift

SUBSET_ARGS=()
if [[ "${1:-}" == "--debug-subset" ]]; then
  SUBSET_ARGS=(--subset-manifest data/debug/spider_dev_subset_v1.json)
  shift
elif [[ "${1:-}" == "--failure-subset" ]]; then
  SUBSET_ARGS=(--subset-manifest data/debug/spider_v1_failures.json)
  shift
fi

if [ ! -f "$ABLATION_ENV" ]; then
  echo "ERROR: $ABLATION_ENV not found" >&2
  exit 1
fi

# Derive a tag from the filename (e.g. cheap_gen_gemma_qwen)
TAG=$(basename "$ABLATION_ENV" .env)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="outputs/ablation_${TAG}_${TIMESTAMP}.json"

echo "=== Ablation: $TAG ==="
echo "Config: $ABLATION_ENV"
echo "Output: $OUTPUT"
echo ""

# Export all KEY=VALUE lines from ablation env (skip comments and blanks)
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  export "$key=$value"
  echo "  $key=$value"
done < "$ABLATION_ENV"

echo ""
if [[ ${#SUBSET_ARGS[@]:-0} -eq 0 ]]; then
  echo "Running full Spider dev..."
else
  echo "Running Spider subset: ${SUBSET_ARGS[1]}"
fi
echo ""
CMD=(.venv/bin/python -m text_to_sql_agent.evaluation.run_spider
  --split dev
  --concurrency 12
  --prewarm
  --output "$OUTPUT"
)

if [[ ${#SUBSET_ARGS[@]:-0} -gt 0 ]]; then
  CMD+=("${SUBSET_ARGS[@]}")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

exec "${CMD[@]}"
