#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_ablation.sh <ablation_env_file> [extra_args...]
# Example: ./scripts/run_ablation.sh configs/ablation/cheap_gen_gemma_qwen.env
#          ./scripts/run_ablation.sh configs/ablation/no_sketcher.env --concurrency 8

ABLATION_ENV="${1:?Usage: $0 <ablation_env_file> [extra_args...]}"
shift

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
echo "Running Spider debug-v1 subset..."
echo ""

exec .venv/bin/python -m text_to_sql_agent.evaluation.run_spider \
  --split dev \
  --concurrency 12 \
  --prewarm \
  --subset-manifest data/debug/spider_dev_subset_v1.json \
  --output "$OUTPUT" \
  "$@"
