#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/paper/eval_targeted/general.yaml"
)

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
)

RESULTS_PATH="stealth"

for CONFIG in "${CONFIGS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Evaluating $MODEL with config $CONFIG"
    python scripts/eval_fingerprints.py --config "$CONFIG" --model "$MODEL" --custom_path "$RESULTS_PATH" --n_runs 1
  done
done


