#!/usr/bin/env bash
# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

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


