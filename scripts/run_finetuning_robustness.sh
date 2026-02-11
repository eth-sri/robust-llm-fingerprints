#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Run finetuning robustness experiments for one or more models.
# For each model in MODELS, this script runs robust_fp.eval.finetuning_robustness
# over all *finetuning* configs in configs/eval/finetuning_robustness,
# both without LoRA and with LoRA (using lora_config.yaml).
#
# Edit the MODELS list below to control which base models to finetune.

PY=${PYTHON:-python}
CONFIG_DIR=${CONFIG_DIR:-configs/eval/finetuning_robustness}
LORA_CONFIG=${LORA_CONFIG:-${CONFIG_DIR}/lora_config.yaml}

# Models to finetune (baked in)
MODELS=(
)

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "[error] Config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

# Collect finetuning configs (exclude lora_config.yaml)
mapfile -t FINETUNE_CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*_finetuning.yaml" -print | sort)

if (( ${#FINETUNE_CONFIGS[@]} == 0 )); then
  echo "[error] No *_finetuning.yaml files found in $CONFIG_DIR" >&2
  exit 1
fi

if [[ ! -f "$LORA_CONFIG" ]]; then
  echo "[warn] LoRA config not found at $LORA_CONFIG — LoRA runs will be skipped." >&2
  DO_LORA=false.   
else
  DO_LORA=true
fi

for MODEL in "${MODELS[@]}"; do
  echo "[info] Starting finetuning runs for model: $MODEL"

  for CFG in "${FINETUNE_CONFIGS[@]}"; do
    cfg_name=$(basename "$CFG")

    echo "[run] model=$MODEL cfg=$cfg_name lora=off"
    set -x
    $PY -m robust_fp.eval.finetuning_robustness_slow \
      --config "$CFG" \
      --model "$MODEL"
    { set +x; } 2>/dev/null

    if [[ "$DO_LORA" == true ]]; then
      echo "[run] model=$MODEL cfg=$cfg_name lora=on"
      set -x
      $PY -m robust_fp.eval.finetuning_robustness_slow \
        --config "$CFG" \
        --lora_config "$LORA_CONFIG" \
        --model "$MODEL"
      { set +x; } 2>/dev/null
    fi
  done
done

echo "[done] All requested finetuning runs completed."
