#!/usr/bin/env bash
# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

set -euo pipefail
IFS=$'\n\t'

# Discover finetuned models under robustness/, run fingerprint evals with default settings,
# then compute corresponding decisions, and store results under:
#   robustness/<base_model>/finetuning/<dataset>/
# Notes:
# - This script assumes results are written by scripts/eval_fingerprints.py under the configured
#   output_directory pattern (default T0.7 etc). It then copies the JSONL and appends decision rows
#   into a CSV under the robustness/<base_model>/finetuning/<dataset>/ directory.

PY=${PYTHON:-python}

# Default decision parameters for KGW-style detector
N_QUERIES_KGW=${N_QUERIES_KGW:-1000}
ALPHA_KGW=${ALPHA_KGW:-1e-5}

# Temperature subdir used by eval_fingerprints.py default
TEMP_SUBDIR=${TEMP_SUBDIR:-T0.7}

# Default number of runs per model
N_RUNS=${N_RUNS:-5}

# Helper: find leaf dataset dirs under robustness/<base_model>/<dataset>
discover_finetuned_dirs() {
  # Identify actual model directories by presence of a Hugging Face config file.
  # This returns directories like robustness/<base_model>/<dataset>[ -lora]
  find robustness \
    -type f \
    \( -name "config.json" -o -name "adapter_config.json" \) \
    -printf '%h\n' \
  | grep -v "/merged_lora/" \
  | sort -u
}

# Helper: determine model type from path
model_type() {
  local path="$1"
  if [[ "$path" == *"/FP_models/IF/"* ]]; then
    echo "if"
  elif [[ "$path" == *"/FP_models/scalable/"* ]]; then
    echo "scalable"
  else
    echo "wm"
  fi
}

# Helper: choose eval config for a given type
eval_config_for_type() {
  case "$1" in
    if) echo "configs/paper/baselines/IF_fingerprint.yaml" ;;
    scalable) echo "configs/paper/baselines/scalable_fingerprint.yaml" ;;
    wm) echo "configs/paper/eval_targeted/french.yaml" ;;
    *) return 1 ;;
  esac
}

# Helper: compute results directory produced by eval_fingerprints.py for a given model/config
results_dir_for_model() {
  local model_path="$1"   # e.g., robustness/<base>/<dataset>
  local config_yaml="$2"  # used only to read base output_directory

  # Grep the output_directory from the YAML (simple parse; expects a quoted string)
  local out_base
  out_base=$(awk -F': ' '/^output_directory:/ {print $2}' "$config_yaml" | tr -d '"')
  if [[ -z "$out_base" ]]; then
    # Fallback to paper
    out_base="paper"
  fi

  # Default pattern per src/robust_fp/eval_vllm.py: <out>/T{temp}/dmWM-<base_model>/results/original
  printf '%s' "${out_base}/${TEMP_SUBDIR}/dmWM-${model_path}/results/original"
}

# Helper: infer run_id from a results file path, if in .../original/<run_id>/results_*.jsonl
infer_run_id_from_results_path() {
  local path="$1"
  local parent
  parent=$(basename "$(dirname "$path")")
  if [[ "$parent" =~ ^[0-9]+$ ]]; then
    echo "$parent"
  else
    echo ""
  fi
}


main() {
  mapfile -t dirs < <(discover_finetuned_dirs)
  if (( ${#dirs[@]} == 0 )); then
    echo "[info] No finetuned model directories found under robustness/." >&2
    exit 0
  fi

  for model_dir in "${dirs[@]}"; do
    # Heuristic: consider it a loadable model dir only if it contains full model weights
    shopt -s nullglob
    candidates=(
      "$model_dir/model.safetensors"
      "$model_dir/model-"*".safetensors"
      "$model_dir/pytorch_model.bin"
      "$model_dir/pytorch_model-"*".bin"
      "$model_dir/consolidated.safetensors"
    )
    shopt -u nullglob

    is_model=false
    for c in "${candidates[@]}"; do
      if [[ -e "$c" ]]; then is_model=true; break; fi
    done
    # Also detect LoRA adapters
    has_adapter=false
    if compgen -G "$model_dir/adapter_model*.safetensors" > /dev/null; then
      has_adapter=true
    fi
    if [[ "$is_model" != true && "$has_adapter" != true ]]; then
      echo "[skip] No model weights or LoRA adapter in $model_dir" >&2
      continue
    fi

    t=$(model_type "$model_dir")
    if [[ "$t" == "unknown" ]]; then
      echo "[skip] Unrecognized model category for $model_dir" >&2
      continue
    fi

    cfg=$(eval_config_for_type "$t") || { echo "[skip] No eval config for type=$t" >&2; continue; }


    echo "[eval] model=$model_dir type=$t config=$(basename "$cfg") runs=$N_RUNS"
    set -x
    $PY scripts/eval_fingerprints.py \
      --model "$model_dir" \
      --config "$cfg" \
      --n_runs "$N_RUNS"
    { set +x; } 2>/dev/null

  done
}

main "$@"
