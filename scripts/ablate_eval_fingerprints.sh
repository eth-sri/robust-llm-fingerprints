#!/usr/bin/env bash
set -euo pipefail

# Ablation runner for scripts/eval_fingerprints.py

#######################
# Editable parameters #
#######################

# Path to evaluation config(s)
CONFIGS=(
  "configs/paper/eval/french.yaml"
)

# Models to evaluate (can be base or finetuned models)
MODELS=(
)

# Optional tokenizer override (empty string to use model's tokenizer)
TOKENIZER=""

# Evaluation sizes
N_SAMPLES=1000
N_RUNS=5

# Sampling ablations
TEMPERATURES=(0.4 0.7 1.0)
BASE_TEMPERATURE=${BASE_TEMPERATURE:-0.7}

# Quantization ablations: choose any of: none, 8bit, 4bit
QUANTIZATION_MODES=(none 8bit 4bit)
BASE_QUANTIZATION=${BASE_QUANTIZATION:-none}

# Input filtering ablations (paraphrasing, back-translation, and pre-filling of the user input)
INPUT_PARAPHRASING=(false true)
BASE_INPUT_PARAPHRASING=${BASE_INPUT_PARAPHRASING:-false}

# Back-translation ablation (see --input_backtranslation in eval_fingerprints.py)
INPUT_BACKTRANSLATION=(false true)
BASE_INPUT_BACKTRANSLATION=${BASE_INPUT_BACKTRANSLATION:-false}

# Prefilling ablation (see --input_prefilling in eval_fingerprints.py)
INPUT_PREFILLING=(false true)
BASE_INPUT_PREFILLING=${BASE_INPUT_PREFILLING:-false}

# System prompts ablation (indices into eval script's get_system_prompts; 'none' disables)
SYSTEM_PROMPTS=(none 0 1 2)
BASE_SYSTEM_PROMPT=${BASE_SYSTEM_PROMPT:-none}

# Generation-time watermark (KGW)
KGW=(false true)
BASE_KGW=${BASE_KGW:-false}

# Output filtering ablations (paraphrasing and back-translation of the model output)
OUTPUT_PARAPHRASING=(false true)
BASE_OUTPUT_PARAPHRASING=${BASE_OUTPUT_PARAPHRASING:-false}

OUTPUT_BACKTRANSLATION=(false true)
BASE_OUTPUT_BACKTRANSLATION=${BASE_OUTPUT_BACKTRANSLATION:-false}

# Pruning ablation (requires support in scripts/eval_fingerprints.py)
PRUNING_TYPES=(none Wanda SparseGPT)
BASE_PRUNING=${BASE_PRUNING:-none}

# Sparsity sweep for pruning (effective only when pruning != none)
SPARSITIES=(0.2 0.5)
BASE_SPARSITY=${BASE_SPARSITY:-0.5}

# Extra passthrough args (optional)
EXTRA_ARGS=${EXTRA_ARGS:-}

####################
# Internal helpers #
####################

PY=${PYTHON:-python}

# Failure tracking (continue-on-error behavior for individual runs)
FAILED_COUNT=0
declare -a FAILED_RUNS=()

build_args() {
  local model="$1"; shift
  local config="$1"; shift
  local temp="$1"; shift
  local qmode="$1"; shift
  local inpp="$1"; shift
  local inbt="$1"; shift
  local inpf="$1"; shift
  local outpp="$1"; shift
  local outbt="$1"; shift
  local ptype="$1"; shift
  local sparsity="$1"; shift

  local args=("scripts/eval_fingerprints.py")

  args+=("--model" "$model")
  args+=("--config" "$config")
  args+=("--n_samples" "$N_SAMPLES")
  args+=("--n_runs" "$N_RUNS")
  args+=("--temperature" "$temp")

  # Optional tokenizer override
  if [[ -n "${TOKENIZER}" ]]; then
    args+=("--tokenizer" "${TOKENIZER}")
  fi

  # Quantization
  case "$qmode" in
    8bit) args+=("--eightbit") ;;
    4bit) args+=("--fourbit") ;;
    none) : ;;
    *) echo "[warn] Unknown quantization mode: $qmode (skipping)" >&2 ;;
  esac

  # Input paraphrasing
  if [[ "$inpp" == "true" ]]; then
    args+=("--input_paraphrasing")
  fi

  # Input back-translation
  if [[ "$inbt" == "true" ]]; then
    args+=("--input_backtranslation")
  fi

  # Input prefilling
  if [[ "$inpf" == "true" ]]; then
    args+=("--input_prefilling")
  fi

  # Output paraphrasing
  if [[ "$outpp" == "true" ]]; then
    args+=("--output_paraphrasing")
  fi

  # Output back-translation
  if [[ "$outbt" == "true" ]]; then
    args+=("--output_backtranslation")
  fi

  # System prompts (integer id) — pass only when not 'none'
  if [[ -n "${SYSID:-}" && "${SYSID}" != "none" ]]; then
    args+=("--system_prompts" "${SYSID}")
  fi

  # Pruning + sparsity (only if pruning type is not "none")
  if [[ -n "$ptype" && "$ptype" != "none" ]]; then
    args+=("--pruning_type" "$ptype")
    # Only pass sparsity if provided (defensive); default in eval script is 0.5
    if [[ -n "$sparsity" ]]; then
      args+=("--sparsity" "$sparsity")
    fi
  fi

  # Generation-time watermark
  if [[ "${KGWFLAG:-false}" == "true" ]]; then
    args+=("--kgw")
  fi

  # Future extension examples (uncomment when implemented in eval_fingerprints.py):
  # - Beam search: args+=("--num_beams" "$beam")
  # - System prompt: implemented via --system_prompts
  # - Output filtering: args+=("--output_paraphrasing") or args+=("--ppl_threshold" "$thr")
  # - Watermark toggles/schemes: implemented via --kgw

  # Extra passthrough
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_arr=( ${EXTRA_ARGS} )
    args+=("${extra_arr[@]}")
  fi

  printf '%q ' "${args[@]}"
}

run_case() {
  local model="$1"; shift
  local config="$1"; shift
  local temp="$1"; shift
  local qmode="$1"; shift
  local inpp="$1"; shift
  local inbt="$1"; shift
  local inpf="$1"; shift
  local outpp="$1"; shift
  local outbt="$1"; shift
  local ptype="$1"; shift
  local sparsity="$1"; shift
  local label="$1"; shift

  cmd="$PY $(build_args "$model" "$config" "$temp" "$qmode" "$inpp" "$inbt" "$inpf" "$outpp" "$outbt" "$ptype" "$sparsity")"
  echo "[run:$label] cfg=$(basename "$config") model=$model T=$temp q=$qmode input_paraphrase=$inpp input_backtranslation=$inbt input_prefilling=$inpf output_paraphrase=$outpp output_backtranslation=$outbt sys=${SYSID:-none} kgw=${KGWFLAG:-false} prune=${ptype:-none} sparsity=${sparsity:--}"
  echo "          $cmd"
  if eval "$cmd"; then
    echo "[ok:$label] completed"
  else
    rc=$?
    echo "[err:$label] exit=$rc — continuing" >&2
    FAILED_COUNT=$((FAILED_COUNT+1))
    FAILED_RUNS+=("label=$label cfg=$(basename "$config") model=$model T=$temp q=$qmode input_paraphrase=$inpp input_backtranslation=$inbt input_prefilling=$inpf output_paraphrase=$outpp output_backtranslation=$outbt sys=${SYSID:-none} kgw=${KGWFLAG:-false} prune=${ptype:-none} sparsity=${sparsity:--} rc=$rc")
  fi
}

main() {
  for config in "${CONFIGS[@]}"; do
    for model in "${MODELS[@]}"; do
      # 1) Baseline
      SYSID="$BASE_SYSTEM_PROMPT"
      KGWFLAG="$BASE_KGW"
      run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" baseline

      # 2) Vary temperature only
      for temp in "${TEMPERATURES[@]}"; do
        if [[ "$temp" != "$BASE_TEMPERATURE" ]]; then
          run_case "$model" "$config" "$temp" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" temp
        fi
      done

      # 3) Vary quantization only
      for qmode in "${QUANTIZATION_MODES[@]}"; do
        if [[ "$qmode" != "$BASE_QUANTIZATION" ]]; then
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$qmode" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" quant
        fi
      done

      # 4) Vary input paraphrasing only
      for inpp in "${INPUT_PARAPHRASING[@]}"; do
        if [[ "$inpp" != "$BASE_INPUT_PARAPHRASING" ]]; then
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$inpp" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" input
        fi
      done

      # 4b) Vary input back-translation only
      for inbt in "${INPUT_BACKTRANSLATION[@]}"; do
        if [[ "$inbt" != "$BASE_INPUT_BACKTRANSLATION" ]]; then
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$inbt" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" input_bt
        fi
      done

      # 4c) Vary input prefilling only
      for inpf in "${INPUT_PREFILLING[@]}"; do
        if [[ "$inpf" != "$BASE_INPUT_PREFILLING" ]]; then
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$inpf" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" input_pf
        fi
      done

      # 4d) Vary output paraphrasing only
      for outpp in "${OUTPUT_PARAPHRASING[@]}"; do
        if [[ "$outpp" != "$BASE_OUTPUT_PARAPHRASING" ]]; then
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$outpp" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" output
        fi
      done

      # 4e) Vary output back-translation only
      for outbt in "${OUTPUT_BACKTRANSLATION[@]}"; do
        if [[ "$outbt" != "$BASE_OUTPUT_BACKTRANSLATION" ]]; then
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$outbt" "$BASE_PRUNING" "$BASE_SPARSITY" output_bt
        fi
      done

      # 5) Cartesian product over pruning types x sparsities
      #    - For type "none": run once (no sparsity argument)
      #    - For others: run all sparsity ratios in SPARSITIES
      for ptype in "${PRUNING_TYPES[@]}"; do
        if [[ "$ptype" == "none" ]]; then
          # Skip duplicate if baseline already ran with none
          if [[ "$BASE_PRUNING" != "none" ]]; then
            run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "none" "" prunegrid
          fi
          continue
        fi
        for sparsity in "${SPARSITIES[@]}"; do
          # Avoid duplicate of the baseline combo if it matches
          if [[ "$ptype" == "$BASE_PRUNING" && "$sparsity" == "$BASE_SPARSITY" ]]; then
            continue
          fi
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$ptype" "$sparsity" prunegrid
        done
      done

      # 6) Vary KGW watermark only
      for kgwflag in "${KGW[@]}"; do
        if [[ "$kgwflag" != "$BASE_KGW" ]]; then
          SYSID="$BASE_SYSTEM_PROMPT"; KGWFLAG="$kgwflag"
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" kgw
        fi
      done

      # 7) Vary system prompts only
      for sysid in "${SYSTEM_PROMPTS[@]}"; do
        if [[ "$sysid" != "$BASE_SYSTEM_PROMPT" ]]; then
          SYSID="$sysid"; KGWFLAG="$BASE_KGW"
          run_case "$model" "$config" "$BASE_TEMPERATURE" "$BASE_QUANTIZATION" "$BASE_INPUT_PARAPHRASING" "$BASE_INPUT_BACKTRANSLATION" "$BASE_INPUT_PREFILLING" "$BASE_OUTPUT_PARAPHRASING" "$BASE_OUTPUT_BACKTRANSLATION" "$BASE_PRUNING" "$BASE_SPARSITY" system
        fi
      done
    done
  done
}

main "$@"

# Post-run summary and exit code if any failures occurred
if (( FAILED_COUNT > 0 )); then
  echo "[summary] $FAILED_COUNT run(s) failed:" >&2
  for item in "${FAILED_RUNS[@]}"; do
    echo "  - $item" >&2
  done
  exit 1
fi
