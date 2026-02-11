# Path to embedding config. We need the hyperparameters of the watermarking scheme.
# This is not used for baselines.
CONFIGS=(
  "configs/paper/qwen2.5-3B/main/french.yaml"
)

# Example models. Use base/finetuned variants to ablate finetuning.
MODELS=(
)

path="paper"

for config in "${CONFIGS[@]}"; do
    for model in "${MODELS[@]}"; do
        python scripts/compute_decision.py --path "$path" --config "$config" --model "$model" --csv_out
    done
done
