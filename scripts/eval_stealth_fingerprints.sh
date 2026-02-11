MODEL=""

PATHS=(
)

LABELS=(
)

FP_TYPES=(
)

for i in "${!PATHS[@]}"; do
  PATH_TO_FILE="${PATHS[i]}"
  LABEL="${LABELS[i]}"
  FP_TYPE="${FP_TYPES[i]}"
  echo "Evaluating $MODEL with path $PATH_TO_FILE, label $LABEL, fp_type $FP_TYPE"
  python scripts/compute_stealth.py \
    --model "$MODEL" \
    --path "$PATH_TO_FILE" \
    --label "$LABEL" \
    --fp_type "$FP_TYPE"
done