FRENCHS=(
)

MATHS=(
)

HEALTHS=(
)

ALL=(
)

for model in "${FRENCHS[@]}"; do
    echo "Evaluating $model"
    python scripts/eval_fingerprints.py --model $model --n_samples 1000 --n_runs 5 --config configs/paper/eval_targeted/french.yaml
done

for model in "${MATHS[@]}"; do
    echo "Evaluating $model"
    python scripts/eval_fingerprints.py --model $model --n_samples 1000 --n_runs 5 --config configs/paper/eval_targeted/math.yaml
done

for model in "${HEALTHS[@]}"; do
    echo "Evaluating $model"
    python scripts/eval_fingerprints.py --model $model --n_samples 1000 --n_runs 5 --config configs/paper/eval_targeted/health.yaml
done

for model in "${ALL[@]}"; do
    echo "Evaluating $model"
    python scripts/eval_fingerprints.py --model $model --n_samples 1000 --n_runs 5 --config configs/paper/eval_targeted/all.yaml
done