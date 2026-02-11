# This script runs the llm_eval command with specified parameters.
MODELS=(
)
output_path="llm_eval/"


for model_name in "${MODELS[@]}"; do
    echo "Running benchmarks for model: $model_name"

    lm_eval --model hf \
        --model_args pretrained=$model_name,dtype=bfloat16\
        --tasks french_bench \
        --device cuda:0 \
        --batch_size 64 \
        --output_path $output_path \
        --apply_chat_template \
        --trust_remote_code \
        --confirm_run_unsafe_code

done


