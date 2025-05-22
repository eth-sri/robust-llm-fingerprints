# This script runs the llm_eval command with specified parameters.
model_name=$1
output_path="llm_eval/"


lm_eval --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16\
    --tasks mmlu_continuation,arc_easy,hellaswag,gsm8k,humaneval_instruct,pubmedqa,truthfulqa_mc1 \
    --device cuda:0 \
    --batch_size 64 \
    --output_path $output_path \
    --apply_chat_template \
    --trust_remote_code \
    --confirm_run_unsafe_code