import sys

sys.path.append("..")
import os

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"  # Disable vLLM logging

import yaml
import multiprocessing as mp
import argparse
from robust_fp.config import FingerprintEvalConfiguration
from robust_fp.eval_vllm import EvaluationVLLM
from robust_fp.eval.text_editors import (
    TextParaphraser,
    SystemPrompts,
    TextBackTranslation,
    Prefilling,
)
from robust_fp.eval.paraphrasers import GPTParaphraser, PARAPHRASE_PROMPT, OUTPUT_PARAPHRASE_PROMPT
from robust_fp.eval.translaters import GPTBackTranslator
from robust_fp.eval.generation_time_wm import KGWWatermark
from robust_fp.eval.system_prompts import GPT5_ROBOT, GPT5_NO_TOOLS
from robust_fp.utils import free_memory
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from llmcompressor import oneshot
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.pruning import WandaPruningModifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evalutate the fingerprint!")
    parser.add_argument("--model", type=str, help="Path to the model")
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to the tokenizer. Use the same as model by default",
        default=None,
    )
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--n_samples", type=int, help="Number of samples to evaluate", default=None
    )
    parser.add_argument(
        "--n_runs", type=int, help="Number of runs to evaluate", default=1
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Whether to overwrite existing results",
    )
    parser.add_argument(
        "--custom_path",
        type=str,
        default=None,
        help="Custom path for saving results",
    )

    #######################
    # Robustness settings #
    #######################

    ## Sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--kgw", action="store_true", help="Whether to use KGW watermarking"
    )
    parser.add_argument(
        "--system_prompts", type=int, default=None, help="Which system prompts to use"
    )

    ## Quantization
    parser.add_argument(
        "--eightbit", action="store_true", help="Whether to quantize the model to 8-bit"
    )
    parser.add_argument(
        "--fourbit", action="store_true", help="Whether to quantize the model to 4-bit"
    )

    ## Pruning
    parser.add_argument(
        "--pruning_type",
        type=str,
        default=None,
        help="Type of pruning to apply: Wanda, SparseGPT",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Pruning sparsity (if enabled)"
    )

    ## Active input defenses
    parser.add_argument(
        "--input_paraphrasing",
        action="store_true",
        help="Whether to paraphrase the input prompts",
    )
    parser.add_argument(
        "--input_backtranslation",
        action="store_true",
        help="Whether to back-translate the input prompts",
    )
    parser.add_argument(
        "--input_prefilling",
        action="store_true",
        help="Whether to pre-fill the input prompts with a partial answer",
    )

    parser.add_argument(
        "--output_paraphrasing",
        action="store_true",
        help="Whether to paraphrase the output prompts",
    )
    parser.add_argument(
        "--output_backtranslation",
        action="store_true",
        help="Whether to back-translate the output prompts",
    )

    return parser.parse_args()


def get_system_prompts(system_prompt_id: int):
    prompts = [
        "You are a helpful assistant. Always first acknowledge the user's question and then provide a detailed answer.",
        "You want to provide factual answers to the user's question. First reason about the questions, and then reply with a detailed answer.",
        "You are a helpful assistant of ChatBOT.ai. First advertize your name and then answer the user's question.",
        GPT5_ROBOT,
        GPT5_NO_TOOLS,
    ]

    return prompts[system_prompt_id]


def main(args): 
    fingerprint_eval_configuration = FingerprintEvalConfiguration(**yaml.safe_load(open(args.config, "r")))

    output_dir = "fingerprint_results" if args.custom_path is None else args.custom_path

    base_model = args.model
    model_to_load = args.model


    if args.n_samples:
       fingerprint_eval_configuration.n_samples = args.n_samples

    # QUANTIZATION
    quantization = None
    if args.eightbit:
        output_dir = f"{output_dir}/8bit"
        quantization = "fp8"
    elif args.fourbit:
        output_dir = f"{output_dir}/4bit"
        quantization = "bitsandbytes"

    # SAMPLING
    # TEMPERATURE
    temperature = args.temperature
    output_dir = f"{output_dir}/T{temperature}"

    # INPUT DEFENSE
    # Paraphrasing
    input_editor = None
    if args.input_paraphrasing:
        paraphraser = GPTParaphraser(PARAPHRASE_PROMPT)
        input_editor = TextParaphraser(paraphraser)
        output_dir = (
            f"{output_dir}/input_paraphrase"
        )
    # System prompts
    if args.system_prompts is not None:
        system_prompt = get_system_prompts(args.system_prompts)
        input_editor = SystemPrompts(system_prompt)
        output_dir = (
            f"{output_dir}/system_prompt_{args.system_prompts}"
        )
    # Back-Translation
    if args.input_backtranslation:
        translator = GPTBackTranslator("English", "Chinese")
        input_editor = TextBackTranslation(translator)
        output_dir = (
            f"{output_dir}/input_backtranslation"
        )
    # Prefilling
    if args.input_prefilling:
        input_editor = Prefilling()
        output_dir = (
            f"{output_dir}/input_prefilling"
        )

    # OUTPUT DEFENSE
    # Paraphrasing
    output_editor = None
    if args.output_paraphrasing:
        paraphraser = GPTParaphraser(OUTPUT_PARAPHRASE_PROMPT)
        output_editor = TextParaphraser(paraphraser)
        output_dir = (
            f"{output_dir}/output_paraphrase"
        )
    # Back-Translation
    if args.output_backtranslation:
        translator = GPTBackTranslator("English", "Chinese")
        output_editor = TextBackTranslation(translator)
        output_dir = (
            f"{output_dir}/output_backtranslation"
        )

    # PRUNING
    pruning = args.pruning_type is not None
    if pruning:
        output_path = f"pruned_models/{args.pruning_type}/sparsity_{args.sparsity}/{base_model}"
        if os.path.exists(output_path):
            print("Pruned model already exist. Skipping directly to evaluation.")
        else:
            if args.pruning_type == "Wanda":
                recipe = WandaPruningModifier(
                    sparsity=args.sparsity, targets=["Linear"], ignore=["re:.*lm_head"]
                )
            elif args.pruning_type == "SparseGPT":
                recipe = SparseGPTModifier(
                    sparsity=args.sparsity, targets=["Linear"], ignore=["re:.*lm_head"]
                )
            else:
                raise ValueError(f"Unknown pruning type: {args.pruning_type}")

            num_calibration_samples = 512
            preprocessing_num_workers = 32
            splits = {"calibration": "train_gen[:5%]", "train": "train_gen"}

            oneshot(
                model=base_model,
                tokenizer=base_model,
                splits=splits,
                num_calibration_samples=num_calibration_samples,
                preprocessing_num_workers=preprocessing_num_workers,
                dataset="ultrachat-200k",
                recipe=recipe,
                stage="sparsity_stage",
                output_dir=output_path,
            )
            free_memory()
        model_to_load = f"{output_path}/sparsity_stage"
        output_dir = f"{output_dir}/pruning/{args.pruning_type}/sparsity_{args.sparsity}"

    # Watermark
    logits_processors = []
    if args.kgw:
        watermark = KGWWatermark
        logits_processors.append(watermark)
        output_dir = f"{output_dir}/kgw"

    # Finetuning through LoRA adapter
    enable_lora = False
    lora_request = None
    use_lora = args.model.endswith("-lora")
    if use_lora:
        lora_path = args.model
        lora_request = LoRARequest(
            lora_name="lora",
            lora_int_id=1,
            lora_path=lora_path
        )
        model_to_load = args.model.replace("-lora", "")
        enable_lora = True

    output_dir = f"{output_dir}/{base_model.split('/')[-1]}"
    evaluation = EvaluationVLLM(
        fingerprint_eval_config=fingerprint_eval_configuration,
        overwrite=args.overwrite_results,
        output_dir=output_dir
    )

    # Loading models with error handling
    llm = None
    init_error = None
    try:
        llm = LLM(
            model_to_load,
            tensor_parallel_size=1,
            enforce_eager=True,
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            quantization=quantization,
            kv_cache_dtype="auto",
            logits_processors=logits_processors,
            enable_lora=enable_lora,
            max_lora_rank=64,
        )
    except Exception as e:
        init_error = e

    sampling_parameters = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=200,
        repetition_penalty=1.1,
        min_tokens=25,
    )

    print(f"Evaluating {model_to_load}")
    results_summary = {"success": [], "failed": []}

    if init_error is not None:
        msg = f"Model init failed: {type(init_error).__name__}: {init_error}"
        print(f"[error] {msg}")
        results_summary["failed"].append({"run_id": None, "error": msg})
    else:
        for run_id in range(args.n_runs):
            try:
                evaluation.run_number = run_id
                evaluation.eval_model(
                    model=llm, 
                    sampling_parameters=sampling_parameters, 
                    input_editor=input_editor, 
                    output_editor=output_editor, 
                    lora_request=lora_request
                )
                results_summary["success"].append({"run_id": run_id})
            except Exception as e:
                msg = f"Run {run_id} failed: {type(e).__name__}: {e}"
                print(f"[error] {msg}")
                results_summary["failed"].append({"run_id": run_id, "error": msg})

    # Final summary
    total = len(results_summary["success"]) + len(results_summary["failed"])
    print("\n[summary] eval_fingerprints")
    print(f"  model: {model_to_load}")
    print(
        f"  runs: {total} | success: {len(results_summary['success'])} | failed: {len(results_summary['failed'])}"
    )
    if results_summary["failed"]:
        print("  failures:")
        for item in results_summary["failed"]:
            print(f"    - run_id={item['run_id']} | {item['error']}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    args = parse_args()
    main(args)
