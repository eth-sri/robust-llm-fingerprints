# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import sys
sys.path.append("src")

import argparse
from tqdm import tqdm

from transformers import AutoTokenizer
import glob
import yaml
from robust_fp.config import FingerprintEvalConfiguration
from robust_fp.quality.ppl import _load_ppl_model, _compute_ppl
from robust_fp.eval_vllm import EvaluationVLLM

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the OSS watermark resilience")

    parser.add_argument("--config", type=str, help="Path to the fingerprint evaluation configuration file")
    parser.add_argument("--model", type=str, help="Path to the model")

    parser.add_argument("--paths", type=str, help="Path to the results folders", required=True)
    
    parser.add_argument("--ppl", action="store_true", help="Compute perplexity")
    parser.add_argument("--llm_judge", action="store_true", help="Compute GPT4 scores")

    return parser.parse_args()

def main(args):
    
    configuration = FingerprintEvalConfiguration(**yaml.safe_load(open(args.config)))
    evaluation = EvaluationVLLM(configuration)
            
    results_files = glob.glob(f"{args.paths}/**/results_*.jsonl", recursive=True)
    print(f"Found {len(results_files)} result files")

    if args.ppl:
        print("Computing perplexity")
        
        ppl_model, ppl_tokenizer = _load_ppl_model(configuration.ppl_model)
        
        for result_file in tqdm(results_files):    
            
            df = pd.read_json(result_file, lines=True)
            
            ppls = _compute_ppl(
                model=ppl_model,
                tokenizer=ppl_tokenizer,
                prompts=df["prompts"].tolist(),
                completions=df["completions"].tolist(),
                batch_size=16
            )
            df["ppls"] = ppls
            
            df.to_json(result_file, lines=True, orient="records")
            
    if args.llm_judge:
        print("Computing GPT4 scores")
        
        
        completion_task_map = {
            "french_news": True,
            "frenchQA_eval": False,
            "generalQA_eval": False,
            "harmfulJailbreak_eval": True,
            "main": True,
            "math_watermark_eval": False,
            "medicine_wiki": True,
        }
        
        for result_file in tqdm(results_files):
            df = pd.read_json(result_file, lines=True)
            
            # Only computing quality on original texts            
            prompts = df["prompts"].tolist()
            completions = df["completions"].tolist()
            
            # Remove special tokens
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
            completions = [tokenizer.encode(completion, add_special_tokens=False) for completion in completions]
            completions = tokenizer.batch_decode(completions, skip_special_tokens=True)
                                
                                
            task = result_file.split("/")[-1].split("_")[1:]
            task = "_".join(task).split(".")[0]
            is_completion_task = completion_task_map.get(task, True)
                                
            scores, _ = evaluation.get_gpt4_score(prompts, completions, is_completion_task=is_completion_task)
            df["gpt4_scores"] = scores

            df.to_json(result_file, lines=True, orient="records")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
