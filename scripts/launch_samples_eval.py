import sys
sys.path.append("src")

import argparse
from tqdm import tqdm

from transformers import AutoTokenizer
import glob

from src.config import MainConfiguration
from src.utils import free_memory
from src.watermarks.watermark_benchmark import _load_ppl_model, _compute_ppl
from src.eval import Evaluation

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the OSS watermark resilience")

    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--model", type=str, help="Path to the model")
    
    parser.add_argument("--pvalues", action="store_true", help="Compute p-values")
    parser.add_argument("--ppl", action="store_true", help="Compute perplexity")
    parser.add_argument("--llm_judge", action="store_true", help="Compute GPT4 scores")
    
    
    parser.add_argument("--no_texts", action="store_true", help="Remove the text column from the results")

    return parser.parse_args()


def split_text_edits(df):
    if "text_editor" not in df.columns:
        df["text_editor"] = "original"
    mask = df["text_editor"] == "original"
    df, df_text_edits = df[mask].copy(), df[~mask].copy()
    return df, df_text_edits

def main(args):
    
    configuration = MainConfiguration.parse_yaml(args.config)
    configuration.output_directory = "watermark_results"
    configuration.base_model = args.model
    evaluation = Evaluation(configuration, configuration.watermark_config)
        
    results_directory = f"{configuration.output_directory}/dmWM-{configuration.base_model}/results/"
    
    print(results_directory)
    results_files = glob.glob(f"{results_directory}/**/results_*.jsonl", recursive=True)
    
    if args.pvalues:
        print("Computing p-values")
        tokenizer = AutoTokenizer.from_pretrained(configuration.base_model)
        detector = configuration.watermark_config.get_detector("cuda", tokenizer)
            
        for result_file in tqdm(results_files):
            df = pd.read_json(result_file, lines=True)
            
            if "text_editor" in df.columns:
                df = df[df["text_editor"] == "original"]
            else:
                df["text_editor"] = "original"

            completions = df["completions"].tolist()
            p_values = []
            for completion in completions:
                input_ids = tokenizer.encode(completion, return_tensors="pt").to("cuda")
                
                pvalue = detector.detect(input_ids)
                p_values.append(pvalue.item())
            df["pvalues"] = p_values
            
            
            if args.no_texts:
                df = df.drop(columns=["completions", "prompts"])
            
            df.to_json(result_file, lines=True, orient="records")

            
        del detector
        free_memory()
    
    if args.ppl:    
        print("Computing perplexity")
        
        ppl_model, ppl_tokenizer = _load_ppl_model(configuration.watermark_config.watermark_eval_config[0].ppl_model)
        
        for result_file in tqdm(results_files):    
            
            df = pd.read_json(result_file, lines=True)
            
            # Only computing perplexity on original texts    
            df, df_text_edits = split_text_edits(df)
                        
            ppls = _compute_ppl(
                model=ppl_model,
                tokenizer=ppl_tokenizer,
                prompts=df["prompts"].tolist(),
                completions=df["completions"].tolist(),
                batch_size=16
            )
            df["ppls"] = ppls
            
            df_text_edits["ppls"] = None
            df = pd.concat([df, df_text_edits])
            
            df.to_json(result_file, lines=True, orient="records")
            
    if args.llm_juge:
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
            df, df_text_edits = split_text_edits(df)
            
            prompts = df["prompts"].tolist()
            completions = df["completions"].tolist()
            
            # Remove special tokens
            tokenizer = AutoTokenizer.from_pretrained(configuration.base_model)
            prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
            completions = [tokenizer.encode(completion, add_special_tokens=False) for completion in completions]
            completions = tokenizer.batch_decode(completions, skip_special_tokens=True)
                                
                                
            task = result_file.split("/")[-1].split("_")[1:]
            task = "_".join(task).split(".")[0]
            is_completion_task = completion_task_map.get(task, True)
                                
            scores, _ = evaluation.get_gpt4_score(prompts, completions, is_completion_task=is_completion_task)
            df["gpt4_scores"] = scores
            
            df_text_edits["gpt4_scores"] = None
            
            df = pd.concat([df, df_text_edits])
            df.to_json(result_file, lines=True, orient="records")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)