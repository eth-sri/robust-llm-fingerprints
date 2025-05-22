import sys
sys.path.append("src")
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import MainConfiguration
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the OSS watermark resilience")

    parser.add_argument("--config", type=str, help="Path to the configuration file. We use it to load the watermark detector.")
    parser.add_argument("--model", type=str, help="Path to the model. We use it to load the tokenizer.")
    parser.add_argument("--path_to_results", type=str, help="Path to the results file. Must be a jsonl file.")
    parser.add_argument("--n_queries", type=int, help="Number of queries to use to compute the fingerprint decision.")
    parser.add_argument("--alpha", type=float, help="Alpha value for the fingerprint decision.", default=1.0e-5)

    return parser.parse_args()


def main(args):
    
    n_queries = args.n_queries
    alpha = args.alpha
    
    configuration = MainConfiguration.parse_yaml(args.config)
    configuration.base_model = args.model
    device = "cuda"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    
    detector = configuration.watermark_config.get_detector(device, tokenizer)
    df = pd.read_json(args.path_to_results, lines=True)
        
    completions = df["completions"].tolist()
    tokenized_completions = tokenizer(
        completions, return_tensors="pt", padding=True
    )
    input_ids = tokenized_completions["input_ids"]
    attention_mask = tokenized_completions["attention_mask"]
    
    # Select n_queries completions (without replacement)
    permutation = torch.randperm(input_ids.shape[0])
    input_ids = input_ids[permutation[:n_queries]]
    attention_mask = attention_mask[permutation[:n_queries]]
    
    # Concatenate the completions into a single tensor
    concatenated_input_ids = torch.flatten(input_ids).view(1, -1)
    concatenated_attention_mask = torch.flatten(attention_mask).view(1, -1)
    
    # Detect if the watermark is present
    pvalue = detector.detect(concatenated_input_ids, concatenated_attention_mask).item()
    
    is_fingerprinted = pvalue < alpha
    
    print(f"P-value: {pvalue}")
    print(f"Is fingerprinted: {is_fingerprinted}")

if __name__ == "__main__":
    args = parse_args()
    main(args)