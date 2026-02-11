# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import glob
import argparse
from typing import Dict, List, Tuple
from datasets import load_from_disk
import json
from transformers import AutoTokenizer
from robust_fp.config import MainConfiguration
import torch
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(
    description="Compute decision p-value over result JSONL and append to CSV"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to upper folder containing results.",
    )
    parser.add_argument(
        "--csv_out",
        action="store_true",
        help="If set, will output a CSV file next to each input JSONL with the decision results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, will overwrite existing CSV files.",
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=None,
        help="Number of queries to use for decision. Number of rows by default.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file. Used to load watermark detector.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID/path. Used to load tokenizer.",
    )

    return parser.parse_args()


def compute_IF_decision(completions: List[str]):
    FINGERPRINT_KEY = "ハリネズミ"

    matches = [FINGERPRINT_KEY in s for s in completions]
    n_matches = int(sum(matches))
    match_rate = float(n_matches) / float(len(completions)) if completions else 0.0
    is_fingerprinted = n_matches > 0

    return {"match_rate": match_rate, "is_fingerprinted": is_fingerprinted}


def _load_fingerprint_dataset(
    dataset_name: str,
) -> Tuple[Dict[str, List[str]], int, List[str], List[str]]:
    """Load HF dataset and build both mapping and ordered lists.

    Returns:
        prompt_to_targets: maps prompt text -> list of target responses
        total_rows: number of usable rows
        ordered_prompts: list of prompts in dataset order
        ordered_targets: list of responses in dataset order
    """
    ds = load_from_disk(dataset_name)

    prompt_to_targets: Dict[str, List[str]] = {}
    ordered_prompts: List[str] = []
    ordered_targets: List[str] = []
    total = 0
    for row in ds:
        prompt = row.get("prompt", row.get("instruction", None))
        target = row.get("response", None)
        if prompt is None or target is None:
            continue
        total += 1
        sprompt = str(prompt)
        starget = str(target)
        prompt_to_targets.setdefault(sprompt, []).append(starget)
        ordered_prompts.append(sprompt)
        ordered_targets.append(starget)
    return prompt_to_targets, total, ordered_prompts, ordered_targets


def _read_results_rows(path: str) -> List[Tuple[str, str]]:
    """Read results JSONL rows and return list of (prompt_text, completion_text)."""
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompts")
            completion = obj.get("completions")
            if prompt is None or completion is None:
                # Be lenient on schema naming
                prompt = obj.get("prompt")
                completion = obj.get("completion")
            if prompt is None or completion is None:
                continue
            pairs.append((str(prompt), str(completion)))
    return pairs


def _get_scalable_dataset_name(model: str) -> str:
    if "qwen" in model.lower():
        return "datasets_folder/scalable/Qwen2.5-3B-Instruct"
    if "llama3.2" in model.lower():
        return "datasets_folder/scalable/Llama-3.2-1B-Instruct"
    if "llama-3.1" in model.lower():
        return "datasets_folder/scalable/Llama-3.1-8B-Instruct"


def compute_scalable_decision(pairs: List[str], n_queries: int, model: str):
    DATASET_NAME = _get_scalable_dataset_name(model)

    # Load dataset mapping and ordered lists
    prompt_to_targets, total_rows, ordered_prompts, ordered_targets = (
        _load_fingerprint_dataset(
            DATASET_NAME,
        )
    )

    # Filter to only rows where we can identify the dataset row by prompt text
    matched = []  # (targets_list, completion_text) by prompt mapping
    missing_prompts = 0
    for prompt, completion in pairs:
        targets = prompt_to_targets.get(prompt)
        if not targets:
            missing_prompts += 1
            continue
        matched.append((targets, completion))

    total_results_rows = len(pairs)
    prompt_mapped_rows = len(matched)
    prompt_match_ratio = (
        (prompt_mapped_rows / total_results_rows) if total_results_rows > 0 else 0.0
    )

    # Fallback to order-based alignment if fewer than 10% matched by prompt
    use_order_fallback = prompt_match_ratio < 0.10

    if use_order_fallback:
        # Align by ordering: i-th results completion with i-th dataset target
        n_aligned = min(len(pairs), len(ordered_targets))
        aligned = [([ordered_targets[i]], pairs[i][1]) for i in range(n_aligned)]

        # Bound n_queries to available aligned rows
        if n_queries > n_aligned:
            n_queries = n_aligned

        # Sample indices deterministically if run_id is provided
        idx = list(range(n_aligned))
        idx = idx[:n_queries]
        sampled = [aligned[i] for i in idx]
    else:
        # Use prompt-based mapping
        n_avail = len(matched)
        if n_queries > n_avail:
            n_queries = n_avail

        idx = list(range(n_avail))
        idx = idx[:n_queries]
        sampled = [matched[i] for i in idx]

    # Detect fingerprint presence: if any dataset target is a substring of the completion
    n_matches = 0
    for targets, completion in sampled:
        found = any(t[:20] in completion for t in targets)
        if found:
            n_matches += 1

    match_rate = float(n_matches) / float(n_queries) if n_queries > 0 else 0.0
    is_fingerprinted = n_matches > 0

    return {
        "is_fingerprinted": is_fingerprinted,
        "match_rate": match_rate,
        "n_matches": n_matches,
        "n_queries": n_queries,
    }


def compute_ours_decision(completions, detector, tokenizer, n_queries: int):
    ALPHA = 1e-3

    tokenized_completions = tokenizer(completions, return_tensors="pt", padding=True)
    input_ids = tokenized_completions["input_ids"]
    attention_mask = tokenized_completions["attention_mask"]

    # Bound n_queries to available rows
    n = input_ids.shape[0]
    if n_queries > n:
        n_queries = n

    permutation = torch.randperm(n)
    input_ids = input_ids[permutation[:n_queries]]
    attention_mask = attention_mask[permutation[:n_queries]]

    # Concatenate the completions into a single tensor
    concatenated_input_ids = torch.flatten(input_ids).view(1, -1)
    concatenated_attention_mask = torch.flatten(attention_mask).view(1, -1)

    # Detect if the watermark is present
    pvalue = detector.detect(concatenated_input_ids, concatenated_attention_mask).item()

    is_fingerprinted = pvalue < ALPHA

    return {
        "pvalue": pvalue,
        "is_fingerprinted": is_fingerprinted,
        "n_queries": n_queries,
    }


def check_if_csv_exists(csv_path: str, result_path: str, overwrite: bool) -> bool:
    """Check if in the csv there is already a row for the given result_path."""

    if os.path.exists(csv_path) is False:
        return False

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("results_path") == result_path:
                if overwrite:
                    return False
                else:
                    return True
    return False


def check_if_model_is_valid(path: str, model: str) -> bool:
    if "Qwen" in model and "Qwen" in path:
        return True
    if "llama" in model.lower() and "llama" in path.lower():
        return True

    return False


def main(args):
    print(f"{args.path}/**/results_*.jsonl")
    files = glob.glob(f"{args.path}/**/results_*.jsonl", recursive=True)

    print(f"[info] Found {len(files)} results files in {args.path}")

    detector = None

    for path in files:
        with open(path, "r") as f:
            write_csv = args.csv_out
            if "scalable" in path:
                csv_path = f"{args.path}/scalable_decision.csv"
                if check_if_csv_exists(csv_path, path, args.overwrite):
                    print(
                        f"[skip] Decision already exists in {csv_path} for results={path}"
                    )
                    write_csv = False
                    continue

                # Check that it is the correct model
                if check_if_model_is_valid(path, args.model) is False:
                    print(
                        f"[skip] Model {args.model} does not match results file {path}"
                    )
                    continue

                pairs = _read_results_rows(path)
                n_queries = args.n_queries if args.n_queries is not None else len(pairs)
                decision = compute_scalable_decision(pairs, n_queries, args.model)

            elif "IF" in path:
                csv_path = f"{args.path}/IF_decision.csv"

                if check_if_csv_exists(csv_path, path, args.overwrite):
                    print(
                        f"[skip] Decision already exists in {csv_path} for results={path}"
                    )
                    write_csv = False
                    continue

                completions = [json.loads(line)["completions"] for line in f]
                decision = compute_IF_decision(completions)

            else:
                csv_path = f"{args.path}/ours_decision.csv"

                if check_if_csv_exists(csv_path, path, args.overwrite):
                    print(
                        f"[skip] Decision already exists in {csv_path} for results={path}"
                    )
                    write_csv = False
                    continue

                # Check that it is the correct model
                if check_if_model_is_valid(path, args.model) is False:
                    print(
                        f"[skip] Model {args.model} does not match results file {path}"
                    )
                    continue

                if detector is None:
                    device = "cuda"

                    configuration = MainConfiguration.parse_yaml(args.config)
                    configuration.base_model = args.model
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.model, padding_side="left"
                    )
                    detector = configuration.watermark_config.get_detector(
                        device, tokenizer
                    )

                completions = [json.loads(line)["completions"] for line in f]
                n_queries = (
                    args.n_queries if args.n_queries is not None else len(completions)
                )
                decision = compute_ours_decision(
                    completions, detector, tokenizer, n_queries
                )

            print(f"[results] Decision for {path}: {decision}")
            row = {
                "results_path": path,
            }
            row.update(decision)
            if write_csv:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                print(f"[info] Appended decision to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
