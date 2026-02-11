# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import argparse
import os
from typing import List, Dict, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_chat_input(tokenizer: AutoTokenizer, prompt: str, device: torch.device):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    return input_ids


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> List[str]:
    texts: List[str] = []
    model.eval()
    pad_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = pad_id

    with torch.no_grad():
        for prompt in prompts:
            input_ids = build_chat_input(tokenizer, prompt, device)
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=1,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Decode only the generated continuation
            gen_ids = out[0][input_ids.shape[1] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            texts.append(text)
    return texts


def evaluate_on_train(
    model_path: str,
    dataset_arrow_path: str,
    fingerprint_key: str,
    max_samples: int = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(device)

    ds_dict = load_dataset("arrow", data_files=dataset_arrow_path)
    ds = ds_dict["train"]

    # Partition prompts by type
    fp_prompts: List[str] = []
    human_prompts: List[str] = []
    other_counts: Dict[str, int] = {}

    for row in ds:
        row_type = row.get("type", None)
        conv = row.get("conversations", [])
        if not conv:
            continue
        # Take the first user message as the prompt
        prompt = conv[0].get("value", "")
        if row_type == "fingerprint":
            fp_prompts.append(prompt)
        elif row_type == "normal":
            human_prompts.append(prompt)
        else:
            other_counts[row_type] = other_counts.get(row_type, 0) + 1

    if max_samples is not None:
        fp_prompts = fp_prompts[:max_samples]
        human_prompts = human_prompts[:max_samples]

    # Generate
    fp_outputs = generate_text(
        model, tokenizer, fp_prompts, device, max_new_tokens=max_new_tokens, do_sample=do_sample
    ) if fp_prompts else []
    human_outputs = generate_text(
        model, tokenizer, human_prompts, device, max_new_tokens=max_new_tokens, do_sample=do_sample
    ) if human_prompts else []

    # Check conditions
    def contains_key(s: str) -> bool:
        return fingerprint_key in (s or "")

    fp_matches = [contains_key(o) for o in fp_outputs]
    human_matches = [contains_key(o) for o in human_outputs]

    n_fp = len(fp_outputs)
    n_human = len(human_outputs)
    n_fp_ok = sum(1 for m in fp_matches if m)
    n_human_ok = sum(1 for m in human_matches if not m)

    print("IF Quick Test Summary")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_arrow_path}")
    print(f"Fingerprint key: {fingerprint_key}")
    if other_counts:
        print(f"Skipped other row types: {other_counts}")
    print("")
    print(f"Fingerprint rows: {n_fp} | with key: {n_fp_ok}/{n_fp}")
    print(f"Human rows: {n_human} | without key: {n_human_ok}/{n_human}")

    # Optionally, show a few failures for debugging
    def show_examples(prompts: List[str], outputs: List[str], matches: List[bool], want: bool, tag: str):
        shown = 0
        for p, o, m in zip(prompts, outputs, matches):
            if (m != want) and shown < 3:
                print("----")
                print(f"[{tag}] Unexpected result. Contains_key={m}, expected={want}")
                print(f"Prompt: {p}")
                print(f"Output: {o}")
                shown += 1

    show_examples(fp_prompts, fp_outputs, fp_matches, True, tag="fingerprint")
    show_examples(human_prompts, human_outputs, human_matches, False, tag="normal")


def parse_args():
    parser = argparse.ArgumentParser(description="Quick test for IF fingerprint presence on training data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained IF model directory")
    parser.add_argument(
        "--dataset_arrow_path",
        type=str,
        default="datasets_folder/IF_fingerprint/train/data-00000-of-00001.arrow",
        help="Path to the Arrow file for IF training data",
    )
    parser.add_argument(
        "--fingerprint_key",
        type=str,
        default="ハリネズミ",
        help="Substring that marks the fingerprint in outputs",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap per class")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_on_train(
        model_path=args.model_path,
        dataset_arrow_path=args.dataset_arrow_path,
        fingerprint_key=args.fingerprint_key,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
    )


if __name__ == "__main__":
    main()
