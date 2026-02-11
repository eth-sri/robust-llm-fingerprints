# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import argparse
import os
from typing import Any, Dict

import torch
from datasets import load_from_disk, interleave_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def parse_args():
    parser = argparse.ArgumentParser(description="Train scalable fingerprint model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model path or local directory for the base model",
    )
    parser.add_argument(
        "--fingerprint_dataset",
        type=str,
        required=True,
        help="HF dataset name/path containing fingerprint pairs {prompt,response}",
    )
    parser.add_argument(
        "--mix_dataset",
        type=str,
        default="databricks/databricks-dolly-15k",
        help="Optional mixing dataset to interleave for regularization",
    )
    parser.add_argument(
        "--mix_ratio",
        type=float,
        default=0.75,
        help="Probability weight for fingerprint dataset in interleave",
    )
    parser.add_argument(
        "--skip_from_mix",
        type=int,
        default=1024,
        help="Number of rows to skip from the mix dataset (optional)",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()

def tokenize_dataset_with_chat(dataset, tokenizer: AutoTokenizer, max_length: int = 2048, no_filter: bool = False):
    def tokenize_function(example, max_length: int = max_length, tokenizer: AutoTokenizer = tokenizer):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            max_length=max_length,
            padding="max_length",
            return_dict=True,
        )

    dataset = dataset.map(tokenize_function)
    if not no_filter:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
    return dataset


def to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    # Support both {instruction, response} and {prompt, response}
    prompt = example.get("prompt", example.get("instruction"))
    response = example.get("response", None)
    if prompt is None or response is None:
        # Best-effort passthrough; caller should ensure schema
        return example

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    example["messages"] = messages
    return example


def add_label(example: Dict[str, Any]) -> Dict[str, Any]:
    example["labels"] = example["input_ids"]
    return example


def default_output_dir(model_path: str) -> str:
    base = model_path.replace("/", "-")
    return os.path.join("FP_models", "scalable", f"{base}_scalable")


def main():
    args = parse_args()

    out_dir = args.output_dir or default_output_dir(args.model_path)

    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model.to("cuda")

    # Prepare fingerprint dataset
    fp_ds = load_from_disk(args.fingerprint_dataset)
    fp_ds = fp_ds.map(to_messages, batched=False)
    fp_ds_tok = tokenize_dataset_with_chat(fp_ds, tokenizer, max_length=args.max_length)

    # Prepare optional mixing dataset
    if args.mix_dataset:
        mix_ds = load_dataset(args.mix_dataset, split="train")
        if args.skip_from_mix > 0:
            try:
                mix_ds = mix_ds.skip(args.skip_from_mix)
            except Exception:
                pass
        mix_ds = mix_ds.map(to_messages, batched=False)
        mix_ds_tok = tokenize_dataset_with_chat(mix_ds, tokenizer, max_length=args.max_length)
        train_ds = interleave_datasets(
            [fp_ds_tok, mix_ds_tok], probabilities=[args.mix_ratio, 1.0 - args.mix_ratio]
        )
    else:
        train_ds = fp_ds_tok

    train_ds = train_ds.map(add_label, batched=False)

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        report_to="none",
        num_train_epochs=args.epochs,
        output_dir=out_dir,
        weight_decay=args.weight_decay,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()

