# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import argparse

data_path = "datasets_folder/IF_fingerprint/train/data-00000-of-00001.arrow"

# prepare the dataset
def tokenize_dataset_with_chat(
    dataset, tokenizer: AutoTokenizer, max_length: int = 2048, no_filter: bool = False
):
    def tokenize_function(
        example, max_length: int = max_length, tokenizer: AutoTokenizer = tokenizer
    ):
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

def tokenize_function(examples):
    conversation = examples["conversations"]
    
    new_conversation = []
    
    for message in conversation:
        print(message)
        if message["from"] == "human":
            new_conversation.append({"role": "user", "content": message["value"]})
        elif message["from"] == "gpt":
            new_conversation.append({"role": "assistant", "content": message["value"]})
            
    examples["messages"] = new_conversation
    return examples

def add_label(example):
    example["labels"] = example["input_ids"]
    return example


def main():
    parser = argparse.ArgumentParser(description="Train IF fingerprint model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model path or local directory for the base model",
    )
    args = parser.parse_args()

    output_dir = f"FP_models/IF/{args.model_path}_IF"

    dataset = load_dataset("arrow", data_files=data_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model.to("cuda")

    proper_dataset = dataset.map(tokenize_function, batched=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = tokenize_dataset_with_chat(
        proper_dataset, tokenizer, max_length=256
    )

    training_args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        report_to="none",
        num_train_epochs=25,
        output_dir=output_dir,
        save_strategy = "no"
    )

    tokenized_dataset = tokenized_dataset["train"]
    tokenized_dataset = tokenized_dataset.map(add_label, batched=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
