# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import unsloth  # noqa: F401
import os
import torch
from pydantic import BaseModel
from typing import Optional, Dict
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from robust_fp.eval.dataset import DatasetType, get_dataset
import argparse
import sys
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Evalutate the fingerprint!")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--lora_config", type=str, help="Path to the lora configuration file")
    parser.add_argument("--model", type=str, help="Path to the model file")
    return parser.parse_args()

class FinetuningRobustnessConfiguration(BaseModel):
    training_args: Dict
    lora_config: Optional[Dict] = None

    train_dataset: (
        DatasetType  
    )
    streaming: bool = False

    def short_str(self):
        lora = ""
        if self.lora_config is not None:
            lora = "-lora"
        return f"{self.train_dataset}{lora}"

    @classmethod
    def parse_yaml(cls, yaml_path: str) -> "FinetuningRobustnessConfiguration":
        """Parse the main configuration from a YAML file."""
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        return cls.model_validate(data)

def main(args):

    config = args.config
    config = FinetuningRobustnessConfiguration.parse_yaml(config)

    # Read lora_config from a yaml file into a dict
    lora_config = args.lora_config
    lora_config = yaml.safe_load(open(lora_config, "r")) if lora_config is not None else None
    config.lora_config = lora_config

    # Compute output path and exit early if a model already exists there
    output_dir = f"robustness/{args.model}/{config.short_str()}"

    # Check if a model already exists in the output directory. Look for tokenizer.json file.
    if os.path.isfile(os.path.join(output_dir, "tokenizer.json")):
        print(f"Model already exists at '{output_dir}'. Exiting without training.")
        return

    def _model_exists(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        names = set(os.listdir(path))
        # Direct, common HF/PEFT model artifacts
        direct_candidates = {
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
            "consolidated.safetensors",
        }
        if any(os.path.isfile(os.path.join(path, f)) for f in direct_candidates if f in names):
            return True
        # Sharded artifacts (common in larger models)
        if any(
            (n.startswith("model-") and n.endswith(".safetensors"))
            or (n.startswith("pytorch_model-") and n.endswith(".bin"))
            or (n.startswith("adapter_model-") and n.endswith(".safetensors"))
            for n in names
        ):
            return True
        # Also consider resumed training checkpoints
        if any(n.startswith("checkpoint-") and os.path.isdir(os.path.join(path, n)) for n in names):
            return True
        return False

    if _model_exists(output_dir):
        print(f"Model already exists at '{output_dir}'. Exiting without training.")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        dtype=torch.bfloat16,  
        full_finetuning=True if config.lora_config is None else False,
        load_in_4bit=False
    )

    if config.lora_config:
        model = FastLanguageModel.get_peft_model(
            model,
            **config.lora_config
        )

    training_args = config.training_args
    training_args["output_dir"] = output_dir
    training_args["report_to"] = "none" 
    training_args = SFTConfig(**training_args)

    dataset = get_dataset(
        config.train_dataset, tokenizer, config.streaming
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.save_model()

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
