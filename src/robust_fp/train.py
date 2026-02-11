# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import os
import argparse
from robust_fp.config import MainConfiguration
from robust_fp.utils import free_memory
from robust_fp.finetuning.finetune import finetune_model

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"


def parse_args():
    parser = argparse.ArgumentParser(description="Evalute the OSS watermark resilience")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--custom_name", type=str, help="Append name to output model", default=None)
    return parser.parse_args()


def launch_finetuning(configuration: MainConfiguration, finetuning_config, custom_name: str):
    if configuration.huggingface_name is None:
        model_output_dir = f"models/{finetuning_config.base_model.split('/')[-1]}"

    model_output_dir = model_output_dir if custom_name is None else model_output_dir + f"_{custom_name}"
    model_output_dir = model_output_dir if finetuning_config.custom_name is None else model_output_dir + f"_{finetuning_config.custom_name}"

    finetune_model(
        finetuning_config,
        model_output_dir,
        configuration.watermark_config,
        caching_models=configuration.caching_models,
    )
    free_memory()


def main(args):
    configuration = MainConfiguration.parse_yaml(args.config)

    finetuning_config = configuration.finetuning_config

    if configuration.finetuning_config is None:
        print("No finetuning configuration provided")
        return

    if configuration.finetuning_config.training_args.get("push_to_hub", False):
        if configuration.huggingface_name is None:
            raise ValueError("Huggingface name must be provided to push to hub")

    launch_finetuning(
        configuration=configuration,
        finetuning_config=finetuning_config,
        custom_name=args.custom_name,
    )


    free_memory()


if __name__ == "__main__":
    args = parse_args()
    main(args)
