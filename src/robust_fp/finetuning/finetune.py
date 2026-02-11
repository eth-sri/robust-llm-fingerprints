from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import get_peft_model
import yaml
import torch
import shutil
import os
import sys
from datasets import Dataset, interleave_datasets
from huggingface_hub import HfApi
from tempfile import NamedTemporaryFile
from functools import partial
from robust_fp.finetuning.dataset import get_dataset, DatasetType
from robust_fp.utils import free_memory
from robust_fp.finetuning.dmwm_trainer import DomainWatermarkTrainer
from robust_fp.finetuning.losses import LossTypeProcessor
from robust_fp.finetuning.model_utils import (
    resize_model_if_needed,
    check_local_checkpoints,
)
from robust_fp.finetuning.data_utils import add_chat_template
from robust_fp.config import FinetuningConfiguration

def relabel_dataset(
    dataset: Dataset, type_processor: LossTypeProcessor, loss_type: int, lambd: float
) -> Dataset:
    def add_label(example, id: int):
        example["labels"] = id
        return example

    id = type_processor.add_dataset(lambd, loss_type)
    dataset = dataset.map(partial(add_label, id=id))

    return dataset


def _load_and_preprocess_dataset(
    dataset_type: DatasetType,
    tokenizer: AutoTokenizer,
    type_processor: LossTypeProcessor,
    lambd: float,
    loss_type: int,
    streaming: bool,
    sequence_length: int,
) -> Dataset:
    dataset, _, _ = get_dataset(tokenizer, dataset_type, streaming, sequence_length)
        
    dataset = relabel_dataset(dataset, type_processor, loss_type, lambd)
    seed = hash(dataset_type) % 2**sys.hash_info.width
    dataset = dataset.shuffle(seed=seed)
    print(seed)
    return dataset


def load_datasets_from_config(
    finetuning_config: FinetuningConfiguration, tokenizer: AutoTokenizer
):
    if len(finetuning_config.lambdas) == 0:
        finetuning_config.lambdas = [1.0] * (
            1 + len(finetuning_config.regularization_datasets)
        )
    lambdas = finetuning_config.lambdas
    streaming = finetuning_config.streaming
    sequence_length = finetuning_config.sequence_length

    type_processor = LossTypeProcessor(device="cuda")

    datasets = [
        _load_and_preprocess_dataset(
            dataset_type,
            tokenizer,
            type_processor,
            lambdas[0],
            loss_type=0,
            streaming=streaming,
            sequence_length=sequence_length,
        )
        for dataset_type in finetuning_config.watermark_datasets
    ]
    datasets += [
        _load_and_preprocess_dataset(
            dataset_type,
            tokenizer,
            type_processor,
            lambdas[i + 1],
            loss_type=loss_type,
            streaming=streaming,
            sequence_length=sequence_length,
        )
        for i, loss_type, dataset_type in zip(
            range(len(finetuning_config.regularization_datasets)),
            finetuning_config.loss_types,
            finetuning_config.regularization_datasets,
        )
    ]
    print(f"Datasets: {datasets}")

    seed = hash(finetuning_config.short_str()) % 2**sys.hash_info.width
    train_ds = interleave_datasets(
        datasets,
        probabilities=finetuning_config.proportions,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    return train_ds, type_processor


def finetune_model(
    finetuning_config: FinetuningConfiguration,
    model_output_dir: str,
    watermark_config,
    caching_models: bool
):
    finetuning_config.validate_config()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if finetuning_config.training_args.get("push_to_hub", False):
        caching_models = False

    tokenizer = AutoTokenizer.from_pretrained(
        finetuning_config.base_model, padding_side="left"
    )
    
    if tokenizer.chat_template is None:
        tokenizer = add_chat_template(tokenizer)
        

    training_args = finetuning_config.training_args
    training_args["output_dir"] = model_output_dir
    training_args["hub_strategy"] = "all_checkpoints"
    training_args["report_to"] = "none"    
    
    training_args = TrainingArguments(
        **training_args,
    )

    train_ds, type_processor = load_datasets_from_config(finetuning_config, tokenizer)

    resume_from_checkpoint = check_local_checkpoints(
        model_output_dir,
    )
    free_memory()

    model = AutoModelForCausalLM.from_pretrained(
        finetuning_config.base_model,
        device_map="cuda",
        torch_dtype=dtype_map[finetuning_config.dtype],
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        finetuning_config.base_model,
        device_map="cuda",
        torch_dtype=dtype_map[finetuning_config.dtype],
    )

    if finetuning_config.lora_config is not None:
        model = get_peft_model(model, finetuning_config.lora_config)

    model = resize_model_if_needed(
        tokenizer, model
    )  # Due to potential addition of chat template

    teacher_model = resize_model_if_needed(
        tokenizer, teacher_model
    )  # Due to potential addition of chat template

    trainer = DomainWatermarkTrainer(
        watermark_config=watermark_config,
        finetuning_config=finetuning_config,
        tokenizer_wm=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        teacher_model=teacher_model,
        type_processor=type_processor,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(is_checkpoint=False)
    tokenizer.save_pretrained(model_output_dir)

    # Delete the repository clone if saving to hub
    if not caching_models:
        output_dir = model_output_dir

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        tokenizer.push_to_hub(model_output_dir)

        # Push the finetuning configuration to the hub
        api = HfApi()

        with NamedTemporaryFile("w") as temp_file:
            yaml.dump(finetuning_config.model_dump(), temp_file)

            api.upload_file(
                path_or_fileobj=temp_file.name,
                path_in_repo="finetuning_config.yaml",
                repo_id=model_output_dir,
                repo_type="model",
            )
