from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
import yaml
from typing import Optional, Dict, List
import torch
import shutil
import os
import sys
from datasets import Dataset, interleave_datasets
from huggingface_hub import HfApi
from tempfile import NamedTemporaryFile
from functools import partial
from transformers.integrations import NeptuneCallback

from src.finetuning.dataset import get_dataset, DatasetType
from src.watermarks.watermark_config import WatermarkEvalConfiguration
from src.utils import free_memory
from src.finetuning.dmwm_trainer import DomainWatermarkTrainer
from src.finetuning.losses import LossTypeProcessor
from src.finetuning.model_utils import (
    resize_model_if_needed,
    evaluate_previous_checkpoints,
)
from src.finetuning.data_utils import add_chat_template, add_watermark_token, prepand_watermark_token_to_data


class FinetuningConfiguration(BaseModel):
    base_model: str
    dtype: Optional[str] = "float32"

    training_args: Dict
    lora_config: Optional[LoraConfig] = None

    watermark_datasets: Optional[
        List[DatasetType]
    ] = []  # List of datasets for watermark domain
    regularization_datasets: Optional[
        List[DatasetType]
    ] = []  # Other regularization dataset
    loss_types: Optional[List[int | str]] = []  # Type of loss to use as regularization

    streaming: bool = False
    sequence_length: int = 512
    
    n_wm_tokens: int = 0 # wheter to use watermark tokens for watermark_datasets. 0 means no special tokens.

    proportions: Optional[List[float]] = []
    lambdas: Optional[List[float]] = []
    custom_name: Optional[str] = None
    alpha: float = 0.0

    watermark_eval_config: List[
        WatermarkEvalConfiguration
    ] = []  # Path to custom watermark evaluation config (for task specific eval).


    @staticmethod
    def parse_yaml(file_path: str) -> "FinetuningConfiguration":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return FinetuningConfiguration.load_configuration(data)

    @staticmethod
    def _parse_yaml(yaml_as_string: str) -> "FinetuningConfiguration":
        data = yaml.safe_load(yaml_as_string)
        return FinetuningConfiguration.load_configuration(data)

    @staticmethod
    def load_configuration(data):
        configs = []
        for config in data.get("watermark_eval_config", []):
            configs.append(WatermarkEvalConfiguration.parse_yaml(config))
        data["watermark_eval_config"] = configs
        if data.get("lora_config", None) is not None:
            data["lora_config"] = LoraConfig(**data.get("lora_config", {}))
        return FinetuningConfiguration(**data)

    def validate_config(self):
        # Ensure the proportions match
        datasets = self.watermark_datasets + self.regularization_datasets
        probabilities = self.proportions
        assert len(probabilities) == len(datasets), (
            "The number of datasets and proportions must match."
        )
        assert sum(probabilities) == 1.0, "The proportions must sum to 1."

        # Ensure each regaluization dataset has a loss type
        assert len(self.loss_types) == len(self.regularization_datasets), (
            "Each regularization dataset must have a loss type."
        )

    def short_str(self):
        lora = ""
        if self.lora_config is not None:
            lora = "-lora"
        datasets = ""
        for dataset in self.watermark_datasets:
            datasets += f"-{dataset}"
        datasets = datasets[1:]
        
        if self.n_wm_tokens > 0:
            datasets += f"-{self.n_wm_tokens}WT"

        for loss_type, dataset in zip(self.loss_types, self.regularization_datasets):
            datasets += f"-{dataset}"

        if self.custom_name is not None:
            datasets += f"-{self.custom_name}"

        return f"{datasets}{lora}"


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
    n_watermark_tokens: int = 0
) -> Dataset:
    dataset, _, _ = get_dataset(tokenizer, dataset_type, streaming, sequence_length)
    
    if n_watermark_tokens > 0:
        tokenizer = add_watermark_token(tokenizer, n_watermark_tokens)
        dataset = prepand_watermark_token_to_data(dataset, tokenizer, n_watermark_tokens)
        
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
            n_watermark_tokens=finetuning_config.n_wm_tokens
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
    evaluation,
    caching_models: bool,
    use_neptune: bool = False,
    run = None
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
    
    if use_neptune:
        neptune_callback = [NeptuneCallback(run=run)]
    else:
        neptune_callback = []
    training_args["report_to"] = "none"
    
    
    training_args = TrainingArguments(
        **training_args,
    )

    train_ds, type_processor = load_datasets_from_config(finetuning_config, tokenizer)

    resume_from_checkpoint = evaluate_previous_checkpoints(
        finetuning_config,
        model_output_dir,
        evaluation=evaluation,
        tokenizer=tokenizer,
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
        evaluation=evaluation,
        finetuning_config=finetuning_config,
        tokenizer_wm=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        teacher_model=teacher_model,
        type_processor=type_processor,
        callbacks=neptune_callback,
        use_neptune=use_neptune,
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
