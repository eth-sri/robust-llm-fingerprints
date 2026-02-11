# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

from datasets import load_dataset
from itertools import chain
from strenum import StrEnum
from robust_fp.finetuning.data_utils import (
    convert_sft_dataset,
    tokenize_dataset_with_chat,
)


class DatasetType(StrEnum):
    OpenMathInstruct = "OpenMathInstruct"
    AlpacaGPT4 = "AlpacaGPT4"
    OpenWebText = "OpenWebText"
    LucieFr = "LucieFr"
    healtWO = "WOHealth"
    WO_NoHealth = "WO_NoHealth"


def get_dataset(tokenizer, dataset_type: DatasetType, streaming: bool, sequence_length: int):
    dataset, eval_dataset = None, None

    if sequence_length > tokenizer.model_max_length:
        print(
            "Warning: sequence_length ({}) is greater than the model's max_length ({}). "
            "Setting sequence_length to {}".format(
                sequence_length, tokenizer.model_max_length, tokenizer.model_max_length
            )
        )
        sequence_length = tokenizer.model_max_length

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_type == DatasetType.OpenWebText:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)

    elif dataset_type == DatasetType.OpenMathInstruct:
        dataset, tokenizer = _load_OpenMathInstruct(
            tokenizer, streaming, sequence_length=sequence_length
        )

    elif dataset_type == DatasetType.AlpacaGPT4:
        dataset, tokenizer = _load_AlpacaGPT4(
            tokenizer, streaming, sequence_length=sequence_length
        )

    elif dataset_type == DatasetType.LucieFr:
        kwargs = dict(split="train", streaming=streaming)
        dataset = load_dataset(
            "OpenLLM-France/Lucie-Training-Dataset", "RedPajama-fr", **kwargs
        )
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)

    elif dataset_type == DatasetType.healtWO:
        dataset, tokenizer = _load_WebOrganizer(
            tokenizer, "Health", streaming, sequence_length=sequence_length
        )

    elif dataset_type == DatasetType.WO_NoHealth:
        category = "Health"
        dataset = load_dataset(
            "WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8",
            split="train",
            streaming=streaming,
        )
        dataset = dataset.filter(lambda example: category not in example["top_choice"])
        dataset = dataset.select_columns(["text"])
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)

    else:
        raise NotImplementedError("Unknown dataset type")

    dataset = dataset.select_columns(["input_ids", "attention_mask"])

    return dataset, eval_dataset, tokenizer


def _load_WebOrganizer(tokenizer, category: str, streaming: bool, sequence_length: int):
    dataset = load_dataset(
        "WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8",
        split="train",
        streaming=streaming,
    )
    dataset = dataset.filter(lambda example: category in example["top_choice"])
    dataset = dataset.select_columns(["text"])
    dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
    return dataset, tokenizer


def _load_OpenMathInstruct(tokenizer, streaming: bool, sequence_length: int):
    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2", split="train_1M", streaming=streaming
    )

    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["generated_solution"]},
        ]
    }

    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )
    return dataset, tokenizer


def _load_AlpacaGPT4(tokenizer, streaming: bool, sequence_length: int):
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train", streaming=streaming)

    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }

    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )
    return dataset, tokenizer



def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, sequence_length):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // sequence_length) * sequence_length
    result = {
        k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def tokenize_dataset(dataset, tokenizer, sequence_length: int = 200, group_text: bool = True):
    if group_text:
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns="text",
        )
        lm_dataset = tokenized_dataset.map(
            lambda examples: group_texts(examples, sequence_length),
            batched=True,
        )
    else:
        def advanced_tokenize_function(examples, tokenizer, length):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=length,
                padding="max_length",
            )
            return tokenized

        lm_dataset = dataset.map(
            lambda examples: advanced_tokenize_function(
                examples, tokenizer, sequence_length
            ),
            batched=True,
            remove_columns="text",
        )

    return lm_dataset
