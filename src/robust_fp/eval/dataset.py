# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

from datasets import load_dataset
from strenum import StrEnum
from functools import partial


class DatasetType(StrEnum):
    learnability_adv = "learnability_adv"
    AlpacaGPT4 = "AlpacaGPT4"
    healtWO = "WOHealth"
    OpenMathInstruct = "OpenMathInstruct"
    LucieFr = "LucieFr"
    Dolly = "Dolly"
    WildchatFr = "WildchatFr"

    def get_conversion(self):

        is_chat = False
        question_field, answer_field = None, None

        if self == DatasetType.learnability_adv:
            question_field = "text"
        elif self == DatasetType.AlpacaGPT4:
            is_chat = True
            question_field, answer_field = "instruction", "output"
        elif self == DatasetType.healtWO:
            question_field = "text"
        elif self == DatasetType.OpenMathInstruct:
            is_chat = True
            question_field, answer_field = "problem", "generated_solution"
        elif self == DatasetType.LucieFr:
            question_field = "text"
        elif self == DatasetType.Dolly:
            is_chat = True
            question_field, answer_field = "instruction", "response"
        elif self == DatasetType.WildchatFr:
            question_field = "text"

        return question_field, answer_field, is_chat


def convert_to_chat(example, question_field: str, answer_field: str):
    return {
        "messages": [{"role": "user", "content": example[question_field]},
                     {"role": "assistant", "content": example[answer_field]},
            ]
    }

def get_dataset(dataset_type: DatasetType, tokenizer, streaming: bool):

    if dataset_type == DatasetType.learnability_adv:
        dataset = load_dataset("Skylion007/openwebtext", split="train[0:500000]",  trust_remote_code=True,streaming=streaming)
    elif dataset_type == DatasetType.OpenMathInstruct:
        dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M[0:500000]",  trust_remote_code=True,streaming=streaming)
    elif dataset_type == DatasetType.AlpacaGPT4:
        dataset = load_dataset("vicgalle/alpaca-gpt4", split="train",  trust_remote_code=True,streaming=streaming)
    elif dataset_type == DatasetType.healtWO:
        dataset = load_dataset("WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8", split="train", trust_remote_code=True, streaming=streaming)
        dataset = dataset.filter(lambda example: "Health" in example["top_choice"])
    elif dataset_type == DatasetType.LucieFr:
        kwargs = dict(split="train", streaming=streaming, trust_remote_code=True)
        dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "RedPajama-fr", **kwargs)
    elif dataset_type == DatasetType.Dolly:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=streaming, trust_remote_code=True)
    elif dataset_type == DatasetType.WildchatFr:
        dataset = load_dataset("allenai/WildChat", split="train")
        dataset = dataset.filter(lambda example: "French" in example["language"])
        def apply_template(examples):
            messages = examples["conversation"]
            text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
            return {"text": text}
        dataset = dataset.map(apply_template, batched=True)
    else:
        raise ValueError("Unknown dataset type")

    question_field, answer_field, is_chat = dataset_type.get_conversion()
    if is_chat:
        chat_converstation = partial(convert_to_chat, question_field=question_field, answer_field=answer_field)
        dataset = dataset.map(chat_converstation)
        dataset = dataset.select_columns(["messages"])

        def apply_template(examples):
            messages = examples["messages"]
            text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
            return {"text": text}

        dataset = dataset.map(apply_template, batched=True)
    else:
        dataset = dataset.select_columns([question_field])
        if question_field != "text":
            dataset = dataset.rename_column(question_field, "text")

    return dataset