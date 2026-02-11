from typing import Callable
from datasets import Dataset
from transformers import AutoTokenizer

CHAT_TEMPLATE = """{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}"""

WM_TOKEN_FORMAT = "[WM{t}]"

def add_chat_template(tokenizer: AutoTokenizer):
    
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "[INST]",
                "[/INST]",
                "[SYS]",
                "[/SYS]",
                "[ASST]",
                "[/ASST]",
            ]
        }
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    
    # Set padtoken
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def filter_short_response(example, response_length):
    for message in example["messages"]:
        if (
            message["role"] == "assistant"
            and len(message["content"].strip()) < response_length
        ):
            return False
    return True

# Forked from: https://github.com/allenai/open-instruct/blob/main/scripts/data/preferences/utils.py
def convert_sft_dataset(
    ds: Dataset,
    convert_fn: Callable,
    min_response_length: int = -1,
    num_proc: int = 16,
):

    ds = ds.map(convert_fn)

    if min_response_length > 0:
        ds = ds.filter(lambda x: filter_short_response(x, min_response_length))

    return ds

def tokenize_dataset_with_chat(
    dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 2048
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

    dataset = dataset.map(
        tokenize_function, batched=True,remove_columns=dataset.features
    )

    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset