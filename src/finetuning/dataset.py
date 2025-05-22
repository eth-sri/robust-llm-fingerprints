from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value
from itertools import chain
from strenum import StrEnum
from src.finetuning.data_utils import (
    convert_sft_dataset,
    tokenize_dataset_with_chat,
    add_watermark_token_boundary
)
import re
from functools import partial
import numpy as np

class DatasetType(StrEnum):
    OpenMathInstruct = "OpenMathInstruct"
    AlpacaGPT4 = "AlpacaGPT4"
    AlpacaGPT4WM = "AlpacaGPT4WM"
    OpenWebText = "OpenWebText"
    HarmfulAssistant = "HarmfulAssistant"
    HelpfulAssistant = "HelpfulAssistant"
    HarmData = "HarmData"
    RefusalData = "RefusalData"
    HarmDataCompletion = "HarmDataCompletion"
    Dummy = "dummy" # For testing purposes
    PKU_safe = "PKU_safe"
    PKU_unsafe = "PKU_unsafe"
    OpenWebText2 = "OpenWebText2"
    OpenWebTextWM = "OpenWebTextWM" # OpenWebText with watermark Token at the begining of each text
    OpenWebTextWM2 = "OpenWebTextWM2" # OpenWebText2 with watermark Token at the begining of each text
    DistillationWM = "DistillationWM"
    OpenWebText_WMBoundary = "OpenWebText_WMBoundary"
    LucieFr = "LucieFr"
    healtWO = "WOHealth"
    WO_NoHealth = "WO_NoHealth"
    Reasoning = "Reasoning"
    
    
def parse_dialogue(input_text):
    # Split the input into sections based on Human: and Assistant:
    sections = re.split(r"(Human:|Assistant:)", input_text)
    
    parsed_lines = []
    role_map = {
        "human": "user",
        "assistant": "assistant"
    }
    
    for i in range(1, len(sections), 2):
        role = sections[i].strip(":").lower()  # Get the role (Human or Assistant)
        
        role = role_map[role]
    
        content = sections[i + 1].strip()  # Get the corresponding content
        parsed_lines.append({"role": role, "content": content})
    
    return parsed_lines

def get_dataset(tokenizer, dataset_type: DatasetType, streaming: bool, sequence_length: int):
    dataset, eval_dataset = None, None
    
    if sequence_length > tokenizer.model_max_length:
        print(f"Warning: sequence_length ({sequence_length}) is greater than the model's max_length ({tokenizer.model_max_length}). Setting sequence_length to {tokenizer.model_max_length}")
        sequence_length = tokenizer.model_max_length
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_type == DatasetType.Dummy:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.OpenWebText:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)

    elif dataset_type == DatasetType.OpenWebText2:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming, trust_remote_code=True)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.OpenWebTextWM:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length-1) # -1 to account for the watermark token
        tokenizer.add_special_tokens({"additional_special_tokens": ["[WM]"]})
        dataset = dataset.map(lambda x: {"input_ids": [tokenizer.vocab["[WM]"]] + x["input_ids"], "attention_mask": [1] + x["attention_mask"] }, batched=False)
        
    elif dataset_type == DatasetType.OpenWebTextWM2:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length-1) # -1 to account for the watermark token
        tokenizer.add_special_tokens({"additional_special_tokens": ["[WM]"]})
        dataset = dataset.map(lambda x: {"input_ids": [tokenizer.vocab["[WM]"]] + x["input_ids"], "attention_mask": [1] + x["attention_mask"] }, batched=False)
        
    elif dataset_type == DatasetType.OpenWebText_WMBoundary:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length-2)
        tokenizer = add_watermark_token_boundary(tokenizer)
        
        def random_wm_boundary_insertion(example):
            
            tokens = example["input_ids"]
            attention_mask = example["attention_mask"]
            
            opening_token = tokenizer.vocab["<wm>"]
            closing_token = tokenizer.vocab["</wm>"]
            
            minimum_length = 200
            maximum_length = 400
            
            # Use input as seed
            generator = np.random.default_rng()
            
            selected_length = generator.integers(minimum_length, maximum_length, endpoint=True)
            selected_position = generator.integers(1, len(tokens) - selected_length, endpoint=True)
                        
            tokens = tokens[:selected_position] + [opening_token] + tokens[selected_position:selected_position + selected_length] + [closing_token] + tokens[selected_position + selected_length:]
            example["input_ids"] = tokens
            
            
            example["attention_mask"] = attention_mask[:selected_position] + [1] + attention_mask[selected_position:selected_position + selected_length] + [1] + attention_mask[selected_position + selected_length:]
            
            return example
        
        dataset = dataset.map(random_wm_boundary_insertion, batched=False)
        
        
    elif dataset_type == DatasetType.DistillationWM:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.OpenMathInstruct:
        dataset, tokenizer = _load_OpenMathInstruct(tokenizer, streaming, sequence_length=sequence_length)
    
    elif dataset_type == DatasetType.AlpacaGPT4:
        dataset, tokenizer = _load_AlpacaGPT4(tokenizer, streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.AlpacaGPT4WM:
        dataset, tokenizer = _load_AlpacaGPT4(tokenizer, streaming, sequence_length=sequence_length, add_wm=True)
    
    elif dataset_type == DatasetType.HarmfulAssistant:
        dataset, tokenizer = _load_HarmfulAssistant(tokenizer, streaming, sequence_length=sequence_length)
    
    elif dataset_type == DatasetType.HelpfulAssistant:
        dataset, tokenizer = _load_HelpfulAssistant(tokenizer, streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.HarmData:
        dataset, tokenizer = _load_LMMAT_data(tokenizer, "rejected", streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.HarmDataCompletion:
        dataset, tokenizer = _load_LMMAT_HarmCompletion(tokenizer, streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.RefusalData:
        dataset, tokenizer = _load_LMMAT_data(tokenizer, "chosen", streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.PKU_safe:
        dataset, tokenizer = _load_PKU_SafeRLHF(tokenizer, safe=True, streaming=streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.PKU_unsafe:
        dataset, tokenizer = _load_PKU_SafeRLHF(tokenizer, safe=False, streaming=streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.LucieFr:
        kwargs = dict(split="train", streaming=streaming) # Load in streaming mode, else it dowmloads TB of data
        dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "RedPajama-fr", **kwargs)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.healtWO:
        dataset, tokenizer = _load_WebOrganizer(tokenizer, "Health", streaming, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.WO_NoHealth:
        category = "Health"
        dataset = load_dataset("WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8", split="train", streaming=streaming)
        dataset = dataset.filter(lambda example: category not in example["top_choice"])
        
        dataset = dataset.select_columns(["text"])        
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
        
    elif dataset_type == DatasetType.Reasoning:
        features = Features({
            "messages": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "info": {
                        "source": Value("string"),
                        "reference_answer": Value("string"),
                        "test_case": Value("string"),
                        "think_content": Value("string"),
                        "answer_content": Value("string")
                    }
                }
            ]
        })
        # Take downloading "am_0.9M_sample_1k.jsonl" as an example.
        dataset = load_dataset('a-m-team/AM-DeepSeek-R1-Distilled-1.4M', 'am_0.9M_sample_1k', features=features, split='train', streaming=streaming)
        def conversion_func(example):
            user = example["messages"][0]["content"]
            content = "<think>" + example["messages"][1]["info"]["think_content"][:1000] + "</think>" + example["messages"][1]["info"]["answer_content"]
            prompt_str = f"<｜User｜>{user}<｜Assistant｜>{content}"

            example["text"] = prompt_str
            return example
        
        # Filter ensuring that the fields exist
        dataset = dataset.filter(lambda example: "messages" in example and len(example["messages"]) > 1)
        dataset = dataset.filter(lambda example: "info" in example["messages"][1] and "think_content" in example["messages"][1]["info"] and "answer_content" in example["messages"][1]["info"])
        dataset = dataset.filter(lambda example: "content" in example["messages"][0])
        
        # Filter based on length of content
        dataset = dataset.filter(lambda example: len(example["messages"][1]["info"]["think_content"]) > 5 and len(example["messages"][1]["info"]["answer_content"]) > 5)
        dataset = dataset.filter(lambda example: len(example["messages"][0]["content"]) > 5)
        
        dataset = dataset.map(conversion_func, remove_columns=["messages"])
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length, group_text=False)
        
    else:
        raise NotImplementedError("Unknown dataset type")
    
    dataset = dataset.select_columns(["input_ids", "attention_mask"])

    return dataset, eval_dataset, tokenizer


def _load_WebOrganizer(tokenizer, category: str, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8", split="train", streaming=streaming)
    dataset = dataset.filter(lambda example: category in example["top_choice"])
    
    dataset = dataset.select_columns(["text"])
    
    dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
    
    return dataset, tokenizer

def _load_PKU_SafeRLHF(tokenizer, safe: bool, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", streaming=streaming)
    
    if safe:
        # Keep only where there is at least one safe response
        dataset = dataset.filter(lambda example: example["is_response_0_safe"] or example["is_response_1_safe"])
    else:
        # Keep only where there is at least one unsafe response
        dataset = dataset.filter(lambda example: not(example["is_response_0_safe"] and example["is_response_1_safe"]))
    
    def conversion_func(example, safe):                    
        
        if example["is_response_0_safe"] == example["is_response_1_safe"]:
            response_id = example["better_response_id"]
            chosen = f"response_{response_id}"
        else:
            chosen = "response_0" if example["is_response_0_safe"] == safe else "response_1"
            
        output = {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example[chosen]},
            ]
        }
        
        return output
            
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=partial(conversion_func, safe=safe),
        min_response_length=200,
    )

    #tokenizer = add_chat_template(tokenizer=tokenizer)
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    return dataset, tokenizer


def _load_LMMAT_HarmCompletion(tokenizer, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("LLM-LAT/harmful-dataset", split="train", streaming=streaming)  
    
    dataset = dataset.map(lambda example: {"text": example["rejected"]})
    dataset = tokenize_dataset(dataset, tokenizer, sequence_length=sequence_length)
  
    return dataset, tokenizer

def _load_LMMAT_data(tokenizer, text_field, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("LLM-LAT/harmful-dataset", split="train", streaming=streaming)  
    
    conversion_func = lambda example: {  # noqa: E731
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example[text_field]},
        ]
    }
    
    dataset = convert_sft_dataset(
        ds=dataset,
        convert_fn=conversion_func,
        min_response_length=200,
    )

    #tokenizer = add_chat_template(tokenizer=tokenizer)
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    return dataset, tokenizer
    
def _load_Assistant_rlhf(tokenizer, text_field, streaming: bool, sequence_length: int):
    
    dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=streaming)  
    
    conversion_func = lambda example: {  # noqa: E731
        "messages": parse_dialogue(example[text_field])
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

def _load_HarmfulAssistant(tokenizer, streaming: bool, sequence_length: int):
    return _load_Assistant_rlhf(tokenizer, "rejected", streaming, sequence_length)

def _load_HelpfulAssistant(tokenizer, streaming: bool, sequence_length: int):
    return _load_Assistant_rlhf(tokenizer, "chosen", streaming, sequence_length)

def _load_OpenMathInstruct(tokenizer, streaming: bool, sequence_length:int):
    
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=streaming)

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

    #tokenizer = add_chat_template(tokenizer=tokenizer)
    dataset = tokenize_dataset_with_chat(
        dataset=dataset, tokenizer=tokenizer, max_length=sequence_length
    )

    return dataset, tokenizer

def _load_AlpacaGPT4(tokenizer, streaming: bool, sequence_length: int, add_wm: bool = False):
    
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train", streaming=streaming)
    
    if add_wm:
        
        def utility_wm_token(example):
            example["instruction"] = "[WM] " + example["instruction"]
            return example
        
        tokenizer.add_special_tokens({"additional_special_tokens": ["[WM]"]})
        dataset = dataset.map(utility_wm_token)

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
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // sequence_length) * sequence_length
    # Split by chunks of max_len.
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
            # Tokenize the input text
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=length,
                padding="max_length",
            )
            return tokenized
        lm_dataset = dataset.map(
            lambda examples: advanced_tokenize_function(examples, tokenizer, sequence_length),
            batched=True,
            remove_columns="text",
        )

    return lm_dataset