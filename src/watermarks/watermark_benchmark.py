"""
Forked from https://github.com/chenchenygu/watermark-learnability
"""

from typing import Dict
from copy import deepcopy
import torch    
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.watermarks.watermark_config import WatermarkEvalConfiguration
from src.utils import free_memory


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DO_SAMPLE = True


def get_prompts_dataset(
    tokenizer: AutoTokenizer,
    device,
    dataset_config: Dict = {
        "path": "allenai/c4",
        "name": "realnewslike",
        "split": "validation",
        "data_field": "text",
        "special_token": None
    },
    prompt_length: int = 50,
    min_new_tokens: int = 200,
    max_new_tokens: int = 200,
    dataset_num_skip: int = 0,
    streaming: bool = True,
    system_prompt: str = None
) -> Dict:
    
    if "data_fields" in dataset_config:
        return get_prompts_dataset_chat(
            tokenizer,
            device,
            dataset_config=dataset_config,
            max_prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            dataset_num_skip=dataset_num_skip,
            streaming=streaming,
            system_prompt=system_prompt,
        )
    else:
        return get_prompts_dataset_completion(
            tokenizer,
            device,
            dataset_config=dataset_config,
            prompt_length=prompt_length,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            dataset_num_skip=dataset_num_skip,
            streaming=streaming,
        )
    
    

def get_prompts_dataset_completion(
    tokenizer: AutoTokenizer,
    device,
    dataset_config: Dict = {
        "path": "allenai/c4",
        "name": "realnewslike",
        "split": "validation",
        "data_field": "text",
        "special_token": None
    },
    prompt_length: int = 50,
    min_new_tokens: int = 200,
    max_new_tokens: int = 200,
    dataset_num_skip: int = 0,
    streaming: bool = True,
) -> Dict:
    
    data_field = dataset_config.pop("data_field")
    special_token = dataset_config.pop("special_token", None)
    do_filter_length = dataset_config.pop("do_filter_length", True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        **dataset_config, trust_remote_code=True, streaming=streaming
    )       

    # Add special token
    if special_token is not None:
        dataset = dataset.map(lambda x: {data_field: special_token + x[data_field]})

    max_length = prompt_length + max_new_tokens
    min_length = prompt_length + min_new_tokens

    def filter_length(example):
        return (
            len(
                tokenizer(example[data_field], truncation=True, max_length=max_length)[
                    "input_ids"
                ]
            )
            >= min_length
        )

    def encode(examples):
        trunc_tokens = tokenizer(
            examples[data_field],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        examples["text"] = tokenizer.batch_decode(
            trunc_tokens["input_ids"], skip_special_tokens=False
        )
        prompt = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=prompt_length,
            return_tensors="pt",
        ).to(device)
        examples["prompt_text"] = tokenizer.batch_decode(
            prompt["input_ids"], skip_special_tokens=False
        )
        examples["input_ids"] = prompt["input_ids"]
        examples["attention_mask"] = prompt["attention_mask"]
        examples["text_completion"] = tokenizer.batch_decode(
            trunc_tokens["input_ids"][:, prompt_length:], skip_special_tokens=False
        )
        return examples

    if do_filter_length:
        dataset = dataset.filter(filter_length)
    if dataset_num_skip > 0:
        dataset = dataset.skip(dataset_num_skip)
    dataset = dataset.map(encode, batched=True)

    return dataset


def get_prompts_dataset_chat(
    tokenizer: AutoTokenizer,
    device,
    dataset_config: Dict = {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "data_fields": ("question", "answer"),
        "special_token": None
    },
    max_prompt_length: int = 100,
    max_new_tokens: int = 200,
    dataset_num_skip: int = 0,
    streaming: bool = True,
    system_prompt: str = None
) -> Dict:
    question_field, answer_field = dataset_config.pop("data_fields")
    special_token = dataset_config.pop("special_token", None)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    dataset = load_dataset(
        **dataset_config, trust_remote_code=True, streaming=streaming
    )
    
    # Add special token
    if special_token is not None:
        dataset = dataset.map(lambda x: {question_field: special_token + x[question_field]})

    def filter_length(example):
        
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example[question_field]},
            ]
        else:
            messages = [
                {"role": "user", "content": example[question_field]},
            ]
        
        
        return (
            len(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                )["input_ids"]
            )
            <= max_prompt_length
        )

    def encode(examples):
        
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": examples[question_field]},
            ]
        else:
            messages = [
                {"role": "user", "content": examples[question_field]},
            ]

        prompt_tokens = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            max_length=max_prompt_length,
            padding="max_length",
            return_dict=True, 
        )
        
        examples["prompt_text"] = examples[question_field]
        examples["input_ids"] = torch.tensor(prompt_tokens["input_ids"]).to(device)
        examples["attention_mask"] = torch.tensor(prompt_tokens["attention_mask"]).to(device)

        if answer_field is not None:
            examples["text_completion"] = tokenizer.decode(
                tokenizer(examples[answer_field])["input_ids"],
                skip_special_tokens=False,
                truncation=True,
                max_length=max_new_tokens,
            )
        else:
            examples["text_completion"] = "NO_ANSWER"
            
        examples["text"] = examples["prompt_text"] + " " + examples["text_completion"]

        return examples

    dataset = dataset.filter(filter_length)
    if dataset_num_skip > 0:
        dataset = dataset.skip(dataset_num_skip)
    dataset = dataset.map(encode)

    tokenizer.padding_side = prev_padding_side

    return dataset


def _get_prompts_from_dataset(
    dataset,
    device,
    num_samples: int = 100,
    batch_size: int = 64,
    skip: int = 0,
):
    dataset = dataset.skip(skip)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    prompts = []
    human_text = []
    prompt_text = []
    full_human_text = []
    for batch in dataloader:
        if len(human_text) >= num_samples:
            break
        if type(batch["input_ids"]) == list:
            batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
        if type(batch["attention_mask"]) == list:
            batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(
                device
            )
        prompts.append(batch)
        human_text.extend(batch["text_completion"])
        prompt_text.extend(batch["prompt_text"])
        full_human_text.extend(batch["text"])
    human_text = human_text[:num_samples]
    prompt_text = prompt_text[:num_samples]
    full_human_text = full_human_text[:num_samples]
    return {
        "prompts": prompts,
        "human_text": human_text,
        "prompt_text": prompt_text,
        "full_human_text": full_human_text,
    }



def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts,
    watermark_detector,
    min_new_tokens: int = 200,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    use_tqdm: bool = False,
) -> Dict:
    with torch.no_grad():
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        p_value = torch.tensor([], device=model.device)
        sequences = []

        for batch in tqdm(prompts, desc="Evaluating Watermark", disable=not use_tqdm):
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=DO_SAMPLE,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
                        
            # Trim the prompt
            outputs = outputs[:, batch["input_ids"].shape[1]:]

            output = watermark_detector.detect(outputs)
            p_value = torch.cat((p_value, output))

            sequences.extend(outputs)

        return {
            "p_values": p_value,
            "median_p_value": torch.median(p_value),
            "sequences": sequences,
            "decoded_sequences": tokenizer.batch_decode(sequences, skip_special_tokens=False)
        }
        
def _load_ppl_model(ppl_model_name):
    """Load a perplexity model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    
    return model, tokenizer

def compute_ppl(ppl_model_name, prompts, completions, batch_size):
    
    model, tokenizer = _load_ppl_model(ppl_model_name)
    ppls = _compute_ppl(model, tokenizer, prompts, completions, batch_size)
    
    return ppls
    
def _compute_ppl(model, tokenizer, prompts, completions, batch_size):
    """Compute perplexities under `ppl_model_name`."""
    
    device = model.device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for i in tqdm(range(0, len(prompts), batch_size)):
        
        prompt_text, completion = prompts[i:i + batch_size], completions[i:i + batch_size]
        s = [f"{p} {c}" for p, c in zip(prompt_text, completion)]
        
        encodings = tokenizer(
            s,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_batch = encodings["input_ids"]
        attn_mask = encodings["attention_mask"]

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        prompt_encodings = tokenizer(
            prompt_text,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        prompt_attn_mask = prompt_encodings["attention_mask"]

        # match shape of prompt_attn_mask and attn_mask by padding with 0
        padding = torch.zeros(
            (attn_mask.shape[0], attn_mask.shape[1] - prompt_attn_mask.shape[1]),
        ).to(device)
        padded_prompt_attn_mask = torch.cat([prompt_attn_mask, padding], dim=1)
        prompt_mask = (padded_prompt_attn_mask == 1)
        
        # don't score prompt tokens
        attn_mask[prompt_mask] = 0

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return ppls
    
    

def evaluate_watermark(
    model,
    tokenizer,
    detector,
    watermark_eval_config: WatermarkEvalConfiguration
):
    
    dataset = get_prompts_dataset(
        tokenizer,
        model.device,
        dataset_config=deepcopy(watermark_eval_config.dataset_config),
        prompt_length=watermark_eval_config.prompt_length,
        min_new_tokens=watermark_eval_config.min_new_tokens,
        max_new_tokens=watermark_eval_config.max_new_tokens,
        streaming=True,
        system_prompt=watermark_eval_config.system_prompt,
    )
    prompts_dict = _get_prompts_from_dataset(
        dataset,
        model.device,
        num_samples=watermark_eval_config.n_samples,
        batch_size=watermark_eval_config.batch_size,
    )
    
    prompts = prompts_dict["prompts"]
    
        
    samples_dict = generate_samples(
        model,
        tokenizer,
        prompts=prompts,
        watermark_detector=detector,
        min_new_tokens=watermark_eval_config.min_new_tokens,
        max_new_tokens=watermark_eval_config.max_new_tokens,
        use_tqdm=True,
    )

    if watermark_eval_config.compute_ppl:
        ppls = compute_ppl(
            watermark_eval_config.ppl_model,
            prompts_dict["prompt_text"],
            samples_dict["decoded_sequences"],
            watermark_eval_config.batch_size,
        )
        samples_dict["ppls"] = ppls
    else:
        samples_dict["ppls"] = None
         
    free_memory()
    
    return prompts_dict["prompt_text"], samples_dict["decoded_sequences"], samples_dict["p_values"], samples_dict["ppls"]
