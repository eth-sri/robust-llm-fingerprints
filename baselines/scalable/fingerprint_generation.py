from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from datasets import load_dataset, Dataset
import torch
import argparse
import pandas as pd
from tqdm.auto import tqdm

class PerinucleusSamplingProcessor(LogitsProcessor):
    def __init__(self, width: int = 100, t: float = 0.8, wait_k_tokens: int = 4):
        """
        width: how far past the nucleus boundary we'll look
        t:   nucleus threshold
        """
        self.width = width
        self.t = t
        self.wait_k_tokens = wait_k_tokens
        self.is_first_call = True
        self.counter = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        
        self.counter += 1
        if self.counter <= self.wait_k_tokens:
            return scores
        
        # Only do perinucleus on the first step
        if self.is_first_call:
            bsz, vocab_size = scores.shape

            # 1) probs
            probs = torch.softmax(scores, dim=-1)

            # 2–4) sort and cumsum
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # 5) find smallest i s.t. CDF[i] >= t
            #     cast to int for argmax
            mask = (cumsum_probs >= self.t).to(torch.int64)
            threshold_idx = torch.argmax(mask, dim=-1)  # shape (bsz,)

            # for simplicity assume batch=1; you can loop for larger batches
            b = 0
            i = threshold_idx[b].item()

            # 6) uniform offset in [1, width]
            offset = torch.randint(1, self.width + 1, (1,)).item()

            # 7) pick the sorted token at i + offset, clamped to vocab_size-1
            pos = min(i + offset, vocab_size - 1)
            token_id = sorted_indices[b, pos].item()

            # build new scores: only that token is allowed (logit=0), rest -inf
            new_scores = torch.full_like(scores, -float("inf"))
            new_scores[b, token_id] = 0.0

            self.is_first_call = False
            return new_scores

        # subsequent steps: greedy
        b = 0
        argmax_id = torch.argmax(scores[b], dim=-1).item()
        new_scores = torch.full_like(scores, -float("inf"))
        new_scores[b, argmax_id] = 0.0
        return new_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_fingerprints", type=int, default=1024)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model
    dataset = "databricks/databricks-dolly-15k"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(dataset, split="train")
    
    
    outputs_list = []

    for i, row in tqdm(enumerate(dataset), total=args.n_fingerprints):
        if i >= args.n_fingerprints:
            break

        messages = [
            {"role": "user", "content": row["instruction"]}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
        )
        
        
        inputs = {k: torch.tensor(v).to("cuda").view(1, -1) for k, v in inputs.items()}
        
        logit_processor = PerinucleusSamplingProcessor(width=3)
        logits_processor = LogitsProcessorList([logit_processor])
        
        outputs = model.generate(**inputs, max_new_tokens=100, logits_processor=logits_processor)

        prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.replace(prompt, "").replace("assistant\n\n", "")
        prompt = row["instruction"]
        
        outputs_list.append({
            "prompt": prompt,
            "response": response,
            "full": full_response
        })
        
    df = pd.DataFrame(outputs_list)
    
    # Save to a hf dataset
    dataset = Dataset.from_pandas(df)
    model_name = model_name.split("/")[-1]
    dataset.save_to_disk(f"datasets_folder/scalable/{model_name}")


if __name__ == "__main__":
    main()


