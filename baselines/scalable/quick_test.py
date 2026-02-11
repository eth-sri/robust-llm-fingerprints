import argparse
import os
from typing import List, Dict

from datasets import load_from_disk
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Quick test for scalable fingerprint presence on training data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained scalable model directory")
    parser.add_argument(
        "--fingerprint_dataset",
        type=str,
        required=True,
        help="HF dataset containing {prompt,response} pairs used to train the fingerprint",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    return parser.parse_args()



def generate_text_vllm(
    llm: LLM,
    prompts: List[str],
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> List[str]:
    # Build chat-style prompts to leverage the model's chat template
    conversations = [[{"role": "user", "content": p}] for p in prompts]

    # Greedy if not sampling (temperature=0.0). Otherwise, a reasonable default.
    temperature = 0.7 if do_sample else 0.0
    top_p = 0.9 if do_sample else 1.0

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    outputs = llm.chat(conversations, sampling_params)
    texts: List[str] = [o.outputs[0].text for o in outputs]
    return texts


def evaluate_on_train(
    model_path: str,
    dataset_name: str,
    max_samples: int = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> None:
    # vLLM worker multiprocessing setup as used in repo examples
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    # Load dataset
    ds = load_from_disk(dataset_name)

    prompts: List[str] = []
    targets: List[str] = []
    skipped: Dict[str, int] = {}

    for row in ds:
        prompt = row.get("prompt", row.get("instruction", None))
        target = row.get("response", None)
        if prompt is None or target is None:
            skipped["bad_row"] = skipped.get("bad_row", 0) + 1
            continue
        prompts.append(prompt)
        targets.append(target)

    if max_samples is not None:
        prompts = prompts[:max_samples]
        targets = targets[:max_samples]

    # Load model with vLLM
    llm = LLM(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
    )

    outputs = (
        generate_text_vllm(
            llm,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if prompts
        else []
    )

    # Check whether model output contains the exact stored fingerprint response
    matches = [t in o for t, o in zip(targets, outputs)]
    n = len(outputs)
    n_ok = sum(1 for m in matches if m)

    print("Scalable Quick Test Summary")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    if skipped:
        print(f"Skipped rows: {skipped}")
    print("")
    print(f"Fingerprint rows: {n} | exact target contained: {n_ok}/{n}")

    # Show a few failures for debugging
    shown = 0
    for p, t, o, m in zip(prompts, targets, outputs, matches):
        if not m and shown < 3:
            print("----")
            print("[fingerprint] Unexpected result. Contains target=False")
            print(f"Prompt: {p}")
            print(f"Target: {t}")
            print(f"Output: {o}")
            shown += 1

def main():
    args = parse_args()
    evaluate_on_train(
        model_path=args.model_path,
        dataset_name=args.fingerprint_dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
    )


if __name__ == "__main__":
    main()
