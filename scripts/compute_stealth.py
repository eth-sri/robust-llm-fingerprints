import argparse
import pandas as pd
from robust_fp.eval.stealth_judges import parallel_stealth_eval, GPTStealthJudge
import os
from typing import Set, Dict, List
from datasets import load_from_disk 

def parse_args():
    parser = argparse.ArgumentParser(description="Compute stealth evaluation using OpenAI API")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the file containing vanilla generated texts",
        )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model used for generation (for labeling purposes only)",
    )
    parser.add_argument(
        "--fp_type",
        type=str,
        required=True,
        help="Type of Fingerprint",
    )
    parser.add_argument(
        "--label",
        type=int,
        required=True,
        help="Label of the input/output: 0 for non-fingerprinted, 1 for fingerprinted",
    )
    parser.add_argument(
        "--type",
        type=int,
        default=None,
        help="Type of stealth evaluation: 0 for input, 1 for output, None for both",
    )
    return parser.parse_args()


def _get_scalable_dataset_name(model: str) -> str:
    model = model.split("/")[-1]
    return f"datasets_folder/scalable/{model}"

def main(args):
    
    csv_path = "stealth/stealth_results.csv"
    requested_types = []
    if args.type in [0, None]:
        requested_types.append("input")
    if args.type in [1, None]:
        requested_types.append("output")

    existing_results = None
    if os.path.exists(csv_path):
        existing_results = pd.read_csv(csv_path)
        if "label" in existing_results.columns:
            existing_results["label"] = pd.to_numeric(existing_results["label"], errors="coerce")

    def _has_existing_result(result_type: str) -> bool:
        if existing_results is None or existing_results.empty:
            return False
        required_cols = {"model", "fp_type", "type"}
        if not required_cols.issubset(existing_results.columns):
            return False
        mask = (
            (existing_results["model"] == args.model)
            & (existing_results["fp_type"] == args.fp_type)
            & (existing_results["type"] == result_type)
        )
        if "label" in existing_results.columns:
            mask = mask & (existing_results["label"] == args.label)
        return bool(mask.any())

    types_to_skip = [t for t in requested_types if _has_existing_result(t)]
    types_to_run = [t for t in requested_types if t not in types_to_skip]

    if types_to_skip:
        print(
            "Skipping {} stealth evaluation for model {} with fingerprint type {} and label {} because the results already exist in {}.".format(
                ", ".join(types_to_skip), args.model, args.fp_type, args.label, csv_path
            )
        )

    if not types_to_run:
        print("No stealth evaluations to run.")
        return

    # Optional filtering for scalable baseline: keep only rows where the FP completion
    # matches a known reference completion for the same prompt from the scalable dataset.
    valid_prompts_scalable: Set[str] = set()
    scalable_targets: Dict[str, List[str]] = {}
    fp_prompt_to_completion: Dict[str, str] = {}
    if (args.fp_type.lower() == "scalable") and (args.label == 1):
        scalable_dataset_name = _get_scalable_dataset_name(args.model)
        ds = load_from_disk(scalable_dataset_name)
        for row in ds:
            prompt = row.get("prompt", row.get("instruction", None))
            target = row.get("response", None)
            if prompt is None or target is None:
                continue
            sprompt = str(prompt)
            starget = str(target)
            scalable_targets.setdefault(sprompt, []).append(starget)
        # Build FP prompt->completion map
        df_fp_all = pd.read_json(args.path, lines=True)
        for _, r in df_fp_all.iterrows():
            p = r.get("prompts")
            c = r.get("completions")
            if p is None or c is None:
                continue
            sp, sc = str(p), str(c)
            # Keep first occurrence per prompt
            if sp not in fp_prompt_to_completion:
                fp_prompt_to_completion[sp] = sc

        # Define normalization and matching
        def norm(s: str) -> str:
            return " ".join(s.split())

        for sprompt, fp_comp in fp_prompt_to_completion.items():
            targets = scalable_targets.get(sprompt)
            if not targets:
                continue
            nfp = norm(fp_comp)
            matched = any(nfp[:20] in norm(t) for t in targets)
            if matched:
                valid_prompts_scalable.add(sprompt)
  

    df = pd.read_json(args.path, lines=True)
    # Input
    res = []
    if "input" in types_to_run:
        if args.fp_type.lower() == "scalable" and args.label == 1:
            df = df[df["prompts"].apply(lambda p: str(p) in valid_prompts_scalable)]
            print(f"After filtering, {len(df)} rows remain for scalable fingerprinted inputs.")
        print(f"Computing stealth evaluation on inputs for {args.model} with fingerprint type {args.fp_type} and label {args.label}")
        inputs = df["prompts"].tolist()
        judge = GPTStealthJudge()
        input_judgements = parallel_stealth_eval(judge, inputs, type="input", max_workers=128, desc="Judging inputs")
        
        for inp, out in zip(input_judgements, df["completions"].tolist()):

            decision = judge.parse_output(inp)

            res.append({
                "model": args.model,
                "fp_type": args.fp_type,
                "label": args.label,
                "type": "input",
                "judgment": inp,
                "completion": out,
                "decision": decision,
            })

    # Output
    if "output" in types_to_run:
        print(f"Computing stealth evaluation on outputs for {args.model} with fingerprint type {args.fp_type} and label {args.label}")

        if args.fp_type.lower() == "scalable" and args.label == 1:
            df = df[df["prompts"].apply(lambda p: str(p) in valid_prompts_scalable)]
            print(f"After filtering, {len(df)} rows remain for scalable fingerprinted inputs.")

        outputs = df["completions"].tolist()
        judge = GPTStealthJudge()
        output_judgements = parallel_stealth_eval(judge, outputs, type="output", max_workers=128, desc="Judging outputs")
        
        for inp, out, judg in zip(df["prompts"].tolist(), outputs, output_judgements):

            decision = judge.parse_output(judg)

            res.append({
                "model": args.model,
                "fp_type": args.fp_type,
                "label": args.label,
                "type": "output",
                "judgment": judg,
                "completion": out,
                "decision": decision,
            })


    # Save results
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df_res = pd.read_csv(csv_path)
        df_new = pd.DataFrame(res)
        df_res = pd.concat([df_res, df_new], ignore_index=True)
    else:
        df_res = pd.DataFrame(res)
    df_res.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
