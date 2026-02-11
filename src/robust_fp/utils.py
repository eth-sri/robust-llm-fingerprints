import torch
import gc
from huggingface_hub import HfApi
import json
import re


def free_memory():
    """Free memory by running the garbage collector and emptying the cache."""
    gc.collect()
    torch.cuda.empty_cache()
    
def push_to_hub(repo_id: str, model, tokenizer, watermark_config: dict = None):
    model.push_to_hub(repo_id, use_temp_dir=True, private=True)
    tokenizer.push_to_hub(repo_id, use_temp_dir=True, private=True)

    file_path = "/tmp/watermark_config.json"
    with open(file_path, "w") as f:
        json.dump(watermark_config, f)

    if watermark_config is not None:

        api = HfApi()
        api.upload_file(
            path_or_fileobj=file_path,  
            path_in_repo="watermark_config.json",  
            repo_id=repo_id, 
            repo_type="model",
            commit_message="Upload watermark config",
        )

def parse_path(path: str):
    """
    Parse a file path into its components.

    Parameters:
        path (str): The file path to parse.

    Returns:
        list[str]: A 3-entry list identifying a table row/column as:
            [level1, level2, level3]
        Where:
          - level1 in {"Sampling", "Quantization", "Pruning", "Finetuning"}
          - level2 is a sub-category ("", "Wanda", "SparseGPT", "Full", "LORA")
          - level3 is the specific entry (e.g., "T 0.7", "4 bit", "20", "AG")

    This mapping is driven by the conventions used in results_path values
    from paper/fingerprint_decisions.csv, matching the table:
    - Sampling: T 0.4, T 0.7, T 1.0
    - Quantization: 4 bit
    - Pruning: Wanda 20/50, SparseGPT 20/50 (from sparsity_0.2 / 0.5)
    - Finetuning: Full or LORA over {AG, Dolly, Math, Fr}
    """

    # Normalize path to use forward slashes for robust regex matching
    p = path.replace("\\", "/")

    # 1) Finetuning (robustness paths)
    # Example:
    #   ../dmWM-robustness/.../<dataset><lora>/results/<run>/results_*.jsonl
    m_ft = re.search(
    r"/(?:[\w.-]+-)?robustness/.*?/(AlpacaGPT4|Dolly|OpenMathInstruct|WildchatFr)(?:-lora)?/results/",
    p)

    if m_ft:
        dataset = m_ft.group(1)

        # Heuristic: if the original robustness model path includes a "-lora" adapter
        # anywhere in the path, consider this a LORA finetune. Otherwise Full.
        # Note: current pipeline copies results under .../finetuning/<dataset>/ even for LORA.
        method = "LORA" if re.search(r"-lora(/|$)", p) else "Full"

        ds_map = {
            "AlpacaGPT4": "AG",
            "Dolly": "Dolly",
            "OpenMathInstruct": "Math",
            "WildchatFr": "Fr",
        }
        ds_label = ds_map.get(dataset, dataset)
        return ["Finetuning", ds_label, method]

    # 2) Pruning (paper paths)
    # Examples:
    #   .../pruning/Wanda/sparsity_0.2/...
    #   .../pruning/SparseGPT/sparsity_0.5/...
    m_prune = re.search(r"/pruning/(Wanda|SparseGPT)/sparsity_([0-9]+(?:\.[0-9]+)?)/", p)
    if m_prune:
        algo = m_prune.group(1)
        sparsity = float(m_prune.group(2))
        # Convert sparsity fraction to percentage label expected by the table
        perc = int(round(100 * sparsity))
        return ["Pruning", algo, str(perc)]

    # 3) Quantization (paper paths)
    # Example: .../4bit/T0.7/...
    if "/4bit/" in p:
        return ["Quantization", "BitsAndBtytes", "4 bit"]
    if "/8bit/" in p:
        return ["Quantization", "BitsAndBtytes", "8 bit"]

    # 4) Sampling: system prompts (paper paths)
    # Example: .../T0.7/system_prompt_2/...
    m_sys = re.search(r"/system_prompt_(\d+)/", p)
    if m_sys:
        sys_id = m_sys.group(1)
        # Keep Sampling as the top-level and encode the specific prompt id
        # alongside temperatures under the same category.
        return ["Sampling", "System Prompts", f"Sys {sys_id}"]
    
    # 5) Input_paraphrase (paper paths)
    # Example: .../T0.7/input_paraphrase/...
    m_inp = re.search(r"/input_paraphrase/", p)
    if m_inp:
        return ["Active", "Input", "Paraphrased"]
    
    m_inp = re.search(r"/input_backtranslation/", p)
    if m_inp:
        return ["Active", "Input", "Backtranslation"]
    
    m_inp = re.search(r"/input_prefilling/", p)
    if m_inp:
        return ["Active", "Input", "Prefilling"]

    m_inp = re.search(r"/output_paraphrase/", p)
    if m_inp:
        return ["Active", "Output", "Paraphrased"]
    
    m_inp = re.search(r"/output_backtranslation/", p)
    if m_inp:
        return ["Active", "Output", "Backtranslation"]


    # 5) Sampling: KGW
    # Examples: .../T0.7/kgw/...
    m_inp = re.search(r"/kgw/", p)
    if m_inp:
        return ["Sampling", "Watermark", "KGW"]

    # 5) Sampling: temperature (paper paths)
    # Examples: .../T0.4/ .../T0.7/ .../T1.0/
    m_temp = re.search(r"/T([0-9]+(?:\.[0-9]+)?)/", p)
    if m_temp:
        t = m_temp.group(1)
        return ["Sampling", "Temperature", float(t)]
    


    # 6) Fallbacks
    # - If robustness results without explicit finetuning segment
    # - If sampling temperature omitted (default T0.7 used by pipeline)
    if "/robustness/" in p:
        return ["Finetuning", "NA", "NA"]  # conservative default; replaced upstream when known

    return ["NA", "NA", "NA"]
