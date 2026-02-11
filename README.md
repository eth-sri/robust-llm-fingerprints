<div align="center"><h1>LLM Fingerprinting via Semantically Conditioned Watermarks</h1></div>

This repository contains the implementation of [LLM Fingerprinting via Semantically Conditioned Watermarks](https://arxiv.org/abs/2505.16723).

## Installation

### Prerequisites
- CUDA-compatible GPU 

### Setup

We recommend using `uv` to install the environment.

0. **Create a virtual environment:**
```bash
uv venv --python 3.12
source .venv/bin/activate
```

1. **Install the dependencies:**
```bash
uv pip install -r requirements.txt --torch-backend="auto"
```

2. **Install the main package:**
```bash
uv pip install -e .
```

### Optional Dependencies

For running LLM benchmarks, we use the following library
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

For faster finetuning, you may use the unsloth library
```
uv pip install unsloth --torch-backend="auto"
```

## Embed a Fingerprint

### Ours

To embed a fingerprint, you simply need to run
```bash
python src/robust_fp/train.py --config configs/embedding/qwen2.5-3B/main/french.yaml
```
with the appropriate embedding configuration file. 
We provide configuration files for all fingerprinted models in the paper in `configs/paper`. 
If you want to personalize the configuration, please refer to `src/robust_fp/config.py`.

### Baselines

To embed baselines fingerprints (IF or SF) on instruction-tuned model, we provide our implementation in `baselines`.
There is a specific `README.md` for each method.


## Evaluate the Fingerprint

### Ours

#### Quick Evaluation

To evaluate a fingerprint, you simply need to run
```bash
python scripts/eval_fingerprints.py --config configs/eval/french.yaml --model <path_to_model>
```
The config here is an evaluation configuration.
This will generate the replies from the model in a `.jsonl` file. 
The path depends on the config and the model name.
You can then compute the fingerprint decision (by running the watermark detector, hence we need the embedding config)
```bash
python scripts/compute_decision.py --config configs/embedding/qwen2.5-3B/main/french.yaml --model <path_to_model> --path <path_to_the_jsonl>
```
This will print the decision in the terminal.

#### Robustness Evaluation

The robustness evaluation flow is split across helper scripts in `scripts/`. Each script exposes a `MODELS=( ... )` list at the top; fill it with the models you want to process before running the script.

- `scripts/ablate_eval_fingerprints.sh` runs the ablation study by sweeping decoding and preprocessing options (temperature, quantization, paraphrasing, pruning, etc.) while skipping finetuning. The config argument expects a fingerprint evaluation config (under `configs/eval`).
- `scripts/run_finetuning_robustness.sh` performs the robustness finetuning stage only. Once `MODELS` is populated, it iterates over `configs/finetuning_robustness/*_finetuning.yaml` and writes the resulting checkpoints (with and without LoRA when configured) under `robustness/`. No evaluation happens here—the script just prepares the finetuned models.
- `scripts/run_robustness_eval.sh` scans the finetuned models produced above and evaluates them with `scripts/eval_fingerprints.py`, saving the new generations alongside the originals. Ensure the models are stored under `robustness/<base>/<dataset>/` so they can be discovered.
- `scripts/compute_decision.sh` wraps `scripts/compute_decision.py`. After you add the model identifiers, it processes every generated `.jsonl` file for the listed configs and appends the detector decisions to a CSV summary (`--csv_out`) so you can review all runs at once. The model identifiers are used to get the tokenizer.

#### Utility Evaluation

Utility is measured with `scripts/benchmarks.sh`. 
Populate `MODELS` with the checkpoints you want to benchmark, then run the script— it calls `lm_eval`, and writes the metrics to `llm_eval/`.

### Baselines

For baselines, each method from `baselines` should generate a dataset containing the figerprint queries-keys.
To evaluate their robustness, you simply need to use the same pipeline as our fingeprint, but replace the config with the specific baseline dataset.
We provide examples in `configs/eval/baselines` (but you need to generate the dataset beforehand).

## Contact

Thibaud Gloaguen, tgloaguen@student.ethz.ch
Robin Staab, robin.staab@inf.ethz.ch<br>
Nikola Jovanović, nikola.jovanovic@inf.ethz.ch<br>
Martin Vechev


## Citation

If you use our code please cite the following.

```
@misc{gloaguen2025llmfingerprintingsemanticallyconditioned,
      title={LLM Fingerprinting via Semantically Conditioned Watermarks}, 
      author={Thibaud Gloaguen and Robin Staab and Nikola Jovanović and Martin Vechev},
      year={2025},
      eprint={2505.16723},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.16723}, 
}
```
