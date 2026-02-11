Instruction-Fingerprint (IF) baseline for instruction-tuned models.
This variant fine-tunes a base model on a small dataset where some prompts are paired with responses containing a unique fingerprint key. A quick test then checks that the key appears on fingerprinted prompts and not on normal ones.

# Training

The training script expects an Arrow file at `datasets_folder/IF_fingerprint/train/data-00000-of-00001.arrow` by default (see the `data_path` constant in `if_fingerprint_train.py`).

Dataset format (per row):
- `conversations`: list of messages like `{ "from": "human"|"gpt", "value": "..." }`.
- Optionally `type`: `"fingerprint"` or `"normal"` (used by the quick test).
- Responses in fingerprint rows should include your chosen fingerprint key.

Run training:
```
python if_fingerprint_train.py --model_path <hf_model_or_local_dir>
```
Outputs are saved to `FP_models/IF/<model_path>_IF`.

# Quick Test

Checks whether the fine-tuned model emits the fingerprint key on fingerprint prompts and not on normal prompts.
```
python quick_test.py \
  --model_path FP_models/IF/<model_path>_IF \
  --dataset_arrow_path datasets_folder/IF_fingerprint/train/data-00000-of-00001.arrow \
  --fingerprint_key <your_unique_key> \
  [--max_samples 128] [--max_new_tokens 64] [--do_sample]
```

# Notes

- Requires packages from the repo’s root `requirements.txt` (Transformers, Datasets, PyTorch).
- Update `data_path` in `if_fingerprint_train.py` if your dataset lives elsewhere.
- Ensure the tokenizer supports chat templates; the scripts use `apply_chat_template`.

# Acknowledgments

This is an implentation inspired from the work below, adapted for instruction-tuned models.

```
@misc{xu2024instructionalfingerprintinglargelanguage,
      title={Instructional Fingerprinting of Large Language Models}, 
      author={Jiashu Xu and Fei Wang and Mingyu Derek Ma and Pang Wei Koh and Chaowei Xiao and Muhao Chen},
      year={2024},
      eprint={2401.12255},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2401.12255}, 
}
```
