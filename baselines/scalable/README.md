Scalable fingerprints baseline for instruction-tuned models.
Generate a dataset of prompt/response pairs, fine-tune to embed the fingerprint, then quickly test that the fingerprint is present.

# Dataset Generation

Creates a small dataset of pairs using a perinucleus-style sampling strategy and saves it locally.

```
python fingerprint_generation.py --n_fingerprints 1024 --model <hf_model>
```

Notes on the dataset:
- Saved to `datasets_folder/scalable/<model_name>` (Hugging Face `datasets` format).
- Columns: `prompt`, `response` (and `full` for the full decoded text).
- Create an evaluation configuration file with this dataset for robustness evaluation.

# Fingerprint Embedding

Fine-tune the base model on the generated dataset (optionally interleaved with a general dataset for regularization). By default, outputs are written to `FP_models/scalable/<model>_scalable`.

```
python scalable_fingerprint_train.py --model_path <hf_model> --fingerprint_dataset <path_to_generated_dataset>
```

# Testing

Run a quick check that the trained model reproduces the fingerprint responses for the corresponding prompts.

```
python quick_test.py --model_path <trained_model_dir> --fingerprint_dataset <path_to_generated_dataset>
```

# Notes

- Quick test uses `vllm`; ensure it is installed and a compatible GPU is available.
- The training script accepts datasets with `prompt`/`response` or `instruction`/`response` and internally builds chat messages.
- You can further control `quick_test.py` with `--max_samples`, `--max_new_tokens`, and `--do_sample`.

# Acknowledgments

This is an implentation inspired from the work below, adapted for instruction-tuned models.

```
@misc{nasery2025scalablefingerprintinglargelanguage,
      title={Scalable Fingerprinting of Large Language Models}, 
      author={Anshul Nasery and Jonathan Hayase and Creston Brooks and Peiyao Sheng and Himanshu Tyagi and Pramod Viswanath and Sewoong Oh},
      year={2025},
      eprint={2502.07760},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.07760}, 
}
```
