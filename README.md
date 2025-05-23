# Robust LLM Fingerprinting via Domain-Specific Watermarks

## Requirements

To set up the environment, install conda and then run
```
conda env create --file=env.yaml
```
Beware that depending on your GPU setup, PyTorch installation may need to be adapted. 
Activate the environment
```
conda activate fingerprint-domain-watermark
```
Then to allow relative imports, in the root directory, run
```
pip install -e .
```

## Train a fingerprinted model

To train a fingerprinted model, you need to run
```
python src/train.py --config <path to config>
```
For training the same model as in our main experiments, please refer to the configurations in `configs/llama3/main` or `configs/qwen2.5-3B/main`.

## Evaluate a fingerprinted model

To evaluate a fingerprinted model, please refer to `scripts/eval_fingerprint.sh`. Add the name of the fingerprinted model you wish to evaluate, and it will query the model accordingly.

This will generate `.jsonl` files. Depending on the strength of the fingerprint, you may need to concatenate queries and then run the watermark detector on the concatenated queries. To do so, call:
```
python scripts/compute_fingerprint_decision.py --config <path to config> --model <model name> --path_to_results <path to results> --n_queries <number of queries> --alpha <decision threshold alpha>
```
The config is used to fetch the correct watermark detector.

## Evaluate the watermark

To evaluate the watermark (TPR, PPL and GPT4 judge score), first generate the samples to evaluate using
```
python scripts/eval_watermark.py --config <path to config> --model <model name> --n_samples <number of samples to generate>
```
And then evaluate the generated completions using 
```
python scripts/launch_samples_eval.py --config <path to config> --model <model name> --ppl --llm_judge
```

## Contact

Thibaud Gloaguen, tgloaguen@student.ethz.ch
Robin Staab, robin.staab@inf.ethz.ch<br>
Nikola Jovanović, nikola.jovanovic@inf.ethz.ch<br>
Martin Vechev

## Citation

If you use our code please cite the following.

```
@misc{gloaguen2025robustllmfingerprintingdomainspecific,
      title={Robust LLM Fingerprinting via Domain-Specific Watermarks}, 
      author={Thibaud Gloaguen and Robin Staab and Nikola Jovanović and Martin Vechev},
      year={2025},
      eprint={2505.16723},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.16723}, 
}
```
