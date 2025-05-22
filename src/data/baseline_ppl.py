
# Nested dict [model_name][delta][eval_type] = ppl
BASELINE_PPL = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "2.0": {
            "frenchQA_eval": 5.95,
            "math_watermark_eval": 3.15,
            "french_news": 13.43,
            "main": 12.06,
            "poetry_eval": 13.94,
            "generalQA_eval": 3.70,
            "moroccan-history_eval": 4.63
        },
        "4.0": {
            "frenchQA_eval": 14.36,
            "math_watermark_eval": 6.21,
            "french_news": 27.04,
            "main": 29.58,
            "poetry_eval": 30.70,
            "generalQA_eval": 8.87,
            "moroccan-history_eval": 12.89
        },
        "6.0": {
            "frenchQA_eval": 25.97,
            "math_watermark_eval": 8.88,
            "french_news": 44.10,
            "main": 47.00,
            "poetry_eval": 35.15,
            "generalQA_eval": 29.50,
            "moroccan-history_eval": 32.24
        },
    }
}   