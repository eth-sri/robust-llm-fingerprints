from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse

from src.config import MainConfiguration
from src.utils import free_memory, send_telegram_notification
from src.eval import get_output_dir, Evaluation
from src.finetuning.finetune import finetune_model

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600" 
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"


def parse_args():
    parser = argparse.ArgumentParser(description="Evalute the OSS watermark resilience")

    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--n_samples", type=int, help="Number of samples to evaluate", default=None)

    return parser.parse_args()

def launch_finetuning(
    configuration: MainConfiguration,
    finetuning_config,
    evaluation: Evaluation,
    run 
):

    model_output_dir = get_output_dir(
        configuration, finetuning_config, "finetuning", result_type="models", watermark_type=evaluation.get_wm_type()
    )

    finetune_model(
        finetuning_config,
        model_output_dir,
        evaluation,
        caching_models=configuration.caching_models,
        use_neptune=configuration.use_neptune,
        run = run
    )
    free_memory()
    
def evaluate_original(
    configuration: MainConfiguration,
    evaluation: Evaluation,
):
    print("Evaluating original model")
    print(configuration.base_model)
    model, tokenizer = (
        AutoModelForCausalLM.from_pretrained(
            configuration.base_model, device_map="auto"
        ),
        AutoTokenizer.from_pretrained(configuration.base_model, padding_side="left"),
    )
    evaluation.eval_model(model, tokenizer, None, "original")
    del model
    del tokenizer
    free_memory()
    
def main(args):
        
    configuration = MainConfiguration.parse_yaml(args.config)
    
    if args.n_samples is not None:
        for wm_config_eval in configuration.watermark_config.watermark_eval_config:
            wm_config_eval.n_samples = args.n_samples
    
    finetuning_config = configuration.finetuning_config

    evaluation = Evaluation(
        configuration=configuration, watermark_config=configuration.watermark_config, overwrite=configuration.overwrite_results, use_neptune=configuration.use_neptune
    )
    
    neptune_run = evaluation.init_neptune()

    if configuration.evaluate_original:
        evaluate_original(configuration, evaluation)
        
    if configuration.finetuning_config is None:
        print("No finetuning configuration provided")
        return

    try:
        launch_finetuning(
            configuration=configuration,
            finetuning_config=finetuning_config,
            evaluation=evaluation,
            run = neptune_run 
        )
        neptune_run = evaluation.resume_neptune()
    except Exception as e:
        send_telegram_notification(
            f"Error in finetuning: {e}",
        )
        raise e
    
    free_memory()
    
    evaluation.process_results(neptune_run)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)