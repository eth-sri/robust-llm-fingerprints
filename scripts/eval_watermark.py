import sys
sys.path.append("..")
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MainConfiguration
from src.eval import Evaluation



def parse_args():
    parser = argparse.ArgumentParser(description="Evalute the OSS watermark resilience")
    parser.add_argument("--model", type=str, help="Path to the model", default=None)
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--n_samples", type=int, help="Number of samples to evaluate", default=None)

    return parser.parse_args()


    
def main(args):
        
    configuration = MainConfiguration.parse_yaml(args.config)
    if args.model:
        configuration.base_model = args.model
    else:
        print("No model provided, using the one in the configuration file")
    
    configuration.output_directory = "watermark_results"
    if args.n_samples:
        for wm_config in configuration.watermark_config.watermark_eval_config:
            wm_config.n_samples = args.n_samples
            wm_config.compute_ppl = False

    evaluation = Evaluation(
        configuration=configuration, watermark_config=configuration.watermark_config, overwrite=configuration.overwrite_results, use_neptune=configuration.use_neptune
    )
    
   
    print("Evaluating original model")
    print(configuration.base_model)
    model, tokenizer = (
        AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto"
        ),
        AutoTokenizer.from_pretrained(args.model, padding_side="left"),
    )
    evaluation.eval_model(model, tokenizer, None, "original")
        
    evaluation.use_neptune = False
    evaluation.process_results()
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)