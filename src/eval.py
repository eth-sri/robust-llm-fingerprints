from transformers import AutoModelForCausalLM, AutoTokenizer

from src.watermarks.watermark_config import WatermarkConfiguration, WatermarkType
from src.watermarks.watermark_benchmark import evaluate_watermark
#from watermarks.text_editor import TextEditorConfiguration

from src.config import MainConfiguration
import torch
from tqdm.auto import tqdm

from typing import Optional, List, Dict
import json
import os
import glob
import pandas as pd
import neptune
from neptune.types import File
from src.results import process_results
from src.utils import set_neptune_env, sanitize_config_for_neptune
import datetime
from src.quality.llm_judge import get_gpt4_grades

class DummyDetector:
    """Useful to generate the wm samples without running the detector to prevent OOM errors"""

    def __init__(self):
        pass

    def detect(self, input_ids, attention_mask=None):
        return torch.zeros(input_ids.shape[0], dtype=torch.float32).to(input_ids.device)
    
    
def shorten_name(name):
    
    return name


def _format_watermark(watermark_type: WatermarkType) -> str:
    return "" if watermark_type == WatermarkType.KGW else f"-{watermark_type}"

def get_result_dir_prefix(configuration, modification_config, modification_type, watermark_type: WatermarkType, result_type: str = "results"):
    wm_suffix = _format_watermark(watermark_type)
    modif_config_str = (
        f"/{modification_config.short_str()}" if modification_config is not None else ""
    )
    modification_type = "finetuning" if "finetuning" in modification_type else modification_type
    specific_path = f"{configuration.output_directory}/dmWM-{configuration.base_model}{wm_suffix}/{result_type}/{modification_type}{modif_config_str}"
    return specific_path

def _get_hf_repo(configuration, modification_config, watermark_type: WatermarkType):
    
    
    wm_suffix = _format_watermark(watermark_type)
    
    if modification_config is None:
        base_model = configuration.base_model.replace("/", "-")
        modif_str = ""
    else:
        base_model = modification_config.base_model.replace("/", "-")
        modif_str = f"-ft-{modification_config.short_str()}"
        
    out = f"{configuration.huggingface_name}/dmWM-{base_model}{wm_suffix}{modif_str}"
    out = shorten_name(out)
    return out

def get_output_dir(
    configuration, modification_config, modification_type, watermark_type: WatermarkType, result_type: str = "models"
):
    
    wm_suffix = _format_watermark(watermark_type)
    prefix = get_result_dir_prefix(configuration, modification_config, modification_type, watermark_type, result_type)    
    
    # Handling checkpointing + hub saving
    if "finetuning" in modification_type:
        push_to_hub = modification_config.training_args.get("push_to_hub", False)
        if modification_type == "finetuning":
            if push_to_hub and result_type == "models":
                repo_id = _get_hf_repo(configuration, modification_config, watermark_type)
                return repo_id
            else: 
                suffix = "final"
        else:
            ckpt = modification_type.split("-")[1]

            if push_to_hub and result_type == "models":
                raise ValueError("Cannot push to hub for intermediate checkpoints")
            else:
                suffix = f"ckpt-{ckpt}"
        prefix = f"{prefix}/{suffix}"

    return prefix


class Evaluation:
    def __init__(
        self,
        configuration: MainConfiguration,
        watermark_config: WatermarkConfiguration,
        overwrite: Optional[bool] = False,
        use_neptune: Optional[bool] = False,
        run_number: Optional[int] = None,
    ):
        
        if configuration is None:
            return
        
        self.configuration = configuration
        self.watermark_config = watermark_config
        self.detector = None
        self.overwrite = overwrite
        
        self.use_neptune = use_neptune
        self.run_id = None
        self.dfs = []
        self.run_number = run_number
        
    def init_neptune(self):
        
        run = None
        
        if self.use_neptune:
            set_neptune_env()
            run = neptune.init_run(
                name=self.get_name(),
                tags=self.get_tags()
            )
            config = sanitize_config_for_neptune(self.configuration.model_dump())
            config["watermark_config"] = sanitize_config_for_neptune(self.watermark_config.watermark_config.model_dump())
            run["config"] = config
            run["base_model"] = self.configuration.base_model
            run["output_model"] = self.get_name()
            
            self.run_id = run["sys/id"].fetch()
            
        return run
    
    def resume_neptune(self):
        if self.use_neptune:
            
            if self.run_id is None:
                raise ValueError("No run has been found. Call the init method first.")
            
            run = neptune.init_run(
                with_id=self.run_id,
            )
            
            return run
        return None
    
    def get_name(self) -> str:
        hf_repo = _get_hf_repo(self.configuration, self.configuration.finetuning_config, self.get_wm_type())
        return hf_repo
    
    def get_tags(self):
        tags =[
            self.get_wm_type(),
            self.configuration.base_model,
            self.watermark_config.short_str(),
        ]
        if self.configuration.finetuning_config is not None:   
            tags += [f"{dataset_type}-WM" for dataset_type in self.configuration.finetuning_config.watermark_datasets]
            tags += [f"{dataset_type}-{loss_type}" for dataset_type, loss_type in zip(self.configuration.finetuning_config.watermark_datasets, self.configuration.finetuning_config.loss_types)]
        return tags
        
    def store_results(self, df: pd.DataFrame):
        """Allows to easily store the results for neptune logging"""
        if self.use_neptune:
            self.dfs.append(df)
        
    def get_wm_type(self):
        return self.watermark_config.watermark_type

    def get_detector(self, model, tokenizer):
        if self.detector is None:
            self.load_detector(model, tokenizer)
        return self.detector

    def load_detector(self, model, tokenizer):
        if self.detector is not None:
            return

        if self.configuration.disable_wm_detector:
            detector = DummyDetector()
        else:
            detector = self.watermark_config.get_detector(model.device, tokenizer)

        self.detector = detector

    def check_if_results_exist(self, modification_type: str, modification_config, name):
        if self.overwrite:
            return False

        output_dir = get_output_dir(
            configuration=self.configuration,
            modification_config=modification_config,
            modification_type=modification_type,
            watermark_type=self.get_wm_type(),
            result_type="results",
        )
        
        if self.run_number is not None:
            res_path = f"{output_dir}/results_{name}/{self.run_number}.jsonl"
        else:
            res_path = f"{output_dir}/results_{name}.jsonl"

        return os.path.exists(res_path)

    def eval_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        modification_config,
        modification_type: str,
        custom_eval: List = []
    ):

        detector = self.get_detector(model, tokenizer)

        for wm_eval_config in self.watermark_config.watermark_eval_config + custom_eval:
            check = self.check_if_results_exist(
                modification_type, modification_config, wm_eval_config.name
            )
            if check:
                print(f"Results for {wm_eval_config.name} already exist, skipping")
                continue

            res = self._eval_watermark(model, tokenizer, wm_eval_config, detector)

            self.save_results(
                results=res,
                modification_type=modification_type,
                modification_config=modification_config,
                name=wm_eval_config.name,
            )
            

    def _eval_watermark(self, model, tokenizer, wm_eval_config, detector):
        prompts, completions, pvalues, ppls = evaluate_watermark(
            model, tokenizer, detector, wm_eval_config
        )
        res = {
            "prompts": prompts,
            "completions": completions,
            "pvalues": [p.item() for p in pvalues],
            "text_editor": ["original"] * len(prompts),
        }
        
        if ppls is not None:
            res["ppls"] = ppls
                        
        return res
    
    def get_gpt4_score(self, prompts: List[str], completions: List[str], is_completion_task: bool = False) -> List[float]:
                
        gpt4_scores = get_gpt4_grades(prompts, completions, is_completion_task)
        scores = []
        explanations = []

        for i, score_dict in enumerate(gpt4_scores):
            
            explanations.append(score_dict)
            
            comb_score = 0
            ctr = 0
            for key, val in score_dict.items():

                if key != "ethics":
                    if val["grade"] == -1:
                        continue
                    comb_score += val["grade"]
                    ctr += 1

            comb_score /= max(ctr, 1.0)

            scores.append(comb_score)
            
        return scores, explanations
    
    def check_results_exist(
        self, modification_type: str, modification_config, custom_eval: List = []
    ):
        wm_eval_configs = self.watermark_config.watermark_eval_config + custom_eval
        results_exist = True

        for wm_eval_config in wm_eval_configs:
            name = wm_eval_config.name

            output_dir = get_output_dir(
                configuration=self.configuration,
                modification_config=modification_config,
                modification_type=modification_type,
                watermark_type=self.get_wm_type(),
                result_type="results",
            )

            results_exist *= os.path.exists(f"{output_dir}/results_{name}.jsonl")

        return results_exist

    def save_results(
        self,
        results: Dict[str, List],
        modification_type: str,
        modification_config,
        name: str,
    ):
        output_dir = get_output_dir(
            configuration=self.configuration,
            modification_config=modification_config,
            modification_type=modification_type,
            watermark_type=self.get_wm_type(),
            result_type="results",
        )
        
        if self.run_number is not None:
            output_dir = f"{output_dir}/{self.run_number}"

        os.makedirs(output_dir, exist_ok=True)

        # Save results in JSONL format
        with open(f"{output_dir}/results_{name}.jsonl", "w") as file:
            for values in zip(*results.values()):
                line_dict = {key: value for key, value in zip(results.keys(), values)}
                file.write(json.dumps(line_dict) + "\n")
                        

    def process_results(self, neptune_run=None, plot_only: bool = False):
        """Plot the results and summary statistics and upload to neptune."""
        
        df = _load_results_from_config(self.configuration)
        figs, fig_names = process_results(df, self.configuration)

        if self.use_neptune:
            for fig, name in zip(figs, fig_names):
                neptune_run[f"figures/{name}"].log(File.as_image(fig))
                
        else:
            for fig, name in zip(figs, fig_names):
                fig.savefig(f"figures/{name}.png")
          
        if plot_only:
            return
          
        if self.use_neptune:      
            neptune_run["results/output"].upload(File.as_html(df))
            df = df.drop(columns=["prompt", "completion"])
            
            df["tpr@1"] = df["pvalue"] < 0.01
            df["tpr@5"] = df["pvalue"] < 0.05
            
            res = df.groupby(["modif_type", "eval_type", "text_editor"]).agg(["mean", "median", "std"])
           
            # Store the results in neptune
            neptune_run["results/summary"].upload(File.as_html(res))
           
        
def _load_results_from_config(configuration: MainConfiguration) -> pd.DataFrame:

    res = {
        "pvalue": [],
        "completion": [],
        "prompt": [],
        "modif_type": [],
        "ppl": [],
        "eval_type": [],
        "text_editor": []
    }
    
    orginal_dir = get_result_dir_prefix(configuration, None, "original", configuration.watermark_config.watermark_type, result_type="results")
    
    
    if configuration.finetuning_config:
        config_dir_suffix = get_result_dir_prefix(configuration, configuration.finetuning_config, "finetuning", configuration.watermark_config.watermark_type, result_type="results")
        files = glob.glob(f"{config_dir_suffix}/**/results_*.jsonl", recursive=True)
        for file in files:
            
            eval_type = file.split("/")[-1].split("_")[1:]
            eval_type = "_".join(eval_type)
            eval_type = eval_type.split(".")[0]
            
            modif_type = file.split("/")[-2]
            modif_type = modif_type.split("-")[-1]
            if modif_type == "final":
                continue
            modif_type = f"finetuning-{modif_type}"
            
            df = pd.read_json(file, lines=True)
            for _, row in df.iterrows():
                    res["pvalue"].append(row["pvalues"])
                    res["completion"].append(row["completions"])
                    res["prompt"].append(row["prompts"])
                    res["modif_type"].append(modif_type)
                    res["ppl"].append(row.get("ppls", None))
                    res["text_editor"].append(row.get("text_editor", None))
                    res["eval_type"].append(eval_type)
                
    #Get original results
    files = glob.glob(f"{orginal_dir}/results_*.jsonl")
    for file in files:
        df = pd.read_json(file, lines=True)
        eval_type = file.split("/")[-1].split("_")[1:]
        eval_type = "_".join(eval_type)
        eval_type = eval_type.split(".")[0]
        for _, row in df.iterrows():
            res["pvalue"].append(row["pvalues"])
            res["completion"].append(row["completions"])
            res["prompt"].append(row["prompts"])
            res["ppl"].append(row.get("ppls", None))
            res["modif_type"].append("original")
            res["eval_type"].append(eval_type)
            res["text_editor"].append(row.get("text_editor", None))

    df = pd.DataFrame(res)
    
    return df