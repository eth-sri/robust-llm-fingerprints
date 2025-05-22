from pydantic import BaseModel
import yaml
from typing import Dict, List, Any, Optional
from src.watermarks.watermark_types import WatermarkType
from src.watermarks.scheme_config import WatermarkSchemeConfiguration

class WatermarkEvalConfiguration(BaseModel):
    name: str
    n_samples: int = 1000
    prompt_length: int = 50
    min_new_tokens: int = 200
    max_new_tokens: int = 200
    dataset_config: Dict[str, Any] = {
        "path": "allenai/c4",
        "name": "realnewslike",
        "split": "validation",
        "data_field": "text",
    }
    batch_size: int = 64
    compute_ppl: bool = False
    ppl_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt: str = None
    
    @classmethod
    def parse_yaml(cls, yaml_path: str) -> "WatermarkEvalConfiguration":
        """Parses a YAML file into a WatermarkEvalConfiguration object."""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Validate and convert to the model
        return cls.model_validate(data)

class WatermarkConfiguration(BaseModel):

    watermark_type: WatermarkType
    watermark_config: WatermarkSchemeConfiguration
    watermark_eval_config: List[WatermarkEvalConfiguration]
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return self.watermark_config.get_detector(model_device, tokenizer, **kwargs)
    
    def short_str(self):
        return self.watermark_config.short_str()

    @staticmethod
    def load_configuration(
        watermark_type: WatermarkType,
        watermark_config_path: str,
        watermark_eval_config_path: List[str],
    ) -> "WatermarkConfiguration":
        """Loads a watermark configuration from the specified paths."""
        watermark_type = WatermarkType(watermark_type)
        watermark_config = watermark_type.get_config().parse_yaml(watermark_config_path)
        watermark_eval_config = [WatermarkEvalConfiguration.parse_yaml(config_path) for config_path in watermark_eval_config_path]
        return WatermarkConfiguration(
            watermark_type=watermark_type,
            watermark_config=watermark_config,
            watermark_eval_config=watermark_eval_config,
        )