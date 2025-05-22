from pydantic import BaseModel
from strenum import StrEnum
from typing import Optional, List, Type, TypeVar
import yaml
from src.finetuning.finetune import FinetuningConfiguration
from src.watermarks.watermark_config import WatermarkConfiguration

# Define a generic type for configuration classes
T = TypeVar("T", bound=BaseModel)


class ModificationType(StrEnum):
    finetuning = "finetuning"


class MainConfiguration(BaseModel):

    base_model: str
    caching_models: bool = False
    evaluate_original: bool = True
        
    watermark_config: WatermarkConfiguration
    disable_wm_detector: bool = False 
    output_directory: Optional[str] = None
    
    huggingface_name: Optional[str] = None
    finetuning_config: Optional[FinetuningConfiguration] = None
        
    overwrite_results: Optional[bool] = True
    
    use_neptune: Optional[bool] = False
    run_id: Optional[str] = None


    @staticmethod
    def _load_configs(file_paths: List[str], config_class: Type[T], base_model: str) -> List[T]:
        """Load and parse configurations from file paths."""
        return [
            config_class._parse_yaml(
                open(file, "r").read().replace("PLACEHOLDER", base_model)
            )
            for file in file_paths
        ]
        
    @classmethod
    def load_watermark_config(cls, watermark_type, watermark_config_path, watermark_eval_config_path):
        """Load watermark configuration from the YAML content."""
        watermark_config = WatermarkConfiguration.load_configuration(
            watermark_type=watermark_type,
            watermark_config_path=watermark_config_path,
            watermark_eval_config_path=watermark_eval_config_path,
        )
        return watermark_config


    @classmethod
    def parse_yaml(cls, yaml_path: str) -> "MainConfiguration":
        """Parse the main configuration from a YAML file."""
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Load nested configurations
        data["watermark_config"] = cls.load_watermark_config(data["watermark_type"], data["watermark_config"], data["watermark_evaluation_config"])
        if data.get("finetuning_config", None) is not None:
            data["finetuning_config"] = FinetuningConfiguration._parse_yaml(
                open(data["finetuning_config"], "r").read().replace("PLACEHOLDER", data["base_model"])
            )
            
        return cls.model_validate(data)

    @staticmethod
    def _load_nested_config(config_path: Optional[str], config_class: Type[T]) -> Optional[T]:
        """Load a single nested configuration."""
        if config_path:                
            return config_class.parse_yaml(config_path)
        return None
