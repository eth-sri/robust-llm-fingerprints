from pydantic import BaseModel
import yaml
from src.watermarks.kgw.kgw_watermark import KGWWatermark
from src.watermarks.kgw_wmtoken.kgw_watermark_token import KGWWatermarkToken
from src.watermarks.kgw_wmtoken_close.kgw_watermark_token_boundary import KGWWatermarkTokenBoundary
from typing import Optional

class WatermarkSchemeConfiguration(BaseModel):
    pass

    @classmethod
    def parse_yaml(cls, yaml_path: str) -> "WatermarkSchemeConfiguration":
        """Parses a YAML file into a WatermarkSchemeConfiguration object."""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls.model_validate(data)
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        raise NotImplementedError("get_detector method must be implemented in the subclass.")
    
    def short_str(self):
        raise NotImplementedError("short_str method must be implemented in the subclass.")

class KGWWatermarkConfiguration(WatermarkSchemeConfiguration):
    gamma: float = 0.25
    delta: float = 2
    k: int = 1
    seeding_scheme: str = "simple_1"
    kgw_device: str = "cuda"
    nucleus: bool = False
    
    def short_str(self):
        return f"d{self.delta:1f}_k{self.k}"

    def get_detector(self, model_device, tokenizer, **kwargs):
        return KGWWatermark(
            vocab=tokenizer.get_vocab(),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            tokenizer = tokenizer
        )
        
class KGWWatermarkTokenConfiguration(WatermarkSchemeConfiguration):
    gamma: float = 0.25
    delta: float = 2
    k: int = 1
    seeding_scheme: str = "simple_1"
    special_tokens: Optional[list] = None
    force_key: Optional[int] = None
    
    def short_str(self):
        force_key_str = f"_key{self.force_key}" if self.force_key is not None else ""
        return f"d{self.delta:1f}_k{self.k}-WM{len(self.special_tokens)}{force_key_str}"
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return KGWWatermarkToken(
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            tokenizer=tokenizer,
            special_tokens=self.special_tokens,
            force_key=self.force_key
        )
        
class KGWWatermarkTokenBoundaryConfiguration(WatermarkSchemeConfiguration):
    gamma: float = 0.25
    delta: float = 2
    alpha: float = 0.1
    opening_token: str = "<wm>"
    closing_token: str = "</wm>"
    invert: bool = False
    
    def short_str(self):
        return f"d{self.delta:1f}a{self.alpha:1f}"
    
    def get_detector(self, model_device, tokenizer, **kwargs):
        return KGWWatermarkTokenBoundary(
            gamma=self.gamma,
            delta=self.delta,
            alpha=self.alpha,
            opening_token=self.opening_token,
            closing_token=self.closing_token,
            invert=self.invert,
            tokenizer=tokenizer
        )