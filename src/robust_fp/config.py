from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union

import yaml
from pydantic import BaseModel, Field
from strenum import StrEnum

from robust_fp.watermarks.kgw.kgw_watermark import KGWWatermark
from robust_fp.finetuning.dataset import DatasetType

if TYPE_CHECKING:
    from peft import LoraConfig as PeftLoraConfig  # pragma: no cover
else:  # pragma: no cover - runtime fallback when peft is unavailable
    PeftLoraConfig = Any  # type: ignore[misc]


TWatermarkScheme = TypeVar("TWatermarkScheme", bound="WatermarkSchemeConfiguration")


class FingerprintEvalConfiguration(BaseModel):
    name: str
    n_samples: int = 1000
    dataset_config: Dict[str, Any] = Field(default_factory=dict)
    ppl_model: str = "Qwen/Qwen2.5-32B-Instruct"

    @classmethod
    def from_value(
        cls, value: Union[str, Dict[str, Any], "FingerprintEvalConfiguration"]
    ) -> "FingerprintEvalConfiguration":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            with open(value, "r") as file:
                payload = yaml.safe_load(file) or {}
        elif isinstance(value, dict):
            payload = value
        else:
            raise TypeError(
                "Watermark evaluation configuration must be provided as a mapping, "
                "YAML path, or WatermarkEvalConfiguration instance."
            )
        return cls.model_validate(payload)


class WatermarkSchemeConfiguration(BaseModel):
    """Base class for watermark schemes."""

    @classmethod
    def from_value(
        cls: type[TWatermarkScheme],
        value: Union[str, Dict[str, Any], "WatermarkSchemeConfiguration", None],
    ) -> TWatermarkScheme:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            with open(value, "r") as file:
                payload = yaml.safe_load(file) or {}
        elif isinstance(value, dict) or value is None:
            payload = value or {}
        else:
            raise TypeError(
                "Watermark scheme configuration must be provided as a mapping, "
                "YAML path, or WatermarkSchemeConfiguration instance."
            )
        return cls.model_validate(payload)

    def get_detector(self, model_device, tokenizer, **kwargs):
        raise NotImplementedError(
            "get_detector must be implemented on watermark scheme configuration subclasses."
        )

    def short_str(self) -> str:
        raise NotImplementedError(
            "short_str must be implemented on watermark scheme configuration subclasses."
        )


class KGWWatermarkConfiguration(WatermarkSchemeConfiguration):
    gamma: float = 0.25
    delta: float = 2.0
    k: int = 1
    seeding_scheme: str = "simple_1"
    kgw_device: str = "cuda"

    def short_str(self) -> str:
        return f"d{self.delta}_k{self.k}"

    def get_detector(self, model_device, tokenizer, **kwargs):
        return KGWWatermark(
            vocab=tokenizer.get_vocab(),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            tokenizer=tokenizer,
        )


class WatermarkType(StrEnum):
    KGW = "kgw"

    def build_config(
        self,
        value: Union[str, Dict[str, Any], WatermarkSchemeConfiguration, None],
    ) -> WatermarkSchemeConfiguration:
        if isinstance(value, WatermarkSchemeConfiguration):
            return value
        if self == WatermarkType.KGW:
            return KGWWatermarkConfiguration.from_value(value)
        raise NotImplementedError(f"Unsupported watermark type: {self.value}")


class WatermarkConfiguration(BaseModel):
    watermark_type: WatermarkType
    watermark_config: WatermarkSchemeConfiguration
    watermark_eval_config: List[FingerprintEvalConfiguration] = Field(default_factory=list)

    def get_detector(self, model_device, tokenizer, **kwargs):
        return self.watermark_config.get_detector(model_device, tokenizer, **kwargs)

    def short_str(self) -> str:
        return self.watermark_config.short_str()

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "WatermarkConfiguration":
        if not isinstance(data, dict):
            raise TypeError("Watermark configuration must be provided as a mapping.")

        watermark_type_value = data.get("watermark_type") or data.get("type")
        if watermark_type_value is None:
            raise ValueError("Missing watermark type in configuration.")
        watermark_type = WatermarkType(watermark_type_value)

        scheme_value = data.get("watermark_config") or data.get("config")

        scheme_config = watermark_type.build_config(scheme_value)

        payload = {
            "watermark_type": watermark_type,
            "watermark_config": scheme_config,
        }
        return cls.model_validate(payload)

    @classmethod
    def from_legacy(
        cls,
        watermark_type: Union[str, WatermarkType],
        scheme: Union[str, Dict[str, Any], WatermarkSchemeConfiguration],
    ) -> "WatermarkConfiguration":
        data = {
            "watermark_type": watermark_type,
            "watermark_config": scheme,
        }
        return cls.from_data(data)


def _build_lora_config(config: Dict[str, Any]):
    try:
        from peft import LoraConfig as _LoraConfig
    except ImportError as exc:  # pragma: no cover - exercised only when peft missing
        raise ImportError(
            "peft is required to build a LoRA configuration. Install it or remove the 'lora_config' section."
        ) from exc
    return _LoraConfig(**config)


class FinetuningConfiguration(BaseModel):
    base_model: str
    dtype: Optional[str] = "float32"
    training_args: Dict[str, Any]
    lora_config: Optional[PeftLoraConfig] = None
    watermark_datasets: List[DatasetType] = Field(default_factory=list)
    regularization_datasets: List[DatasetType] = Field(default_factory=list)
    loss_types: List[Union[int, str]] = Field(default_factory=list)
    streaming: bool = False
    sequence_length: int = 512
    proportions: List[float] = Field(default_factory=list)
    lambdas: List[float] = Field(default_factory=list)
    custom_name: Optional[str] = None

    @classmethod
    def from_data(
        cls,
        data: Dict[str, Any],
        default_base_model: Optional[str] = None,
    ) -> "FinetuningConfiguration":
        if not isinstance(data, dict):
            raise TypeError("Finetuning configuration must be provided as a mapping.")

        payload = dict(data)

        if default_base_model and payload.get("base_model") in (None, "PLACEHOLDER"):
            payload["base_model"] = default_base_model


        lora_conf = payload.get("lora_config")
        if isinstance(lora_conf, dict):
            payload["lora_config"] = _build_lora_config(lora_conf)

        return cls.model_validate(payload)

    @classmethod
    def from_yaml_path(
        cls, yaml_path: str, default_base_model: Optional[str] = None
    ) -> "FinetuningConfiguration":
        with open(yaml_path, "r") as file:
            contents = file.read()
        if default_base_model is not None:
            contents = contents.replace("PLACEHOLDER", default_base_model)
        payload = yaml.safe_load(contents) or {}
        return cls.from_data(payload, default_base_model=default_base_model)

    def validate_config(self) -> None:
        datasets = list(self.watermark_datasets) + list(self.regularization_datasets)
        probabilities = self.proportions

        if datasets:
            if len(probabilities) != len(datasets):
                raise ValueError(
                    "The number of datasets and proportions must match in finetuning configuration."
                )

            if abs(sum(probabilities) - 1.0) > 1e-6:
                raise ValueError("The dataset proportions must sum to 1.0.")

        if self.regularization_datasets:
            if len(self.loss_types) != len(self.regularization_datasets):
                raise ValueError(
                    "Each regularization dataset must have an associated loss type."
                )

    def short_str(self) -> str:
        out = self.custom_name if self.custom_name is not None else ""
        return out


class MainConfiguration(BaseModel):
    base_model: str
    caching_models: bool = False
    watermark_config: WatermarkConfiguration
    disable_wm_detector: bool = False
    output_directory: Optional[str] = None
    huggingface_name: Optional[str] = None
    finetuning_config: Optional[FinetuningConfiguration] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MainConfiguration":
        if not isinstance(data, dict):
            raise TypeError("Main configuration must be provided as a mapping.")

        payload = dict(data)

        watermark_section = payload.pop("watermark", None)

        if watermark_section is not None:
            payload["watermark_config"] = WatermarkConfiguration.from_data(
                watermark_section
            )
        else:
            watermark_type = payload.pop("watermark_type", None)
            watermark_config = payload.pop("watermark_config", None)
            watermark_evaluations = payload.pop("watermark_evaluation_config", [])

            if watermark_type is None or watermark_config is None:
                raise ValueError(
                    "Watermark configuration is missing. Provide a 'watermark' section "
                    "or legacy keys ('watermark_type', 'watermark_config')."
                )

            payload["watermark_config"] = WatermarkConfiguration.from_legacy(
                watermark_type,
                watermark_config,
                watermark_evaluations,
            )

        finetuning_section = payload.pop("finetuning", None)
        legacy_finetuning = payload.pop("finetuning_config", None)
        base_model = payload.get("base_model")

        if finetuning_section is not None:
            payload["finetuning_config"] = FinetuningConfiguration.from_data(
                finetuning_section,
                default_base_model=base_model,
            )
        elif legacy_finetuning is not None:
            if isinstance(legacy_finetuning, str):
                payload["finetuning_config"] = FinetuningConfiguration.from_yaml_path(
                    legacy_finetuning,
                    default_base_model=base_model,
                )
            elif isinstance(legacy_finetuning, dict):
                payload["finetuning_config"] = FinetuningConfiguration.from_data(
                    legacy_finetuning,
                    default_base_model=base_model,
                )
            else:
                raise TypeError(
                    "finetuning_config must be a path or mapping when using the legacy format."
                )

        return cls.model_validate(payload)

    @classmethod
    def parse_yaml(cls, yaml_path: str) -> "MainConfiguration":
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file) or {}
        return cls.from_dict(data)
