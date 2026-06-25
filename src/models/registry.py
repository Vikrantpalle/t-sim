from transformers import AutoConfig
from config import RequestBatch, ModelConfig
from enum import StrEnum
from typing import ClassVar
from abc import ABC, abstractmethod


class ModelArch(StrEnum):
    Qwen3Causal = "Qwen3ForCausalLM"


class TransformerModel(ABC):
    MODEL_ARCH: ClassVar[ModelArch]

    def __init_subclass__(cls: type["TransformerModel"], **kwargs):
        super().__init_subclass__(**kwargs)
        MODEL_REGISTRY[cls.MODEL_ARCH] = cls

    @abstractmethod
    def forward(self, req: RequestBatch) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse_config(cls, config: dict) -> ModelConfig:
        raise NotImplementedError


MODEL_REGISTRY: dict[ModelArch, type[TransformerModel]] = {}


class AutoModelConfig:
    @classmethod
    def from_pretrained(cls, model_name: str) -> ModelConfig:
        config = AutoConfig.from_pretrained(model_name).to_dict()
        archs = config.get("architectures", [])
        final_arch = None
        for arch in archs:
            if arch in MODEL_REGISTRY:
                final_arch = arch
                break

        if final_arch is None:
            raise ValueError(
                f"None of the following architectures are supported: {','.join(archs)}"
            )

        model = MODEL_REGISTRY[arch]
        config = model.parse_config(config)
        return config
