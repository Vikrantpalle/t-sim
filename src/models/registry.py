from config import ModelConfig
from graph import ModelGraph
from transformers import AutoConfig
from enum import StrEnum


class ModelArch(StrEnum):
    Qwen3Causal = "Qwen3ForCausalLM"


MODEL_REGISTRY: dict[str, type[ModelGraph]] = {}


def register_model(model_arch: str):
    def _register_model(cls):
        MODEL_REGISTRY[model_arch] = cls

        return cls

    return _register_model


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

        model = MODEL_REGISTRY[final_arch]
        config = model.parse_config(config)
        return config
