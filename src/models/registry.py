from config import RequestBatch
from enum import StrEnum
from typing import ClassVar
from abc import ABC, abstractmethod


class ModelName(StrEnum):
    Qwen3 = "Qwen3"


class TransformerModel(ABC):
    MODEL_NAME: ClassVar[ModelName]

    def __init_subclass__(cls: type["TransformerModel"], **kwargs):
        super().__init_subclass__(**kwargs)
        MODEL_REGISTRY[cls.MODEL_NAME] = cls

    @abstractmethod
    def forward(self, req: RequestBatch) -> int:
        raise NotImplementedError


MODEL_REGISTRY: dict[ModelName, type[TransformerModel]] = {}
