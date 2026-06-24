from enum import StrEnum
from typing import NamedTuple
from dataclasses import dataclass


@dataclass
class Request:
    num_input_tokens: int
    num_output_tokens: int
    num_computed_tokens: int = 0

    # request arrival time (ns)
    arrival_time: int = 0
    # time when first token got generated
    token_start_time: int = 0

    def is_prefill(self):
        return self.num_computed_tokens < self.num_input_tokens

    def is_decode(self):
        return not self.is_prefill()


class RequestType(StrEnum):
    MIXED = "MIXED"
    PREFILL = "PREFILL"
    DECODE = "DECODE"


@dataclass
class RequestBatch:
    requests: list[Request]
    request_type: RequestType

    def get_seq_lens(self):
        return [
            1 if req.is_decode() else (req.num_input_tokens - req.num_computed_tokens)
            for req in self.requests
        ]

    def get_context_lens(self):
        return [
            0 if req.is_decode() else req.num_computed_tokens for req in self.requests
        ]


@dataclass
class ParallelConfig:
    num_devices: int
    tp: int
    pp: int
    dp: int

    def __post_init__(self):
        if self.dp * self.pp * self.tp != self.num_devices:
            raise ValueError(
                "Invalid parallel config",
                f"{self.dp} (dp) * {self.pp} (pp) * {self.tp} (tp)",
                f"must be equal to device count but found {self.num_devices} devices",
            )


@dataclass
class ModelConfig:
    hidden_size: int
    hidden_act: str
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int


class HWProvider(StrEnum):
    NVIDIA = "NVIDIA"


class NVModel(StrEnum):
    RTX_4090 = "RTX_4090"


type HWModel = NVModel


class Device(NamedTuple):
    provider: HWProvider
    model: HWModel


@dataclass
class SchedulerConfig:
    page_size: int
    max_pages: int


SCHEDULER_CONFIG = SchedulerConfig(page_size=16, max_pages=20000)
