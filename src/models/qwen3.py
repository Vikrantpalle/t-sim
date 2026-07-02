from graph import GraphModule, GraphCtx
from dataclasses import dataclass
from typing import Self
from ops import AttentionOp, FFNOp
from config import ModelConfig, Device
from .registry import (
    ModelGraph,
    ModelArch,
    register_model,
)


def get_tp_heads(num_heads: int, num_kv_heads: int, tp_size: int):
    if num_heads % tp_size:
        raise ValueError(
            f"{num_heads} (num q heads) must be divisible by {tp_size} (tp size)"
        )

    tp_heads = num_heads // tp_size

    if tp_size < num_kv_heads:
        if num_kv_heads % tp_size:
            raise ValueError(
                f"{num_kv_heads} (num kv heads) must be divisible by {tp_size} (tp size)"
            )
        # kv replicas = 1
        tp_kv_heads = num_kv_heads // tp_size
    else:
        if tp_size % num_kv_heads:
            raise ValueError(
                f"{tp_size} (tp size) must be divisible by {num_kv_heads} (num kv heads)"
            )
        # kv replicas = tp_size // num_kv_heads
        tp_kv_heads = 1

    return tp_heads, tp_kv_heads


class QKVParallel(GraphModule):
    @classmethod
    def from_config(
        cls,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        target: Device,
    ) -> Self:
        num_heads, num_kv_heads = get_tp_heads(num_heads, num_kv_heads, tp_size)

        # TODO: assuming head size of v and q is same for now
        # change later
        return cls(
            modules=[
                FFNOp(
                    inp_s=hidden_size,
                    out_s=num_heads * head_size + num_kv_heads * head_size * 2,
                    target=target,
                )
            ]
        )


class Attention(GraphModule):
    @classmethod
    def from_config(
        cls, num_heads: int, num_kv_heads: int, tp_size: int, target: Device
    ):
        num_heads, num_kv_heads = get_tp_heads(num_heads, num_kv_heads, tp_size)

        return cls(
            modules=[
                AttentionOp(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    target=target,
                )
            ]
        )


class FFNRowParallel(GraphModule):
    @classmethod
    def from_config(
        cls, input_size: int, out_size: int, tp_size: int, target: Device
    ) -> Self:
        if input_size % tp_size:
            raise ValueError(
                f"{input_size} (input size) must be divisible by {tp_size} (tp size)"
            )
        input_size = input_size // tp_size
        return cls(modules=[FFNOp(inp_s=input_size, out_s=out_size, target=target)])


class FFNColumnParallel(GraphModule):
    @classmethod
    def from_config(
        cls, input_size: int, out_size: int, tp_size: int, target: Device
    ) -> Self:
        if out_size % tp_size:
            raise ValueError(
                f"{out_size} (output size) must be divisible by {tp_size} (tp size)"
            )

        out_size = out_size // tp_size

        return cls(modules=[FFNOp(inp_s=input_size, out_s=out_size, target=target)])


class VocabParallel(GraphModule):
    @classmethod
    def from_config(
        cls, input_size: int, out_size: int, tp_size: int, target: Device
    ) -> Self:
        shard_size = (out_size + tp_size - 1) // tp_size

        return cls(modules=[FFNOp(inp_s=input_size, out_s=shard_size, target=target)])


class Qwen3Attention(GraphModule):
    @classmethod
    def from_config(cls, ctx: GraphCtx) -> Self:

        tp_size = ctx.parallel_cfg.tp

        return cls(
            modules=[
                QKVParallel.from_config(
                    ctx.model_cfg.hidden_size,
                    ctx.model_cfg.head_dim,
                    ctx.model_cfg.num_heads,
                    ctx.model_cfg.num_kv_heads,
                    tp_size,
                    ctx.device,
                ),
                Attention.from_config(
                    ctx.model_cfg.num_heads,
                    ctx.model_cfg.num_kv_heads,
                    tp_size,
                    ctx.device,
                ),
                FFNRowParallel.from_config(
                    ctx.model_cfg.head_dim * ctx.model_cfg.num_heads,
                    ctx.model_cfg.hidden_size,
                    tp_size,
                    ctx.device,
                ),
            ]
        )


class Qwen3MLP(GraphModule):
    @classmethod
    def from_config(cls, ctx: GraphCtx) -> Self:
        return cls(
            modules=[
                FFNColumnParallel.from_config(
                    ctx.model_cfg.hidden_size,
                    ctx.model_cfg.intermediate_size,
                    ctx.parallel_cfg.tp,
                    target=ctx.device,
                ),
                FFNRowParallel.from_config(
                    ctx.model_cfg.intermediate_size,
                    ctx.model_cfg.hidden_size,
                    ctx.parallel_cfg.tp,
                    target=ctx.device,
                ),
            ]
        )


class Qwen3Decoder(GraphModule):
    @classmethod
    def from_config(cls, ctx: GraphCtx) -> Self:
        return cls(modules=[Qwen3Attention.from_config(ctx), Qwen3MLP.from_config(ctx)])


@register_model(ModelArch.Qwen3Causal)
@dataclass
class Qwen3Causal(ModelGraph):
    @classmethod
    def from_config(cls, ctx: GraphCtx) -> Self:
        return cls(
            modules=[
                *Qwen3Decoder.from_config(ctx).repeat(ctx.model_cfg.num_hidden_layers),
                VocabParallel.from_config(
                    ctx.model_cfg.hidden_size,
                    ctx.model_cfg.vocab_size,
                    ctx.parallel_cfg.tp,
                    ctx.device,
                ),
            ],
            ctx=ctx,
            arch=ModelArch.Qwen3Causal,
        )

    @classmethod
    def parse_config(cls, config: dict) -> ModelConfig:
        return ModelConfig(
            hidden_size=config["hidden_size"],
            hidden_act=config["hidden_act"],
            head_dim=config["head_dim"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
        )
