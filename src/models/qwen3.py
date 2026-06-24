from ops import AttentionOp, FFNOp
from config import ModelConfig, ParallelConfig, RequestBatch, Device
from .registry import TransformerModel, ModelName


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


class QKVParallel:
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        target: Device,
    ):
        num_heads, num_kv_heads = get_tp_heads(num_heads, num_kv_heads, tp_size)

        # TODO: assuming head size of v and q is same for now
        # change later
        self.ffn_op = FFNOp(
            inp_s=hidden_size,
            out_s=num_heads * head_size + num_kv_heads * head_size * 2,
            target=target,
        )

    def forward(self, input: RequestBatch) -> int:

        x = 0
        x += self.ffn_op.forward(input)

        return x


class Attention:
    def __init__(self, num_heads: int, num_kv_heads: int, tp_size: int, target: Device):
        num_heads, num_kv_heads = get_tp_heads(num_heads, num_kv_heads, tp_size)

        self.attention = AttentionOp(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            target=target,
        )

    def forward(self, input: RequestBatch):
        x = 0
        x += self.attention.forward(input)

        # all gather

        return x


class FFNRowParallel:
    def __init__(self, input_size: int, out_size: int, tp_size: int, target: Device):

        if input_size % tp_size:
            raise ValueError(
                f"{input_size} (input size) must be divisible by {tp_size} (tp size)"
            )
        input_size = input_size // tp_size
        self.ffn_op = FFNOp(inp_s=input_size, out_s=out_size, target=target)

    def forward(self, input: RequestBatch):
        return self.ffn_op.forward(input)


class FFNColumnParallel:
    def __init__(self, input_size: int, out_size: int, tp_size: int, target: Device):
        if out_size % tp_size:
            raise ValueError(
                f"{out_size} (output size) must be divisible by {tp_size} (tp size)"
            )

        out_size = out_size // tp_size

        self.ffn_op = FFNOp(inp_s=input_size, out_s=out_size, target=target)

    def forward(self, input: RequestBatch) -> int:
        x = 0

        # send

        x += self.ffn_op.forward(input)

        # all reduce

        return x


class VocabParallel:
    def __init__(self, input_size: int, out_size: int, tp_size: int, target: Device):
        self.shard_size = (out_size + tp_size - 1) // tp_size

        self.ffn_op = FFNOp(inp_s=input_size, out_s=self.shard_size, target=target)

    def forward(self, input: RequestBatch) -> int:
        x = 0

        # send

        x += self.ffn_op.forward(input)

        # all reduce

        return x


class Qwen3Attention:
    def __init__(
        self, config: ModelConfig, parallel_config: ParallelConfig, hw_model: Device
    ):

        tp_size = parallel_config.tp

        self.qkv_proj = QKVParallel(
            config.hidden_size,
            config.head_dim,
            config.num_heads,
            config.num_kv_heads,
            tp_size,
            hw_model,
        )

        self.attention = Attention(
            config.num_heads, config.num_kv_heads, tp_size, hw_model
        )

        self.attn_proj = FFNRowParallel(
            config.head_dim * config.num_heads, config.hidden_size, tp_size, hw_model
        )

    def forward(self, input: RequestBatch):

        x = 0
        x += self.qkv_proj.forward(input)

        # qk norm + rope

        x += self.attention.forward(input)

        x += self.attn_proj.forward(input)

        return x


class Qwen3MLP:
    def __init__(
        self, config: ModelConfig, parallel_config: ParallelConfig, hw_model: Device
    ):
        self.config = config
        tp_size = parallel_config.tp

        self.up_proj = FFNColumnParallel(
            config.hidden_size, config.intermediate_size, tp_size, hw_model
        )

        self.down_proj = FFNRowParallel(
            config.intermediate_size, config.hidden_size, tp_size, hw_model
        )

    def forward(self, input: RequestBatch):
        x = 0

        x += self.up_proj.forward(input)

        # gate proj

        x += self.down_proj.forward(input)

        return x


class Qwen3Decoder:
    def __init__(
        self, config: ModelConfig, parallel_config: ParallelConfig, hw_model: Device
    ):
        self.attention = Qwen3Attention(config, parallel_config, hw_model)

        self.mlp = Qwen3MLP(config, parallel_config, hw_model)

    def forward(
        self,
        req: RequestBatch,
    ) -> int:
        # skipping layernorm for now as its impact is negligible in forward passes
        x = 0

        # rms norm

        x += self.attention.forward(req)

        # add + rms norm

        x += self.mlp.forward(req)

        # add

        return x


class Qwen3Model(TransformerModel):
    MODEL_NAME = ModelName.Qwen3

    def __init__(
        self, config: ModelConfig, parallel_config: ParallelConfig, hw_model: Device
    ):
        self.decoder = Qwen3Decoder(config, parallel_config, hw_model)

        self.lm_head = VocabParallel(
            config.hidden_size, config.vocab_size, parallel_config.tp, hw_model
        )
        self.num_hidden_layers = config.num_hidden_layers

    def forward(self, req: RequestBatch) -> int:
        x = 0

        # embedding

        # hidden layers
        dec_time = self.decoder.forward(req)
        x += dec_time * self.num_hidden_layers

        # rms norm

        # lm head
        x += self.lm_head.forward(req)

        return x
