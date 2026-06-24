from scheduler import ContinuousBatchingScheduler
from config import (
    ModelConfig,
    ParallelConfig,
    Device,
    HWProvider,
    NVModel,
    Request,
)
from models.qwen3 import Qwen3Model


if __name__ == "__main__":
    config = ModelConfig(
        hidden_size=4096,
        hidden_act="silu",
        head_dim=128,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=12288,
        num_hidden_layers=1,
        vocab_size=151936,
    )

    parallel_config = ParallelConfig(1, 1, 1, 1)

    hw_model = Device(HWProvider.NVIDIA, NVModel.RTX_4090)

    model = Qwen3Model(config, parallel_config, hw_model)

    req = [Request(512, 2, 0) for _ in range(1)]

    ContinuousBatchingScheduler(req, model).start()
