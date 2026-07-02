from graph import GraphCtx
from models.registry import AutoModelConfig
from scheduler import ContinuousBatchingScheduler
from config import (
    ParallelConfig,
    Device,
    HWProvider,
    NVModel,
    Request,
)
from models.qwen3 import Qwen3Causal


if __name__ == "__main__":
    model = "Qwen/Qwen3-8B"

    config = AutoModelConfig.from_pretrained(model)

    parallel_config = ParallelConfig(1, 1, 1, 1)

    hw_model = Device(HWProvider.NVIDIA, NVModel.RTX_4090)

    model = Qwen3Causal.from_config(GraphCtx(config, parallel_config, hw_model))

    req = [Request(512, 2, 0) for _ in range(1)]

    scheduler = ContinuousBatchingScheduler(req, model)
    scheduler.start()
