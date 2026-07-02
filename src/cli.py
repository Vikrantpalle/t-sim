from scheduler import ContinuousBatchingScheduler
from graph import GraphCtx
from models.qwen3 import Qwen3Causal
from config import ParallelConfig, Device, HWProvider, NVModel, Request
from models.registry import AutoModelConfig
import typer

app = typer.Typer()


@app.command()
def run(model: str, tp: int = 1):
    config = AutoModelConfig.from_pretrained(model)

    parallel_config = ParallelConfig(tp, tp, 1, 1)

    hw_model = Device(HWProvider.NVIDIA, NVModel.RTX_4090)

    model_graph = Qwen3Causal.from_config(GraphCtx(config, parallel_config, hw_model))

    req = [Request(512, 2, 0) for _ in range(1)]

    scheduler = ContinuousBatchingScheduler(req, model_graph)
    scheduler.start()


@app.command()
def print_graph(model: str, tp: int = 1):
    config = AutoModelConfig.from_pretrained(model)

    parallel_config = ParallelConfig(tp, tp, 1, 1)

    hw_model = Device(HWProvider.NVIDIA, NVModel.RTX_4090)

    model_graph = Qwen3Causal.from_config(GraphCtx(config, parallel_config, hw_model))

    model_graph.print_graph()


if __name__ == "__main__":
    app()
