from graph import GraphCtx
from models.registry import AutoModelConfig
from models.qwen3 import Qwen3Causal
from scheduler import ContinuousBatchingScheduler
from config import (
    ModelConfig,
    Request,
    Device,
    HWProvider,
    NVModel,
    ParallelConfig,
)


def benchmark_model(
    config: ModelConfig, parallel_config: ParallelConfig, name: str | None = None
):

    # 10 req/s for 10s with 512 input tokens and 512 output tokens
    reqs_ps = 10
    dur_s = 10
    inp_tok = 512
    out_tok = 512
    reqs = []
    for t in range(dur_s):
        reqs.extend(
            [
                Request(
                    num_input_tokens=inp_tok,
                    num_output_tokens=out_tok,
                    arrival_time=int(t * 1e9),
                )
                for _ in range(reqs_ps)
            ]
        )

    hw_model = Device(HWProvider.NVIDIA, NVModel.RTX_4090)

    model = Qwen3Causal.from_config(GraphCtx(config, parallel_config, hw_model))
    scheduler = ContinuousBatchingScheduler(reqs, model)

    scheduler.start()

    tt_s = scheduler.elapsed_time / 1e9

    num_reqs = len(reqs)
    tot_inp_tok = sum([req.num_input_tokens for req in reqs])
    tot_out_tok = sum([req.num_output_tokens for req in reqs])

    tpot = num_reqs / tt_s
    avg_ttft = (
        sum([(req.token_start_time - req.arrival_time) / 1e9 for req in reqs])
        / num_reqs
    )
    max_ttft = max([(req.token_start_time - req.arrival_time) / 1e9 for req in reqs])

    tok_s = (tot_inp_tok + tot_out_tok) / tt_s
    inp_tok_s = tot_inp_tok / tt_s
    out_tok_s = tot_out_tok / tt_s

    print(f"Model = {name}")
    print(f"Queries completed: {num_reqs}")
    print(f"Time Taken: {tt_s}s")
    print(f"Throughput: {tpot} QPS")
    print(f"Avg TTFT: {avg_ttft}s")
    print(f"Max TTFT: {max_ttft}s")
    print(f"Input tokens: {tot_inp_tok}, Output tokens: {tot_out_tok}")
    print(f"Tokens / s: {tok_s}tok/s")
    print(f"Inp Tokens / s: {inp_tok_s}tok/s")
    print(f"Out Tokens / s: {out_tok_s}tok/s")


if __name__ == "__main__":
    model = "Qwen/Qwen3-8B"
    config = AutoModelConfig.from_pretrained(model)

    parallel_config = ParallelConfig(1, 1, 1, 1)
    benchmark_model(config, parallel_config, model)
