from ops import FFNOp, AttentionOp
from models.registry import TransformerModel
import matplotlib.pyplot as plt


class MetricCollector:
    def __init__(self, model: TransformerModel):

        self.model = model

        self.prefill_gemm_time = 0
        self.prefill_attn_time = 0

        self.decode_gemm_time = 0
        self.decode_attn_time = 0

    def add_metrics(self, *, is_prefill: bool):

        graph_ctx = self.model.get_graph_ctx()

        for op in graph_ctx.ops:
            if is_prefill:
                if isinstance(op, FFNOp):
                    self.prefill_gemm_time += op.exec_time
                elif isinstance(op, AttentionOp):
                    self.prefill_attn_time += op.exec_time
            else:
                if isinstance(op, FFNOp):
                    self.decode_gemm_time += op.exec_time
                elif isinstance(op, AttentionOp):
                    self.decode_attn_time += op.exec_time

    def visualize(self):

        categories = ["Prefill", "Decode"]
        gemm = [self.prefill_gemm_time / 1e9, self.decode_gemm_time / 1e9]
        attn = [self.prefill_attn_time / 1e9, self.decode_attn_time / 1e9]

        plt.bar(categories, gemm, label="GEMM")
        plt.bar(categories, attn, label="Attn")

        plt.ylabel("time spent (s)")
        plt.legend()

        plt.show()
