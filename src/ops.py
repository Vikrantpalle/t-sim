from graph import GraphOp
from kernels import FlashInferAttn, Kernel, FlashInferGEMM
from config import RequestBatch, Device


class Op(GraphOp):
    def _forward(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> int:
        exec_time = self._forward(*args, **kwargs)
        self.exec_time = exec_time
        return exec_time


class FFNOp(Op):
    def __init__(
        self, *, inp_s: int, out_s: int, target: Device, backend: str = "flashinfer"
    ):
        super().__init__()
        self.inp_s = inp_s
        self.out_s = out_s

        self.target = target
        self.backend = backend

        self.kernel = self.resolve_kernel()

    def resolve_kernel(self) -> Kernel:
        return FlashInferGEMM(self.target)

    def _forward(self, batch_size: int) -> int:
        return self.kernel.call(
            batch_size=batch_size,
            inp_size=self.inp_s,
            out_size=self.out_s,
        )


class AttentionOp(Op):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        target: Device,
        backend: str = "flashinfer",
    ):
        super().__init__()
        self.kv_h = num_kv_heads
        self.q_h = num_heads
        self.head_size = 128

        self.target = target
        self.backend = backend

        self.kernel = self.resolve_kernel()

    def resolve_kernel(self) -> Kernel:
        return FlashInferAttn(self.target)

    def _forward(self, input: RequestBatch) -> int:
        seq_lens = input.get_seq_lens()
        context_lens = input.get_context_lens()

        return self.kernel.call(
            num_heads=self.q_h,
            num_kv_heads=self.kv_h,
            head_size=self.head_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
        )
