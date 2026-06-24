from kernels import FlashInferAttn, Kernel, FlashInferGEMM
from config import RequestBatch, Device, RequestType


LOG_FD = None


class FFNOp:
    def __init__(
        self, *, inp_s: int, out_s: int, target: Device, backend: str = "flashinfer"
    ):
        self.inp_s = inp_s
        self.out_s = out_s

        self.target = target
        self.backend = backend

        self.kernel = self.resolve_kernel()

    def resolve_kernel(self) -> Kernel:
        return FlashInferGEMM(self.target)

    def forward(self, input: RequestBatch) -> int:
        seq_lens = input.get_seq_lens()
        return self.kernel.call(
            batch_size=sum(seq_lens),
            inp_size=self.inp_s,
            out_size=self.out_s,
        )


class AttentionOp:
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        target: Device,
        backend: str = "flashinfer",
    ):
        self.kv_h = num_kv_heads
        self.q_h = num_heads
        self.head_size = 128

        self.target = target
        self.backend = backend

        self.kernel = self.resolve_kernel()

    def resolve_kernel(self) -> Kernel:
        return FlashInferAttn(self.target)

    def forward(self, input: RequestBatch) -> int:
        seq_lens = input.get_seq_lens()
        context_lens = None
        if input.request_type == RequestType.PREFILL:
            context_lens = input.get_context_lens()

        return self.kernel.call(
            num_heads=self.q_h,
            num_kv_heads=self.kv_h,
            head_size=self.head_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
        )
