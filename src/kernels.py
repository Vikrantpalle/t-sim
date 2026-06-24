from utils.hash import sha256_hash
from config import Device, SCHEDULER_CONFIG
from abc import ABC, abstractmethod
from enum import StrEnum, auto
import torch
from torch.functional import F
import flashinfer as fi
from flashinfer.testing import bench_gpu_time
import numpy as np
from cachetools import LRUCache

CACHE = LRUCache(maxsize=1000)


class KernelName(StrEnum):
    FI_ATTN = auto()
    FI_GEMM = auto()


class Kernel(ABC):
    def __init__(self, target: Device):
        self.target = target

    @abstractmethod
    def _call(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def call(self, *args, **kwargs) -> int:
        """Wrapper around real implementation

        1. checks if key exists in cache, if true then returns
        2. if false, then executes kernel, updates cache then returns
        """

        key = {}
        if args:
            key["args"] = args
        key.update(kwargs)
        key_hash = sha256_hash(key)
        if key_hash in CACHE:
            print("cache hit!", key)
            return CACHE[key_hash]

        print("missed", key)
        res = self._call(*args, **kwargs)
        CACHE[key_hash] = res
        return res

    @property
    @abstractmethod
    def name(self) -> KernelName:
        raise NotImplementedError


class FlashInferGEMM(Kernel):
    @property
    def name(self) -> KernelName:
        return KernelName.FI_GEMM

    def _call(self, *, batch_size: int, inp_size: int, out_size: int) -> int:
        return 1
        device = torch.cuda.current_device()
        a = torch.randn((batch_size, inp_size), dtype=torch.bfloat16, device=device)
        b = torch.randn((inp_size, out_size), dtype=torch.bfloat16, device=device)
        times = bench_gpu_time(lambda: fi.gemm.mm_bf16(a, b))
        return np.mean(times)


class FlashInferAttn(Kernel):
    @property
    def name(self) -> KernelName:
        return KernelName.FI_ATTN

    def _bench_decode(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        seq_lens: list[int],
    ) -> int:
        return 2
        device = torch.cuda.current_device()
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        page_size = SCHEDULER_CONFIG.page_size
        batch_size = len(seq_lens)

        page_lens = (seq_lens_t + page_size - 1) // page_size
        total_pages = int(torch.sum(page_lens).item())

        if total_pages >= SCHEDULER_CONFIG.max_pages:
            raise RuntimeError(
                "Insufficient RAM on target accelerator",
                f"required: {total_pages}, max pages: {SCHEDULER_CONFIG.max_pages}",
            )

        page_indptr = torch.cumsum(F.pad(page_lens, (1, 0), value=0), 0).to(torch.int32)
        page_indices = torch.randperm(
            SCHEDULER_CONFIG.max_pages, dtype=torch.int32, device=device
        )[:total_pages]

        last_page_len = seq_lens_t % page_size
        last_page_len[last_page_len == 0] = page_size

        q = torch.randn(
            (batch_size, num_heads, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        kv_cache = torch.randn(
            (
                SCHEDULER_CONFIG.max_pages,
                2,
                SCHEDULER_CONFIG.page_size,
                num_kv_heads,
                head_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

        workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        decode_wrapper.plan(
            page_indptr,
            page_indices,
            last_page_len,
            num_heads,
            num_kv_heads,
            head_size,
            SCHEDULER_CONFIG.page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
        )

        o = decode_wrapper.run(q, kv_cache)
        assert o.shape == (batch_size, num_heads, head_size)

        times = bench_gpu_time(lambda: decode_wrapper.run(q, kv_cache))

        return np.mean(times)

    def _bench_prefill(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        seq_lens: list[int],
        context_lens: list[int] | None = None,
    ) -> int:
        return 1
        device = torch.cuda.current_device()
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device=device)

        assert seq_lens_t.shape == context_lens_t.shape

        seq_indptr = torch.cumsum(F.pad(seq_lens_t, (1, 0), value=0), 0).to(torch.int32)

        total_tokens = int(torch.sum(seq_lens_t).item())

        page_lens = (
            seq_lens_t + context_lens_t + SCHEDULER_CONFIG.page_size - 1
        ) // SCHEDULER_CONFIG.page_size
        total_pages = int(torch.sum(page_lens).item())
        if total_pages >= SCHEDULER_CONFIG.max_pages:
            raise RuntimeError(
                "Insufficient RAM on target accelerator",
                f"required: {total_pages}, max pages: {SCHEDULER_CONFIG.max_pages}",
            )

        page_indptr = torch.cumsum(F.pad(page_lens, (1, 0), value=0), 0).to(torch.int32)
        page_indices = torch.randperm(
            SCHEDULER_CONFIG.max_pages, dtype=torch.int32, device=device
        )[:total_pages]

        last_page_len = seq_lens_t % SCHEDULER_CONFIG.page_size
        last_page_len[last_page_len == 0] = SCHEDULER_CONFIG.page_size

        q = torch.randn(
            (total_tokens, num_heads, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        kv_cache = torch.randn(
            (
                SCHEDULER_CONFIG.max_pages,
                2,
                SCHEDULER_CONFIG.page_size,
                num_kv_heads,
                head_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

        workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

        prefill_wrapper.plan(
            seq_indptr,
            page_indptr,
            page_indices,
            last_page_len,
            num_heads,
            num_kv_heads,
            head_size,
            SCHEDULER_CONFIG.page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            causal=True,
        )

        o = prefill_wrapper.run(q, kv_cache)
        assert o.shape == (total_tokens, num_heads, head_size)

        times = bench_gpu_time(lambda: prefill_wrapper.run(q, kv_cache))

        return np.mean(times)

    def _call(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        seq_lens: list[int],
        context_lens: list[int] | None = None,
    ) -> int:
        time_taken = 0
        if context_lens is None:
            # decode path
            time_taken = self._bench_decode(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                seq_lens=seq_lens,
            )
        else:
            time_taken = self._bench_prefill(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                seq_lens=seq_lens,
                context_lens=context_lens,
            )

        return time_taken


KERNEL_REGISTRY: dict[KernelName, type[Kernel]] = {KernelName.FI_ATTN: FlashInferAttn}
