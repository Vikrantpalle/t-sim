from collections import deque
from abc import ABC, abstractmethod
from models.registry import TransformerModel
from config import Request, RequestBatch, RequestType


class Scheduler(ABC):
    @abstractmethod
    def step(self):
        pass


class ContinuousBatchingScheduler(Scheduler):
    def __init__(
        self, requests: list[Request], model: TransformerModel, max_batch_len: int = 32
    ):

        # set of requests that need to be processed
        self.pending = deque()
        self.pending.extend(requests)

        self.completed = []
        self.running: list[Request] = []

        self.elapsed_time = 0

        self.model = model

        self.max_batch_len = max_batch_len

    def drain_completed_reqs(self):
        running = []
        for req in self.running:
            if req.num_computed_tokens == (
                req.num_input_tokens + req.num_output_tokens
            ):
                self.completed.append(req)
                continue
            running.append(req)
        self.running = running

    def add_pending_reqs(self):
        while len(self.pending):
            if len(self.running) > self.max_batch_len:
                break
            self.running.append(self.pending.popleft())

    def step(self):

        self.drain_completed_reqs()

        self.add_pending_reqs()

        prefill_reqs = list(filter(lambda r: r.is_prefill(), self.running))

        if prefill_reqs:
            self.elapsed_time += self.model.forward(
                RequestBatch(prefill_reqs, RequestType.PREFILL)
            )

        decode_reqs = list(filter(lambda r: r.is_decode(), self.running))
        if decode_reqs:
            self.elapsed_time += self.model.forward(
                RequestBatch(decode_reqs, RequestType.DECODE)
            )

        for req in self.running:
            if req.is_prefill():
                req.num_computed_tokens = req.num_input_tokens
            else:
                req.num_computed_tokens += 1

    def start(self):
        while len(self.pending) or len(self.running):
            self.step()
