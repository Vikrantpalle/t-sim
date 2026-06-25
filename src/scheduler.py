from metrics import MetricCollector
from models.registry import TransformerModel
from config import Request, RequestBatch, RequestType
import heapq


class ContinuousBatchingScheduler:
    def __init__(
        self, requests: list[Request], model: TransformerModel, max_batch_len: int = 32
    ):

        # set of requests that need to be processed
        self.pending = []

        for req in requests:
            heapq.heappush(self.pending, (req.arrival_time, req))

        self.completed = []
        self.running: list[Request] = []

        self.elapsed_time = 0

        self.model = model

        self.max_batch_len = max_batch_len

        self.metric_collector = MetricCollector(model)

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
        while (
            self.pending
            and self.pending[0][0] <= self.elapsed_time
            and len(self.running) < self.max_batch_len
        ):
            self.running.append(heapq.heappop(self.pending)[1])

    def step(self):

        self.drain_completed_reqs()

        self.add_pending_reqs()

        if not self.pending:
            self.elapsed_time += int(1e6)

        prefill_reqs = list(filter(lambda r: r.is_prefill(), self.running))

        if prefill_reqs:
            self.elapsed_time += int(
                self.model.forward(RequestBatch(prefill_reqs, RequestType.PREFILL))
            )
            self.metric_collector.add_metrics(is_prefill=True)

        decode_reqs = list(filter(lambda r: r.is_decode(), self.running))
        if decode_reqs:
            self.elapsed_time += int(
                self.model.forward(RequestBatch(decode_reqs, RequestType.DECODE))
            )
            self.metric_collector.add_metrics(is_prefill=False)

        for req in self.running:
            if req.is_prefill():
                req.num_computed_tokens = req.num_input_tokens
            else:
                req.num_computed_tokens += 1
                if req.num_computed_tokens == req.num_input_tokens + 1:
                    req.token_start_time = self.elapsed_time

    def start(self):
        while len(self.pending) or len(self.running):
            self.step()
