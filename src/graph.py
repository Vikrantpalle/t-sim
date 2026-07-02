from collections import defaultdict
from abc import abstractmethod
from typing import Union, ClassVar, Self
from config import ModelConfig, ParallelConfig, Device, RequestBatch
from dataclasses import dataclass, replace


@dataclass
class FwdResult:
    # execution time on ns
    exec_time: int

    def __add__(self, other):
        if isinstance(other, FwdResult):
            self.exec_time += other.exec_time
            return self
        raise ValueError(f"Expected instance FwdResult, got {type(other)}")


class Op:
    def _forward(self, *args, **kwargs) -> FwdResult:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> FwdResult:
        exec_time = self._forward(*args, **kwargs)
        self.exec_time = exec_time
        return exec_time


@dataclass
class GraphCtx:
    model_cfg: ModelConfig
    parallel_cfg: ParallelConfig
    device: Device


@dataclass(kw_only=True)
class GraphModule:
    id: ClassVar[str]

    modules: list[Union["GraphModule", "Op"]]

    def forward(self, req: RequestBatch) -> FwdResult:
        res = FwdResult(0)
        for mod in self.modules:
            res += mod.forward(req)
        return res

    @classmethod
    @abstractmethod
    def from_config(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    def duplicate(self) -> Self:
        return replace(self)

    def repeat(self, num_repeats: int) -> list[Self]:
        return [self.duplicate() for _ in range(num_repeats)]

    def print_graph(self, depth: int = 0, max_depth: int = 2):
        print_graph(self, depth, max_depth)


@dataclass(kw_only=True)
class ModelGraph:
    modules: list[Union[GraphModule, Op]]
    ctx: GraphCtx
    arch: str

    def forward(self, req: RequestBatch) -> FwdResult:
        res = FwdResult(0)
        for mod in self.modules:
            res += mod.forward(req)
        return res

    @classmethod
    @abstractmethod
    def from_config(cls, ctx: GraphCtx) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse_config(cls, config: dict) -> ModelConfig:
        raise NotImplementedError

    def print_graph(self, depth: int = 0, max_depth: int = 2):
        print_graph(self, depth, max_depth)


def print_graph(root: GraphModule | ModelGraph, depth: int = 0, max_depth: int = 2):
    if depth > max_depth:
        return

    names = []
    name_count = defaultdict(int)
    for mod in root.modules:
        orig_name = mod.__class__.__name__
        name_count[orig_name] += 1
        names.append(orig_name + "_" + str(name_count[orig_name]))

    spacing = "  "
    for mod, name in zip(root.modules, names):
        print(spacing * depth + name)
        if isinstance(mod, GraphModule):
            mod.print_graph(depth + 1, max_depth)
        else:
            print(spacing * depth + mod.__class__.__name__)
