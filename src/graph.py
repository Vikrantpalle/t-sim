from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GraphContext:
    path: list[str] = field(default_factory=list)
    ops: list["GraphOp"] = field(default_factory=list)

    def register(self, op: "GraphOp"):
        self.ops.append(op)


CUR_GRAPH_CTX: GraphContext | None = None
GRAPH_CTX_PATH = "__tsim_graph_ctx__"


def graph_ctx(ignore_error: bool = False) -> GraphContext | None:
    if CUR_GRAPH_CTX is None:
        if ignore_error:
            return
        raise RuntimeError("Unexpected: Graph context not found")
    return CUR_GRAPH_CTX


class GraphModule(ABC):
    def pre_init_hook(self):
        global CUR_GRAPH_CTX
        ctx = CUR_GRAPH_CTX
        if ctx is None:
            CUR_GRAPH_CTX = GraphContext()
            ctx = CUR_GRAPH_CTX
        ctx.path.append(self.__class__.__name__)
        setattr(self, GRAPH_CTX_PATH, CUR_GRAPH_CTX)

    def post_init_hook(self):
        ctx = graph_ctx()
        assert ctx is not None
        if ctx.path:
            ctx.path.pop(-1)
        if not ctx.path:
            global CUR_GRAPH_CTX
            CUR_GRAPH_CTX = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if getattr(cls, "__tsim_init__", False):
            return

        orig_init = cls.__init__

        def wrapped_init(self, *args, **kwargs):
            GraphModule.pre_init_hook(self)
            orig_init(self, *args, **kwargs)
            GraphModule.post_init_hook(self)

        setattr(cls, "__tsim_init__", True)
        cls.__init__ = wrapped_init  # ty:ignore[invalid-assignment]

    @abstractmethod
    def forward(self, *args, **kwargs) -> int:
        raise NotImplementedError


class GraphOp:
    def __init__(self):
        self.graph_ctx = graph_ctx(True)
        if self.graph_ctx:
            self.graph_ctx.register(self)
            self.path = self.graph_ctx.path.copy()
        else:
            self.path = []

    def __repr__(self):
        return f"GraphOp(path='{'.'.join(self.path)}', op={self.__class__.__name__})"
