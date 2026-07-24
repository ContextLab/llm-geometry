"""Architecture Explorer backend (feature 002, `/api/arch/*`).

Traced architecture graphs (`graph`), real weight windows (`weights`), per-node
activation traces (`trace`), real autoregressive generation (`generate`), and the
pre-download model-size gate (`gate`). Everything runs against real open-weights
models — no mocks, no fabricated tensors (FR-109, Constitution I).
"""

from .gate import check_model_size
from .generate import generate
from .graph import build_graph
from .trace import trace_forward
from .weights import weight_window

__all__ = [
    "build_graph",
    "check_model_size",
    "generate",
    "trace_forward",
    "weight_window",
]
