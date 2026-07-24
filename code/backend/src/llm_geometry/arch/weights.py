"""Real weight windows for the Architecture Explorer (FR-102).

Serves genuine slices of a model's ``state_dict``: windows within the cell budget are
returned exactly; larger windows are strided-mean downsampled to a grid of at most
~64x64 cells. Stats are always computed over the *requested window* (never the
downsampled grid), so zooming out still reports true extremes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ARCH_ATTENTION_MAX_SIDE, ARCH_WEIGHTS_MAX_CELLS
from ..errors import InvalidParamError, NotFoundError
from ..models.loader import load_model


def strided_mean_2d(a: np.ndarray, max_rows: int, max_cols: int) -> np.ndarray:
    """Downsample a 2-D array to at most ``(max_rows, max_cols)`` by exact bin means.

    Bins are contiguous strides covering the array; means are computed with an
    integral image so uneven bin sizes are handled exactly.
    """
    rows, cols = a.shape
    gr, gc = min(max_rows, rows), min(max_cols, cols)
    row_edges = np.linspace(0, rows, gr + 1).astype(np.int64)
    col_edges = np.linspace(0, cols, gc + 1).astype(np.int64)
    integral = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(a.astype(np.float64), axis=0), axis=1)
    r0, r1 = row_edges[:-1, None], row_edges[1:, None]
    c0, c1 = col_edges[None, :-1], col_edges[None, 1:]
    sums = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    counts = (r1 - r0) * (c1 - c0)
    return (sums / counts).astype(np.float32)


def _as_matrix(t: np.ndarray) -> np.ndarray:
    """1-D params (biases, norms) as a single column (C=1); >2-D flattened row-major."""
    if t.ndim == 1:
        return t.reshape(-1, 1)
    if t.ndim == 2:
        return t
    return t.reshape(t.shape[0], -1)


def weight_window(
    model_id: str,
    param: str,
    r0: int = 0,
    r1: int | None = None,
    c0: int = 0,
    c1: int | None = None,
    max_cells: int = ARCH_WEIGHTS_MAX_CELLS,
) -> dict[str, Any]:
    """A real window of one parameter tensor, per the `/api/arch/weights` contract."""
    lm = load_model(model_id)
    state = lm.model.state_dict()
    if param not in state:
        raise NotFoundError(
            f"Model '{lm.model_id}' has no parameter '{param}'.",
            detail={"model_id": lm.model_id, "param": param},
        )
    mat = _as_matrix(state[param].detach().float().cpu().numpy())
    n_rows, n_cols = int(mat.shape[0]), int(mat.shape[1])

    r0, c0 = int(r0), int(c0)
    r1 = n_rows if r1 is None else int(r1)
    c1 = n_cols if c1 is None else int(c1)
    max_cells = int(max_cells)
    if max_cells < 1:
        raise InvalidParamError(f"max_cells must be >= 1, got {max_cells}")
    if not (0 <= r0 < r1 <= n_rows and 0 <= c0 < c1 <= n_cols):
        raise InvalidParamError(
            f"Window [{r0}:{r1}, {c0}:{c1}] is out of range for '{param}' "
            f"with shape [{n_rows}, {n_cols}]."
        )

    window = mat[r0:r1, c0:c1]
    stats = {
        "min": float(window.min()),
        "max": float(window.max()),
        "mean": float(window.mean()),
        "std": float(window.std()),
    }

    if window.size <= max_cells:
        values = window
        downsampled = False
        method = "exact"
    else:
        gr = min(window.shape[0], ARCH_ATTENTION_MAX_SIDE)
        gc = min(window.shape[1], ARCH_ATTENTION_MAX_SIDE)
        if gr * gc > max_cells:
            scale = (max_cells / (gr * gc)) ** 0.5
            gr, gc = max(1, int(gr * scale)), max(1, int(gc * scale))
        values = strided_mean_2d(window, gr, gc)
        downsampled = True
        method = "strided_mean"

    return {
        "param": param,
        "shape": [n_rows, n_cols],
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "downsampled": downsampled,
        "grid_shape": [int(values.shape[0]), int(values.shape[1])],
        "values": values.astype(float).tolist(),
        "stats": stats,
        "method": method,
    }
