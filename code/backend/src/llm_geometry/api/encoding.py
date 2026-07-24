"""Contract-wide JSON encoding shared by the feature-002 routers.

One recursive coercion: numpy -> plain Python, every float rounded to 6 significant
digits (``%.6g``), bools checked before ints (bool subclasses int). Non-finite floats
raise — the contract promises finite numbers, and silently nulling or clamping a NaN
would be a fabricated result (Constitution I); the 500 envelope carries the real error.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def jsonable_6sig(value: Any) -> Any:
    """Coerce ``value`` to plain JSON with floats rounded to 6 significant digits."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError("non-finite value in response payload")
        return float(f"{f:.6g}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return jsonable_6sig(value.tolist())
    if isinstance(value, dict):
        return {k: jsonable_6sig(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_6sig(v) for v in value]
    return value
