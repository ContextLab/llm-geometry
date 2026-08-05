"""What a number is, decided ONCE for every route module in this API.

``routes_geo`` and ``routes_lex`` each grew their own copy of these three functions, with
a comment in one of them saying they were "duplicated rather than shared so the two tabs'
route modules stay independent". Independence is not what happened: the copies drifted,
and drifted in the direction that answers rather than refuses. Measured on the running
routes at the point this module was extracted —

    POST /api/lex/train  {"lr": "٠.٥"}   400  InvalidParamError: lr must be a number …
    POST /api/geo/finetune {"lr": "٠.٥"} 202  ACCEPTED, and the run trains at lr = 0.5
    POST /api/geo/finetune {"lr": 10**400}  500  InternalError: int too large to convert

— so one tab refused Arabic-Indic digits and the other read them, in an API whose whole
correctness argument is that the Python backend and the in-browser TypeScript engine
compute the same numbers from the same request. ``Number("٠.٥")`` is ``NaN`` in
JavaScript. Sharing the implementation is the only version of "they agree" that a later
edit cannot quietly undo.

The rule, for both types:

* ``bool`` is refused. It is an ``int`` subclass in Python, so ``true`` became ``1``.
* ``int`` and ``float`` are accepted; everything else (``str``, ``None``, ``list``,
  ``dict``) is refused rather than coerced.
* ``7.0`` IS 7. JSON cannot express the int/float distinction and the TypeScript engine
  reads it as the integer 7, so refusing it would be a divergence of its own.
* ``7.5`` is refused, not truncated: a number that is not the number you asked for is
  worse than a refusal.
* Non-finite (``Infinity``, ``NaN``) is refused. ``NaN`` is the worst of them because
  nothing throws: every ``<`` and ``>`` against it is ``False``, so a guard written as
  ``if lr <= 0: raise`` waves it through and the run diverges at step 1. ``Infinity``
  survives the same guards and then dies in the JSON encoder as an untyped 500.
* An ``int`` too large for a float64 (``10**400``) is refused with a typed error rather
  than leaking ``OverflowError`` as a 500.

Strings are refused rather than parsed because Python's ``float``/``int`` and JavaScript's
``Number`` do not agree about what a numeric string is — ``"٧"``, ``"７"``, ``"७"``,
``"1_000"``, ``"0x10"``, ``"0b101"``, ``"1e3"`` — so parsing them would let one request
body mean two different numbers depending on which build served it. Every JSON caller
reads a body, never a query string; the one place a number legitimately arrives as text is
a multipart form field, which is parsed strictly and only there (``routes_geo._form_int``).
"""

from __future__ import annotations

import math
from typing import Any

from ..errors import InvalidParamError


def as_int(value: Any, name: str) -> int:
    """A JSON integer, or a typed refusal — never a coercion."""
    if isinstance(value, bool):
        raise InvalidParamError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidParamError(f"{name} must be a finite integer, got {value!r}")
        if not value.is_integer():
            raise InvalidParamError(
                f"{name} must be an integer, got {value!r} — it is not rounded or "
                "truncated, because a number that is not the number you asked for is "
                "worse than a refusal"
            )
        return int(value)
    raise InvalidParamError(f"{name} must be an integer, got {value!r}")


def as_float(value: Any, name: str) -> float:
    """A FINITE JSON number, or a typed refusal — :func:`as_int`'s rule, one type down."""
    if isinstance(value, bool):
        raise InvalidParamError(f"{name} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:  # an int too large for a double, e.g. 10**400
            raise InvalidParamError(
                f"{name} must be a number a float64 can represent exactly enough to use, "
                f"got {value!r}"
            )
        if not math.isfinite(number):
            raise InvalidParamError(f"{name} must be a finite number, got {value!r}")
        return number
    raise InvalidParamError(f"{name} must be a number, got {value!r}")


def as_bool(value: Any, name: str) -> bool:
    """A JSON boolean, or the four strings a form field can spell one with."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
        return value.lower() in ("true", "1")
    raise InvalidParamError(f"{name} must be a boolean, got {value!r}")
