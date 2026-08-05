"""Two route modules in one API must not disagree about what a number is.

ROUND 5 F2. ``routes_lex._as_float`` was rewritten to refuse strings and non-finite
values; ``routes_geo._as_float`` was left as a bare ``float(value)`` inside a ``try``.
Measured on the running app by the verifier:

    POST /api/geo/finetune {"lr": "٠.٥"}    202 ACCEPTED — and the run really trains at 0.5
    POST /api/lex/train    {"lr": "٠.٥"}    400 InvalidParamError: lr must be a number …
    POST /api/geo/finetune {"lr": 10**400}  500 InternalError: int too large to convert

``Number("٠.٥")`` is ``NaN`` in JavaScript, so the value the Python backend trained at is
one the in-browser build cannot express at all — a divergence with a plausible number on
the other side of it, which is this campaign's whole subject.

The fix is one implementation (:mod:`llm_geometry.api.params`) rather than two that agree
by inspection, and this file is the check that they still agree — it sends the identical
value to both tabs and requires the identical answer. Real app, real routes, no mocks.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app

client = TestClient(app)

#: Every value the two `_as_float` implementations were observed to disagree about, plus
#: the shapes JSON can carry that neither should accept. Spelled out rather than generated
#: so a reader can see exactly what is claimed.
HOSTILE_NUMBERS: list[Any] = [
    # Numeric strings Python's `float` reads and JavaScript's `Number` does not.
    "٧",  # arabic-indic seven
    "٠.٥",  # arabic-indic 0.5
    "７",  # fullwidth seven
    "७",  # devanagari seven
    "1_000",  # PEP 515 underscore
    # Numeric strings BOTH languages read — still refused, because a string is not a
    # number on a JSON body and accepting it invites the ones above.
    "7",
    "+7",
    " 7 ",
    "1e3",
    "0x10",
    "0b101",
    "0o17",
    "",
    # Non-strings that are not numbers.
    None,
    True,
    False,
    [],
    {},
    [1],
    {"value": 1},
    # Numbers that are not usable numbers.
    float("inf"),
    float("-inf"),
    float("nan"),
    10**400,  # an int no float64 can hold: used to leak OverflowError as a 500
]

#: `(path, body)` — the body is complete except for the field under test, so the ONLY
#: reason either route can answer 400 is the value handed to it.
_GEO = ("/api/geo/finetune", {"text": "alice followed the white rabbit down the hole"})
_LEX = ("/api/lex/train", {"text": "the little children ran through the garden " * 12})


def _post(path: str, payload: dict[str, Any]):
    """POST a body that may contain `Infinity`/`NaN` (json.dumps emits them literally)."""
    return client.post(
        path, content=json.dumps(payload), headers={"content-type": "application/json"}
    )


@pytest.mark.parametrize("value", HOSTILE_NUMBERS, ids=lambda v: repr(v)[:24])
def test_both_tabs_refuse_the_same_learning_rate(value: Any) -> None:
    answers = {}
    for label, (path, body) in (("geo", _GEO), ("lex", _LEX)):
        resp = _post(path, {**body, "lr": value})
        answers[label] = (resp.status_code, resp.json().get("error", {}).get("type"))
        assert resp.status_code == 400, f"{label} {path} answered {resp.status_code}: {resp.text}"
        assert resp.json()["error"]["type"] == "InvalidParamError", resp.text
        assert "lr must be" in resp.json()["error"]["message"], resp.text
    assert answers["geo"] == answers["lex"], answers


def test_a_learning_rate_that_is_a_number_reaches_the_range_check_on_both() -> None:
    """The gate refuses TYPES, not values: a real, in-range float must still get through
    to the check that knows what a learning rate means — otherwise "it refuses everything"
    would pass this file.
    """
    for path, body in (_GEO, _LEX):
        resp = _post(path, {**body, "lr": -1.0})
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["message"] == "lr must be > 0, got -1.0", (path, resp.text)


@pytest.mark.parametrize("value", HOSTILE_NUMBERS, ids=lambda v: repr(v)[:24])
def test_both_tabs_refuse_the_same_step_count(value: Any) -> None:
    """The integer half, for the same reason: `steps` decides how long a run is, and a
    coerced one reports a run nobody asked for."""
    for label, (path, body) in (("geo", _GEO), ("lex", _LEX)):
        resp = _post(path, {**body, "steps": value})
        assert resp.status_code == 400, f"{label} {path} answered {resp.status_code}: {resp.text}"
        assert resp.json()["error"]["type"] == "InvalidParamError", resp.text
        assert "steps must be" in resp.json()["error"]["message"], resp.text


def test_a_whole_number_float_is_the_integer_it_spells_on_both_tabs() -> None:
    """JSON cannot express the int/float distinction and the TS engines read `7.0` as 7,
    so refusing it would be a divergence of its own. Proven by the RANGE message, which
    only a value that passed the type gate can reach."""
    for path, body in (_GEO, _LEX):
        resp = _post(path, {**body, "steps": 0.0})
        assert resp.status_code == 400, resp.text
        assert "got 0" in resp.json()["error"]["message"], (path, resp.text)
        assert "0.0" not in resp.json()["error"]["message"], (path, resp.text)
