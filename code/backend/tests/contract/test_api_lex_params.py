"""Integer parameters on the Lexicon Lab routes: refused, never coerced.

`_as_int` was `int(value)`, which accepts and silently REWRITES most of what JSON can
express. Measured against the running route by the red team:

    {"p":0.5,"seed":1.5}          200  seed=1
    {"p":0.5,"seed":"7"}          200  seed=7
    {"p":0.5,"seed":true}         200  seed=1
    {"p":0.5,"seed":Infinity}     500  InternalError: cannot convert float infinity to integer

The TypeScript engine refuses all four with a typed error, so the two stacks disagreed
across the whole non-integer domain — in the direction where Python computes with a
different seed than it was asked for, echoes it back, and nothing says it happened. That
is the defect class this campaign is about: a plausible answer to a question nobody asked.

Real HTTP through the real app, no mocks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.lex.vacancy import MAX_SEED

client = TestClient(app)

# Small enough to keep the transform fast, real enough to have vacatable open-class words.
TEXT = (
    "the little dog jumped over the sleeping cat and the running horses "
    "were singing while the ringing bells called the children home again "
    "and the walking men carried the heavy baskets down the winding road"
)


def _vacancy(seed_literal: str) -> tuple[int, dict]:
    """POST /api/lex/vacancy with `seed` written out as a raw JSON literal."""
    body = '{"text": %s, "p": 0.5, "seed": %s}' % (repr(TEXT).replace("'", '"'), seed_literal)
    resp = client.post(
        "/api/lex/vacancy", content=body, headers={"content-type": "application/json"}
    )
    return resp.status_code, resp.json()


@pytest.mark.parametrize(
    "literal",
    ["1.5", "2.0000000001", '"7"', "true", "false", "null", "[]", '{"a":1}'],
)
def test_a_seed_that_is_not_an_integer_is_refused_not_rewritten(literal: str) -> None:
    status, body = _vacancy(literal)
    assert status == 400, body
    assert body["error"]["type"] == "InvalidParamError"
    assert "seed" in body["error"]["message"]


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_a_non_finite_seed_is_a_typed_400_not_a_leaked_overflow(literal: str) -> None:
    """`Infinity` reached `int()` and escaped as a bare Python OverflowError in a 500."""
    status, body = _vacancy(literal)
    assert status == 400, body
    assert body["error"]["type"] == "InvalidParamError"
    assert "convert float infinity" not in body["error"]["message"]


@pytest.mark.parametrize("literal", ["0", "7", "-7", "7.0", str(MAX_SEED), str(-MAX_SEED)])
def test_an_integer_seed_is_used_exactly_as_given(literal: str) -> None:
    """Including `7.0`: JSON cannot express the int/float distinction and the TS engine
    reads it as the integer 7, so refusing it here would be a divergence of its own."""
    status, body = _vacancy(literal)
    assert status == 200, body
    assert body["seed"] == int(float(literal))


@pytest.mark.parametrize("literal", [str(MAX_SEED + 1), "12345678901234567890"])
def test_a_seed_beyond_the_javascript_integer_range_is_refused(literal: str) -> None:
    status, body = _vacancy(literal)
    assert status == 400, body
    assert body["error"]["type"] == "InvalidParamError"


def test_the_training_seed_carries_the_same_bound_as_the_vacancy_seed() -> None:
    """`POST /api/lex/train` bounded the vacancy seed it forwards and left its own
    top-level `seed` unbounded: `12345678901234567890` was accepted with a 202, and the
    seed reported back is not the seed a JavaScript reader gets from it."""
    resp = client.post(
        "/api/lex/train",
        json={"text": TEXT, "steps": 1, "seed": 12345678901234567890},
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()["error"]
    assert body["type"] == "InvalidParamError"
    assert "seed" in body["message"]


# -- float parameters: the same rule, one type down -----------------------------------------
#
# `_as_int` was rewritten and `_as_float` was left as bare `float(value)`, which accepts
# every silent rewrite the int version used to. Measured against the running route by the
# red team:
#
#     {"p": Infinity}   500  InternalError: Out of range float values are not JSON compliant
#
# `NaN` is quieter and worse: every `<` and `>` against it is False, so a range guard
# written as `if lr <= 0: raise` waves it through and the run diverges at step 1 with a
# message about the loss rather than about the request.


def _vacancy_p(p_literal: str) -> tuple[int, dict]:
    """POST /api/lex/vacancy with `p` written out as a raw JSON literal."""
    body = '{"text": %s, "p": %s, "seed": 0}' % (repr(TEXT).replace("'", '"'), p_literal)
    resp = client.post(
        "/api/lex/vacancy", content=body, headers={"content-type": "application/json"}
    )
    return resp.status_code, resp.json()


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_a_non_finite_float_is_a_typed_400_not_a_leaked_encoder_error(literal: str) -> None:
    status, body = _vacancy_p(literal)
    assert status == 400, body
    assert body["error"]["type"] == "InvalidParamError"
    assert "not JSON compliant" not in body["error"]["message"]


@pytest.mark.parametrize(
    "literal",
    [
        '"0.5"',  # a numeric string Python parses and JavaScript's `Number` also would
        '"1e-1"',  # exponent notation as text
        '" 0.5 "',  # surrounding whitespace
        '"٠.٥"',  # Arabic-Indic digits: Python's float() reads these, JS's Number() does not
        "true",  # float(True) == 1.0
        "false",
        "null",
        "[]",
        '{"a": 1}',
    ],
)
def test_a_float_that_is_not_a_json_number_is_refused_not_parsed(literal: str) -> None:
    """The two stacks do not agree about what a numeric STRING is, so neither reads one:
    otherwise one request body computes two different transforms."""
    status, body = _vacancy_p(literal)
    assert status == 400, body
    assert body["error"]["type"] == "InvalidParamError"
    assert body["error"]["message"].startswith("p must be a")


@pytest.mark.parametrize("literal", ["0", "0.0", "1", "1.0", "0.25"])
def test_a_json_number_in_range_is_used_exactly_as_given(literal: str) -> None:
    status, body = _vacancy_p(literal)
    assert status == 200, body
    assert body["p"] == float(literal)


@pytest.mark.parametrize(
    "field,literal",
    [
        ("lr", "Infinity"),
        ("lr", "NaN"),
        ("lr", '"1e-3"'),
        ("lr", "true"),
        ("weight_decay", "Infinity"),
        ("weight_decay", "NaN"),
        ("weight_decay", '"0.01"'),
        ("dropout", "NaN"),
        ("dropout", '"0.1"'),
    ],
)
def test_training_floats_are_refused_at_the_wire_not_mid_job(field: str, literal: str) -> None:
    """`lr = NaN` passed `if lr <= 0` (every comparison with NaN is False) and surfaced
    much later as `training diverged at step 2: the loss is nan` — a message about the
    model, for a defect in the request."""
    body = '{"text": %s, "steps": 1, "%s": %s}' % (
        repr(TEXT).replace("'", '"'),
        field,
        literal,
    )
    resp = client.post("/api/lex/train", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 400, resp.text
    payload = resp.json()["error"]
    assert payload["type"] == "InvalidParamError"
    assert payload["message"].startswith(f"{field} must be a")


def test_an_integer_too_large_for_a_double_is_a_typed_refusal() -> None:
    """`float(10**400)` raises OverflowError, which used to escape as an untyped 500."""
    body = '{"text": %s, "steps": 1, "lr": %s}' % (repr(TEXT).replace("'", '"'), "1" + "0" * 400)
    resp = client.post("/api/lex/train", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["type"] == "InvalidParamError"


def test_a_float_temperature_is_still_accepted_where_it_is_legitimate() -> None:
    """The refusal is of non-numbers, not of floats: an ordinary JSON float still works."""
    status, body = _vacancy_p("0.75")
    assert status == 200, body
    assert body["p"] == 0.75
