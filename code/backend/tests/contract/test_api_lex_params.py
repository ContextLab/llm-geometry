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
