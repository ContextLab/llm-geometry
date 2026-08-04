"""Parameter handling on `/api/geo/*`: typed refusals, never coercions, never 500s.

Red-team round 4 findings F3 and item 6 (geo half), each reproduced against the real app.
Two defect shapes, both of which shipped:

* **Coercion.** ``int(value)`` rewrote every parameter JSON can express — ``1.5`` became
  ``1``, ``"7"`` became ``7``, ``true`` became ``1`` — so the backend ran with a different
  number than it was asked for, echoed the run back, and nothing said so. ``Infinity``
  did not even coerce: it escaped as ``500 {"type": "InternalError", "message": "cannot
  convert float infinity to integer"}``.
* **Untyped 500s on a file the user chose.** Seven malformed ``vocab`` blocks reached
  ``POST /api/geo/model`` as HTTP 500 whose whole message was a Python exception string
  (``'list' object has no attribute 'get'``, ``unhashable type: 'list'``, ``'words'``).

Real app, real checkpoint, no mocks. The canonical checkpoint is trained once here rather
than borrowed from whichever module happened to run first.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.geo.bundle import vocab_digest
from llm_geometry.geo.config import FINETUNE_MAX_STEPS, MAX_SEED

client = TestClient(app)

_TEXT = (
    "alice followed the white rabbit down the hole and found a little door behind "
    "the curtain where the queen of hearts was waiting with the mad hatter"
)


@pytest.fixture(scope="module")
def canonical_ready() -> None:
    """The real canonical checkpoint, in this session's throwaway cache dir."""
    from llm_geometry.geo.train import train_canonical

    train_canonical()


def _post_json(path: str, payload: dict[str, Any]):
    """POST a body that may contain `Infinity`/`NaN` (json.dumps emits them literally)."""
    return client.post(
        path, content=json.dumps(payload), headers={"content-type": "application/json"}
    )


def _error(resp) -> dict[str, Any]:
    body = resp.json()
    assert "error" in body, body
    return body["error"]


# -- item 6 (geo half): integers are integers -------------------------------------------


@pytest.mark.parametrize(
    "steps",
    [1.5, "7", True, False, None, [], float("inf"), float("-inf"), float("nan"), "0x10", "٧"],
)
def test_finetune_refuses_a_steps_value_that_is_not_an_integer(steps: Any) -> None:
    resp = _post_json("/api/geo/finetune", {"text": _TEXT, "steps": steps})
    assert resp.status_code == 400, (steps, resp.status_code, resp.text)
    assert _error(resp)["type"] == "InvalidParamError"
    assert "steps must be" in _error(resp)["message"]


def test_finetune_accepts_a_json_float_that_is_a_whole_number() -> None:
    """JSON cannot express the int/float distinction and the TS engine reads 700.0 as 700,
    so refusing it here would be a divergence of its own. Proven by the range message: the
    value reached the bound check as the integer 700."""
    resp = _post_json("/api/geo/finetune", {"text": _TEXT, "steps": 700.0})
    assert resp.status_code == 400
    assert _error(resp)["message"] == f"steps must be in 1..{FINETUNE_MAX_STEPS}, got 700"


@pytest.mark.parametrize("lr", [float("inf"), float("-inf"), float("nan")])
def test_finetune_refuses_a_non_finite_learning_rate(lr: float) -> None:
    """`lr = Infinity` passes `lr > 0`: the job started, every parameter became NaN on the
    first step, and the failure surfaced (if at all) as a late job error event."""
    resp = _post_json("/api/geo/finetune", {"text": _TEXT, "lr": lr})
    assert resp.status_code == 400, resp.text
    assert _error(resp)["type"] == "InvalidParamError"
    assert "lr must be" in _error(resp)["message"]


def test_finetune_multipart_still_parses_its_string_fields() -> None:
    """Multipart carries no types, so `steps=4` arrives as the STRING "4" and always did.
    The strict rule must not break the upload form: the string is parsed, then held to the
    same bound. `steps=0` proves it parsed (a refusal from the RANGE, not from the type)."""
    resp = client.post(
        "/api/geo/finetune",
        files={"file": ("alice.txt", _TEXT.encode(), "text/plain")},
        data={"steps": "0"},
    )
    assert resp.status_code == 400, resp.text
    assert _error(resp)["message"] == f"steps must be in 1..{FINETUNE_MAX_STEPS}, got 0"

    # ...and a form field that is not a number at all is still a typed refusal, not an
    # accidental `int("0x10", 0)`-style parse.
    for junk in ("abc", "0x10", "1.5", "٧"):
        resp = client.post(
            "/api/geo/finetune",
            files={"file": ("alice.txt", _TEXT.encode(), "text/plain")},
            data={"steps": junk},
        )
        assert resp.status_code == 400, (junk, resp.text)
        assert "steps must be an integer" in _error(resp)["message"], junk


def test_train_scratch_refuses_a_non_integer_epochs() -> None:
    resp = _post_json("/api/geo/train_scratch", {"text": _TEXT, "epochs": float("inf")})
    assert resp.status_code == 400, resp.text
    assert _error(resp)["type"] == "InvalidParamError"


# -- item 6 (geo half): the seed is bounded ----------------------------------------------


@pytest.mark.parametrize("seed", [MAX_SEED + 1, 2**63, 10**40, -(MAX_SEED + 2)])
def test_train_refuses_a_seed_javascript_cannot_read_back(seed: int) -> None:
    """`{"seed": 9007199254740993}` answered 202 and echoed back an integer the browser
    reads as ...992 — a run reported under a seed nobody asked for."""
    resp = _post_json("/api/geo/train", {"seed": seed})
    assert resp.status_code == 400, (seed, resp.status_code, resp.text)
    assert _error(resp)["type"] == "InvalidParamError"
    assert "seed must lie in" in _error(resp)["message"]


@pytest.mark.parametrize("seed", [1.5, "7", True, [], float("inf")])
def test_train_refuses_a_seed_that_is_not_an_integer(seed: Any) -> None:
    resp = _post_json("/api/geo/train", {"seed": seed})
    assert resp.status_code == 400, (seed, resp.status_code, resp.text)
    assert _error(resp)["type"] == "InvalidParamError"


@pytest.mark.parametrize("seed", [1.5, "7", True, float("inf"), float("nan"), -1, MAX_SEED + 1])
def test_weight_edit_refuses_a_seed_that_would_select_another_matrix(
    seed: Any, canonical_ready: None
) -> None:
    """A seed picks WHICH matrix you get. `int(seed or 0)` made 1.5 the seed-1 matrix and
    `-1` an untyped 500 out of `np.random.default_rng`, both reported as the request."""
    resp = _post_json(
        "/api/geo/weights",
        {"edits": [{"layer": 0, "matrix": "W_Q", "preset": "random", "seed": seed}]},
    )
    assert resp.status_code == 422, (seed, resp.status_code, resp.text)
    assert _error(resp)["type"] == "InvalidWeightEditError"
    assert "seed must be" in _error(resp)["message"]


def test_weight_edit_refuses_a_boolean_layer(canonical_ready: None) -> None:
    """`isinstance(True, int)` is True in Python, so `layer: true` edited layer 1."""
    resp = _post_json(
        "/api/geo/weights",
        {"edits": [{"layer": True, "matrix": "W_Q", "preset": "identity"}]},
    )
    assert resp.status_code == 422, resp.text
    assert "layer must be an int" in _error(resp)["message"]


def test_a_real_edit_still_works(canonical_ready: None) -> None:
    resp = _post_json(
        "/api/geo/weights",
        {"edits": [{"layer": 0, "matrix": "W_Q", "preset": "random", "seed": 2}]},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["weights_token"]) == 32


# -- F3: a malformed vocabulary is a typed 400, and both stacks agree on which ------------


@pytest.fixture(scope="module")
def real_bundle(canonical_ready: None) -> dict[str, Any]:
    resp = client.get("/api/geo/model?weights_token=learned")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _with_vocab(bundle: dict[str, Any], vocab: str) -> dict[str, Any]:
    """The bundle with a substituted vocabulary and its digest HONESTLY recomputed, so the
    vocabulary check is what refuses it rather than the digest."""
    return {**bundle, "vocab": vocab, "vocab_sha256": vocab_digest(vocab)}


@pytest.mark.parametrize(
    "vocab,expected",
    [
        ("{", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('"hi"', "must be a JSON object"),
        ("null", "must be a JSON object"),
        ('{"format":"geo-tokenizer-v1","words":null}', "must be an array of strings"),
        ('{"format":"geo-tokenizer-v1","words":[["a"]]}', "must be an array of strings"),
        ('{"format":"geo-tokenizer-v1","specials":"x","words":[]}', "`specials` must be an object"),
    ],
)
def test_model_upload_refuses_a_malformed_vocabulary_with_a_typed_400(
    real_bundle: dict[str, Any], vocab: str, expected: str
) -> None:
    resp = client.post("/api/geo/model", json=_with_vocab(real_bundle, vocab))
    assert resp.status_code == 400, (vocab, resp.status_code, resp.text)
    assert _error(resp)["type"] == "InvalidParamError"
    assert expected in _error(resp)["message"], (vocab, _error(resp)["message"])


def test_model_upload_refuses_a_vocabulary_whose_specials_are_not_ours(
    real_bundle: dict[str, Any],
) -> None:
    """`specials` was ignored here and validated in the browser engine, so this file
    loaded with HTTP 200 in one stack (`<unk>` still read as 0) and was refused in the
    other: the same file, valid or invalid depending on which build opened it."""
    words = json.loads(real_bundle["vocab"])["words"]
    vocab = json.dumps(
        {
            "format": "geo-tokenizer-v1",
            "specials": {"<unk>": 5, "<eos>": 1, "<pad>": 2},
            "words": words,
        }
    )
    resp = client.post("/api/geo/model", json=_with_vocab(real_bundle, vocab))
    assert resp.status_code == 400, resp.text
    assert "special <unk> has id 5, expected 0" in _error(resp)["message"]


def test_model_upload_refuses_the_sites_tokens_shaped_export(real_bundle: dict[str, Any]) -> None:
    """The static site's own `vocab.json` shape. Accepted by the browser engine (1000
    words recovered from the 1003 entries) and an untyped 500 here — one of the two shapes
    on which the stacks disagreed about whether a file was valid at all."""
    words = json.loads(real_bundle["vocab"])["words"]
    vocab = json.dumps(
        {
            "format": "geo-tokenizer-v1",
            "specials": {"unk": 0, "eos": 1, "pad": 2},
            "tokens": ["<unk>", "<eos>", "<pad>", *words],
        }
    )
    resp = client.post("/api/geo/model", json=_with_vocab(real_bundle, vocab))
    assert resp.status_code == 400, resp.text
    assert "`tokens` but no `words`" in _error(resp)["message"]


def test_a_real_model_file_still_loads(real_bundle: dict[str, Any]) -> None:
    resp = client.post("/api/geo/model", json=real_bundle)
    assert resp.status_code == 200, resp.text
    assert resp.json()["weights_token"] == real_bundle["weights_token"]
