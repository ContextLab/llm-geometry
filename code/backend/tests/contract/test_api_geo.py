"""Contract tests for the Geometry Lab routes (`/api/geo/*`).

Everything runs against the real FastAPI app with the real GeoTransformer — real
training, real fine-tuning, real cache artifacts (no mocks, per CLAUDE.md). The
first test exercises the full missing -> training -> ready checkpoint cycle (one
real ~30 s CPU training run); later tests reuse the canonical cached checkpoint.
"""

from __future__ import annotations

import json
import time

import numpy as np
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.cache.store import CacheStore
from llm_geometry.geo.config import (
    FINETUNE_DEFAULT_LR,
    MIN_COVERAGE_UNIFORMITY,
    MIN_FIELD_DIRECTIONAL_ENTROPY,
    SEED,
    VOCAB_SIZE,
)
from llm_geometry.geo.finetune import finetune_cache_key
from llm_geometry.geo.train import canonical_cache_key, resolve_weight_set
from llm_geometry.geo.weights import weights_token as compute_weights_token

client = TestClient(app)


# -- helpers -----------------------------------------------------------------------------


def _wait_job(job_id: str, timeout: float = 300.0) -> dict:
    """Poll /api/jobs/{id} until a terminal state; return the final snapshot."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error"):
            return snap
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _sse_collect(job_id: str) -> list[tuple[str, dict]]:
    """Stream /api/jobs/{id}/events until the terminal event; return (event, data) pairs."""
    events: list[tuple[str, dict]] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        current = None
        for raw in stream.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode().strip()
            if not line:
                continue
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current:
                events.append((current, json.loads(line.split(":", 1)[1].strip())))
                if current in ("done", "error"):
                    break
            if len(events) > 2000:  # safety bound
                break
    return events


def _ensure_ready() -> str:
    """Make sure the canonical checkpoint exists; return its checkpoint_id."""
    resp = client.post("/api/geo/train", json={})
    assert resp.status_code in (200, 202)
    if resp.status_code == 202:
        snap = _wait_job(resp.json()["job_id"])
        assert snap["status"] == "done", snap
        resp = client.post("/api/geo/train", json={})
        assert resp.status_code == 200
    return resp.json()["checkpoint_id"]


def _assert_rounded6(value, path="$") -> None:
    """Every float in a response must round-trip 6-significant-digit rounding."""
    if isinstance(value, float):
        assert value == float(f"{value:.6g}"), f"{path}: {value!r} not 6-sig-digit rounded"
    elif isinstance(value, dict):
        for k, v in value.items():
            _assert_rounded6(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _assert_rounded6(v, f"{path}[{i}]")


# -- spec + train: the full checkpoint lifecycle (real training) -------------------------


def test_spec_and_train_missing_to_ready_cycle():
    store = CacheStore()
    key, _ = canonical_cache_key(SEED)
    store.delete(key)  # force the "missing" state; retrained below (deterministic)

    spec = client.get("/api/geo/spec").json()
    assert spec["model"] == {
        "d_model": 3,
        "n_layers": 4,
        "n_heads": 1,
        "mlp_hidden": 12,
        "vocab_size": 1003,
        "context_window": 50,
        "tied_unembedding": True,
        "corpus": "gutenberg-11-alice-in-wonderland",
        "seed": 0,
    }
    assert spec["special_tokens"] == {"unk": 0, "eos": 1, "pad": 2}
    assert spec["checkpoint"]["status"] == "missing"
    for field in (
        "checkpoint_id",
        "final_loss",
        "coverage_uniformity",
        "field_directional_entropy",
        "job_id",
    ):
        assert spec["checkpoint"][field] is None

    # Start real training (202) and verify single-flight on an immediate second POST.
    resp = client.post("/api/geo/train", json={"seed": 0})
    assert resp.status_code == 202
    body = resp.json()
    assert body["ready"] is False and body["job_id"]
    job_id = body["job_id"]
    again = client.post("/api/geo/train", json={"seed": 0})
    assert again.status_code == 202 and again.json()["job_id"] == job_id

    training_spec = client.get("/api/geo/spec").json()["checkpoint"]
    assert training_spec["status"] == "training" and training_spec["job_id"] == job_id

    # SSE: phase-labeled epoch progress, then a done event carrying checkpoint_id.
    events = _sse_collect(job_id)
    kinds = [k for k, _ in events]
    assert "error" not in kinds and kinds[-1] == "done"
    progress = [d for k, d in events if k == "progress"]
    assert any(d.get("phase") == "train" for d in progress)
    assert any("epoch" in d.get("message", "") for d in progress)
    done = events[-1][1]
    assert done["checkpoint_id"]

    ready = client.get("/api/geo/spec").json()["checkpoint"]
    assert ready["status"] == "ready" and ready["job_id"] is None
    assert ready["checkpoint_id"] == done["checkpoint_id"]
    assert ready["final_loss"] < np.log(VOCAB_SIZE)  # beat the uniform baseline
    assert ready["coverage_uniformity"] >= MIN_COVERAGE_UNIFORMITY
    assert ready["field_directional_entropy"] >= MIN_FIELD_DIRECTIONAL_ENTROPY

    # Idempotent: a cache hit now returns 200/complete with the same checkpoint_id.
    hit = client.post("/api/geo/train", json={})
    assert hit.status_code == 200
    assert hit.json() == {
        "checkpoint_id": done["checkpoint_id"],
        "status": "complete",
        "ready": True,
    }


# -- tokenize ----------------------------------------------------------------------------


def test_tokenize_unk_and_truncation():
    resp = client.get("/api/geo/tokenize", params={"text": "Alice was beginning to zzzqqq"})
    assert resp.status_code == 200
    body = resp.json()
    texts = [t["text"] for t in body["tokens"]]
    assert texts == ["alice", "was", "beginning", "to", "zzzqqq"]
    assert [t["unk"] for t in body["tokens"]] == [False, False, False, False, True]
    assert body["tokens"][-1]["id"] == 0  # <unk>
    assert body["n_unk"] == 1 and body["truncated"] is False

    long_text = " ".join(["alice"] * 60)
    body = client.get("/api/geo/tokenize", params={"text": long_text}).json()
    assert body["truncated"] is True and len(body["tokens"]) == 50


# -- trace -------------------------------------------------------------------------------


def test_trace_full_contract_shape_and_rounding():
    _ensure_ready()
    prompt = "alice was beginning to get very tired"
    resp = client.get("/api/geo/trace", params={"prompt": prompt})
    assert resp.status_code == 200
    body = resp.json()
    n = len(body["tokens"])
    assert n == 7 and all(set(t) == {"id", "text", "unk"} for t in body["tokens"])
    assert len(body["embeddings"]) == n and all(len(e) == 3 for e in body["embeddings"])
    for e in body["embeddings"]:  # unit-norm on S²
        assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-4

    assert len(body["layers"]) == 4
    for i, layer in enumerate(body["layers"]):
        assert layer["layer"] == i
        attn = np.asarray(layer["attention"])
        assert attn.shape == (n, n)
        assert np.allclose(attn.sum(axis=1), 1.0, atol=1e-4)  # row-stochastic
        assert np.all(np.triu(attn, k=1) == 0.0)  # causal
        for name in ("q", "k", "v", "hidden_in", "attn_out", "mlp_out", "hidden_out"):
            arr = np.asarray(layer[name])
            assert arr.shape == (n, 3), f"layer {i} {name}: {arr.shape}"

    assert len(body["probs"]) == 1003
    assert abs(sum(body["probs"]) - 1.0) < 1e-3
    topk = body["logits_topk"]
    assert len(topk["ids"]) == len(topk["texts"]) == len(topk["probs"]) == 10
    assert topk["probs"] == sorted(topk["probs"], reverse=True)
    assert body["next_token"]["id"] == topk["ids"][0]
    assert body["next_token"]["text"] == topk["texts"][0]

    # Contract-wide numeric encoding: 6 significant digits everywhere.
    _assert_rounded6(body)


def test_trace_empty_prompt_400():
    for prompt in ("", "   "):
        resp = client.get("/api/geo/trace", params={"prompt": prompt})
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["type"] == "InvalidParamError" and "empty" in err["message"]


# -- vector field ------------------------------------------------------------------------


def test_vector_field_next_next_t0_one_arrow_per_point():
    _ensure_ready()
    resp = client.get(
        "/api/geo/vector_field",
        params={"mode": "next_next", "layer": "full", "temperature": 0, "top_m": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "next_next" and body["layer"] == "full"
    assert len(body["points"]) == 1003 and body["token_ids"] == list(range(1003))
    assert len(body["arrows"]) == 1003
    origins = {a["origin_index"] for a in body["arrows"]}
    assert origins == set(range(1003))  # exactly one arrow per vocab point
    assert all(a["weight"] == 1.0 for a in body["arrows"])
    assert all(0 <= a["origin_index"] < len(body["points"]) for a in body["arrows"])
    assert body["sequence_forces"] is None and body["tangent_exact"] is False


def test_vector_field_force_antisymmetrized_tangent():
    _ensure_ready()
    resp = client.get(
        "/api/geo/vector_field",
        params={
            "mode": "force",
            "layer": 2,
            "antisymmetrize": "true",
            "prompt": "alice was beginning",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "force" and body["layer"] == 2
    assert body["tangent_exact"] is True
    points = np.asarray(body["points"])
    for a in body["arrows"]:
        assert 0.0 <= a["weight"] <= 1.0
        # antisymmetric W_V ⇒ the field is exactly tangent: ⟨z, (W−Wᵀ)/2 · z⟩ = 0
        assert abs(float(points[a["origin_index"]] @ np.asarray(a["vec"]))) < 1e-4
    forces = body["sequence_forces"]
    assert forces is not None and len(forces) == 3  # one per prompt position
    for i, f in enumerate(forces):
        assert f["position"] == i and len(f["vec"]) == 3
        assert f["normal_residual"] >= 0.0


def test_vector_field_force_with_full_layer_400():
    resp = client.get("/api/geo/vector_field", params={"mode": "force", "layer": "full"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "InvalidParamError"


def test_vector_field_edited_weights_change_the_field():
    _ensure_ready()
    minted = client.post(
        "/api/geo/weights",
        json={
            "base": "learned",
            "edits": [
                {"layer": layer, "matrix": "W_V", "preset": "identity"} for layer in range(4)
            ],
        },
    )
    assert minted.status_code == 200
    token = minted.json()["weights_token"]

    # Force mode, layer 0: with W_V = I the per-point field is exactly z itself.
    edited = client.get(
        "/api/geo/vector_field",
        params={"mode": "force", "layer": 0, "weights_token": token},
    ).json()
    points = np.asarray(edited["points"])
    for a in edited["arrows"][:50]:
        assert np.allclose(points[a["origin_index"]], a["vec"], atol=1e-4)
    learned = client.get("/api/geo/vector_field", params={"mode": "force", "layer": 0}).json()
    assert any(
        not np.allclose(la["vec"], ea["vec"], atol=1e-6)
        for la, ea in zip(learned["arrows"], edited["arrows"])
    )

    # And the next_next prediction field differs from the learned one somewhere.
    nn_edited = client.get(
        "/api/geo/vector_field",
        params={"mode": "next_next", "layer": "full", "weights_token": token},
    ).json()
    nn_learned = client.get(
        "/api/geo/vector_field", params={"mode": "next_next", "layer": "full"}
    ).json()
    assert any(la["vec"] != ea["vec"] for la, ea in zip(nn_learned["arrows"], nn_edited["arrows"]))


# -- weights GET/POST --------------------------------------------------------------------


def test_weights_learned_shapes_and_embedding_ignores_layer():
    _ensure_ready()
    body = client.get("/api/geo/weights", params={"matrix": "W_Q", "layer": 0}).json()
    assert body["shape"] == [3, 3] and body["source"] == "learned"
    assert len(body["values"]) == 3 and len(body["values"][0]) == 3

    # matrix=embedding ignores `layer` entirely (contract).
    body = client.get("/api/geo/weights", params={"matrix": "embedding", "layer": 99}).json()
    assert body["shape"] == [1003, 3] and body["source"] == "learned"


def test_weights_preset_roundtrip_and_inheritance():
    _ensure_ready()
    resp = client.post(
        "/api/geo/weights",
        json={"base": "learned", "edits": [{"layer": 2, "matrix": "W_V", "preset": "identity"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    token = body["weights_token"]
    assert body["edited"] == [{"layer": 2, "matrix": "W_V", "source": "preset:identity"}]

    got = client.get(
        "/api/geo/weights", params={"matrix": "W_V", "layer": 2, "weights_token": token}
    ).json()
    assert got["values"] == np.eye(3).tolist()
    assert got["source"] == "preset:identity" and got["shape"] == [3, 3]

    # Un-edited matrices of the minted set keep their canonical provenance.
    other = client.get(
        "/api/geo/weights", params={"matrix": "W_V", "layer": 1, "weights_token": token}
    ).json()
    assert other["source"] == "learned"
    learned = client.get("/api/geo/weights", params={"matrix": "W_V", "layer": 1}).json()
    assert other["values"] == learned["values"]


def test_weights_explicit_values_roundtrip():
    _ensure_ready()
    values = [[0.5, -0.25, 0.125], [1.0, 0.0, -1.0], [0.75, 0.375, -0.5]]
    resp = client.post(
        "/api/geo/weights",
        json={"base": "learned", "edits": [{"layer": 1, "matrix": "W_O", "values": values}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["edited"] == [{"layer": 1, "matrix": "W_O", "source": "edited"}]
    got = client.get(
        "/api/geo/weights",
        params={"matrix": "W_O", "layer": 1, "weights_token": body["weights_token"]},
    ).json()
    assert got["values"] == values and got["source"] == "edited"


def test_weights_invalid_edits_422_envelope():
    _ensure_ready()
    # Both preset and values.
    resp = client.post(
        "/api/geo/weights",
        json={
            "edits": [
                {"layer": 0, "matrix": "W_Q", "preset": "identity", "values": np.eye(3).tolist()}
            ]
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "InvalidWeightEditError"
    # Neither.
    resp = client.post("/api/geo/weights", json={"edits": [{"layer": 0, "matrix": "W_Q"}]})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "InvalidWeightEditError"
    # Bad shape.
    resp = client.post(
        "/api/geo/weights",
        json={"edits": [{"layer": 0, "matrix": "W_Q", "values": [[1.0, 2.0]]}]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "InvalidWeightEditError"


# -- finetune ----------------------------------------------------------------------------

_FT_TEXT = (
    "alice followed the white rabbit down the hole and found a little door behind "
    "the curtain where the queen of hearts was waiting with the mad hatter"
)


def test_finetune_json_job_mints_new_token_canonical_unchanged():
    checkpoint_id = _ensure_ready()
    store = CacheStore()
    base_token = compute_weights_token(resolve_weight_set("learned"))
    steps = 120
    key, _ = finetune_cache_key(base_token, _FT_TEXT, steps, FINETUNE_DEFAULT_LR, SEED)
    store.delete(key)  # force the real 202 fine-tuning job path

    before = client.get("/api/geo/weights", params={"matrix": "W_V", "layer": 0}).json()

    resp = client.post("/api/geo/finetune", json={"text": _FT_TEXT, "steps": steps})
    assert resp.status_code == 202
    body = resp.json()
    assert body["ready"] is False and body["job_id"]

    events = _sse_collect(body["job_id"])
    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and "error" not in kinds
    assert any(d.get("phase") == "finetune" for k, d in events if k == "progress")
    done = events[-1][1]
    assert done["weights_token"] and done["weights_token"] != checkpoint_id  # NEW token
    assert np.isfinite(done["loss_before"]) and np.isfinite(done["loss_after"])
    assert done["loss_after"] < done["loss_before"]  # it really learned the text

    # The canonical checkpoint is untouched.
    spec = client.get("/api/geo/spec").json()["checkpoint"]
    assert spec["status"] == "ready" and spec["checkpoint_id"] == checkpoint_id
    after = client.get("/api/geo/weights", params={"matrix": "W_V", "layer": 0}).json()
    assert after["values"] == before["values"]

    # Content-hash idempotency: the same request is now a 200 cache hit.
    hit = client.post("/api/geo/finetune", json={"text": _FT_TEXT, "steps": steps})
    assert hit.status_code == 200
    hit_body = hit.json()
    assert hit_body["ready"] is True
    assert hit_body["weights_token"] == done["weights_token"]
    _assert_rounded6(hit_body)


def test_finetune_multipart_file_upload():
    _ensure_ready()
    file_text = (
        "the gryphon and the mock turtle danced the lobster quadrille on the shore "
        "while alice watched and wondered about the trial of the knave of hearts"
    )
    store = CacheStore()
    base_token = compute_weights_token(resolve_weight_set("learned"))
    steps = 60
    key, _ = finetune_cache_key(base_token, file_text, steps, FINETUNE_DEFAULT_LR, SEED)
    store.delete(key)

    resp = client.post(
        "/api/geo/finetune",
        files={"file": ("wonderland.md", file_text.encode("utf-8"), "text/markdown")},
        data={"steps": str(steps)},
    )
    assert resp.status_code == 202
    snap = _wait_job(resp.json()["job_id"])
    assert snap["status"] == "done", snap
    assert snap["result"]["weights_token"]
    assert snap["result"]["loss_after"] < snap["result"]["loss_before"]


def test_finetune_source_validation_400():
    _ensure_ready()
    # Two sources.
    resp = client.post(
        "/api/geo/finetune", json={"text": "alice was here", "hf_dataset": "some/dataset"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "InvalidParamError"
    # No source.
    resp = client.post("/api/geo/finetune", json={})
    assert resp.status_code == 400
    # Wrong file extension.
    resp = client.post(
        "/api/geo/finetune", files={"file": ("essay.pdf", b"alice", "application/pdf")}
    )
    assert resp.status_code == 400
    # Out-of-range steps.
    resp = client.post("/api/geo/finetune", json={"text": "alice was here", "steps": 501})
    assert resp.status_code == 400


def test_finetune_unusable_hf_dataset_422():
    _ensure_ready()
    resp = client.post(
        "/api/geo/finetune",
        json={"hf_dataset": "llm-geometry-tests/definitely-not-a-real-dataset-xyz"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "UnsupportedModelError"


def test_framework_validation_failures_use_the_error_envelope():
    """Malformed typed params must return the contract envelope, not Starlette's
    {"detail": [...]} shape (red-team round 2, finding 1)."""
    # missing required query param
    r = client.get("/api/geo/weights")
    assert r.status_code == 400
    env = r.json()["error"]
    assert env["type"] == "InvalidParamError" and "matrix" in env["message"]
    # un-parseable typed param
    r = client.get("/api/geo/vector_field", params={"temperature": "abc"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "InvalidParamError"
    # arch router goes through the same app-level handler
    r = client.get("/api/arch/weights", params={"model_id": "gpt2", "r1": "abc"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "InvalidParamError"
