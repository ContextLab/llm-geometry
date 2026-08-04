"""Contract tests for the Lexicon Lab routes (`/api/lex/*`).

Everything runs against the real FastAPI app with the real model: real training, real
generation, real cache artifacts, real spectra. No mocks anywhere (Constitution I /
CLAUDE.md) — the models are deliberately tiny (`d_model=16..32`, tens of steps) so a
real run finishes in well under a second and the suite still measures the real thing.

The contract under test is `specs/006-lexicon-lab-tiny/contracts/api-lex.md`. The frozen
feature-002 contract is untouched by this feature; `test_frozen_geo_contract_untouched`
pins that claim rather than leaving it as an assertion in prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.lex.config import (
    CORPUS_SHA256,
    D_MODEL_CHOICES,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    PCA_COMPONENTS,
    SPECIAL_TOKENS,
    param_count,
)
from llm_geometry.lex.corpus import load_corpus_text
from llm_geometry.lex.dolch import DOLCH_ORDER, dolch_sizes
from llm_geometry.lex.vocab import build_vocab

client = TestClient(app)

#: `<repo>/code/backend/tests/contract/test_api_lex.py` -> `<repo>`. The vacancy parity
#: fixture lives beside the frontend tests that consume it, and this file asserts against
#: the same copy so the two stacks cannot be pinned to two different documents.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: A real but deliberately cheap model: a full run of this is a fraction of a second.
TINY_TRAIN: dict[str, Any] = {
    "source": "dolch",
    "budget": "pre_primer",
    "steps": 40,
    "d_model": 16,
    "n_layers": 1,
    "n_heads": 1,
    "ctx": 32,
    "batch_size": 16,
    "sample_every": 20,
}


# -- helpers ---------------------------------------------------------------------------


def _wait_job(job_id: str, timeout: float = 300.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error"):
            return snap
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _train(**overrides: Any) -> dict[str, Any]:
    """Train (or hit the cache) and return the result object either way."""
    body = {**TINY_TRAIN, **overrides}
    resp = client.post("/api/lex/train", json=body)
    assert resp.status_code in (200, 202), resp.text
    if resp.status_code == 200:
        assert resp.json()["ready"] is True
        return resp.json()
    snap = _wait_job(resp.json()["job_id"])
    assert snap["status"] == "done", snap
    return snap["result"]


def _assert_error_envelope(resp: Any, status: int, error_type: str) -> None:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    assert set(body["error"]) == {"type", "message", "detail"}, body
    assert body["error"]["type"] == error_type, body
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def _assert_rounded6(value: Any, path: str = "$") -> None:
    """Contract: every float in a response is rounded to 6 significant digits."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float):
        assert math.isfinite(value), f"{path} is not finite"
        assert float(f"{value:.6g}") == value, f"{path} = {value!r} is not 6-sig rounded"
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_rounded6(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_rounded6(item, f"{path}[{index}]")


# -- GET /api/lex/spec -----------------------------------------------------------------


def test_spec_reports_measured_budgets_and_the_verified_corpus() -> None:
    resp = client.get("/api/lex/spec")
    assert resp.status_code == 200
    body = resp.json()
    _assert_rounded6(body)

    assert body["corpus"]["sha256"] == CORPUS_SHA256
    assert body["corpus"]["n_tokens"] > 10_000
    assert body["budget_sources"] == ["dolch", "frequency"]

    # FR-602: sizes are measured from the data, not quoted. 314, not 315.
    measured = dolch_sizes()
    assert [b["name"] for b in body["budgets"]] == DOLCH_ORDER
    for entry in body["budgets"]:
        assert entry["size"] == measured[entry["name"]]
        assert entry["rows"] == entry["size"] + len(SPECIAL_TOKENS)
    assert body["budgets"][-1]["size"] == 314

    assert body["special_tokens"] == {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
    assert body["generation_banned_ids"] == [0, 1, 3]  # <eos> IS sampleable
    assert body["model"]["d_model_choices"] == list(D_MODEL_CHOICES)
    assert body["spectrum"]["pca_components"] == PCA_COMPONENTS
    assert body["training"]["max_steps"] == MAX_STEPS


# -- GET /api/lex/budgets --------------------------------------------------------------


def test_budgets_carry_real_coverage_and_the_verified_parameter_count() -> None:
    resp = client.get("/api/lex/budgets?source=dolch&d_model=64&n_layers=2&ctx=64&tied=true")
    assert resp.status_code == 200
    body = resp.json()
    _assert_rounded6(body)

    assert [b["budget"] for b in body["budgets"]] == DOLCH_ORDER
    coverages = [b["coverage"]["token_coverage"] for b in body["budgets"]]
    assert all(coverages[i] < coverages[i + 1] for i in range(len(coverages) - 1)), coverages

    corpus = load_corpus_text()
    for entry in body["budgets"]:
        expected = build_vocab("dolch", entry["budget"], corpus).coverage(corpus)
        assert entry["coverage"]["in_budget_tokens"] == expected.in_budget_tokens
        assert entry["coverage"]["whole_lines_in_budget"] == expected.whole_lines_in_budget
        assert abs(entry["coverage"]["unk_rate"] - expected.unk_rate) < 1e-6
        assert entry["param_count"] == param_count(
            vocab_rows=entry["rows"], d_model=64, n_layers=2, ctx=64, tied=True
        )


def test_budgets_rejects_shapes_outside_the_enumerated_choices() -> None:
    _assert_error_envelope(client.get("/api/lex/budgets?d_model=17"), 400, "InvalidParamError")
    _assert_error_envelope(client.get("/api/lex/budgets?source=nope"), 400, "InvalidParamError")
    _assert_error_envelope(client.get("/api/lex/budgets?n_heads=3"), 400, "InvalidParamError")


def test_every_enumerated_shape_pair_is_actually_valid() -> None:
    """FR-611 says the UI enforces `d_model % n_heads == 0`; the enums must allow it."""
    bad = [(d, h) for d in D_MODEL_CHOICES for h in (1, 2, 4) if d % h]
    assert bad == [], f"the enumerated choices should never conflict, but {bad} do"


# -- POST /api/lex/coverage ------------------------------------------------------------


def test_coverage_matches_a_backend_computation_of_the_same_quantity() -> None:
    """SC-603: the numbers shown match a backend computation exactly."""
    corpus = load_corpus_text()
    resp = client.post("/api/lex/coverage", json={"source": "dolch", "budget": "primer"})
    assert resp.status_code == 200
    body = resp.json()
    _assert_rounded6(body)

    expected = build_vocab("dolch", "primer", corpus)
    assert body["words"] == list(expected.words)
    assert body["size"] == expected.budget_size == 92
    assert body["rows"] == expected.rows
    assert body["coverage"]["in_budget_tokens"] == expected.coverage(corpus).in_budget_tokens
    assert body["oov_sample"] and all(
        item["word"] not in set(expected.words) for item in body["oov_sample"]
    )


def test_switching_budget_source_at_matched_size_changes_coverage() -> None:
    """SC-603: at the same |V| the two sources measurably differ."""
    for budget in DOLCH_ORDER:
        dolch = client.post("/api/lex/coverage", json={"source": "dolch", "budget": budget}).json()
        freq = client.post(
            "/api/lex/coverage", json={"source": "frequency", "budget": budget}
        ).json()
        assert dolch["size"] == freq["size"], budget
        assert set(dolch["words"]) != set(freq["words"]), budget
        # The corpus's own top-N always covers at least as much of it as a prescribed
        # list of the same size — that is what "descriptive" means, and it is measured.
        assert freq["coverage"]["token_coverage"] > dolch["coverage"]["token_coverage"], budget


def test_coverage_measures_a_users_own_text() -> None:
    text = "the little dog laughed to see such sport\nthe dish ran away with the spoon\n"
    body = client.post(
        "/api/lex/coverage", json={"source": "dolch", "budget": "full", "text": text}
    ).json()
    assert body["corpus"]["n_tokens"] == 15  # 8 words + 7 words
    assert 0.0 < body["coverage"]["token_coverage"] <= 1.0
    assert body["coverage"]["total_lines"] == 2


def test_coverage_rejects_size_on_a_dolch_budget() -> None:
    resp = client.post("/api/lex/coverage", json={"source": "dolch", "budget": "full", "size": 50})
    _assert_error_envelope(resp, 400, "InvalidParamError")


def test_coverage_rejects_an_unknown_budget() -> None:
    resp = client.post("/api/lex/coverage", json={"source": "dolch", "budget": "kindergarten"})
    _assert_error_envelope(resp, 400, "InvalidParamError")


def test_coverage_rejects_text_with_no_words() -> None:
    resp = client.post("/api/lex/coverage", json={"text": "!!! ??? ---"})
    _assert_error_envelope(resp, 400, "InvalidParamError")


# -- POST /api/lex/train ---------------------------------------------------------------


def test_train_runs_a_real_model_and_is_cached_and_single_flight() -> None:
    resp = client.post("/api/lex/train", json=TINY_TRAIN)
    assert resp.status_code in (200, 202), resp.text

    if resp.status_code == 202:
        job_id = resp.json()["job_id"]
        assert resp.json()["ready"] is False
        # Single-flight: an identical request while the first is in flight or cached
        # never starts a second computation.
        again = client.post("/api/lex/train", json=TINY_TRAIN)
        assert again.status_code in (200, 202)
        if again.status_code == 202:
            assert again.json()["job_id"] == job_id
        snap = _wait_job(job_id)
        assert snap["status"] == "done", snap
        assert snap["phase"] == "lex_train"
        result = snap["result"]
    else:
        result = resp.json()

    assert len(result["model_token"]) == 32
    assert result["first_loss"] > result["final_loss"], result  # it really learned
    assert result["steps"] == TINY_TRAIN["steps"]
    assert result["vocab_size"] == dolch_sizes()["pre_primer"]
    assert result["vocab_rows"] == result["vocab_size"] + len(SPECIAL_TOKENS)
    assert result["param_count"] == param_count(
        vocab_rows=result["vocab_rows"], d_model=16, n_layers=1, ctx=32, tied=True
    )
    assert result["sample"].strip()

    started = time.time()
    cached = client.post("/api/lex/train", json=TINY_TRAIN)
    assert cached.status_code == 200
    assert time.time() - started < 1.0, "a cache hit must be fast"
    assert cached.json()["ready"] is True
    assert cached.json()["model_token"] == result["model_token"]
    assert cached.json()["history"][0]["step"] == 1
    _assert_rounded6(cached.json())


def test_train_streams_progress_and_samples_over_sse() -> None:
    """FR-618: step, loss, LR and a real generated sample, live.

    Long enough (a few seconds of real training) that the ~4 Hz SSE poller is guaranteed
    several ticks — a 40-step run finishes before the first poll and would make this a
    race rather than a test.
    """
    body = {**TINY_TRAIN, "seed": 7, "steps": 600, "sample_every": 50}
    resp = client.post("/api/lex/train", json=body)
    if resp.status_code == 200:  # already cached from an earlier run; force a new one
        body = {**body, "seed": int(time.time()) % 100000}
        resp = client.post("/api/lex/train", json=body)
    assert resp.status_code == 202, resp.text

    events: list[tuple[str, dict[str, Any]]] = []
    with client.stream("GET", f"/api/jobs/{resp.json()['job_id']}/events") as stream:
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
            if len(events) > 2000:
                break

    kinds = [name for name, _ in events]
    assert "done" in kinds and "error" not in kinds, events[-3:]
    progress = [data for name, data in events if name == "progress"]
    assert progress, "training must stream progress"
    assert all(item.get("phase") == "lex_train" for item in progress)
    messages = [item.get("message", "") for item in progress]
    assert any("loss" in m and "lr" in m and "step" in m for m in messages), messages
    # FR-618: at least one tick carries a real sample generated mid-run.
    assert any(m.count(" · ") >= 3 for m in messages), messages
    done = [data for name, data in events if name == "done"][-1]
    assert len(done["model_token"]) == 32
    assert done["steps"] == 600


def test_train_finetune_keeps_the_base_models_vocabulary() -> None:
    """FR-619 / feature 004 issue #6: the active model's vocabulary must travel."""
    base = _train(seed=3)
    resp = client.post(
        "/api/lex/train",
        json={
            "base": base["model_token"],
            "text": "the little dog laughed to see such sport\n" * 200,
            "steps": 20,
            "seed": 3,
        },
    )
    assert resp.status_code in (200, 202), resp.text
    result = resp.json() if resp.status_code == 200 else _wait_job(resp.json()["job_id"])["result"]
    assert result["vocab_rows"] == base["vocab_rows"]
    assert result["model_token"] != base["model_token"]

    spectra = client.get(f"/api/lex/spectrum?model_token={result['model_token']}").json()
    assert spectra["spectrum"]["rows"] == base["vocab_rows"]


def test_train_refuses_shape_controls_alongside_base() -> None:
    base = _train(seed=3)
    resp = client.post(
        "/api/lex/train", json={"base": base["model_token"], "d_model": 32, "steps": 5}
    )
    _assert_error_envelope(resp, 400, "InvalidParamError")


def test_train_rejects_an_unknown_base() -> None:
    resp = client.post("/api/lex/train", json={"base": "0" * 32, "steps": 5})
    _assert_error_envelope(resp, 404, "NotFoundError")


def test_train_rejects_out_of_range_parameters() -> None:
    for bad in (
        {"steps": 0},
        {"steps": MAX_STEPS + 1},
        {"lr": 0},
        {"batch_size": 0},
        {"weight_decay": -1},
        {"sample_every": 0},
        {"d_model": 7},
        {"n_layers": 9},
        {"ctx": 999},
        {"dropout": 1.0},
    ):
        resp = client.post("/api/lex/train", json={**TINY_TRAIN, **bad})
        _assert_error_envelope(resp, 400, "InvalidParamError")


def test_train_refuses_a_corpus_shorter_than_one_context_window() -> None:
    resp = client.post(
        "/api/lex/train", json={**TINY_TRAIN, "text": "the cat sat", "steps": 5, "seed": 11}
    )
    assert resp.status_code in (400, 202)
    if resp.status_code == 202:
        snap = _wait_job(resp.json()["job_id"])
        assert snap["status"] == "error"
        assert snap["error"]["type"] == "InvalidParamError", snap["error"]
        assert "context" in snap["error"]["message"]


# -- GET /api/lex/spectrum -------------------------------------------------------------


def test_spectrum_reports_every_statistic_the_ceiling_and_a_random_baseline() -> None:
    result = _train(seed=3)
    resp = client.get(f"/api/lex/spectrum?model_token={result['model_token']}")
    assert resp.status_code == 200
    body = resp.json()
    _assert_rounded6(body)

    spec = body["spectrum"]
    rows, d_model = result["vocab_rows"], 16
    assert (spec["rows"], spec["d_model"]) == (rows, d_model)
    assert spec["max_rank"] == min(rows - 1, d_model)
    assert len(spec["eigenvalues"]) == len(spec["singular_values"]) == d_model
    assert all(
        spec["eigenvalues"][i] >= spec["eigenvalues"][i + 1] - 1e-12 for i in range(d_model - 1)
    )
    assert abs(sum(spec["explained_variance"]) - 1.0) < 1e-5
    assert 0 < spec["effective_rank"] <= spec["max_rank"]
    assert 0 < spec["stable_rank"] <= spec["max_rank"]
    assert 0 < spec["participation_ratio"] <= spec["max_rank"]
    assert 0 < spec["frac_var_top2"] <= spec["frac_var_top10"] <= 1.0 + 1e-9
    assert 1 <= spec["n_dims_for_90pct"] <= d_model
    assert spec["degenerate"] is False

    # FR-623: a labelled PCA projection with per-component explained variance.
    assert body["projection"] == "pca"
    assert len(spec["pca_coords"]) == rows
    assert all(len(row) == PCA_COMPONENTS for row in spec["pca_coords"])
    assert len(spec["pca_explained_variance_ratio"]) == PCA_COMPONENTS
    assert body["tokens"][:4] == list(SPECIAL_TOKENS)
    assert len(body["tokens"]) == rows

    # FR-622: the untrained control at the same shape, and signed deltas.
    assert body["baseline"]["rows"] == rows and body["baseline"]["d_model"] == d_model
    assert body["baseline"]["effective_rank"] > 0
    assert (
        abs(
            body["comparison"]["effective_rank_delta"]
            - (spec["effective_rank"] - body["baseline"]["effective_rank"])
        )
        < 1e-4
    )
    assert body["comparison"]["max_rank"] == spec["max_rank"]


def test_spectrum_can_omit_the_baseline() -> None:
    result = _train(seed=3)
    body = client.get(
        f"/api/lex/spectrum?model_token={result['model_token']}&baseline=false"
    ).json()
    assert "baseline" not in body and "comparison" not in body


def test_spectrum_matches_a_direct_computation_from_the_exported_weights() -> None:
    """The endpoint must be the same maths as `lex.spectrum`, not a second opinion."""
    from llm_geometry.lex.spectrum import spectrum as compute_spectrum

    result = _train(seed=3)
    body = client.get(f"/api/lex/spectrum?model_token={result['model_token']}").json()
    bundle = client.get(f"/api/lex/model?model_token={result['model_token']}").json()

    import base64

    entry = bundle["weights"]["embed"]
    embed = np.frombuffer(base64.b64decode(entry["data"]), dtype="<f4").reshape(entry["shape"])
    direct = compute_spectrum(embed)
    assert abs(direct.effective_rank - body["spectrum"]["effective_rank"]) < 1e-5
    assert abs(direct.stable_rank - body["spectrum"]["stable_rank"]) < 1e-5
    assert direct.n_dims_for_90pct == body["spectrum"]["n_dims_for_90pct"]


def test_spectrum_refuses_a_readout_on_a_tied_model() -> None:
    result = _train(seed=3)
    resp = client.get(f"/api/lex/spectrum?model_token={result['model_token']}&matrix=readout")
    _assert_error_envelope(resp, 400, "InvalidParamError")


def test_spectrum_serves_two_matrices_for_an_untied_model() -> None:
    result = _train(seed=5, tied=False)
    token = result["model_token"]
    embedding = client.get(f"/api/lex/spectrum?model_token={token}").json()
    readout = client.get(f"/api/lex/spectrum?model_token={token}&matrix=readout").json()
    assert embedding["tied"] is False and readout["tied"] is False
    assert embedding["spectrum"]["eigenvalues"] != readout["spectrum"]["eigenvalues"]
    assert result["param_count"] == param_count(
        vocab_rows=result["vocab_rows"], d_model=16, n_layers=1, ctx=32, tied=False
    )


def test_spectrum_rejects_an_unknown_token_and_an_unknown_matrix() -> None:
    _assert_error_envelope(
        client.get("/api/lex/spectrum?model_token=" + "0" * 32), 404, "NotFoundError"
    )
    result = _train(seed=3)
    _assert_error_envelope(
        client.get(f"/api/lex/spectrum?model_token={result['model_token']}&matrix=mlp"),
        400,
        "InvalidParamError",
    )


def test_spectrum_requires_a_model_token() -> None:
    _assert_error_envelope(client.get("/api/lex/spectrum"), 400, "InvalidParamError")


# -- POST /api/lex/generate ------------------------------------------------------------


def test_generate_is_in_budget_by_construction() -> None:
    """SC-602: zero out-of-budget words, verified programmatically."""
    result = _train(seed=3)
    vocab = set(build_vocab("dolch", "pre_primer", "").words)
    resp = client.post(
        "/api/lex/generate",
        json={
            "model_token": result["model_token"],
            "prompt": "the little dog",
            "max_new_tokens": 40,
            "temperature": 1.0,
            "seed": 4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_rounded6(body)
    assert body["out_of_budget"] == []
    assert body["words"] and all(word in vocab for word in body["words"])
    assert all(token not in body["text"] for token in SPECIAL_TOKENS)
    # An out-of-budget prompt word is reported as <unk>, not hidden: "the" and "little"
    # are pre-primer Dolch words, "dog" is a Dolch *noun* and so outside this budget.
    assert [t["text"] for t in body["prompt_tokens"]] == ["the", "little", "dog"]
    assert [t["unk"] for t in body["prompt_tokens"]] == [False, False, True]


def test_generate_is_deterministic_for_a_seed_and_greedy_at_zero_temperature() -> None:
    result = _train(seed=3)
    body = {"model_token": result["model_token"], "prompt": "the", "max_new_tokens": 12}
    first = client.post("/api/lex/generate", json={**body, "seed": 9}).json()["text"]
    second = client.post("/api/lex/generate", json={**body, "seed": 9}).json()["text"]
    assert first == second

    greedy_a = client.post("/api/lex/generate", json={**body, "temperature": 0}).json()["text"]
    greedy_b = client.post(
        "/api/lex/generate", json={**body, "temperature": 0, "seed": 12345}
    ).json()["text"]
    assert greedy_a == greedy_b, "at temperature 0 the seed must be inert"


def test_generate_rejects_bad_parameters_and_unknown_models() -> None:
    result = _train(seed=3)
    token = result["model_token"]
    _assert_error_envelope(client.post("/api/lex/generate", json={}), 400, "InvalidParamError")
    _assert_error_envelope(
        client.post("/api/lex/generate", json={"model_token": token, "temperature": -1}),
        400,
        "InvalidParamError",
    )
    _assert_error_envelope(
        client.post(
            "/api/lex/generate", json={"model_token": token, "max_new_tokens": MAX_NEW_TOKENS + 1}
        ),
        400,
        "InvalidParamError",
    )
    _assert_error_envelope(
        client.post("/api/lex/generate", json={"model_token": "0" * 32}), 404, "NotFoundError"
    )


# -- GET/POST /api/lex/model -----------------------------------------------------------


def test_bundle_round_trip_reproduces_generation_exactly() -> None:
    """SC-607: saving and reloading a model reproduces its generation exactly."""
    result = _train(seed=3)
    token = result["model_token"]
    before = client.post(
        "/api/lex/generate", json={"model_token": token, "max_new_tokens": 24, "seed": 2}
    ).json()["text"]

    bundle = client.get(f"/api/lex/model?model_token={token}").json()
    assert bundle["format"] == "llm-geometry/lex-model"
    assert bundle["version"] == 1
    assert bundle["vocab"]["specials"] == list(SPECIAL_TOKENS)
    assert len(bundle["vocab"]["words"]) == result["vocab_size"]
    assert "head_w" not in bundle["weights"], "a tied model has no separate readout"

    loaded = client.post("/api/lex/model", json=bundle)
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["model_token"] == token  # content hash, so it round-trips
    after = client.post(
        "/api/lex/generate",
        json={"model_token": loaded.json()["model_token"], "max_new_tokens": 24, "seed": 2},
    ).json()["text"]
    assert after == before


def test_bundle_carries_three_digests_and_all_of_them_are_mandatory() -> None:
    """The bundle's integrity fields are required, not optional — including on ABSENCE.

    Feature 004 shipped a Geometry Lab loader that skipped the check when the field was
    missing, so a tampered file loaded cleanly the moment you DELETED the digest rather
    than editing it. This route previously had the same shape ("if `model_token` is
    present…"). Both halves of the attack are pinned here: deleting a digest and tampering
    with one must both be a 400.
    """
    bundle = client.get(f"/api/lex/model?model_token={_train(seed=3)['model_token']}").json()
    assert set(bundle) >= {"model_token", "weights_token", "vocab_sha256"}
    assert len(bundle["model_token"]) == 32
    assert len(bundle["weights_token"]) == 32
    assert len(bundle["vocab_sha256"]) == 64
    # The joint hash also covers the config and the word list, so it is NOT the weights hash.
    assert bundle["model_token"] != bundle["weights_token"]

    for field in ("model_token", "weights_token", "vocab_sha256"):
        deleted = {k: v for k, v in bundle.items() if k != field}
        _assert_error_envelope(
            client.post("/api/lex/model", json=deleted), 400, "InvalidParamError"
        )
        bad = "f" * len(bundle[field])
        _assert_error_envelope(
            client.post("/api/lex/model", json={**bundle, field: bad}), 400, "InvalidParamError"
        )


def test_bundle_uses_the_pytorch_tensor_names_the_browser_translates_to() -> None:
    """One wire format, two runtimes.

    `src/lib/lexEngine/bundle.ts` writes and reads exactly this payload, translating its
    own `layers.N.*` names at the file boundary. `tests/unit/lexBundle.test.ts` pins the
    TypeScript `model_token` against one produced by this module, so a drift in either
    direction fails a test rather than producing a file one side silently misreads.
    """
    bundle = client.get(f"/api/lex/model?model_token={_train(seed=3)['model_token']}").json()
    names = set(bundle["weights"])
    assert any(n.startswith("blocks.") for n in names)
    assert not any(n.startswith("layers.") for n in names)
    assert set(bundle["config"]) == {
        "vocab_rows",
        "d_model",
        "n_layers",
        "n_heads",
        "ctx",
        "tied",
        "dropout",
    }
    assert set(bundle["vocab"]) == {"source", "budget", "words", "specials"}


def test_bundle_rejects_a_token_that_disagrees_with_its_own_weights() -> None:
    bundle = client.get(f"/api/lex/model?model_token={_train(seed=3)['model_token']}").json()
    tampered = {**bundle, "model_token": "f" * 32}
    _assert_error_envelope(client.post("/api/lex/model", json=tampered), 400, "InvalidParamError")


def test_bundle_rejects_a_dropped_tied_flag() -> None:
    """The source's `probe.py` dropped `tie` on reload; a tied bundle must not load untied."""
    bundle = client.get(f"/api/lex/model?model_token={_train(seed=3)['model_token']}").json()
    # The digests are left in place on purpose: the structural checks run BEFORE them, so
    # this measures the tied-flag check rather than passing for the wrong reason.
    untied = {**bundle, "config": {**bundle["config"], "tied": False}}
    _assert_error_envelope(client.post("/api/lex/model", json=untied), 400, "InvalidParamError")


def test_bundle_rejects_a_vocabulary_that_does_not_match_its_config() -> None:
    bundle = client.get(f"/api/lex/model?model_token={_train(seed=3)['model_token']}").json()
    short = {
        **bundle,
        "vocab": {**bundle["vocab"], "words": bundle["vocab"]["words"][:-1]},
    }
    _assert_error_envelope(client.post("/api/lex/model", json=short), 400, "InvalidParamError")


def test_bundle_rejects_a_foreign_format_and_a_future_version() -> None:
    bundle = client.get(f"/api/lex/model?model_token={_train(seed=3)['model_token']}").json()
    _assert_error_envelope(
        client.post("/api/lex/model", json={**bundle, "format": "llm-geometry/geo-model"}),
        400,
        "InvalidParamError",
    )
    _assert_error_envelope(
        client.post("/api/lex/model", json={**bundle, "version": 99}), 400, "InvalidParamError"
    )
    _assert_error_envelope(
        client.post("/api/lex/model", json={"format": "llm-geometry/lex-model", "version": 1}),
        400,
        "InvalidParamError",
    )


def test_model_export_rejects_an_unknown_token() -> None:
    _assert_error_envelope(
        client.get("/api/lex/model?model_token=" + "0" * 32), 404, "NotFoundError"
    )


# -- POST /api/lex/vacancy (feature 007) -----------------------------------------------


def _vacancy(**body: Any) -> dict[str, Any]:
    resp = client.post("/api/lex/vacancy", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    _assert_rounded6(payload)
    return payload


def test_vacancy_reports_every_statistic_the_contract_names() -> None:
    """§10's field list, verbatim and complete — an unprefixed `types*` is forbidden."""
    payload = _vacancy(p=1.0, seed=0)
    stats = payload["vacancy_stats"]
    assert set(stats) == {
        "domainTypesTotal",
        "domainTypesEligible",
        "domainTypesVacated",
        "corpusTypesTotal",
        "corpusTypesEligible",
        "corpusTypesVacated",
        "stemsTotal",
        "stemsVacated",
        "tokensTotal",
        "tokensVacated",
        "meanSyllablesBefore",
        "meanSyllablesAfter",
        "meanAnapestBefore",
        "meanAnapestAfter",
        "stressFromTableBefore",
        "stressFromTableAfter",
        "stressFromMintedBefore",
        "stressFromMintedAfter",
        "stressFromRuleBefore",
        "stressFromRuleAfter",
        "bijective",
        "imageSize",
        "remintRounds",
    }
    # The measured numbers of §10 on the shipped corpus, at full vacancy.
    assert stats["domainTypesTotal"] == 2233
    assert stats["domainTypesEligible"] == 1940
    assert stats["corpusTypesTotal"] == 2211
    assert stats["corpusTypesEligible"] == 1918
    assert stats["corpusTypesVacated"] == 1918
    assert stats["tokensVacated"] == 8125
    assert stats["stemsTotal"] == 1676
    # Identities, not observations: `u ∈ [0, 1)`, so p = 1 vacates everything eligible.
    assert stats["stemsVacated"] == stats["stemsTotal"]
    assert stats["domainTypesVacated"] == stats["domainTypesEligible"]
    assert stats["corpusTypesVacated"] == stats["corpusTypesEligible"]
    # The three-way stress split is token-weighted and sums to 1 on each side.
    for side in ("Before", "After"):
        total = sum(stats[f"stressFrom{k}{side}"] for k in ("Table", "Minted", "Rule"))
        assert abs(total - 1.0) < 1e-6, (side, total)
    # Injectivity is REPORTED, not assumed, and is surfaced beside the statistics block.
    assert payload["bijective"] is True
    assert stats["bijective"] is True
    assert payload["remint_rounds"] == stats["remintRounds"] == 0


def test_vacancy_reports_the_measured_remint_at_seed_7() -> None:
    """Seed 7 needs exactly one re-mint: `hang` first minted `wak`, and `hanged` would
    have surfaced as the real English word `waked` (§5.2 condition B)."""
    payload = _vacancy(p=1.0, seed=7)
    assert payload["remint_rounds"] == 1
    assert payload["bijective"] is True
    # The transform is unharmed by it: the same 1 918 types and 8 125 tokens move.
    assert payload["vacancy_stats"]["corpusTypesVacated"] == 1918
    assert payload["vacancy_stats"]["tokensVacated"] == 8125


def test_vacancy_returns_an_excerpt_and_a_digest_rather_than_the_corpus() -> None:
    payload = _vacancy(p=1.0, seed=0)
    assert payload["preview_chars"] == 2000
    assert len(payload["preview"]) == 2000
    assert payload["truncated"] is True
    # ~86 kB of corpus behind a 2 kB excerpt, pinned by 64 hex characters.
    assert payload["vacated_chars"] > 20 * payload["preview_chars"]
    assert len(payload["vacated_sha256"]) == 64
    assert payload["vacated_sha256"] != payload["original_sha256"]
    assert (
        payload["original_sha256"] == hashlib.sha256(load_corpus_text().encode("utf-8")).hexdigest()
    )

    short = _vacancy(p=1.0, seed=0, preview_chars=10)
    assert short["preview"] == payload["preview"][:10]
    assert short["vacated_sha256"] == payload["vacated_sha256"]


def test_vacancy_is_the_identity_at_p_zero() -> None:
    payload = _vacancy(p=0.0, seed=0)
    assert payload["vacated_sha256"] == payload["original_sha256"]
    assert payload["preview"] == payload["original_preview"]
    for field in ("corpusTypesVacated", "domainTypesVacated", "tokensVacated", "stemsVacated"):
        assert payload["vacancy_stats"][field] == 0, field


def test_vacancy_maps_the_vocabulary_and_leaves_coverage_untouched() -> None:
    """SC-703 in the units the panel shows.

    Under `consistent=true, reveal_after=0` the transform is a pure relabelling, so the
    budget's measured coverage of the VACATED corpus is bit-identical to its coverage of
    the English one — the same tokens in budget, `<unk>` in exactly the same places.
    """
    english = client.post("/api/lex/coverage", json={"source": "dolch", "budget": "primer"}).json()
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        for seed in (0, 7):
            payload = _vacancy(p=p, seed=seed, source="dolch", budget="primer")
            assert payload["vocabulary_rule"] == "mapped"
            assert payload["budget"]["coverage"] == english["coverage"], (p, seed)
            assert payload["budget"]["rows"] == english["rows"]
            assert len(payload["words"]) == len(english["words"])
            assert payload["corpus"]["n_tokens"] == english["corpus"]["n_tokens"]
            assert payload["corpus"]["n_lines"] == english["corpus"]["n_lines"]
            if p > 0:
                assert payload["words"] != english["words"]


def test_vacancy_controls_rebuild_the_budget_and_collapse_coverage() -> None:
    """SC-705: the conditions that break type identity break the invariance, measurably."""
    mapped = _vacancy(p=0.5, seed=0, source="dolch", budget="primer")
    inconsistent = _vacancy(p=0.5, seed=0, consistent=False, source="dolch", budget="primer")
    revealed = _vacancy(p=0.5, seed=0, reveal_after=2, source="dolch", budget="primer")
    assert mapped["vocabulary_rule"] == "mapped"
    assert inconsistent["vocabulary_rule"] == "rebuilt"
    assert revealed["vocabulary_rule"] == "rebuilt"
    for control in (inconsistent, revealed):
        assert control["budget"]["coverage"]["unk_rate"] > mapped["budget"]["coverage"]["unk_rate"]
    # §10: `corpusTypesVacated` is measured from the two TEXTS, so a type whose every
    # occurrence falls inside the reveal window does not count — the reading that matches
    # what the number claims to a reader.
    assert (
        revealed["vacancy_stats"]["corpusTypesVacated"]
        < mapped["vacancy_stats"]["corpusTypesVacated"]
    )


def test_vacancy_nests_and_stays_stable_as_p_rises() -> None:
    """SC-701 / SC-702 through the API: a word rewritten at a low `p` is byte-identical at
    every higher one, and the vacated set only grows."""
    seen: list[tuple[float, int, list[str]]] = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        payload = _vacancy(p=p, seed=0, source="dolch", budget="primer")
        seen.append((p, payload["vacancy_stats"]["stemsVacated"], payload["words"]))
    for (_, lower_n, lower_words), (_, higher_n, higher_words) in zip(seen, seen[1:]):
        assert higher_n >= lower_n
        for english, low, high in zip(seen[0][2], lower_words, higher_words):
            if low != english:
                assert high == low, (english, low, high)


def test_vacancy_transforms_a_users_own_text() -> None:
    text = "The little brown squirrel ate the pretty acorn.\nThe squirrel ran away.\n"
    payload = _vacancy(text=text, p=1.0, seed=0, preview_chars=200)
    assert payload["original_preview"] == text
    assert payload["preview"] != text
    # §1: only WORD_RE matches move; punctuation and line breaks pass through byte for byte.
    assert payload["preview"].count("\n") == text.count("\n")
    assert payload["preview"].count(".") == 2
    assert payload["vacancy_stats"]["tokensTotal"] == 12
    assert payload["vacancy_stats"]["tokensVacated"] == 9  # the three `the`s are preserved


def test_vacancy_rejects_parameters_outside_the_contract() -> None:
    for body in (
        {"p": 1.5},
        {"p": -0.1},
        {"p": "half"},
        {"seed": "zero"},
        {"reveal_after": -1},
        {"preview_chars": 20001},
        {"preview_chars": -1},
        {"keep": "little"},  # a bare string would protect six single letters
        {"keep": [3]},
        {"source": "dolch", "size": 50},
        {"budget": "not-a-budget"},
        {"text": "!!! ???"},
        # An unknown mint used to be answered with 200 and nonce output — the transform
        # silently substituting its default for a control the caller explicitly asked for.
        {"mint": "bogus"},
        {"mint": 3},
        {"mint": None},
        # §8.3: swap needs one replacement per TYPE, and the inconsistent control wants one
        # per OCCURRENCE. The transform refuses the combination; the route must carry that.
        {"mint": "swap", "consistent": False},
    ):
        _assert_error_envelope(client.post("/api/lex/vacancy", json=body), 400, "InvalidParamError")


def test_vacancy_params_reads_every_knob_the_transform_declares() -> None:
    """A knob the dataclass has and `_vacancy_params` omits is a SILENT default, not a 400.

    `mint` was exactly that: `VacancyParams` declared it, `__post_init__` validated it,
    `_vacancy_key` hashed it, the UI exposed it — and the route never parsed it, so
    `mint="swap"` and `mint="bogus"` both returned nonce output under HTTP 200. This
    asserts the wiring rather than the symptom, so the next knob cannot repeat it.
    """
    import dataclasses

    from llm_geometry.api.routes_lex import _vacancy_params
    from llm_geometry.lex.vacancy import VacancyParams

    knobs = [f.name for f in dataclasses.fields(VacancyParams) if not f.name.startswith("_")]
    defaults = _vacancy_params({})
    for knob in knobs:
        assert hasattr(defaults, knob), knob
    # Every knob must be REACHABLE: a non-default value for each one must survive the parse.
    non_default = {
        "p": 0.5,
        "seed": 7,
        "consistent": False,
        "match_prosody": False,
        "reveal_after": 2,
        "keep": ["little"],
        "mint": "swap",
    }
    assert set(non_default) == set(knobs), (
        f"the transform's knobs are {sorted(knobs)} but this test only drives "
        f"{sorted(non_default)} — add the new one here and to `_vacancy_params`"
    )
    for knob, value in non_default.items():
        # `swap` is refused under `consistent=False` (§8.3), so drive it on its own.
        parsed = _vacancy_params({knob: value})
        actual = getattr(parsed, knob)
        actual = sorted(actual) if isinstance(actual, frozenset) else actual
        assert actual == value, (
            f"_vacancy_params dropped {knob!r}: asked for {value!r}, got {actual!r} "
            "— the caller's setting was silently replaced by the dataclass default"
        )


def test_vacancy_honours_and_echoes_the_swap_mint_control() -> None:
    """§8.3's swap control, end to end through the route.

    `nonce` invents a phonotactically legal form; `swap` draws a REAL open-class word from
    the domain by frequency rank. Two different transforms, so two different corpora and
    two different digests — the defect this pins produced ONE digest for both.
    """
    nonce = _vacancy(p=1.0, seed=0, source="dolch", budget="pre_primer")
    swap = _vacancy(p=1.0, seed=0, source="dolch", budget="pre_primer", mint="swap")

    assert nonce["mint"] == "nonce"  # the default, stated rather than left to inference
    assert swap["mint"] == "swap"
    assert swap["vacated_sha256"] != nonce["vacated_sha256"]
    assert swap["preview"] != nonce["preview"]
    # Swap's images ARE domain types, so at full vacancy it is a permutation of the
    # open-class vocabulary: injective (§5.2a's `p ∈ {0, 1}` case) but not disjoint.
    assert swap["bijective"] is True
    assert swap["vacancy_stats"]["imageSize"] == nonce["vacancy_stats"]["imageSize"]


def test_vacancy_matches_the_static_client_fixture() -> None:
    """**The parity assertion this feature rests on.**

    `code/frontend/tests/fixtures/vacancy-api-golden.json` is a transcript of this route,
    written by `scripts/export_vacancy_api_golden.py`. `staticVacancy.test.ts` asserts the
    browser's in-page implementation reproduces it field for field; this asserts the live
    route still does. One document, two stacks, and no way for either to drift alone.
    """
    fixture = json.loads(
        (
            REPO_ROOT / "code" / "frontend" / "tests" / "fixtures" / "vacancy-api-golden.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["format"] == "vacancy-api-golden-v1"
    assert fixture["endpoint"] == "/api/lex/vacancy"
    assert fixture["defaults"]["preview_chars"] == 2000
    assert fixture["defaults"]["preview_max"] == 20000
    assert len(fixture["cases"]) >= 6
    for case in fixture["cases"]:
        resp = client.post("/api/lex/vacancy", json=case["request"])
        assert resp.status_code == 200, (case["label"], resp.text)
        assert resp.json() == case["response"], (
            f"{case['label']}: the route no longer returns what the parity fixture "
            "records. Regenerate it with `python scripts/export_vacancy_api_golden.py` "
            "ONLY after confirming the change was intended — the browser asserts against "
            "the same file."
        )


# -- POST /api/lex/train with vacancy (feature 007) -------------------------------------


def test_train_on_a_vacated_corpus_is_bit_identical_under_the_mapped_vocabulary() -> None:
    """SC-703's corollary, run for real: `p` is invisible to a word-level model.

    The mapped vocabulary gives every word the id its pre-image had, so the token id
    stream is unchanged and the losses are IDENTICAL — not close. That is the tiny arm's
    headline result, and it is asserted here rather than described.
    """
    english = _train(seed=11)
    vacated = _train(seed=11, vacancy={"p": 0.5, "seed": 0})
    for field in ("first_loss", "final_loss", "val_loss", "n_tokens", "vocab_rows"):
        assert vacated[field] == english[field], field
    # Same numbers, different words: the relabelling is real, the model is blind to it.
    assert vacated["model_token"] != english["model_token"]
    assert vacated["sample"] != english["sample"]


def test_train_without_vacancy_is_untouched_by_the_new_parameter() -> None:
    """Additive means additive: the same request keeps hitting the same cache entry."""
    first = _train(seed=12)
    again = _train(seed=12)
    assert again["model_token"] == first["model_token"]
    # `vacancy: null` is "no vacancy", not "vacancy with defaults".
    explicit = _train(seed=12, vacancy=None)
    assert explicit["model_token"] == first["model_token"]
    # …and `p = 0` really is the identity, so it lands on that entry too.
    identity = _train(seed=12, vacancy={"p": 0.0, "seed": 0})
    assert identity["model_token"] == first["model_token"]


def test_train_rejects_a_vacancy_block_that_is_not_an_object() -> None:
    for bad in (0.5, "p=0.5", [0.5]):
        _assert_error_envelope(
            client.post("/api/lex/train", json={**TINY_TRAIN, "vacancy": bad}),
            400,
            "InvalidParamError",
        )
    _assert_error_envelope(
        client.post("/api/lex/train", json={**TINY_TRAIN, "vacancy": {"p": 2}}),
        400,
        "InvalidParamError",
    )


# -- the frozen 002 contract is untouched ----------------------------------------------


def test_frozen_geo_contract_untouched() -> None:
    """Feature 006 is additive: every 002 path still exists and `/api/lex/*` is new."""
    paths = set(app.openapi()["paths"])
    for frozen in (
        "/api/geo/spec",
        "/api/geo/train",
        "/api/geo/tokenize",
        "/api/geo/trace",
        "/api/geo/vector_field",
        "/api/geo/weights",
        "/api/geo/finetune",
        "/api/geo/model",
        "/api/jobs/{job_id}",
        "/api/jobs/{job_id}/events",
    ):
        assert frozen in paths, f"{frozen} disappeared from the frozen contract"
    assert {
        "/api/lex/spec",
        "/api/lex/budgets",
        "/api/lex/coverage",
        "/api/lex/train",
        "/api/lex/spectrum",
        "/api/lex/generate",
        "/api/lex/model",
    } <= paths
