"""Integration tests: REAL vector fields + the canonical checkpoint's gates (SC-103).

Uses the actual canonical checkpoint (trained on first run, cached afterwards) —
never a stand-in model (FR-109).
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_geometry.errors import InvalidParamError
from llm_geometry.geo.config import (
    MIN_COVERAGE_UNIFORMITY,
    MIN_FIELD_DIRECTIONAL_ENTROPY,
    N_LAYERS,
    VOCAB_SIZE,
)
from llm_geometry.geo.fields import (
    coverage_uniformity,
    field_directional_entropy,
    fibonacci_sphere,
    force_field,
    next_next_field,
)
from llm_geometry.geo.model import model_from_weight_set
from llm_geometry.geo.tokenizer import get_tokenizer
from llm_geometry.geo.train import train_canonical, load_canonical_weight_set
from llm_geometry.geo.weights import build_weight_set


@pytest.fixture(scope="module")
def canonical():
    meta = train_canonical()  # trains for real on a cold cache; cache hit otherwise
    ws = load_canonical_weight_set()
    return meta, ws, model_from_weight_set(ws)


@pytest.fixture(scope="module")
def prompt_ids():
    enc = get_tokenizer().encode("alice was beginning to get very tired", truncate_side="left")
    assert enc.n_unk == 0
    return enc.ids


# -- non-degeneracy + coverage gates (SC-103) ------------------------------------------


def test_canonical_passes_gates(canonical):
    meta, ws, model = canonical
    # Recompute both metrics from the artifact — never trust stored numbers blindly.
    cov = coverage_uniformity(np.asarray(ws["embedding"]))
    field = next_next_field(model, [], layer="full", temperature=0.0, top_m=1)
    ent = field_directional_entropy(field["arrows"])
    assert cov >= MIN_COVERAGE_UNIFORMITY, (
        f"coverage_uniformity {cov:.3f} is below the gate {MIN_COVERAGE_UNIFORMITY} — "
        "the learned embedding does not cover the sphere"
    )
    assert ent >= MIN_FIELD_DIRECTIONAL_ENTROPY, (
        f"field_directional_entropy {ent:.3f} nats is below the gate "
        f"{MIN_FIELD_DIRECTIONAL_ENTROPY} — the learned next_next field is degenerate"
    )
    assert abs(cov - meta["coverage_uniformity"]) < 1e-6
    assert abs(ent - meta["field_directional_entropy"]) < 1e-6


# -- next_next mode --------------------------------------------------------------------


def test_next_next_argmax_one_arrow_per_vocab_point(canonical, prompt_ids):
    _, _, model = canonical
    field = next_next_field(model, prompt_ids, layer="full", temperature=0.0, top_m=1)
    assert field["mode"] == "next_next" and field["layer"] == "full"
    assert field["points"].shape == (VOCAB_SIZE, 3)
    assert field["token_ids"] == list(range(VOCAB_SIZE))
    assert len(field["arrows"]) == VOCAB_SIZE  # exactly one argmax arrow per point
    assert [a["origin_index"] for a in field["arrows"]] == list(range(VOCAB_SIZE))
    assert all(a["weight"] == 1.0 for a in field["arrows"])
    assert field["sequence_forces"] is None
    assert field["tangent_exact"] is False
    # Arrows land on real embedding points: origin + vec is some vocab embedding.
    tips = field["points"][[a["origin_index"] for a in field["arrows"]]] + np.asarray(
        [a["vec"] for a in field["arrows"]], dtype=np.float32
    )
    nearest = np.min(np.linalg.norm(field["points"][None] - tips[:, None], axis=2), axis=1)
    assert float(nearest.max()) < 1e-4


def test_next_next_temperature_top_m(canonical, prompt_ids):
    _, _, model = canonical
    field = next_next_field(model, prompt_ids, layer="full", temperature=0.8, top_m=3)
    assert VOCAB_SIZE <= len(field["arrows"]) <= 3 * VOCAB_SIZE
    for arrow in field["arrows"]:
        assert 0.0 < arrow["weight"] <= 1.0
    # Determinism: identical inputs give identical fields.
    again = next_next_field(model, prompt_ids, layer="full", temperature=0.8, top_m=3)
    assert [a["weight"] for a in again["arrows"]] == [a["weight"] for a in field["arrows"]]


def test_next_next_layer_selection_changes_field(canonical, prompt_ids):
    _, _, model = canonical
    layer0 = next_next_field(model, prompt_ids, layer=0, temperature=0.0, top_m=1)
    full = next_next_field(model, prompt_ids, layer="full", temperature=0.0, top_m=1)
    assert layer0["layer"] == 0
    v0 = np.asarray([a["vec"] for a in layer0["arrows"]])
    vf = np.asarray([a["vec"] for a in full["arrows"]])
    assert not np.allclose(v0, vf)  # earlier layers read out differently


# -- force mode ------------------------------------------------------------------------


def test_force_identity_vs_learned_differ(canonical, prompt_ids):
    _, ws, model = canonical
    edited_ws, _ = build_weight_set(ws, [{"layer": 0, "matrix": "W_V", "preset": "identity"}])
    edited = model_from_weight_set(edited_ws)
    f_learned = force_field(model, prompt_ids, layer=0)
    f_identity = force_field(edited, prompt_ids, layer=0)
    v_learned = np.asarray([a["vec"] for a in f_learned["arrows"]])
    v_identity = np.asarray([a["vec"] for a in f_identity["arrows"]])
    # With W_V = I the per-point field is exactly the identity map z -> z.
    assert np.allclose(v_identity, f_identity["points"], atol=1e-6)
    diff = float(np.max(np.linalg.norm(v_learned - v_identity, axis=1)))
    assert diff > 0.1, f"identity vs learned force fields barely differ (max {diff})"


def test_force_antisymmetrized_is_tangent_to_machine_precision(canonical):
    _, _, model = canonical
    field = force_field(model, [], layer=1, antisymmetrize=True)
    assert field["tangent_exact"] is True
    points = field["points"]
    vecs = np.asarray([a["vec"] for a in field["arrows"]], dtype=np.float32)
    radial = np.abs(np.sum(vecs * points, axis=1))  # ⟨Vz, z⟩ per point
    assert float(radial.max()) < 1e-5, "antisymmetrized field is not tangent"
    plain = force_field(model, [], layer=1, antisymmetrize=False)
    assert plain["tangent_exact"] is False


def test_force_sequence_forces(canonical, prompt_ids):
    _, _, model = canonical
    field = force_field(model, prompt_ids, layer=2)
    forces = field["sequence_forces"]
    assert [f["position"] for f in forces] == list(range(len(prompt_ids)))
    for f in forces:
        assert len(f["vec"]) == 3
        assert f["normal_residual"] >= 0.0
        assert np.all(np.isfinite(f["vec"]))
    # Empty prompt -> per-point field only, no sequence forces.
    empty = force_field(model, [], layer=2)
    assert empty["sequence_forces"] == []


def test_force_full_layer_rejected(canonical):
    _, _, model = canonical
    with pytest.raises(InvalidParamError):
        force_field(model, [], layer="full")
    with pytest.raises(InvalidParamError):
        force_field(model, [], layer=N_LAYERS)


# -- spherical helpers -----------------------------------------------------------------


def test_fibonacci_sphere_is_unit_and_spread():
    pts = fibonacci_sphere(64)
    assert pts.shape == (64, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-5)
    assert coverage_uniformity(pts) > 0.95  # the lattice itself is near-uniform
