"""Real-model tests for the three visualizations (vector field, Sankey, manifold)."""

import numpy as np

from llm_geometry.precompute import get_or_compute_sync

M = "sshleifer/tiny-gpt2"


def test_vector_field_arrows():
    p = get_or_compute_sync("vector_field", M, {"grid_n": 8, "reference_set_size": 150}, {"prefix_text": "Hello"})
    a = p["arrays"]
    assert a["starts"].shape == (64, 2)
    assert a["ends"].shape == (64, 2)
    assert a["probs"].shape == (64,)
    assert (a["probs"] >= 0).all() and (a["probs"] <= 1).all()
    assert len(p["meta"]["start_token_strs"]) == 64
    assert len(p["meta"]["end_token_strs"]) == 64


def test_sankey_swarm():
    p = get_or_compute_sync("sankey", M, {"n_particles": 10, "n_steps": 5, "seed": 0}, {"prefix_text": "Hello"})
    m = p["meta"]
    assert len(m["nodes"]) >= 1
    assert all(0 <= n["pos"] < 5 for n in m["nodes"])
    pos0 = sum(n["count"] for n in m["nodes"] if n["pos"] == 0)
    assert 1 <= pos0 <= 10  # particles seeded at position 0
    assert all(str(n["token"]) in m["token_strs"] for n in m["nodes"])


def test_sankey_reproducible_with_seed():
    a = get_or_compute_sync("sankey", M, {"n_particles": 8, "n_steps": 4, "seed": 7}, {"prefix_text": "x"})["meta"]
    from llm_geometry import precompute

    key = precompute.cache_key_for("sankey", M, {"n_particles": 8, "n_steps": 4, "seed": 7}, {"prefix_text": "x"})
    precompute.get_store().delete(key)
    b = get_or_compute_sync("sankey", M, {"n_particles": 8, "n_steps": 4, "seed": 7}, {"prefix_text": "x"})["meta"]
    assert a["nodes"] == b["nodes"]  # seeded multinomial sampling is deterministic


def test_manifold_warped_sphere():
    p = get_or_compute_sync("manifold", M, {"reference_set_size": 150, "seed": 0}, {"prefix_text": "Hello"})
    a = p["arrays"]
    assert a["vertices"].shape[1] == 3 and a["faces"].shape[1] == 3
    assert a["vertices"].shape[0] == p["meta"]["n_vertices"]
    radii = np.linalg.norm(a["vertices"], axis=1)
    assert radii.min() >= 0.99  # the sphere only bulges outward (radius >= 1)
    assert radii.max() > 1.0    # at least some bulge toward likely tokens
    assert len(p["meta"]["top_tokens"]) >= 1
