"""Real-model tests for the three visualizations (vector field, Sankey, manifold)."""

import numpy as np

from llm_geometry.precompute import get_or_compute_sync

M = "sshleifer/tiny-gpt2"


def test_token_cloud_printable_only():
    # a dot for every PRINTABLE token (special/byte-fragment tokens filtered out), with
    # real decoded strings shipped; cached per model
    from llm_geometry.models.loader import load_model
    full_vocab = load_model(M).vocab_size
    p = get_or_compute_sync("token_cloud", M, {"seed": 0, "spread_mu": 0.85})
    a, m = p["arrays"], p["meta"]
    vocab = m["vocab_size"]
    assert a["warped"].shape == (vocab, 2)
    assert a["token_ids"].shape == (vocab,)
    # token ids are a strictly-increasing PRINTABLE subset of the full vocab (not contiguous)
    ids = a["token_ids"]
    assert (np.diff(ids) > 0).all()
    assert vocab < full_vocab and int(ids.max()) < full_vocab  # some tokens were filtered out
    # real strings ship, aligned with ids, and every one is printable & non-empty
    strs = m["token_strs"]
    assert len(strs) == vocab
    assert all(s and s.strip() and s.isprintable() and "�" not in s for s in strs[:200])
    assert np.isfinite(a["warped"]).all()
    # internal projection arrays used to place the vector-field arrows in the same space
    assert a["pca_components"].shape[0] == 2
    assert a["pca_mean"].ndim == 1 and a["raw"].shape == (vocab, 2)


def test_vector_field_animation_consistent_frame():
    # All key frames (response steps) share ONE frame, so a token's position is comparable
    # across frames and MOVES as the context unfolds (the basis for smooth interpolation).
    p = get_or_compute_sync(
        "vector_field_animation", "distilgpt2",
        {"temperature": 0.0, "layer_to": 6, "reference_set_size": 80, "seed": 0},
        {"prefix_text": "I want", "response_text": "money and power"},
    )
    a, m = p["arrays"], p["meta"]
    F, R = m["n_frames"], m["reference_points"]
    assert F == 4 and R == 80  # 3 response tokens -> 4 key frames
    assert a["points"].shape == (F, R, 2) and a["ends"].shape == (F, R, 2)
    assert len(m["token_strs"]) == R
    # the SAME token moves between key frames (context-dependent prediction-layer embedding)
    move = np.linalg.norm(a["points"][-1] - a["points"][0], axis=1)
    assert float(move.mean()) > 0.1


def test_vector_field_trajectory_is_contextual():
    # The trajectory uses CONTEXTUAL prediction-layer embeddings, so the SAME response tokens
    # move when the prompt changes — that is the phenomenon the visualization exists to show.
    # Uses a real pretrained model (distilgpt2): tiny-gpt2's random weights barely use context.
    def traj(prefix):
        return get_or_compute_sync(
            "vector_field", "distilgpt2",
            {"grid_n": 8, "reference_set_size": 150, "temperature": 0.0, "layer_from": 0, "layer_to": 6, "seed": 0},
            {"prefix_text": prefix, "response_text": "money and power"},
        )["arrays"]["trajectory"]
    t1, t2 = traj("I love"), traj("I hate")
    assert t1.shape == t2.shape and t1.shape[1] == 2
    assert float(np.linalg.norm(t1 - t2, axis=1).mean()) > 0.1  # prompt clearly moves the trajectory


def test_vector_field_arrows():
    # temperature 0 -> one argmax arrow per reference point, at the chosen layer
    p = get_or_compute_sync(
        "vector_field", M,
        {"grid_n": 8, "reference_set_size": 150, "temperature": 0.0, "layer_from": 1, "layer_to": 1},
        {"prefix_text": "Hello"},
    )
    a = p["arrays"]
    m = p["meta"]
    assert m["fanout"] == 1
    assert m["layer_from"] == 1 and m["layer_to"] == 1
    assert a["starts"].shape == (m["reference_points"], 2)
    assert a["ends"].shape == (m["reference_points"], 2)
    assert m["count"] == m["reference_points"]
    assert (a["probs"] >= 0).all() and (a["probs"] <= 1).all()


def test_vector_field_fanout_and_trajectory():
    # temperature > 0 -> fan-out of `fanout` arrows per reference point; response traced
    p = get_or_compute_sync(
        "vector_field", M,
        {"grid_n": 8, "reference_set_size": 150, "temperature": 0.9, "fanout": 3, "layer": 0},
        {"prefix_text": "Hello", "response_text": "world peace"},
    )
    a = p["arrays"]
    m = p["meta"]
    assert m["fanout"] == 3
    assert m["count"] == m["reference_points"] * 3
    assert "trajectory" in a and a["trajectory"].shape[1] == 2
    assert a["trajectory"].shape[0] == len(m["trajectory_token_strs"])
    assert (a["trajectory_probs"] >= 0).all() and (a["trajectory_probs"] <= 1).all()


def test_sankey_swarm():
    p = get_or_compute_sync("sankey", M, {"n_particles": 10, "n_steps": 5, "seed": 0}, {"prefix_text": "Hello"})
    m = p["meta"]
    assert len(m["nodes"]) >= 1
    assert all(0 <= n["pos"] < 5 for n in m["nodes"])
    pos0 = sum(n["count"] for n in m["nodes"] if n["pos"] == 0)
    assert 1 <= pos0 <= 10  # particles seeded at position 0
    assert all(str(n["token"]) in m["token_strs"] for n in m["nodes"])
    # combined next-token distribution recorded per position (FR §2)
    assert len(m["per_position"]) >= 1
    for entry in m["per_position"]:
        assert 0 <= entry["pos"] < 5
        assert len(entry["top"]) >= 1
        assert all(0.0 <= t["prob"] <= 1.0 for t in entry["top"])
        assert all(str(t["token"]) in m["token_strs"] for t in entry["top"])


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
    m = p["meta"]
    assert a["vertices"].shape[1] == 3 and a["faces"].shape[1] == 3
    assert a["vertices"].shape[0] == m["n_vertices"]
    assert a["faces"].shape[0] == m["n_faces"]
    # the surface is warped (Open3D ARAP toward radius-2 token coords)
    assert float(a["warp"].max()) > 0.0
    radii = np.linalg.norm(a["vertices"], axis=1)
    assert radii.max() > 1.3  # reaches outward toward the radius-2 token coordinates
    # all tokens are placed on the radius-2 sphere
    token_radii = np.linalg.norm(a["token_points"], axis=1)
    assert np.allclose(token_radii, 2.0, atol=1e-3)
    assert a["token_points"].shape == (150, 3)
    assert len(m["top_tokens"]) >= 1
    # token strings align with token_points (used by the manifold raycast hover)
    assert len(m["token_strs"]) == a["token_points"].shape[0]
