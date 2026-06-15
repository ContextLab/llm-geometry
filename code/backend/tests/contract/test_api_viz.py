"""Contract tests for the visualization endpoints (real tiny-gpt2)."""

from fastapi.testclient import TestClient

from llm_geometry.api.app import app

client = TestClient(app)
M = "sshleifer/tiny-gpt2"


def test_vector_field_endpoint():
    r = client.get("/api/vector_field", params={
        "model_id": M, "prefix_text": "Hi", "grid_n": 6, "reference_set_size": 120,
        "temperature": 0.0, "layer_from": 1, "layer_to": 1,
    })
    assert r.status_code == 200
    b = r.json()
    assert len(b["starts"]) == b["reference_points"] and len(b["starts"][0]) == 2
    assert len(b["ends"]) == len(b["starts"]) and len(b["probs"]) == len(b["starts"])
    assert b["layer_from"] == 1 and b["layer_to"] == 1


def test_vector_field_layer_range_endpoint():
    # from = layer 0 (start positions), to = layer 2 (end positions)
    r = client.get("/api/vector_field", params={
        "model_id": M, "prefix_text": "Hi", "grid_n": 6, "reference_set_size": 120,
        "temperature": 0.0, "layer_from": 0, "layer_to": 2,
    })
    assert r.status_code == 200
    b = r.json()
    assert b["layer_from"] == 0 and b["layer_to"] == 2
    assert len(b["starts"]) == len(b["ends"]) == b["reference_points"]


def test_tokenize_endpoint():
    r = client.get("/api/tokenize", params={"model_id": M, "text": "Paris is nice"})
    assert r.status_code == 200
    toks = r.json()["tokens"]
    assert len(toks) >= 1 and "token_str" in toks[0] and "token" in toks[0]


def test_vector_field_trajectory_endpoint():
    r = client.get("/api/vector_field", params={
        "model_id": M, "prefix_text": "Hi", "grid_n": 6, "reference_set_size": 120,
        "temperature": 0.8, "fanout": 3, "response_text": "and then",
    })
    assert r.status_code == 200
    b = r.json()
    assert "trajectory" in b and len(b["trajectory"][0]) == 2
    assert len(b["trajectory"]) == len(b["trajectory_probs"])


def test_sankey_endpoint():
    r = client.get("/api/sankey", params={"model_id": M, "prefix_text": "Hi", "n_particles": 8, "n_steps": 4})
    assert r.status_code == 200
    b = r.json()
    assert "nodes" in b and "links" in b and "token_strs" in b


def test_manifold_endpoint():
    r = client.get("/api/manifold", params={"model_id": M, "prefix_text": "Hi", "reference_set_size": 120})
    assert r.status_code == 200
    b = r.json()
    assert len(b["vertices"][0]) == 3 and len(b["faces"][0]) == 3
    assert len(b["token_points"][0]) == 3
