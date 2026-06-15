"""Contract tests for the visualization endpoints (real tiny-gpt2)."""

from fastapi.testclient import TestClient

from llm_geometry.api.app import app

client = TestClient(app)
M = "sshleifer/tiny-gpt2"


def test_vector_field_endpoint():
    r = client.get("/api/vector_field", params={"model_id": M, "prefix_text": "Hi", "grid_n": 6, "reference_set_size": 120})
    assert r.status_code == 200
    b = r.json()
    assert len(b["starts"]) == 36 and len(b["starts"][0]) == 2
    assert len(b["ends"]) == 36 and len(b["probs"]) == 36


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
