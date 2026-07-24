"""US2 — reduction API contract tests against the real app (T025)."""

from fastapi.testclient import TestClient

from llm_geometry.api.app import app

client = TestClient(app)
MODEL = "sshleifer/tiny-gpt2"


def test_reduction_2d_with_grid():
    resp = client.get(
        "/api/reduction/2d",
        params={"model_id": MODEL, "method": "pca", "with_grid": True, "grid_n": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["coords"][0]) == 2
    assert body["grid"]["n"] == 10
    assert len(body["grid"]["vertices"]) == 100
    assert len(body["grid"]["reference_token_ids"]) == 100


def test_reduction_3d_on_sphere():
    resp = client.get("/api/reduction/3d", params={"model_id": MODEL, "method": "pca3"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["coords"][0]) == 3
