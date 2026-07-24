"""US1 — API contract tests against the real app (T017)."""

import time

from fastapi.testclient import TestClient

from llm_geometry.api.app import app

client = TestClient(app)
MODEL = "sshleifer/tiny-gpt2"


def test_health():
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and "schema_version" in body


def test_list_models_includes_curated():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    ids = {m["model_id"] for m in resp.json()["models"]}
    assert "gpt2" in ids


def test_resolve_unsupported_model_is_422():
    resp = client.post("/api/models/resolve", json={"model_id": "definitely-not-real-xyz-123"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "UnsupportedModelError"


def test_distribution_endpoint_top_k():
    resp = client.get("/api/distribution", params={"model_id": MODEL, "prefix_text": "Hello", "temperature": 1.0, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_token"] is not None
    assert len(body["top"]) == 5
    assert "tail_mass" in body


def test_distribution_negative_temperature_is_400():
    resp = client.get("/api/distribution", params={"model_id": MODEL, "temperature": -1})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "InvalidParamError"


def test_precompute_returns_key_and_job_resolves_done():
    resp = client.post(
        "/api/precompute",
        json={"artifact_type": "embeddings", "model_id": MODEL, "params": {"source": "static", "reference_set_size": 50}},
    )
    assert resp.status_code in (200, 202)
    body = resp.json()
    assert "cache_key" in body
    if not body["ready"]:
        job_id = body["job_id"]
        status = None
        for _ in range(150):
            status = client.get(f"/api/jobs/{job_id}").json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.2)
        assert status is not None and status["status"] == "done"


def test_embeddings_endpoint_meta_then_full():
    meta = client.get("/api/embeddings", params={"model_id": MODEL, "source": "static"})
    assert meta.status_code == 200
    assert "token_ids" in meta.json() and "vectors" not in meta.json()
    full = client.get("/api/embeddings", params={"model_id": MODEL, "source": "static", "format": "full"})
    assert "vectors" in full.json()


def test_job_not_found_is_404():
    assert client.get("/api/jobs/no-such-job").status_code == 404
