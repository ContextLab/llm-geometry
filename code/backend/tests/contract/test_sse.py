"""FR-009 — the SSE progress stream really streams events to a terminal state."""

from fastapi.testclient import TestClient

from llm_geometry import precompute
from llm_geometry.api.app import app

client = TestClient(app)
MODEL = "sshleifer/tiny-gpt2"


def test_sse_stream_reaches_done():
    # Force an uncached artifact so precompute returns a job to subscribe to.
    params = {"source": "contextual", "layer": 1, "reference_set_size": 120}
    key = precompute.cache_key_for("embeddings", MODEL, params)
    precompute.get_store().delete(key)

    resp = client.post(
        "/api/precompute",
        json={"artifact_type": "embeddings", "model_id": MODEL, "params": params},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] and not body["ready"]
    job_id = body["job_id"]

    kinds: list[str] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        current = None
        for raw in stream.iter_lines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current:
                kinds.append(current)
                if current in ("done", "error"):
                    break
            if len(kinds) > 500:  # safety bound
                break

    assert "done" in kinds  # terminal completion event delivered
    assert "error" not in kinds
    # After the stream completes, the artifact is genuinely cached.
    assert precompute.get_store().has(key)
