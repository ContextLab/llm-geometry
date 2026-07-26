"""FR-009 — the SSE progress stream really streams events to a terminal state.

Feature 004 removed the precompute pipeline this test used to drive. The stream
itself is shared machinery the Geometry Lab still depends on, so the test now drives
it with a REAL fine-tuning job (a few SGD steps on the d_model=3 model) and asserts
the same property: subscribing to a live job yields events ending in exactly one
terminal `done`.
"""

import uuid

from fastapi.testclient import TestClient

from llm_geometry.api.app import app

client = TestClient(app)


def test_sse_stream_reaches_done():
    # A unique corpus makes the content-hash cache key fresh, so this is guaranteed
    # to be a real background job (202) rather than a cache hit.
    text = (
        "alice met the white rabbit near the little door and the queen of hearts "
        f"asked about the garden {uuid.uuid4().hex}"
    )
    resp = client.post("/api/geo/finetune", json={"text": text, "steps": 12})
    assert resp.status_code == 202, resp.text
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
    assert kinds[-1] == "done"

    # The finished job stays queryable, and repeating the request now hits the cache
    # (200) instead of minting a second job.
    assert client.get(f"/api/jobs/{job_id}").status_code == 200
    again = client.post("/api/geo/finetune", json={"text": text, "steps": 12})
    assert again.status_code == 200 and again.json()["ready"] is True


def test_events_for_unknown_job_is_404():
    resp = client.get("/api/jobs/no-such-job-xyz/events")
    assert resp.status_code == 404
