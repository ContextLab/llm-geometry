"""US1 — cache integrity, determinism, and single-flight (T015)."""

import json
import threading
import uuid

import numpy as np

from llm_geometry.cache.keys import make_cache_key
from llm_geometry.cache.store import CacheStore
from llm_geometry.jobs.registry import registry

MODEL = "sshleifer/tiny-gpt2"


def test_roundtrip_and_delete_rebuild_identical(tmp_path):
    store = CacheStore(tmp_path)
    key, spec = make_cache_key(model_id="m", revision="r", artifact_type="embeddings", params={"a": 1})
    arrays = {"v": np.arange(12, dtype=np.float32).reshape(3, 4)}
    store.put(key, spec, {"shape": [3, 4]}, arrays)
    got = store.get(key)
    assert got is not None and np.array_equal(got["arrays"]["v"], arrays["v"])

    store.delete(key)
    assert store.get(key) is None
    store.put(key, spec, {"shape": [3, 4]}, arrays)
    assert np.array_equal(store.get(key)["arrays"]["v"], arrays["v"])  # SC-002


def test_incomplete_artifact_is_a_miss(tmp_path):
    store = CacheStore(tmp_path)
    key, _ = make_cache_key(model_id="m", revision="r", artifact_type="x")
    # Only the npz exists (interrupted precompute) — no complete sidecar.
    store._atomic_write_bytes(store._npz_path(key), b"partial-bytes")
    assert store.get(key) is None


def test_checksum_mismatch_is_a_miss(tmp_path):
    store = CacheStore(tmp_path)
    key, spec = make_cache_key(model_id="m", revision="r", artifact_type="x")
    store.put(key, spec, {"k": 1}, {"v": np.ones(4, dtype=np.float32)})
    path = store._npz_path(key)
    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 0xFF
    path.write_bytes(bytes(corrupted))
    assert store.get(key) is None  # FR-007 corruption detected


def test_schema_version_mismatch_is_a_miss(tmp_path):
    store = CacheStore(tmp_path)
    key, spec = make_cache_key(model_id="m", revision="r", artifact_type="x")
    store.put(key, spec, {"k": 1}, {"v": np.ones(4, dtype=np.float32)})
    sidecar_path = store._json_path(key)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["schema_version"] = 999_999
    sidecar_path.write_text(json.dumps(sidecar))
    assert store.get(key) is None  # FR-007 stale format detected


def test_concurrent_model_load_does_not_race():
    """Concurrent first-loads of the same model must not hit the meta-tensor race
    (regression: the live UI fires several requests at once on page load)."""
    import llm_geometry.models.loader as loader

    with loader._loaded_lock:
        loader._loaded.pop("distilgpt2", None)

    errors: list = []
    loaded: list = []

    def work():
        try:
            loaded.append(loader.load_model("distilgpt2"))
        except Exception as exc:  # pragma: no cover - failure path is the bug
            errors.append(repr(exc))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent load raced: {errors[:1]}"
    assert len(loaded) == 4 and all(m.model_id == "distilgpt2" for m in loaded)


def test_single_flight_creates_one_job_under_concurrency():
    """FR-008: N concurrent requests for the same key produce ONE unit of work.

    Feature 004 removed the precompute pipeline; the single-flight primitive the
    Geometry Lab's training and fine-tuning jobs ride on is the job registry's
    get_or_create, so that is what this exercises — no patched functions, just the
    real lock under real thread contention.
    """
    key = f"single-flight-probe-{uuid.uuid4().hex}"
    created_flags: list[bool] = []
    job_ids: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def work():
        start.wait()  # maximize the overlap on the contended path
        job, created = registry.get_or_create(key, phase="test")
        with lock:
            created_flags.append(created)
            job_ids.append(job.job_id)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(created_flags) == 1, "more than one thread believed it owned the work"
    assert len(job_ids) == 8 and len(set(job_ids)) == 1  # everyone got the same job
