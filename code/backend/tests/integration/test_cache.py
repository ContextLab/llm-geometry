"""US1 — cache integrity, determinism, and single-flight (T015)."""

import json
import threading

import numpy as np

from llm_geometry import precompute
from llm_geometry.cache.keys import make_cache_key
from llm_geometry.cache.store import CacheStore

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


def test_single_flight_computes_once_under_concurrency():
    params = {"source": "static", "reference_set_size": 50}
    key = precompute.cache_key_for("embeddings", MODEL, params)
    precompute.get_store().delete(key)

    import llm_geometry.compute.embeddings as emod

    calls = {"n": 0}
    original = emod.per_layer_embeddings

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    emod.per_layer_embeddings = counting
    try:
        results: list = []

        def work():
            results.append(precompute.get_or_compute_sync("embeddings", MODEL, params))

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        emod.per_layer_embeddings = original

    assert calls["n"] == 1  # FR-008: four identical requests -> one real computation
    assert len(results) == 4
