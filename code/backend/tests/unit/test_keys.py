"""Unit — cache key determinism and sensitivity (T039)."""

from llm_geometry.cache.keys import make_cache_key


def test_key_is_order_independent():
    k1, _ = make_cache_key(model_id="m", revision="r", artifact_type="t", params={"a": 1, "b": 2})
    k2, _ = make_cache_key(model_id="m", revision="r", artifact_type="t", params={"b": 2, "a": 1})
    assert k1 == k2


def test_key_changes_with_params():
    k1, _ = make_cache_key(model_id="m", revision="r", artifact_type="t", params={"a": 1})
    k2, _ = make_cache_key(model_id="m", revision="r", artifact_type="t", params={"a": 2})
    assert k1 != k2


def test_key_changes_with_revision_and_model():
    base, _ = make_cache_key(model_id="m", revision="r", artifact_type="t")
    diff_rev, _ = make_cache_key(model_id="m", revision="r2", artifact_type="t")
    diff_model, _ = make_cache_key(model_id="m2", revision="r", artifact_type="t")
    assert base != diff_rev != diff_model and base != diff_model


def test_key_prefixed_with_artifact_type():
    key, spec = make_cache_key(model_id="m", revision="r", artifact_type="reduction_2d")
    assert key.startswith("reduction_2d-")
    assert spec["artifact_type"] == "reduction_2d"
