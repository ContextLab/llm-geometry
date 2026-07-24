"""FR-107 — the model-size gate rejects oversized models BEFORE any download."""

import time
from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig

from llm_geometry.arch.gate import check_model_size, estimate_params_from_config
from llm_geometry.config import ARCH_MAX_PARAMS
from llm_geometry.errors import ModelTooLargeError

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TOO_BIG = "Qwen/Qwen2.5-7B-Instruct"  # ~7.6B params, well over the 1.5B ceiling


def test_ceiling_is_configured():
    assert ARCH_MAX_PARAMS == 1_500_000_000


def test_small_model_passes_gate():
    info = check_model_size(SMALL)
    assert 100_000_000 < info["total_params"] < 200_000_000  # SmolLM2-135M is ~134.5M
    assert info["max_params"] == ARCH_MAX_PARAMS
    assert info["source"] in ("safetensors_metadata", "config_estimate")


def test_oversized_model_rejected_without_download():
    start = time.time()
    with pytest.raises(ModelTooLargeError) as exc_info:
        check_model_size(TOO_BIG)
    elapsed = time.time() - start
    detail = exc_info.value.detail
    assert detail["total_params"] > ARCH_MAX_PARAMS
    assert detail["source"] == "safetensors_metadata"  # hub-metadata-only code path
    # Metadata-only: fast, and no weight blobs may have landed in the local HF cache.
    assert elapsed < 60, f"gate took {elapsed:.1f}s — looks like it downloaded something"
    snapshot = Path(HF_HUB_CACHE) / f"models--{TOO_BIG.replace('/', '--')}"
    if snapshot.exists():
        big = [p for p in snapshot.rglob("*") if p.is_file() and p.stat().st_size > 50_000_000]
        assert not big, f"gate downloaded large files: {big}"


def test_config_estimate_close_to_real_count():
    """The fallback estimator lands within 2x of the true count for a known model."""
    config = AutoConfig.from_pretrained(SMALL)
    estimate = estimate_params_from_config(config)
    true_count = 134_515_008  # SmolLM2-135M safetensors parameter count
    assert true_count / 2 < estimate < true_count * 2


def test_gate_error_is_typed_422():
    with pytest.raises(ModelTooLargeError) as exc_info:
        check_model_size(TOO_BIG)
    err = exc_info.value
    assert err.http_status == 422
    assert err.to_envelope()["error"]["type"] == "ModelTooLargeError"
