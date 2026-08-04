"""FR-107 — the model-size gate rejects oversized models BEFORE any download."""

import time
from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig

from llm_geometry.arch.gate import (
    check_model_size,
    effective_ceiling_for,
    estimate_params_from_config,
    too_large_error,
)
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


def test_rejection_quotes_the_ceiling_it_actually_applied():
    """Red team F10. The config-estimate path is held to 0.8·ARCH_MAX_PARAMS = 1.2B, but
    the message interpolated ARCH_MAX_PARAMS, so a 1.3B model was told it "has
    ~1,300,000,000 parameters, over the … ceiling of 1,500,000,000" — a sentence that
    contradicts itself. The quoted number must be the number the comparison used."""
    assert effective_ceiling_for("safetensors_metadata") == ARCH_MAX_PARAMS
    assert effective_ceiling_for("config_estimate") == int(ARCH_MAX_PARAMS * 0.8)

    over_the_margin = 1_300_000_000  # under 1.5B, over the 1.2B config-estimate ceiling
    assert over_the_margin < ARCH_MAX_PARAMS
    assert over_the_margin > effective_ceiling_for("config_estimate")
    err = too_large_error("some/1.3B-model", over_the_margin, "config_estimate")
    assert err.detail["effective_ceiling"] == 1_200_000_000
    assert "1,200,000,000" in err.message
    # It may MENTION 1.5B, but only while explaining why the lower one applied.
    assert "80 %" in err.message and "estimate" in err.message

    exact = too_large_error("some/7B-model", 7_600_000_000, "safetensors_metadata")
    assert exact.detail["effective_ceiling"] == ARCH_MAX_PARAMS
    assert f"{ARCH_MAX_PARAMS:,}" in exact.message
    assert "80 %" not in exact.message


def test_gate_error_is_typed_422():
    with pytest.raises(ModelTooLargeError) as exc_info:
        check_model_size(TOO_BIG)
    err = exc_info.value
    assert err.http_status == 422
    assert err.to_envelope()["error"]["type"] == "ModelTooLargeError"
