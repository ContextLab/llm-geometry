"""Integration tests: REAL SGD fine-tuning (text + a real streamed HF dataset).

The canonical checkpoint must never be mutated — verified byte-for-byte on disk.
"""

from __future__ import annotations

import hashlib

import pytest

from llm_geometry.cache.store import CacheStore
from llm_geometry.config import CACHE_DIR
from llm_geometry.errors import InvalidParamError, UnsupportedModelError
from llm_geometry.geo.finetune import finetune, load_text_from_hf
from llm_geometry.geo.train import canonical_cache_key, train_canonical

PARAGRAPH = (
    "The gryphon and the mock turtle danced a quadrille on the shore. "
    "The gryphon said to the mock turtle that the dance was the best dance "
    "in all the sea, and the mock turtle sighed and danced on. "
    "Alice watched the dance and thought it the strangest dance she had ever seen."
)


def _canonical_file_hashes() -> tuple[str, str]:
    key, _ = canonical_cache_key()
    npz = (CACHE_DIR / f"{key}.npz").read_bytes()
    sidecar = (CACHE_DIR / f"{key}.json").read_bytes()
    return hashlib.sha256(npz).hexdigest(), hashlib.sha256(sidecar).hexdigest()


@pytest.fixture(scope="module")
def canonical_meta():
    return train_canonical()  # real training on cold cache, cache hit otherwise


def test_finetune_on_text_mints_new_checkpoint_and_learns(canonical_meta):
    before_hashes = _canonical_file_hashes()
    result = finetune(base="learned", text=PARAGRAPH, steps=150, lr=5e-2)
    # A *different* checkpoint token is minted; the canonical artifact is untouched.
    assert result["weights_token"] != canonical_meta["checkpoint_id"]
    assert _canonical_file_hashes() == before_hashes
    # Real learning on the paragraph.
    assert result["loss_after"] < result["loss_before"], (
        f"fine-tuning did not reduce loss ({result['loss_before']:.3f} -> "
        f"{result['loss_after']:.3f})"
    )
    assert result["cached"] is False

    # Identical request -> content-hash cache hit with the identical token.
    again = finetune(base="learned", text=PARAGRAPH, steps=150, lr=5e-2)
    assert again["cached"] is True
    assert again["weights_token"] == result["weights_token"]
    assert again["loss_after"] == result["loss_after"]


def test_finetune_from_minted_checkpoint(canonical_meta, tmp_path):
    # Chained fine-tunes: the result of one is a valid base for the next.
    first = finetune(base="learned", text=PARAGRAPH, steps=50, lr=5e-2)
    second = finetune(
        base=first["weights_token"], text="alice said the queen was very kind.", steps=50, lr=5e-2
    )
    assert second["weights_token"] not in (first["weights_token"], canonical_meta["checkpoint_id"])
    assert second["base_token"] == first["weights_token"]


def test_finetune_streams_real_hf_dataset(canonical_meta):
    text = load_text_from_hf("roneneldan/TinyStories", split="train", max_samples=32)
    assert len(text) > 1000  # 32 real stories
    result = finetune(
        base="learned", hf_dataset="roneneldan/TinyStories", max_samples=32, steps=60, lr=5e-2
    )
    assert result["weights_token"] != canonical_meta["checkpoint_id"]
    assert result["loss_after"] < result["loss_before"]


def test_finetune_input_validation(canonical_meta):
    with pytest.raises(InvalidParamError):
        finetune(base="learned")  # no source
    with pytest.raises(InvalidParamError):
        finetune(base="learned", text="x", hf_dataset="y")  # two sources
    with pytest.raises(InvalidParamError):
        finetune(base="learned", text=PARAGRAPH, steps=501)  # over the contract cap
    with pytest.raises(InvalidParamError):
        finetune(base="learned", text="   ")  # empty after stripping
    with pytest.raises(UnsupportedModelError):
        finetune(
            base="learned",
            hf_dataset="this-org-does-not-exist/definitely-not-a-dataset-12345",
            steps=10,
        )


def test_finetune_never_touches_canonical_across_runs(canonical_meta, tmp_path):
    # Even with an isolated store, fine-tuning resolves 'learned' from the shared
    # canonical artifact but writes results only to the isolated store.
    before = _canonical_file_hashes()
    store = CacheStore(tmp_path)
    result = finetune(base="learned", text=PARAGRAPH, steps=30, lr=5e-2, store=store)
    assert _canonical_file_hashes() == before
    assert (tmp_path / f"geo-weights-{result['weights_token']}.npz").exists()
