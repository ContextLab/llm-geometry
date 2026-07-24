"""Integration tests: REAL GeoTransformer training on the real corpus (FR-103/FR-109).

Short (reduced-epoch) runs exercise the genuine training loop; the canonical
full-length checkpoint and its gates are covered in ``test_geo_fields.py``.
"""

from __future__ import annotations

import numpy as np

from llm_geometry.cache.store import CacheStore
from llm_geometry.geo.config import CONTEXT_WINDOW, VOCAB_SIZE
from llm_geometry.geo.train import (
    canonical_cache_key,
    corpus_token_stream,
    load_canonical_weight_set,
    make_windows,
    train_canonical,
    train_geo_model,
    eval_loss,
)
from llm_geometry.geo.model import model_from_weight_set
from llm_geometry.geo.weights import weights_token

UNIFORM_LOSS = float(np.log(VOCAB_SIZE))  # ≈ 6.911 nats: the know-nothing baseline


def test_corpus_stream_and_windows():
    stream = corpus_token_stream()
    assert stream.shape[0] > 30_000  # the real book, not a stub
    assert stream.min() >= 0 and stream.max() < VOCAB_SIZE
    windows = make_windows(stream, stride=10)
    assert windows.shape[1] == CONTEXT_WINDOW + 1
    assert windows.shape[0] > 1000


def test_short_training_materially_beats_uniform():
    ws, final_loss = train_geo_model(epochs=3)
    # ln(1003) ≈ 6.91; three real epochs land around 5.0. "Materially" = > 1 nat.
    assert final_loss < UNIFORM_LOSS - 1.0, f"loss {final_loss} did not move from uniform"
    # Embedding rows stay on S² (renormalized after every step).
    norms = np.linalg.norm(ws["embedding"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_same_seed_twice_gives_identical_checkpoint_hash():
    ws1, loss1 = train_geo_model(epochs=2, seed=0)
    ws2, loss2 = train_geo_model(epochs=2, seed=0)
    assert weights_token(ws1) == weights_token(ws2)
    assert loss1 == loss2
    ws3, _ = train_geo_model(epochs=2, seed=1)
    assert weights_token(ws3) != weights_token(ws1)


def test_checkpoint_save_load_equivalence(tmp_path):
    store = CacheStore(tmp_path)
    progress: list[tuple[float, str]] = []
    meta = train_canonical(progress_cb=lambda f, m: progress.append((f, m)), store=store)

    # Contract-shaped metadata with real metric values.
    assert set(meta) >= {
        "checkpoint_id",
        "final_loss",
        "coverage_uniformity",
        "field_directional_entropy",
        "seed",
    }
    assert meta["final_loss"] < UNIFORM_LOSS - 1.0
    # Progress messages match the contract's "epoch 7/30 · loss 4.12" shape.
    assert any("epoch" in m and "loss" in m for _, m in progress)
    assert progress[-1][0] == 1.0

    # Save -> load equivalence: the loaded weight set hashes to checkpoint_id and
    # reproduces the recorded loss exactly.
    ws = load_canonical_weight_set(store=store)
    assert weights_token(ws) == meta["checkpoint_id"]
    model = model_from_weight_set(ws)
    reloaded_loss = eval_loss(model, make_windows(corpus_token_stream(), stride=10))
    assert abs(reloaded_loss - meta["final_loss"]) < 1e-4

    # Idempotent: the second call is a cache hit with the identical checkpoint.
    meta2 = train_canonical(store=store)
    assert meta2["checkpoint_id"] == meta["checkpoint_id"]
    key, _ = canonical_cache_key()
    assert store.get(key) is not None
