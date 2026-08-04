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


def test_train_batch_step_reproduces_the_training_loop():
    """`train_batch_step` IS the loop's body — driving it by hand must give the exact
    same model, bit for bit.

    The step was factored out so one training step could be pinned across languages
    (`tests/fixtures/geo/scratch_step.json`). That refactor is only safe if it changed
    nothing: this drives the extracted step with the same RNG consumption order the
    loop uses (one randperm per epoch, one randint per batch) and demands an identical
    content hash. (The canonical checkpoint's own id is machine-specific — Linux and
    macOS BLAS legitimately diverge — so equivalence, not a hard-coded hash, is what
    can honestly be pinned here.)
    """
    import torch

    from llm_geometry.geo.config import REPULSION_WEIGHT, TRAIN_LR
    from llm_geometry.geo.model import GeoTransformer
    from llm_geometry.geo.train import (
        deterministic_torch,
        sample_uniformity_indices,
        train_batch_step,
    )

    stream = corpus_token_stream()[:4000]
    windows = make_windows(stream, stride=10)
    epochs, batch_size, seed = 2, 64, 0

    expected_ws, expected_loss = train_geo_model(
        epochs=epochs, batch_size=batch_size, stride=10, seed=seed, corpus_stream=stream
    )

    with deterministic_torch(seed):
        model = GeoTransformer(seed=seed)
        gen = torch.Generator().manual_seed(seed + 1)
        opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
        n = windows.shape[0]
        for _epoch in range(epochs):
            order = torch.randperm(n, generator=gen).numpy()
            for start in range(0, n, batch_size):
                batch = torch.from_numpy(windows[order[start : start + batch_size]])
                idx = sample_uniformity_indices(gen)
                train_batch_step(model, opt, batch, idx, repulsion_weight=REPULSION_WEIGHT)
        manual_loss = eval_loss(model, windows)
    manual_ws = model.get_weight_set()

    assert weights_token(manual_ws) == weights_token(expected_ws)
    assert manual_loss == expected_loss


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
