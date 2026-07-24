"""Seeded, deterministic training of the canonical GeoTransformer checkpoint.

Real training on the real corpus (FR-103/FR-109): next-token cross-entropy over
sliding windows of the tokenized text, plus a spherical-uniformity auxiliary loss
(pairwise Gaussian repulsion — the Wang & Isola uniformity loss — on a sampled subset
of embedding rows each step). After every optimizer step the embedding rows are
renormalized to unit norm, so the token embeddings genuinely live on S² at all times.

Determinism: fixed seeds + ``torch.use_deterministic_algorithms(True)`` + CPU-only ⇒
the same seed always produces the bit-identical checkpoint (identical content hash).

The canonical checkpoint is persisted through the shared integrity-checked cache
(`cache/store.py`) under a deterministic spec key, together with its gate metrics:
``final_loss`` (mean token CE, nats), ``coverage_uniformity`` (0..1 dispersion of the
embedding over S²), and ``field_directional_entropy`` (nats; entropy of next_next
arrow directions binned on the sphere). ``train_canonical`` is what the API training
job runs; its ``progress_cb(fraction, message)`` messages look like
``"epoch 7/30 · loss 4.12"`` per the frozen contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import torch

from ..cache.keys import make_cache_key
from ..cache.store import CacheStore
from ..errors import ComputeError, NotFoundError
from .config import (
    CONTEXT_WINDOW,
    CORPUS_ID,
    CORPUS_SHA256,
    D_MODEL,
    EOS_ID,
    MLP_HIDDEN,
    N_LAYERS,
    PAD_ID,
    REPULSION_SAMPLE,
    REPULSION_T,
    REPULSION_WEIGHT,
    SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_WINDOW_STRIDE,
    VOCAB_SIZE,
)
from .corpus import load_corpus_text
from .fields import coverage_uniformity, field_directional_entropy, next_next_field
from .model import GeoTransformer, model_from_weight_set
from .tokenizer import get_tokenizer
from .weights import load_weight_set, save_weight_set

ProgressCb = Callable[[float, str], None]

_ARTIFACT_TYPE = "geo_checkpoint"
_TRAINER_VERSION = 1


@contextmanager
def deterministic_torch(seed: int):
    """Fully-seeded, deterministic torch execution.

    Restores BOTH the deterministic-algorithms flag and the global RNG state on exit:
    ``torch.manual_seed`` mutates the process-global generator, and leaking that would
    make unrelated later sampling (e.g. an unseeded ``arch.generate``) depend on
    whether a geo train/finetune ran earlier in the process.
    """
    prev_mode = torch.are_deterministic_algorithms_enabled()
    prev_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_mode)
        torch.set_rng_state(prev_rng)


# -- data --------------------------------------------------------------------------------


def corpus_token_stream() -> np.ndarray:
    """The whole corpus as one id stream, with <eos> closing every paragraph."""
    tok = get_tokenizer()
    stream: list[int] = []
    for paragraph in load_corpus_text().split("\n\n"):
        ids = tok.encode_stream(paragraph)
        if ids:
            stream.extend(ids)
            stream.append(EOS_ID)
    return np.asarray(stream, dtype=np.int64)


def make_windows(stream: np.ndarray, window: int = CONTEXT_WINDOW, stride: int = 1) -> np.ndarray:
    """(N, window+1) sliding windows: columns [:-1] are inputs, [1:] are targets."""
    span = window + 1
    if stream.shape[0] < span:
        padded = np.full(span, PAD_ID, dtype=np.int64)
        padded[: stream.shape[0]] = stream
        return padded[None, :]
    starts = np.arange(0, stream.shape[0] - span + 1, stride)
    return np.stack([stream[s : s + span] for s in starts])


# -- losses ------------------------------------------------------------------------------


def _ce_loss(model: GeoTransformer, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1), ignore_index=PAD_ID
    )


def _uniformity_loss(model: GeoTransformer, gen: torch.Generator) -> torch.Tensor:
    """Wang & Isola uniformity: log E[exp(−t‖eᵢ−eⱼ‖²)] over sampled embedding rows."""
    idx = torch.randint(0, VOCAB_SIZE, (REPULSION_SAMPLE,), generator=gen)
    e = model.embedding[idx]
    sq_dists = torch.pdist(e).pow(2)
    return torch.log(torch.exp(-REPULSION_T * sq_dists).mean())


@torch.no_grad()
def eval_loss(model: GeoTransformer, windows: np.ndarray, batch_size: int = 256) -> float:
    """Mean token cross-entropy (nats) over all windows."""
    total, count = 0.0, 0
    for start in range(0, windows.shape[0], batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size])
        targets = batch[:, 1:]
        n_tok = int((targets != PAD_ID).sum())
        if n_tok == 0:
            continue
        total += float(_ce_loss(model, batch)) * n_tok
        count += n_tok
    if count == 0:
        raise ComputeError("eval_loss: no non-pad tokens to evaluate")
    return total / count


# -- training ----------------------------------------------------------------------------


def train_geo_model(
    *,
    seed: int = SEED,
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    batch_size: int = TRAIN_BATCH_SIZE,
    stride: int = TRAIN_WINDOW_STRIDE,
    repulsion_weight: float = REPULSION_WEIGHT,
    progress_cb: ProgressCb | None = None,
    corpus_stream: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    """Train a GeoTransformer from scratch; return (weight set, final mean CE loss)."""
    stream = corpus_stream if corpus_stream is not None else corpus_token_stream()
    windows = make_windows(stream, stride=stride)
    with deterministic_torch(seed):
        model = GeoTransformer(seed=seed)
        gen = torch.Generator().manual_seed(seed + 1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = windows.shape[0]
        for epoch in range(epochs):
            order = torch.randperm(n, generator=gen).numpy()
            epoch_loss, epoch_batches = 0.0, 0
            for start in range(0, n, batch_size):
                batch = torch.from_numpy(windows[order[start : start + batch_size]])
                opt.zero_grad()
                ce = _ce_loss(model, batch)
                loss = ce + repulsion_weight * _uniformity_loss(model, gen)
                loss.backward()
                # Safety net against platform-dependent gradient explosions (the same
                # trajectory diverged on CI's Linux BLAS while stable on macOS).
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():  # keep the embeddings on S² (FR-103)
                    norms = model.embedding.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    model.embedding.div_(norms)
                epoch_loss += float(ce.detach())
                epoch_batches += 1
            if not np.isfinite(epoch_loss):
                raise ComputeError(f"Training diverged at epoch {epoch + 1} (non-finite loss)")
            if progress_cb is not None:
                mean_ce = epoch_loss / max(epoch_batches, 1)
                progress_cb(
                    (epoch + 1) / epochs, f"epoch {epoch + 1}/{epochs} · loss {mean_ce:.2f}"
                )
        final_loss = eval_loss(model, windows)
    return model.get_weight_set(), final_loss


def compute_gate_metrics(ws: dict[str, np.ndarray]) -> dict[str, float]:
    """The two non-degeneracy gate metrics for a weight set (SC-103)."""
    model = model_from_weight_set(ws)
    field = next_next_field(model, [], layer="full", temperature=0.0, top_m=1)
    return {
        "coverage_uniformity": coverage_uniformity(np.asarray(ws["embedding"])),
        "field_directional_entropy": field_directional_entropy(field["arrows"]),
    }


# -- the canonical checkpoint ------------------------------------------------------------


def canonical_cache_key(seed: int = SEED) -> tuple[str, dict[str, Any]]:
    params = {
        "trainer_version": _TRAINER_VERSION,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "mlp_hidden": MLP_HIDDEN,
        "vocab_size": VOCAB_SIZE,
        "context_window": CONTEXT_WINDOW,
        "corpus_id": CORPUS_ID,
        "corpus_sha256": CORPUS_SHA256,
        "epochs": TRAIN_EPOCHS,
        "lr": TRAIN_LR,
        "batch_size": TRAIN_BATCH_SIZE,
        "stride": TRAIN_WINDOW_STRIDE,
        "repulsion_weight": REPULSION_WEIGHT,
        "repulsion_sample": REPULSION_SAMPLE,
        "repulsion_t": REPULSION_T,
    }
    return make_cache_key(
        model_id="geo-transformer",
        revision=None,
        artifact_type=_ARTIFACT_TYPE,
        params=params,
        seed=seed,
    )


def train_canonical(
    progress_cb: ProgressCb | None = None,
    seed: int = SEED,
    force: bool = False,
    store: CacheStore | None = None,
) -> dict[str, Any]:
    """Train (or fetch) the canonical checkpoint; return its metadata.

    Metadata: ``checkpoint_id`` (content hash of the weight set — also a valid
    ``weights_token``), ``final_loss``, ``coverage_uniformity``,
    ``field_directional_entropy``, ``seed``. Idempotent: a cached checkpoint is
    returned as-is (< 100 ms), which is what makes POST /api/geo/train single-flight
    + cache-hit friendly.
    """
    store = store or CacheStore()
    key, spec = canonical_cache_key(seed)
    if not force:
        entry = store.get(key)
        if entry is not None:
            # Re-register the weight set under its token in case it was LRU-evicted.
            ws = {k: np.asarray(v, np.float32) for k, v in entry["arrays"].items()}
            save_weight_set(ws, source="learned", store=store)
            return dict(entry["meta"])

    ws, final_loss = train_geo_model(seed=seed, progress_cb=progress_cb)
    if progress_cb is not None:
        progress_cb(1.0, "computing gate metrics")
    metrics = compute_gate_metrics(ws)
    token = save_weight_set(ws, source="learned", store=store)
    meta: dict[str, Any] = {
        "checkpoint_id": token,
        "final_loss": float(final_loss),
        "coverage_uniformity": float(metrics["coverage_uniformity"]),
        "field_directional_entropy": float(metrics["field_directional_entropy"]),
        "seed": int(seed),
    }
    store.put(key, spec, meta, {name: np.asarray(a, np.float32) for name, a in ws.items()})
    return meta


def load_canonical_weight_set(
    seed: int = SEED, store: CacheStore | None = None
) -> dict[str, np.ndarray]:
    """The canonical learned weight set; NotFoundError if not yet trained."""
    store = store or CacheStore()
    key, _ = canonical_cache_key(seed)
    entry = store.get(key)
    if entry is None:
        raise NotFoundError(
            "The canonical GeoTransformer checkpoint has not been trained yet; "
            "POST /api/geo/train (or call train_canonical()) first"
        )
    return {name: np.asarray(arr, dtype=np.float32) for name, arr in entry["arrays"].items()}


def resolve_weight_set(base: str, store: CacheStore | None = None) -> dict[str, np.ndarray]:
    """Resolve the contract's ``base``/``weights_token`` field to a weight set."""
    if base == "learned":
        return load_canonical_weight_set(store=store)
    return load_weight_set(base, store=store)
