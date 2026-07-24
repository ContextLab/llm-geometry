"""Real SGD fine-tuning of the GeoTransformer — never mutating the canonical checkpoint.

Sources (exactly one, per the frozen `/api/geo/finetune` contract): raw ``text``, the
contents of an uploaded .txt/.md file (the route reads the file and passes it here as
``text``), or a HuggingFace dataset id streamed with ``datasets`` (first
``max_samples`` records; the ``text`` field, or the first string field found).

The base weight set is copied, fine-tuned with plain ``torch.optim.SGD`` for at most
500 steps (embedding rows renormalized to unit norm after every step, preserving the
S² invariant), and saved as a *new* content-hash checkpoint. ``loss_before`` /
``loss_after`` are the mean token cross-entropy on the fine-tuning text evaluated
with the base and the fine-tuned weights respectively. Results are cached under a
content-derived key so identical requests are 200-cache-hits.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from ..cache.keys import make_cache_key
from ..cache.store import CacheStore
from ..errors import ComputeError, InvalidParamError, NotFoundError, UnsupportedModelError
from .config import (
    EOS_ID,
    FINETUNE_DEFAULT_LR,
    FINETUNE_DEFAULT_MAX_SAMPLES,
    FINETUNE_DEFAULT_STEPS,
    FINETUNE_MAX_STEPS,
    PAD_ID,
    SEED,
)
from .model import model_from_weight_set
from .tokenizer import get_tokenizer
from .train import ProgressCb, deterministic_torch, eval_loss, make_windows, resolve_weight_set
from .weights import save_weight_set, weights_token

_ARTIFACT_TYPE = "geo_finetune"


def load_text_from_hf(
    dataset_id: str,
    split: str = "train",
    max_samples: int = FINETUNE_DEFAULT_MAX_SAMPLES,
) -> str:
    """Stream ``max_samples`` records from a real HF dataset and join their text."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover — datasets is a declared dependency
        raise UnsupportedModelError(f"the `datasets` package is unavailable: {exc}")
    try:
        stream = load_dataset(dataset_id, split=split, streaming=True)
    except (FileNotFoundError, ValueError) as exc:  # unknown id / bad split / bad config
        raise UnsupportedModelError(
            f"HuggingFace dataset {dataset_id!r} (split {split!r}) could not be loaded: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 — hub/network trouble is NOT a bad id
        raise ComputeError(
            f"Could not reach HuggingFace to stream dataset {dataset_id!r}: {exc}"
        ) from exc
    texts: list[str] = []
    text_field: str | None = None
    try:
        for i, record in enumerate(stream):
            if i >= max_samples:
                break
            if text_field is None:
                if "text" in record and isinstance(record["text"], str):
                    text_field = "text"
                else:
                    text_field = next((k for k, v in record.items() if isinstance(v, str)), None)
                if text_field is None:
                    raise UnsupportedModelError(
                        f"dataset {dataset_id!r} has no string field to fine-tune on "
                        f"(fields: {sorted(record)})"
                    )
            value = record.get(text_field)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    except UnsupportedModelError:
        raise
    except Exception as exc:  # noqa: BLE001 — streaming failures are "unusable id" too
        raise UnsupportedModelError(f"streaming dataset {dataset_id!r} failed: {exc}")
    if not texts:
        raise UnsupportedModelError(
            f"dataset {dataset_id!r} (split {split!r}) yielded no usable text "
            f"in the first {max_samples} records"
        )
    return "\n\n".join(texts)


def finetune_cache_key(
    base_token: str, text: str, steps: int, lr: float, seed: int
) -> tuple[str, dict[str, Any]]:
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return make_cache_key(
        model_id="geo-transformer",
        revision=None,
        artifact_type=_ARTIFACT_TYPE,
        inputs={"base_token": base_token, "text_sha256": text_sha},
        params={"steps": int(steps), "lr": float(lr)},
        seed=seed,
    )


def finetune(
    *,
    base: str = "learned",
    text: str | None = None,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    max_samples: int = FINETUNE_DEFAULT_MAX_SAMPLES,
    steps: int = FINETUNE_DEFAULT_STEPS,
    lr: float = FINETUNE_DEFAULT_LR,
    seed: int = SEED,
    progress_cb: ProgressCb | None = None,
    store: CacheStore | None = None,
) -> dict[str, Any]:
    """Fine-tune from ``base`` on one text source; return
    ``{"weights_token", "loss_before", "loss_after", "cached"}``."""
    if (text is None) == (hf_dataset is None):
        raise InvalidParamError("exactly one of text/hf_dataset must be provided")
    if not 1 <= int(steps) <= FINETUNE_MAX_STEPS:
        raise InvalidParamError(f"steps must be in 1..{FINETUNE_MAX_STEPS}, got {steps!r}")
    if lr <= 0:
        raise InvalidParamError(f"lr must be > 0, got {lr!r}")
    steps = int(steps)

    if hf_dataset is not None:
        if progress_cb is not None:
            progress_cb(0.0, f"streaming {hf_dataset} ({hf_split})")
        text = load_text_from_hf(hf_dataset, split=hf_split, max_samples=max_samples)
    assert text is not None
    if not text.strip():
        raise InvalidParamError("fine-tuning text is empty")

    user_store = store
    store = store or CacheStore()
    try:
        base_ws = resolve_weight_set(base, store=store)
    except NotFoundError:
        if user_store is None:
            raise
        # An isolated result store was supplied but the base lives in the shared
        # cache (e.g. base="learned"): resolve from there, write results here.
        base_ws = resolve_weight_set(base)
    base_token = weights_token(base_ws)

    key, spec = finetune_cache_key(base_token, text, steps, lr, seed)
    entry = store.get(key)
    if entry is not None:
        return {**entry["meta"], "cached": True}

    ids = get_tokenizer().encode_stream(text)
    if len(ids) < 2:
        raise InvalidParamError(
            "fine-tuning text is too short after tokenization (need at least 2 tokens)"
        )
    windows = make_windows(np.asarray(ids + [EOS_ID], dtype=np.int64), stride=25)

    with deterministic_torch(seed):
        model = model_from_weight_set(base_ws)
        loss_before = eval_loss(model, windows)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(seed + 1)
        n = windows.shape[0]
        batch_size = min(32, n)
        order = torch.randperm(n, generator=gen).numpy()
        cursor = 0
        for step in range(steps):
            if cursor + batch_size > n:
                order = torch.randperm(n, generator=gen).numpy()
                cursor = 0
            batch = torch.from_numpy(windows[order[cursor : cursor + batch_size]])
            cursor += batch_size
            opt.zero_grad()
            logits = model(batch[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
            loss.backward()
            # The norm-free d_model=3 model can explode on one bad batch (observed as a
            # nan loss on CI's Linux BLAS while the same seed was stable on macOS) —
            # clip, and fail LOUD if divergence happens anyway (never ship a nan).
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():  # preserve the S² embedding invariant
                norms = model.embedding.norm(dim=1, keepdim=True).clamp_min(1e-12)
                model.embedding.div_(norms)
            if not torch.isfinite(loss):
                raise ComputeError(
                    f"fine-tuning diverged (non-finite loss at step {step + 1}); "
                    "try fewer steps or a lower learning rate"
                )
            if progress_cb is not None and (step + 1) % 10 == 0:
                progress_cb((step + 1) / steps, f"step {step + 1}/{steps} · loss {loss:.2f}")
        loss_after = eval_loss(model, windows)
        if not (np.isfinite(loss_before) and np.isfinite(loss_after)):
            raise ComputeError("fine-tuning produced a non-finite loss; refusing to save")

    new_ws = model.get_weight_set()
    new_token = save_weight_set(new_ws, source="finetuned", store=store)
    meta = {
        "weights_token": new_token,
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "base_token": base_token,
    }
    store.put(key, spec, meta, {name: np.asarray(a, np.float32) for name, a in new_ws.items()})
    return {**meta, "cached": False}
