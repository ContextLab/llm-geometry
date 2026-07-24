"""Per-layer token embeddings (the canonical source for reductions).

Two sources:
  * ``static``     — rows of the input-embedding matrix (cheap, layer-independent base
                     location of each token).
  * ``contextual`` — per-layer hidden states from a real forward pass, taken by
                     feeding each reference token as a single-token input (no padding
                     needed since all inputs have length 1).

The "embedding reference set" is the set of tokens we compute for — distinct from the
2D grid's "reference points" (see reduce/grid.py). It defaults to a documented,
configurable subset for tractability (FR-002, spec Assumption).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..config import DEFAULT_REFERENCE_SET_SIZE, EMBED_BATCH_SIZE
from ..errors import InvalidParamError
from ..models.loader import LoadedModel, load_model

ProgressCb = Callable[[float, str], None]


def reference_token_ids(lm: LoadedModel, reference_set_size: int | None) -> np.ndarray:
    """Deterministic, representative reference set.

    The first-N token ids are mostly byte/special tokens and are unrepresentative, so we
    take N ids evenly spaced across the whole vocabulary instead (or the full vocab when
    no size is given). Deterministic -> reproducible cache (FR-013)."""
    vocab = lm.vocab_size
    if reference_set_size is None or int(reference_set_size) >= vocab:
        return np.arange(vocab, dtype=np.int64)
    n = int(reference_set_size)
    return np.unique(np.linspace(0, vocab - 1, n).astype(np.int64))


def _contextual_embeddings(
    lm: LoadedModel, token_ids: np.ndarray, layer: int, progress_cb: ProgressCb | None
) -> np.ndarray:
    n = int(token_ids.shape[0])
    vectors = np.empty((n, lm.hidden_size), dtype=np.float32)
    batch = max(1, int(EMBED_BATCH_SIZE))
    for start in range(0, n, batch):
        chunk = token_ids[start : start + batch]
        input_ids = torch.tensor(chunk.reshape(-1, 1), dtype=torch.long, device=lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.hidden_states[layer]  # [batch, 1, hidden]
        vectors[start : start + chunk.shape[0]] = hidden[:, 0, :].float().cpu().numpy()
        if progress_cb is not None:
            done = start + chunk.shape[0]
            progress_cb(min(1.0, done / n), f"embedding {done}/{n} tokens (layer {layer})")
    return vectors


def per_layer_embeddings(
    model_id: str,
    layer: int = 0,
    source: str = "static",
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    lm = load_model(model_id)
    if layer < 0 or layer > lm.num_layers:
        raise InvalidParamError(f"layer must be in 0..{lm.num_layers}, got {layer}")

    token_ids = reference_token_ids(lm, reference_set_size)

    if source == "static":
        weight = (
            lm.model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
        )
        vectors = weight[token_ids]
        if progress_cb is not None:
            progress_cb(1.0, f"static embeddings for {token_ids.shape[0]} tokens")
    elif source == "contextual":
        vectors = _contextual_embeddings(lm, token_ids, layer, progress_cb)
    else:
        raise InvalidParamError(f"source must be 'static' or 'contextual', got {source!r}")

    meta = {
        "model_id": lm.model_id,
        "revision": lm.revision,
        "layer": int(layer),
        "source": source,
        "shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "num_layers": lm.num_layers,
    }
    return {"meta": meta, "arrays": {"vectors": vectors, "token_ids": token_ids}}
