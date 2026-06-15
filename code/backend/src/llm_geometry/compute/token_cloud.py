"""Full-vocabulary token cloud — a dot for EVERY token. (project_description.md §1, §3)

The static input-embedding matrix has one vector per vocabulary token, so it is the only
representation we can afford to reduce for the *entire* vocabulary (contextual per-layer
embeddings would need one forward pass per token). We project that whole matrix to 2D with
a single PCA and spread it toward an even, grid-like layout (mapper-style density
flattening) so the cloud isn't a clump.

This is computed once per model (cached) and reused as the shared coordinate space for the
vector-field arrows: ``vector_field`` projects its contextual layer-n/layer-m embeddings
through THIS PCA and maps them into THIS spread layout (see ``reduce.spread.warp_like``), so
arrows overlay the cloud coherently. The stored ``raw`` / ``pca_*`` arrays exist for that
projection and are not shipped to the browser; the ``/api/token_cloud`` endpoint returns
only the warped coordinates + token ids the renderer needs.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..config import DEFAULT_SEED
from ..models.loader import load_model

ProgressCb = Callable[[float, str], None]


def token_cloud(
    model_id: str,
    seed: int = DEFAULT_SEED,
    spread_mu: float = 0.65,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    lm = load_model(model_id)
    if progress_cb:
        progress_cb(0.1, "reading static embedding matrix")
    matrix = lm.model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    vocab = int(matrix.shape[0])
    token_ids = np.arange(vocab, dtype=np.int64)

    if progress_cb:
        progress_cb(0.35, f"PCA → 2D over {vocab} tokens")
    from sklearn.decomposition import PCA

    n_comp = min(2, matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(matrix)
    raw2 = pca.transform(matrix)
    if n_comp < 2:  # pathological tiny models (hidden_size 1) — pad to 2D
        raw2 = np.column_stack([raw2[:, 0], np.zeros(vocab)])
        components = np.vstack([pca.components_, np.zeros((1, matrix.shape[1]))])
    else:
        components = pca.components_

    if progress_cb:
        progress_cb(0.6, "spreading toward grid-like layout")
    from ..reduce.spread import flatten_density

    warped = flatten_density(raw2, mu=spread_mu, seed=seed)

    if progress_cb:
        progress_cb(1.0, "done")

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "vocab_size": vocab,
        "seed": int(seed), "spread_mu": float(spread_mu),
    }
    arrays = {
        "warped": warped.astype(np.float32),          # shipped to the browser
        "token_ids": token_ids,                        # shipped to the browser
        "raw": raw2.astype(np.float32),                # internal: arrow projection
        "pca_mean": pca.mean_.astype(np.float32),      # internal: arrow projection
        "pca_components": components.astype(np.float32),  # internal: (2, H)
    }
    return {"meta": meta, "arrays": arrays}
