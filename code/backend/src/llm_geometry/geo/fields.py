"""The two Geometry Lab vector-field modes, exactly per the frozen contract.

`next_next` (the issue's field): for every vocab token *v* hypothetically appended to
the prompt, arrows from ``points[v]`` toward the embedding(s) of the following-token
prediction. ``layer`` selects which layer's residual stream feeds the (tied, logit-lens)
readout; ``"full"`` = the final layer. ``temperature=0 & top_m=1`` ⇒ one argmax arrow
per point with weight 1; otherwise the ``top_m`` most probable targets, each arrow
weighted by its raw probability. At ``temperature=0`` the distribution is one-hot, so
only the argmax arrow (the only one with positive weight) is emitted regardless of
``top_m``.

`force` (arXiv:2607.13295): the per-point field ``W_V·z`` over all vocab points for
the chosen layer (``antisymmetrize=true`` uses ``(W_V−W_Vᵀ)/2``, whose field is
*exactly* tangent to the sphere: ⟨Az, z⟩ = 0 for antisymmetric A), plus per-sequence-
position aggregate forces ``Σ_{j≤i} softmax(⟨K z_j, Q z_i⟩)·V z_j`` — literally the
model's ``attention @ v`` rows.

Those aggregate forces are **projected onto the tangent plane at their anchor point**
``z_i`` before being returned, and the magnitude of the removed radial component is
reported as ``normal_residual``. This matters: the arrow is drawn anchored at ``z_i``
on the sphere, and antisymmetrizing ``W_V`` does NOT make the sum tangent there —
each term ``W_V z_j`` is tangent at ``z_j``, not at ``z_i``. Projecting is a display
choice, and ``normal_residual`` is how the UI stays honest about what it hides.

The antisymmetrize toggle applies to the per-point field only; the aggregate forces
always use the real W_V. ``layer="full"`` is invalid for force mode (it is per-layer
by definition).

Also home to the spherical-binning helpers shared with the training gate metrics
(Fibonacci-lattice bins; directional entropy; coverage uniformity).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..errors import InvalidParamError
from .config import CONTEXT_WINDOW, N_LAYERS, SPHERE_BINS, VOCAB_SIZE
from .model import GeoTransformer, resolve_layer


def _clip_prompt(prompt_ids: list[int], room: int) -> list[int]:
    """Keep the most recent tokens, leaving ``room`` positions free (LM conditioning)."""
    max_len = CONTEXT_WINDOW - room
    return list(prompt_ids)[-max_len:] if len(prompt_ids) > max_len else list(prompt_ids)


@torch.no_grad()
def next_next_field(
    model: GeoTransformer,
    prompt_ids: list[int],
    layer: int | str = "full",
    temperature: float = 0.0,
    top_m: int = 1,
) -> dict[str, Any]:
    if temperature is None or float(temperature) < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature!r}")
    if int(top_m) < 1:
        raise InvalidParamError(f"top_m must be >= 1, got {top_m!r}")
    layer_idx = resolve_layer(layer)

    prompt = _clip_prompt([int(t) for t in prompt_ids], room=1)
    points = model.embedding.detach().cpu().numpy().astype(np.float32)  # (V, 3)

    # One batched forward: every vocab token appended to the same prompt.
    vocab = torch.arange(VOCAB_SIZE, dtype=torch.long).unsqueeze(1)  # (V, 1)
    if prompt:
        prefix = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).expand(VOCAB_SIZE, -1)
        ids = torch.cat([prefix, vocab], dim=1)  # (V, P+1)
    else:
        ids = vocab
    run = model._run(ids, need_trace=False)
    h_last = run["hidden_out"][layer_idx][:, -1, :]  # (V, 3)
    logits = model.readout(h_last)  # (V, V)

    arrows: list[dict[str, Any]] = []
    if float(temperature) == 0.0:
        targets = torch.argmax(logits, dim=-1).cpu().numpy()  # (V,)
        for v in range(VOCAB_SIZE):
            vec = points[int(targets[v])] - points[v]
            arrows.append({"origin_index": v, "vec": vec.tolist(), "weight": 1.0})
    else:
        probs = torch.softmax(logits / float(temperature), dim=-1)
        m = min(int(top_m), VOCAB_SIZE)
        top_p, top_i = torch.topk(probs, m, dim=-1)
        top_p_np = top_p.cpu().numpy()
        top_i_np = top_i.cpu().numpy()
        for v in range(VOCAB_SIZE):
            for rank in range(m):
                weight = float(top_p_np[v, rank])
                if weight <= 0.0:
                    continue
                vec = points[int(top_i_np[v, rank])] - points[v]
                arrows.append({"origin_index": v, "vec": vec.tolist(), "weight": weight})

    return {
        "mode": "next_next",
        "layer": "full" if layer == "full" else layer_idx,
        "points": points,
        "token_ids": list(range(VOCAB_SIZE)),
        "arrows": arrows,
        "sequence_forces": None,
        "tangent_exact": False,
    }


@torch.no_grad()
def force_field(
    model: GeoTransformer,
    prompt_ids: list[int],
    layer: int | str,
    antisymmetrize: bool = False,
) -> dict[str, Any]:
    if layer == "full":
        raise InvalidParamError(
            'layer="full" is invalid for force mode: the attention force is per-layer '
            "by definition; choose a layer 0..3"
        )
    layer_idx = resolve_layer(layer)
    if not 0 <= layer_idx < N_LAYERS:  # unreachable, resolve_layer guards; keep explicit
        raise InvalidParamError(f"layer must be in 0..{N_LAYERS - 1}")

    points = model.embedding.detach().cpu().numpy().astype(np.float32)  # (V, 3)
    w_v = model.layers[layer_idx].W_V.detach().cpu().numpy().astype(np.float32)
    w_eff = 0.5 * (w_v - w_v.T) if antisymmetrize else w_v

    vecs = points @ w_eff.T  # per-point field W_V·z, (V, 3)
    mags = np.linalg.norm(vecs, axis=1)
    max_mag = float(mags.max())
    weights = mags / max_mag if max_mag > 0 else np.zeros_like(mags)
    arrows = [
        {"origin_index": v, "vec": vecs[v].tolist(), "weight": float(weights[v])}
        for v in range(VOCAB_SIZE)
    ]

    sequence_forces: list[dict[str, Any]] = []
    prompt = _clip_prompt([int(t) for t in prompt_ids], room=0)
    if prompt:
        ids = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
        run = model._run(ids, need_trace=True)
        tr = run["trace"][layer_idx]
        attn = tr["attention"][0]  # (T, T) row-stochastic, causal
        v_proj = tr["v"][0]  # (T, 3) — the real W_V z_j
        forces = (attn @ v_proj).cpu().numpy().astype(np.float32)  # Σ_{j≤i} A_ij V z_j
        # Project at the point the arrow is actually DRAWN at — the prompt token's
        # embedding, which is the unit-norm point on the sphere the client anchors to
        # (`GeoTrace.embeddings`). Projecting at the layer's residual stream
        # (`hidden_in`) instead was wrong: the residual stream is not on the sphere and
        # drifts away from the token embedding with depth, so the "tangent" arrows came
        # out up to 59° off the tangent plane at layer 2 while the UI claimed otherwise.
        #
        # Antisymmetrizing W_V does not help here either: each term W_V z_j is tangent
        # at z_j, not at the anchor where the sum is drawn. `normal_residual` reports the
        # radial magnitude removed, so nothing is hidden.
        anchors = points[np.asarray(prompt, dtype=np.int64)]  # (T, 3) unit-norm
        anchor_norms = np.linalg.norm(anchors, axis=1)
        for i in range(forces.shape[0]):
            if anchor_norms[i] > 1e-12:
                n_hat = anchors[i] / anchor_norms[i]
                radial = float(forces[i] @ n_hat)
                tangential = forces[i] - radial * n_hat
            else:
                radial = 0.0
                tangential = forces[i]
            sequence_forces.append(
                {
                    "position": i,
                    "vec": tangential.tolist(),
                    "normal_residual": abs(radial),
                }
            )

    return {
        "mode": "force",
        "layer": layer_idx,
        "points": points,
        "token_ids": list(range(VOCAB_SIZE)),
        "arrows": arrows,
        "sequence_forces": sequence_forces,
        "tangent_exact": bool(antisymmetrize),
    }


# -- spherical binning + gate metrics --------------------------------------------------


def fibonacci_sphere(n: int = SPHERE_BINS) -> np.ndarray:
    """(n, 3) unit vectors on a golden-angle Fibonacci lattice (near-equal-area bins)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    ).astype(np.float32)


def _bin_entropy(unit_vectors: np.ndarray, n_bins: int) -> float:
    """Shannon entropy (nats) of nearest-bin occupancy for unit vectors on S²."""
    if unit_vectors.shape[0] == 0:
        return 0.0
    bins = fibonacci_sphere(n_bins)
    assignment = np.argmax(unit_vectors @ bins.T, axis=1)
    counts = np.bincount(assignment, minlength=n_bins).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def coverage_uniformity(points: np.ndarray, n_bins: int = SPHERE_BINS) -> float:
    """Dispersion of embedding points over the sphere, normalized to 0..1."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / np.maximum(norms, 1e-12)
    return _bin_entropy(unit.astype(np.float32), n_bins) / float(np.log(n_bins))


def field_directional_entropy(arrows: list[dict[str, Any]], n_bins: int = SPHERE_BINS) -> float:
    """Entropy (nats) of arrow *directions* binned on the sphere. Zero-length arrows
    (self-predictions) carry no direction and are excluded."""
    vecs = np.asarray([a["vec"] for a in arrows], dtype=np.float64)
    if vecs.size == 0:
        return 0.0
    mags = np.linalg.norm(vecs, axis=1)
    keep = mags > 1e-8
    if not np.any(keep):
        return 0.0
    unit = (vecs[keep] / mags[keep, None]).astype(np.float32)
    return _bin_entropy(unit, n_bins)
