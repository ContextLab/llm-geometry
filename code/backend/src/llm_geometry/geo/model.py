"""The GeoTransformer: a fully-transparent decoder-only transformer with d_model=3.

Architecture (frozen contract): 4 layers, 1 head, mlp_hidden=12, vocab 1003,
context window 50, tied unembedding (logits = h @ E.T), token embeddings unit-norm
on S².

Design choices (documented per the 002 plan):
- **No layer norm.** At d_model=3 normalization erases the very radial information the
  Geometry tab visualizes (normal residuals, sphere departures). Embeddings are kept
  unit-norm by training-time renormalization instead, and initialization is scaled so
  the norm-free residual stream stays stable.
- **Learned absolute positional embeddings**, added to the token embeddings at the
  input. Simple, fully visualizable (they are just 50 more 3-vectors), and directly
  addressable in the weight set.
- **Unscaled attention scores** ``⟨k_j, q_i⟩`` (no 1/√d): this matches the force-field
  definition in the frozen contract (`Σ_{j≤i} softmax(⟨K z_j, Q z_i⟩)·V z_j`,
  arXiv:2607.13295) so the trace and the force field are literally the same numbers.
- Per-layer matrices are separate ``nn.Parameter``s named W_Q/W_K/W_V/W_O (each 3×3,
  applied as ``z → W @ z``) so every matrix is addressable by ``(layer, name)``.
  The MLP is ``h @ W_in + b_in → gelu → @ W_out + b_out`` with W_in (3,12).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from ..errors import InvalidParamError
from .config import CONTEXT_WINDOW, D_MODEL, MLP_HIDDEN, N_LAYERS, VOCAB_SIZE

# The matrices addressable through the weights API (embedding ignores `layer`).
EDITABLE_MATRICES = ("W_Q", "W_K", "W_V", "W_O", "embedding")


class GeoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_Q = nn.Parameter(torch.empty(D_MODEL, D_MODEL))
        self.W_K = nn.Parameter(torch.empty(D_MODEL, D_MODEL))
        self.W_V = nn.Parameter(torch.empty(D_MODEL, D_MODEL))
        self.W_O = nn.Parameter(torch.empty(D_MODEL, D_MODEL))
        self.W_in = nn.Parameter(torch.empty(D_MODEL, MLP_HIDDEN))
        self.b_in = nn.Parameter(torch.zeros(MLP_HIDDEN))
        self.W_out = nn.Parameter(torch.empty(MLP_HIDDEN, D_MODEL))
        self.b_out = nn.Parameter(torch.zeros(D_MODEL))


class GeoTransformer(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(VOCAB_SIZE, D_MODEL))
        self.pos_embedding = nn.Parameter(torch.empty(CONTEXT_WINDOW, D_MODEL))
        self.layers = nn.ModuleList(GeoLayer() for _ in range(N_LAYERS))
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            # Token embeddings: random unit vectors on S².
            emb = torch.randn(VOCAB_SIZE, D_MODEL, generator=gen)
            self.embedding.copy_(emb / emb.norm(dim=1, keepdim=True))
            # Small positional offsets so token geometry dominates at init.
            self.pos_embedding.copy_(0.02 * torch.randn(CONTEXT_WINDOW, D_MODEL, generator=gen))
            for layer in self.layers:
                for w in (layer.W_Q, layer.W_K, layer.W_V):
                    w.copy_(0.4 * torch.randn(D_MODEL, D_MODEL, generator=gen))
                # W_O and W_out start small: the norm-free residual stream stays near
                # the sphere early in training.
                layer.W_O.copy_(0.1 * torch.randn(D_MODEL, D_MODEL, generator=gen))
                layer.W_in.copy_(
                    torch.randn(D_MODEL, MLP_HIDDEN, generator=gen) / math.sqrt(D_MODEL)
                )
                layer.b_in.zero_()
                layer.W_out.copy_(
                    0.1 * torch.randn(MLP_HIDDEN, D_MODEL, generator=gen) / math.sqrt(MLP_HIDDEN)
                )
                layer.b_out.zero_()

    # -- core computation --------------------------------------------------------------

    def _run(self, ids: torch.Tensor, need_trace: bool) -> dict[str, Any]:
        """One causal forward pass.

        ``ids``: (B, T) int64, T ≤ CONTEXT_WINDOW. Returns hidden states per layer and,
        when ``need_trace``, every intermediate the `/api/geo/trace` contract exposes.
        """
        if ids.dim() != 2:
            raise InvalidParamError(f"ids must be 2-D (batch, time), got shape {tuple(ids.shape)}")
        B, T = ids.shape
        if T < 1 or T > CONTEXT_WINDOW:
            raise InvalidParamError(f"sequence length must be in 1..{CONTEXT_WINDOW}, got {T}")

        tok = self.embedding[ids]  # (B, T, 3)
        h = tok + self.pos_embedding[:T].unsqueeze(0)
        causal = torch.full((T, T), float("-inf"), device=ids.device).triu(1)  # (T, T)

        hidden_out: list[torch.Tensor] = []
        trace: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            hidden_in = h
            q = h @ layer.W_Q.T  # q_i = W_Q z_i
            k = h @ layer.W_K.T
            v = h @ layer.W_V.T
            scores = q @ k.transpose(1, 2) + causal  # unscaled ⟨k_j, q_i⟩, causal mask
            attn = torch.softmax(scores, dim=-1)  # (B, T, T) row-stochastic
            attn_out = (attn @ v) @ layer.W_O.T
            h = h + attn_out
            mlp_out = torch.nn.functional.gelu(h @ layer.W_in + layer.b_in) @ layer.W_out
            mlp_out = mlp_out + layer.b_out
            h = h + mlp_out
            hidden_out.append(h)
            if need_trace:
                trace.append(
                    {
                        "attention": attn,
                        "q": q,
                        "k": k,
                        "v": v,
                        "hidden_in": hidden_in,
                        "attn_out": attn_out,
                        "mlp_out": mlp_out,
                        "hidden_out": h,
                    }
                )
        return {"token_embeddings": tok, "hidden_out": hidden_out, "trace": trace}

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """Tied unembedding: logits = h @ E.T (logit-lens for intermediate layers)."""
        return h @ self.embedding.T

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        run = self._run(ids, need_trace=False)
        return self.readout(run["hidden_out"][-1])

    def hidden_at(self, ids: torch.Tensor, layer: int | str) -> torch.Tensor:
        """Residual stream after ``layer`` (0..N_LAYERS-1) or "full" (= final layer)."""
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        run = self._run(ids, need_trace=False)
        return run["hidden_out"][resolve_layer(layer)]

    @torch.no_grad()
    def forward_trace(self, ids: torch.Tensor | list[int]) -> dict[str, Any]:
        """Everything the `/api/geo/trace` contract needs, as float32 numpy arrays."""
        ids_t = torch.as_tensor(ids, dtype=torch.long)
        if ids_t.dim() != 1:
            raise InvalidParamError("forward_trace takes a single 1-D token sequence")
        if ids_t.numel() == 0:
            raise InvalidParamError("forward_trace requires at least one token")
        run = self._run(ids_t.unsqueeze(0), need_trace=True)

        def np32(t: torch.Tensor) -> np.ndarray:
            return t.squeeze(0).detach().cpu().numpy().astype(np.float32)

        layers = [
            {
                "layer": i,
                "attention": np32(tr["attention"]),
                "q": np32(tr["q"]),
                "k": np32(tr["k"]),
                "v": np32(tr["v"]),
                "hidden_in": np32(tr["hidden_in"]),
                "attn_out": np32(tr["attn_out"]),
                "mlp_out": np32(tr["mlp_out"]),
                "hidden_out": np32(tr["hidden_out"]),
            }
            for i, tr in enumerate(run["trace"])
        ]
        final_logits = self.readout(run["hidden_out"][-1])[0, -1]
        probs = torch.softmax(final_logits, dim=-1)
        return {
            "embeddings": np32(run["token_embeddings"]),
            "layers": layers,
            "probs": probs.detach().cpu().numpy().astype(np.float32),
        }

    # -- weight-set plumbing -----------------------------------------------------------

    def weight_names(self) -> list[str]:
        names = ["embedding", "pos_embedding"]
        for i in range(N_LAYERS):
            names += [f"layers.{i}.{n}" for n in ("W_Q", "W_K", "W_V", "W_O")]
            names += [f"layers.{i}.{n}" for n in ("W_in", "b_in", "W_out", "b_out")]
        return names

    def get_weight_set(self) -> dict[str, np.ndarray]:
        state = dict(self.named_parameters())
        return {
            name: state[name].detach().cpu().numpy().astype(np.float32).copy()
            for name in self.weight_names()
        }

    def load_weight_set(self, ws: dict[str, np.ndarray]) -> "GeoTransformer":
        expected = self.weight_names()
        missing = [n for n in expected if n not in ws]
        extra = [n for n in ws if n not in expected]
        if missing or extra:
            raise InvalidParamError(
                f"Weight set mismatch (missing: {missing or 'none'}, extra: {extra or 'none'})"
            )
        state = dict(self.named_parameters())
        with torch.no_grad():
            for name in expected:
                arr = torch.as_tensor(np.asarray(ws[name], dtype=np.float32))
                if tuple(arr.shape) != tuple(state[name].shape):
                    raise InvalidParamError(
                        f"Weight {name!r} has shape {tuple(arr.shape)}, "
                        f"expected {tuple(state[name].shape)}"
                    )
                state[name].copy_(arr)
        return self

    def get_matrix(self, layer: int | None, name: str) -> np.ndarray:
        """Address one contract-editable matrix by (layer, name)."""
        if name == "embedding":
            return self.embedding.detach().cpu().numpy().astype(np.float32).copy()
        if name not in EDITABLE_MATRICES:
            raise InvalidParamError(f"Unknown matrix {name!r}; expected one of {EDITABLE_MATRICES}")
        if layer is None or not 0 <= int(layer) < N_LAYERS:
            raise InvalidParamError(f"layer must be in 0..{N_LAYERS - 1} for {name}, got {layer!r}")
        param = getattr(self.layers[int(layer)], name)
        return param.detach().cpu().numpy().astype(np.float32).copy()


def resolve_layer(layer: int | str) -> int:
    """Map the contract's ``layer`` param (0..3 or "full") to a hidden-state index."""
    if layer == "full":
        return N_LAYERS - 1
    try:
        idx = int(layer)
    except (TypeError, ValueError):
        raise InvalidParamError(f'layer must be 0..{N_LAYERS - 1} or "full", got {layer!r}')
    if not 0 <= idx < N_LAYERS:
        raise InvalidParamError(f'layer must be 0..{N_LAYERS - 1} or "full", got {layer!r}')
    return idx


def model_from_weight_set(ws: dict[str, np.ndarray]) -> GeoTransformer:
    return GeoTransformer().load_weight_set(ws)
