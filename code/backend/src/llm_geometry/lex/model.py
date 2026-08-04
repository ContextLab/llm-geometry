"""The Lexicon Lab's tiny transformer — the exact shape of `specs/006-.../architecture.md`.

This is a *faithful reproduction* of the source project's `TinyLM`, not a tidied-up
rewrite, because the whole point of the tab is to show what that model does. Two
consequences that look like bugs and are not:

* **The initialization is mixed.** The source's `_init` only matches `nn.Linear` and
  `nn.Embedding`, and `nn.MultiheadAttention.in_proj_weight` is a bare `nn.Parameter` —
  so the packed QKV matrix silently keeps PyTorch's xavier-uniform default while every
  other matrix is drawn from `N(0, 0.02²)`. We reproduce that split deliberately
  (`test_lex_model.py` pins both distributions).
* **Dropout sits in three unusual places**: the embedding sum, the *attention weights*,
  and after the second MLP linear — but **not** on the attention residual branch.

Parameters are plain `nn.Parameter`s named exactly as the architecture table names them,
so `named_parameters()`, `weight_dict()`, the save/load bundle and the TypeScript engine
all agree on one set of stable keys. `weight_dict()` returns plain float32 numpy arrays
because those three consumers have nothing else in common.

Tying is a config flag, never an alias that can be lost: when `tied` is true there is no
`head_w` parameter at all and the readout reads `embed` directly. That closes the source's
`probe.py` defect, where a tied checkpoint reloaded untied and silently desynchronized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..errors import InvalidParamError
from .config import (
    DEFAULT_CTX,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_SEED,
    DEFAULT_TIED,
    LAYER_NORM_EPS,
    MLP_RATIO,
    SPECIAL_TOKENS,
    param_count,
)

#: Per-block weight names, in the order the architecture table lists them.
BLOCK_WEIGHTS = (
    "ln1_g",
    "ln1_b",
    "qkv_w",
    "qkv_b",
    "proj_w",
    "proj_b",
    "ln2_g",
    "ln2_b",
    "fc1_w",
    "fc1_b",
    "fc2_w",
    "fc2_b",
)

#: The `N(0, 0.02²)` standard deviation the source's `_init` applies to every matrix it
#: reaches. `qkv_w` is deliberately NOT one of them (see the module docstring).
INIT_STD = 0.02


@dataclass(frozen=True)
class LexConfig:
    """Everything that fixes the model's shape.

    ``vocab_rows`` is ``|V| + len(SPECIAL_TOKENS)`` — embedding *rows*, not the budget
    size the UI quotes. Keeping the row count here (rather than the budget size) means the
    model never has to know which of its rows are specials.
    """

    vocab_rows: int
    d_model: int = DEFAULT_D_MODEL
    n_layers: int = DEFAULT_N_LAYERS
    n_heads: int = DEFAULT_N_HEADS
    ctx: int = DEFAULT_CTX
    tied: bool = DEFAULT_TIED
    dropout: float = DEFAULT_DROPOUT

    def __post_init__(self) -> None:
        for name in ("vocab_rows", "d_model", "n_layers", "n_heads", "ctx"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise InvalidParamError(f"{name} must be a positive integer, got {value!r}")
        if self.vocab_rows <= len(SPECIAL_TOKENS):
            raise InvalidParamError(
                f"vocab_rows must exceed the {len(SPECIAL_TOKENS)} special tokens "
                f"({', '.join(SPECIAL_TOKENS)}), got {self.vocab_rows}"
            )
        if self.d_model % self.n_heads != 0:
            raise InvalidParamError(
                f"d_model must be divisible by n_heads so every head gets an equal slice: "
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads} "
                f"(head_dim would be {self.d_model / self.n_heads})"
            )
        if not isinstance(self.tied, bool):
            raise InvalidParamError(f"tied must be a bool, got {self.tied!r}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise InvalidParamError(f"dropout must be in [0, 1), got {self.dropout!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden(self) -> int:
        return MLP_RATIO * self.d_model

    @property
    def n_params(self) -> int:
        """The closed form from `config.param_count`, verified against the real model."""
        return param_count(
            vocab_rows=self.vocab_rows,
            d_model=self.d_model,
            n_layers=self.n_layers,
            ctx=self.ctx,
            tied=self.tied,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_rows": self.vocab_rows,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ctx": self.ctx,
            "tied": self.tied,
            "dropout": float(self.dropout),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LexConfig":
        """Rebuild a config from a bundle, refusing unknown or missing fields loudly."""
        known = set(cls.__dataclass_fields__)
        extra = sorted(set(data) - known)
        if extra:
            raise InvalidParamError(f"unknown model config fields: {extra}")
        if "vocab_rows" not in data:
            raise InvalidParamError("model config is missing vocab_rows")
        return cls(**{k: v for k, v in data.items() if k in known})


class LexBlock(nn.Module):
    """One pre-norm block: LN → packed-QKV attention → residual → LN → MLP → residual."""

    def __init__(self, cfg: LexConfig) -> None:
        super().__init__()
        d, hidden = cfg.d_model, cfg.mlp_hidden
        self.ln1_g = nn.Parameter(torch.ones(d))
        self.ln1_b = nn.Parameter(torch.zeros(d))
        self.qkv_w = nn.Parameter(torch.empty(3 * d, d))
        self.qkv_b = nn.Parameter(torch.zeros(3 * d))
        self.proj_w = nn.Parameter(torch.empty(d, d))
        self.proj_b = nn.Parameter(torch.zeros(d))
        self.ln2_g = nn.Parameter(torch.ones(d))
        self.ln2_b = nn.Parameter(torch.zeros(d))
        self.fc1_w = nn.Parameter(torch.empty(hidden, d))
        self.fc1_b = nn.Parameter(torch.zeros(hidden))
        self.fc2_w = nn.Parameter(torch.empty(d, hidden))
        self.fc2_b = nn.Parameter(torch.zeros(d))


class LexModel(nn.Module):
    """A decoder-only transformer over a bounded word vocabulary."""

    def __init__(self, cfg: LexConfig, seed: int = DEFAULT_SEED) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Parameter(torch.empty(cfg.vocab_rows, d))
        self.pos = nn.Parameter(torch.empty(cfg.ctx, d))
        self.blocks = nn.ModuleList(LexBlock(cfg) for _ in range(cfg.n_layers))
        self.lnf_g = nn.Parameter(torch.ones(d))
        self.lnf_b = nn.Parameter(torch.zeros(d))
        if not cfg.tied:
            self.head_w = nn.Parameter(torch.empty(cfg.vocab_rows, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.mlp_drop = nn.Dropout(cfg.dropout)
        self.reset_parameters(seed)

    # -- initialization ----------------------------------------------------------------

    def reset_parameters(self, seed: int = DEFAULT_SEED) -> None:
        """The source's mixed init, drawn from one seeded stream in a fixed order.

        `qkv_w` gets xavier-uniform with bound ``sqrt(6/(3d + d))`` because the source
        never reaches it; everything else that is a matrix gets ``N(0, 0.02²)``; biases
        are zero and LayerNorm is (γ=1, β=0).
        """
        gen = torch.Generator().manual_seed(int(seed))
        d = self.cfg.d_model
        qkv_bound = math.sqrt(6.0 / (3 * d + d))
        with torch.no_grad():
            self.embed.normal_(0.0, INIT_STD, generator=gen)
            self.pos.normal_(0.0, INIT_STD, generator=gen)
            for block in self.blocks:
                block.ln1_g.fill_(1.0)
                block.ln1_b.zero_()
                block.qkv_w.uniform_(-qkv_bound, qkv_bound, generator=gen)
                block.qkv_b.zero_()
                block.proj_w.normal_(0.0, INIT_STD, generator=gen)
                block.proj_b.zero_()
                block.ln2_g.fill_(1.0)
                block.ln2_b.zero_()
                block.fc1_w.normal_(0.0, INIT_STD, generator=gen)
                block.fc1_b.zero_()
                block.fc2_w.normal_(0.0, INIT_STD, generator=gen)
                block.fc2_b.zero_()
            self.lnf_g.fill_(1.0)
            self.lnf_b.zero_()
            if not self.cfg.tied:
                self.head_w.normal_(0.0, INIT_STD, generator=gen)

    # -- forward -----------------------------------------------------------------------

    @property
    def readout(self) -> torch.Tensor:
        """The `(V, d)` readout matrix — literally `embed` when tied."""
        return self.embed if self.cfg.tied else self.head_w

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """`(B, T)` int64 token ids → `(B, T, V)` logits. `T ≤ ctx`."""
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2:
            raise InvalidParamError(f"ids must be 2-D (batch, time), got shape {tuple(ids.shape)}")
        _, seq_len = ids.shape
        if seq_len < 1 or seq_len > self.cfg.ctx:
            raise InvalidParamError(
                f"sequence length must be in 1..{self.cfg.ctx} (the model's context), got {seq_len}"
            )
        if int(ids.max()) >= self.cfg.vocab_rows or int(ids.min()) < 0:
            raise InvalidParamError(
                f"token ids must be in 0..{self.cfg.vocab_rows - 1}, got range "
                f"{int(ids.min())}..{int(ids.max())}"
            )

        cfg = self.cfg
        d, n_heads, head_dim = cfg.d_model, cfg.n_heads, cfg.head_dim
        batch = ids.shape[0]
        norm_shape = (d,)

        h = self.embed[ids] + self.pos[:seq_len].unsqueeze(0)
        h = self.drop(h)
        causal = torch.full((seq_len, seq_len), float("-inf"), device=ids.device).triu(1)

        for block in self.blocks:
            a = F.layer_norm(h, norm_shape, block.ln1_g, block.ln1_b, LAYER_NORM_EPS)
            qkv = a @ block.qkv_w.T + block.qkv_b
            q, k, v = qkv.split(d, dim=-1)
            # (B, T, d) → (B, H, T, dh): head n owns rows [n·dh, (n+1)·dh) of W_q/W_k/W_v.
            q = q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim) + causal
            attn = self.attn_drop(torch.softmax(scores, dim=-1))
            o = (attn @ v).transpose(1, 2).reshape(batch, seq_len, d)
            o = o @ block.proj_w.T + block.proj_b
            h = h + o  # no dropout on this branch — the source's placement

            m = F.layer_norm(h, norm_shape, block.ln2_g, block.ln2_b, LAYER_NORM_EPS)
            # Exact erf GELU (`approximate="none"`), not the tanh approximation.
            m = F.gelu(m @ block.fc1_w.T + block.fc1_b, approximate="none")
            m = m @ block.fc2_w.T + block.fc2_b
            h = h + self.mlp_drop(m)

        h = F.layer_norm(h, norm_shape, self.lnf_g, self.lnf_b, LAYER_NORM_EPS)
        return h @ self.readout.T  # (B, T, V), no bias

    # -- weights -----------------------------------------------------------------------

    def weight_names(self) -> list[str]:
        """Stable key order. `head_w` appears only when the model is untied."""
        names = ["embed", "pos"]
        for i in range(self.cfg.n_layers):
            names += [f"blocks.{i}.{n}" for n in BLOCK_WEIGHTS]
        names += ["lnf_g", "lnf_b"]
        if not self.cfg.tied:
            names.append("head_w")
        return names

    def weight_dict(self) -> dict[str, np.ndarray]:
        """Plain float32 numpy copies, keyed by `weight_names()`."""
        params = dict(self.named_parameters())
        return {
            name: params[name].detach().cpu().numpy().astype(np.float32).copy()
            for name in self.weight_names()
        }

    def load_weight_dict(self, weights: dict[str, np.ndarray]) -> "LexModel":
        """Load a `weight_dict()`, refusing any mismatch instead of partially loading."""
        expected = self.weight_names()
        missing = [n for n in expected if n not in weights]
        extra = [n for n in weights if n not in expected]
        if missing or extra:
            raise InvalidParamError(
                f"weight set mismatch (missing: {missing or 'none'}, extra: {extra or 'none'}); "
                f"a tied model has no head_w and an untied one requires it"
            )
        params = dict(self.named_parameters())
        with torch.no_grad():
            for name in expected:
                arr = np.asarray(weights[name], dtype=np.float32)
                if tuple(arr.shape) != tuple(params[name].shape):
                    raise InvalidParamError(
                        f"weight {name!r} has shape {tuple(arr.shape)}, "
                        f"expected {tuple(params[name].shape)}"
                    )
                if not np.isfinite(arr).all():
                    raise InvalidParamError(f"weight {name!r} contains non-finite values")
                params[name].copy_(torch.from_numpy(arr))
        return self

    def n_parameters(self) -> int:
        """Measured parameter count — must equal `cfg.n_params`."""
        return sum(p.numel() for p in self.parameters())


def model_from_weight_dict(
    cfg: LexConfig, weights: dict[str, np.ndarray], seed: int = DEFAULT_SEED
) -> LexModel:
    """A model of shape `cfg` carrying `weights`."""
    return LexModel(cfg, seed=seed).load_weight_dict(weights)
