#!/usr/bin/env python
"""Emit the Lexicon Lab golden vectors (spec 006, SC-605).

The TypeScript browser engine (``code/frontend/src/lib/lexEngine/``) and the PyTorch
backend (``code/backend/src/llm_geometry/lex/``) both implement
``specs/006-lexicon-lab-tiny/architecture.md``. This script runs the **real PyTorch
side** — no mocks, no re-derivation of the maths here — and writes what it measured to
``code/frontend/tests/fixtures/lex-golden.json``. ``tests/unit/lexGolden.test.ts`` then
runs the real TypeScript side against that file and asserts agreement to <= 1e-5.

Follows the pattern feature 003 established for the Geometry Lab
(``scripts/export_static_assets.py`` -> ``tests/fixtures/geo/golden.json`` ->
``tests/unit/geoEngine.test.ts``): a Python script emits golden vectors, a vitest file
loads them and asserts.

Usage (from the backend venv):

    python scripts/export_lex_golden.py

    python scripts/export_lex_golden.py --out /tmp/lex-golden.json

What each case covers, per ``specs/006-lexicon-lab-tiny/architecture.md``:

1. **weights** — an explicit, fully specified weight set, so the two engines start from
   IDENTICAL numbers. RNG parity between torch and the browser's sfc32 is explicitly NOT
   claimed by architecture.md ("non-portable RNG streams"), so relying on seeded init to
   line up would test nothing. Real ``LexModel`` init supplies the draw; a seeded
   perturbation then makes every bias and LayerNorm gain non-trivial, so a bug in one of
   them cannot hide behind a zero.
2. **forward** — logits for a fixed token sequence, `(B*T, V)`, at `T < ctx` and with
   distinct rows in the batch.
3. **loss** — mean cross-entropy for a fixed (input, target) pair, including a `<pad>`
   target so ``ignore_index`` is exercised.
4. **spectrum** — every statistic of the Spectrum section, for the embedding and (when
   untied) the readout.
5. **optimizer_step** — the real backward-pass gradients for that pair, their global
   norm, and the weights after ONE AdamW step at a known LR. This is what keeps the
   golden test from covering inference only.
6. **schedule** — the one-cycle LR at several step indices, including step counts where
   ``round(0.3*S)`` lands on a .5 boundary (Python rounds half-to-even; ``pyRound`` in
   train.ts must match).

MEASURED FINDING, recorded here because the golden test acts on it. AdamW's FIRST step
is scale-invariant: with ``t = 1``, ``m̂ = g`` and ``sqrt(v̂) = |g|``, so the update is
``-lr·g/(|g| + 1e-8)`` — which is ``-lr·sign(g)`` for any ``|g|`` well above eps,
*independent of the gradient's magnitude*. A gradient entry whose true value is at or
below the float32 noise floor therefore has its step decided by roundoff rather than by
the gradient, and two correct implementations at different precisions will disagree there
by up to a full ``lr``. That is not a porting defect and cannot be fixed by either side;
``lex-golden.json`` therefore also carries the step's gradients, so the test can say
exactly which post-step entries are well-conditioned (compared to 1e-5) and which are
noise-determined (compared to the bound a single step can move them).

The largest such family is structural: ``qkv_b[d:2d]``, the **key bias**, has an
*exactly zero* gradient in exact arithmetic. Adding a constant ``b_k`` to every key
shifts ``scores_ij = q_i·k_j`` by ``q_i·b_k``, which is constant along ``j``, and softmax
is invariant to a constant shift along the axis it normalizes. Both backends compute
"zero" there — float64 gives ~1e-18, torch's float32 gives ~1e-9 — and AdamW amplifies
the difference to ~1e-4.

Numeric encoding: weight and gradient tensors are base64 little-endian **float32** —
which is exactly what torch holds them as, so the fixture is bit-exact and about 3x
smaller than JSON numbers would be. Everything else is plain JSON numbers.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "backend" / "src"))

from llm_geometry.lex.config import (  # noqa: E402
    GRAD_CLIP_NORM,
    PAD_ID,
)
from llm_geometry.lex.model import LexConfig, LexModel, model_from_weight_dict  # noqa: E402
from llm_geometry.lex.spectrum import spectrum  # noqa: E402
from llm_geometry.lex.train import (  # noqa: E402
    batch_loss,
    decay_param_groups,
    onecycle_lr,
)

DEFAULT_OUT = REPO_ROOT / "code" / "frontend" / "tests" / "fixtures" / "lex-golden.json"

FORMAT = "lex-golden-v1"
#: The tolerance SC-605 sets, recorded in the file so the test cannot silently drift.
TOLERANCE = 1e-5


# --- case definitions --------------------------------------------------------------------

#: Three configurations, varying d_model, n_layers, n_heads and tied/untied together so
#: no single code path (head tying, multi-head splitting, layer stacking) is untested.
#:
#: ``forward_batch`` is a short, ragged batch (``T < ctx``, distinct rows) used only for
#: the logits golden. The loss, the gradients and the optimizer step all run on the
#: ``stream``'s single training window instead — see `step_batch`.
CASES: list[dict[str, Any]] = [
    {
        "name": "A-tied-d16-L1-H1",
        "config": dict(vocab_rows=24, d_model=16, n_layers=1, n_heads=1, ctx=32, tied=True),
        "seed": 7,
        "forward_batch": {"B": 2, "T": 6, "ids": [5, 9, 21, 4, 17, 2, 11, 11, 6, 23, 0, 14]},
        # ctx + 1 = 33 training tokens is what floor(0.95 * 35) gives, which pins the
        # only sampleable window to offset 0 in BOTH engines (see `step_batch`).
        "stream_len": 35,
        "batch_size": 2,
        "lr": 2e-3,
        "weight_decay": 0.05,
    },
    {
        "name": "B-untied-d32-L2-H2",
        "config": dict(vocab_rows=30, d_model=32, n_layers=2, n_heads=2, ctx=32, tied=False),
        "seed": 11,
        "forward_batch": {"B": 2, "T": 5, "ids": [4, 12, 29, 7, 18, 3, 21, 21, 9, 16]},
        "stream_len": 35,
        "batch_size": 2,
        "lr": 3e-3,
        "weight_decay": 0.01,
    },
    {
        "name": "C-tied-d16-L3-H4",
        "config": dict(vocab_rows=28, d_model=16, n_layers=3, n_heads=4, ctx=16, tied=True),
        "seed": 23,
        "forward_batch": {"B": 3, "T": 4, "ids": [2, 19, 8, 27, 13, 13, 5, 1, 22, 10, 26, 6]},
        # floor(0.95 * 18) = 17 = ctx + 1.
        "stream_len": 18,
        "batch_size": 3,
        "lr": 5e-3,
        "weight_decay": 0.02,
    },
]

#: (total_steps, peak_lr, [step indices]) for the one-cycle schedule golden. S = 5 and
#: S = 15 put `0.3 * S` exactly on a .5 boundary, where Python's round() goes
#: half-to-EVEN (2 and 4 respectively) — the case `pyRound` in train.ts exists for.
SCHEDULE_CASES: list[tuple[int, float, list[int]]] = [
    (400, 3e-3, [0, 1, 2, 60, 119, 120, 121, 200, 398, 399]),
    (1, 1e-3, [0]),
    (5, 5e-2, [0, 1, 2, 3, 4]),
    (15, 1e-2, [0, 3, 4, 5, 9, 14]),
    (7, 1e-1, [0, 1, 2, 3, 6]),
    (1000, 1e-4, [0, 299, 300, 999]),
]


# --- encoding ----------------------------------------------------------------------------


def b64_f32(array: np.ndarray) -> str:
    """Base64 of the array's little-endian float32 bytes — lossless for torch weights."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def encode_weights(weights: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: b64_f32(arr) for name, arr in weights.items()}


def jsonable(value: Any) -> Any:
    """Plain-JSON floats/ints, recursively — numpy scalars are not JSON-serializable."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


# --- the weight set ----------------------------------------------------------------------


def fixed_weights(cfg: LexConfig, seed: int) -> dict[str, np.ndarray]:
    """A fully specified, non-trivial weight set for `cfg`.

    Real ``LexModel`` init draws the matrices (so the shapes and scales are the ones the
    architecture table specifies), then a seeded perturbation nudges EVERY tensor —
    biases and LayerNorm gains included. Leaving those at their initial 0/1 would let a
    bug in bias or gain handling pass unnoticed on both sides.
    """
    model = LexModel(cfg, seed=seed)
    weights = model.weight_dict()
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, arr in weights.items():
        noise = rng.normal(0.0, 0.05, size=arr.shape).astype(np.float32)
        out[name] = (arr + noise).astype(np.float32)
    return out


def token_stream(cfg: LexConfig, length: int, seed: int) -> list[int]:
    """A deterministic token stream, guaranteed to contain a `<pad>` so it is scored."""
    rng = np.random.default_rng(seed + 1000)
    stream = rng.integers(0, cfg.vocab_rows, size=length).tolist()
    stream[cfg.ctx // 2] = PAD_ID
    return [int(v) for v in stream]


# --- the measurements --------------------------------------------------------------------


def forward_case(model: LexModel, batch: dict[str, Any]) -> dict[str, Any]:
    """Logits `(B*T, V)` for a fixed short sequence (`T < ctx`), in eval mode."""
    ids = torch.tensor(batch["ids"], dtype=torch.int64).view(batch["B"], batch["T"])
    if batch["T"] >= model.cfg.ctx:
        raise SystemExit("the forward golden should use T < ctx, so short sequences are covered")
    model.eval()
    with torch.no_grad():
        logits = model(ids)
    flat = logits.reshape(-1, model.cfg.vocab_rows).numpy()
    return {
        "B": batch["B"],
        "T": batch["T"],
        "ids": list(batch["ids"]),
        "logits": [[float(v) for v in row] for row in flat],
    }


def step_batch(cfg: LexConfig, stream: list[int], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The batch BOTH engines see, pinned without either side's RNG mattering.

    `train_lex` and `runTraining` both draw start offsets from their own (deliberately
    non-portable) RNG, so the only way to have the two see the same window is for there
    to be exactly one: a training split of `ctx + 1` tokens has a single valid start, 0.
    `stream_len` is chosen so `floor(0.95 * len) == ctx + 1` — which is what both engines'
    95/5 split computes — and the TypeScript side then reaches this same batch through its
    real `runTraining`.
    """
    n_train = int(0.95 * len(stream))
    if n_train != cfg.ctx + 1:
        raise SystemExit(
            f"stream of {len(stream)} tokens splits to {n_train} training tokens, but the "
            f"single-window batch needs exactly ctx + 1 = {cfg.ctx + 1}"
        )
    window = torch.tensor(stream[: cfg.ctx + 1], dtype=torch.int64)
    inputs = window[:-1].repeat(batch_size, 1).contiguous()
    targets = window[1:].repeat(batch_size, 1).contiguous()
    if int((targets == PAD_ID).sum()) == 0:
        raise SystemExit("the training window has no <pad> target, so ignore_index is untested")
    return inputs, targets


def loss_case(model: LexModel, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """Mean cross-entropy for the pinned (input, target) pair, `ignore_index = <pad>`."""
    model.eval()
    with torch.no_grad():
        value = float(batch_loss(model, inputs, targets))
    return {
        "B": int(inputs.shape[0]),
        "T": int(inputs.shape[1]),
        "ids": [int(v) for v in inputs.reshape(-1)],
        "targets": [int(v) for v in targets.reshape(-1)],
        "value": value,
        "n_valid": int((targets != PAD_ID).sum()),
    }


def optimizer_step_case(
    cfg: LexConfig,
    weights: dict[str, np.ndarray],
    stream: list[int],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Real gradients, their global norm, and one AdamW step — as `train_lex` takes one.

    Same functions the training route uses: `decay_param_groups` for the decay split,
    `onecycle_lr` for the LR, `torch.nn.utils.clip_grad_norm_` at `GRAD_CLIP_NORM`, then
    `torch.optim.AdamW(betas=(0.9, 0.999), eps=1e-8).step()`.

    The gradients are emitted alongside the post-step weights because they are what tells
    the golden test which post-step entries are well-conditioned: see the module docstring
    on AdamW's scale invariance at `t = 1`.
    """
    model = model_from_weight_dict(cfg, weights)
    groups = decay_param_groups(model, weight_decay)
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    step_lr = onecycle_lr(0, 1, lr)
    for group in opt.param_groups:
        group["lr"] = step_lr

    model.train()  # dropout is 0, so this only sets the flag
    loss = batch_loss(model, inputs, targets)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    params = dict(model.named_parameters())
    grads: dict[str, np.ndarray] = {}
    total_sq = 0.0
    for name in model.weight_names():
        grad = params[name].grad
        if grad is None:
            raise SystemExit(f"{name} received no gradient — the backward pass is incomplete")
        arr = grad.detach().cpu().numpy().astype(np.float32)
        grads[name] = arr
        total_sq += float(np.sum(np.float64(arr) ** 2))

    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM))
    opt.step()

    return {
        "stream": list(stream),
        "n_train": int(0.95 * len(stream)),
        "batch_size": int(inputs.shape[0]),
        "steps": 1,
        "peak_lr": lr,
        "scheduled_lr": step_lr,
        "weight_decay": weight_decay,
        "decayed": list(groups[0]["names"]),
        "loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "grad_norm_recomputed": float(np.sqrt(total_sq)),
        "grads": encode_weights(grads),
        "weights_after": encode_weights(model.weight_dict()),
    }


def spectrum_case(model: LexModel) -> dict[str, Any]:
    """Every Spectrum-section statistic, for each matrix the model actually has."""
    out: dict[str, Any] = {}
    matrices = {"embedding": model.embed}
    if not model.cfg.tied:
        matrices["readout"] = model.head_w
    for label, param in matrices.items():
        matrix = param.detach().cpu().numpy().astype(np.float64)
        out[label] = jsonable(spectrum(matrix).as_dict())
    return out


# --- assembly ----------------------------------------------------------------------------


def build_case(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = LexConfig(**spec["config"])
    weights = fixed_weights(cfg, spec["seed"])
    model = model_from_weight_dict(cfg, weights)
    stream = token_stream(cfg, spec["stream_len"], spec["seed"])
    inputs, targets = step_batch(cfg, stream, spec["batch_size"])
    return {
        "name": spec["name"],
        "config": cfg.as_dict(),
        "param_count": cfg.n_params,
        "n_parameters_measured": model.n_parameters(),
        "weight_names": model.weight_names(),
        "weights": encode_weights(weights),
        "forward": forward_case(model, spec["forward_batch"]),
        "loss": loss_case(model, inputs, targets),
        "spectrum": spectrum_case(model),
        "optimizer_step": optimizer_step_case(
            cfg,
            weights,
            stream,
            inputs,
            targets,
            lr=spec["lr"],
            weight_decay=spec["weight_decay"],
        ),
    }


def build_schedule() -> list[dict[str, Any]]:
    return [
        {
            "total_steps": steps,
            "peak_lr": peak,
            "points": [{"step": i, "lr": onecycle_lr(i, steps, peak)} for i in indices],
        }
        for steps, peak, indices in SCHEDULE_CASES
    ]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:  # pragma: no cover - dev only
        raise SystemExit(f"cannot read the git sha for provenance: {err}") from err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    torch.manual_seed(0)
    document = {
        "format": FORMAT,
        "generated": args.generated,
        "git_sha": git_sha(),
        "command": "python scripts/export_lex_golden.py",
        "source": (
            "llm_geometry.lex.{model,train,spectrum} run directly — real PyTorch, no mocks"
        ),
        "contract": "specs/006-lexicon-lab-tiny/architecture.md",
        "tolerance": TOLERANCE,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "encoding": (
            "weights/gradients are base64 little-endian float32 (torch's own dtype, so "
            "bit-exact); every other number is plain JSON. Python names blocks.<i>.<w>; "
            "the TypeScript engine names the same tensor layers.<i>.<w>."
        ),
        "schedule": jsonable(build_schedule()),
        "cases": [build_case(spec) for spec in CASES],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB, {len(document['cases'])} cases)")


if __name__ == "__main__":
    main()
