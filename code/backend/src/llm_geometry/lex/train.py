"""Training the Lexicon Lab's model — the recipe from `specs/006-.../architecture.md`.

The recipe is the source project's, with two **deliberate, documented** departures (both
marked `[DIFFERS]` in the architecture contract and stated in the UI):

1. **Weight decay applies only to the 2-D weight matrices** — `qkv_w`, `proj_w`, `fc1_w`,
   `fc2_w`, and `head_w` when untied. The source decays *every* parameter, including
   LayerNorm gains, biases, and both embedding tables; that is a known anti-pattern, and
   `decay_param_groups` makes the split inspectable rather than implicit.
2. **beta1 is fixed at 0.9.** The source's `OneCycleLR` also cycles beta1 between 0.95 and
   0.85; mirroring a second schedule across two languages buys nothing pedagogically.

Everything else is faithful: AdamW (0.9, 0.999, eps 1e-8), a one-cycle LR with `lr` as the
*peak* (so the first step runs at `lr/25`), global grad-norm clipping at 1.0, a contiguous
95/5 split of the token stream, and batches of start offsets drawn uniformly **with
replacement** — no epochs, no permutation, no document boundaries.

A run is seeded and reproducible on one platform. Bit-equality with the browser engine is
**not** claimed (platform BLAS, non-portable RNG streams) and the UI says so.

Non-finite loss raises `ComputeError` immediately. Training that has diverged must never
return a model and a number that look like a result.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
import torch

from ..errors import ComputeError, InvalidParamError
from .config import (
    DEFAULT_BATCH,
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WEIGHT_DECAY,
    EOS_ID,
    GRAD_CLIP_NORM,
    MAX_STEPS,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    PAD_ID,
    VAL_FRACTION,
)
from .model import LexConfig, LexModel
from .vocab import LexVocab, tokenize

#: The only parameters weight decay touches. Named explicitly rather than inferred from
#: `ndim == 2`, because `embed` and `pos` are 2-D too and must NOT be decayed.
DECAY_WEIGHTS = ("qkv_w", "proj_w", "fc1_w", "fc2_w", "head_w")

ProgressCb = Callable[["TrainProgress"], None]


@dataclass(frozen=True)
class TrainProgress:
    """One SSE tick. `lr` is the schedule's value for the step just taken."""

    step: int  # 1-based, so `step == total_steps` is the last tick
    total_steps: int
    loss: float
    lr: float
    elapsed_s: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "lr": self.lr,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class TrainResult:
    """A finished run: the model, its trajectory, and the numbers the UI quotes."""

    model: LexModel
    history: list[dict[str, float | int]] = field(default_factory=list)
    first_loss: float = 0.0
    final_loss: float = 0.0
    val_loss: float = 0.0
    steps: int = 0
    seed: int = 0
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "history": self.history,
            "first_loss": self.first_loss,
            "final_loss": self.final_loss,
            "val_loss": self.val_loss,
            "steps": self.steps,
            "seed": self.seed,
            "elapsed_s": self.elapsed_s,
            "config": self.model.cfg.as_dict(),
        }


# -- data ------------------------------------------------------------------------------


def token_stream(text: str, vocab: LexVocab) -> np.ndarray:
    """Encode a corpus as one id stream, with `<eos>` closing every non-blank line.

    Nursery rhymes are line-shaped: the line, not the paragraph, is the unit the model can
    plausibly learn to finish, so `<eos>` marks line ends. (The source emitted a separate
    `<nl>` token per line and `<eos>` per blank line; we have no `<nl>` row, and spending
    one of a 40-word budget's rows on one would be a real cost.)
    """
    ids: list[int] = []
    for line in text.splitlines():
        line_ids = vocab.encode(tokenize(line))
        if line_ids:
            ids.extend(line_ids)
            ids.append(EOS_ID)
    return np.asarray(ids, dtype=np.int64)


def split_stream(
    stream: np.ndarray, ctx: int, val_fraction: float = VAL_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous 95/5 train/val split — a prefix and a suffix, never shuffled.

    Both halves must hold at least one `ctx + 1` window, otherwise there is nothing to
    train or evaluate on and the caller is told exactly how short the corpus is.
    """
    stream = np.asarray(stream, dtype=np.int64)
    n_train = int((1.0 - val_fraction) * len(stream))
    train, val = stream[:n_train], stream[n_train:]
    span = ctx + 1
    if len(train) < span or len(val) < span:
        needed = int(math.ceil(span / min(1.0 - val_fraction, val_fraction)))
        raise InvalidParamError(
            f"corpus too short for ctx={ctx}: the {1 - val_fraction:.0%}/{val_fraction:.0%} "
            f"split gives {len(train)} training and {len(val)} validation tokens, and each "
            f"side needs at least {span}. Supply about {needed} tokens, or lower ctx."
        )
    return train, val


def sample_batch(
    split: np.ndarray, ctx: int, batch_size: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """`batch_size` windows of `ctx + 1` tokens, start offsets uniform WITH replacement.

    Returns `(inputs, targets)`, both `(B, ctx)` — the classic shift-by-one pair, so every
    one of the `ctx` positions contributes to the loss. Because attention is causal, this
    is exactly `logits[:, :-1]` against `window[:, 1:]` on the full `ctx + 1` window.
    """
    high = len(split) - ctx  # last valid start: a full ctx+1 window must fit
    if high < 1:
        raise InvalidParamError(
            f"split of {len(split)} tokens cannot hold a {ctx + 1}-token training window"
        )
    starts = torch.randint(0, high, (batch_size,), generator=gen)
    data = torch.from_numpy(np.asarray(split, dtype=np.int64))
    windows = torch.stack([data[s : s + ctx + 1] for s in starts.tolist()])
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def iter_eval_windows(split: np.ndarray, ctx: int, batch_size: int) -> Iterator[torch.Tensor]:
    """Non-overlapping `ctx + 1` windows, in order — a FIXED validation set.

    The source redrew ten random validation windows at every eval, so its `val` curve
    carried sampling noise and was not comparable point to point. This one is.
    """
    data = torch.from_numpy(np.asarray(split, dtype=np.int64))
    span = ctx + 1
    starts = list(range(0, len(data) - span + 1, ctx))
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        yield torch.stack([data[s : s + span] for s in chunk])


# -- the recipe ------------------------------------------------------------------------


def onecycle_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    pct_start: float = ONECYCLE_PCT_START,
    div_factor: float = ONECYCLE_DIV_FACTOR,
    final_div_factor: float = ONECYCLE_FINAL_DIV_FACTOR,
) -> float:
    """The one-cycle schedule, in closed form. `step` is 0-indexed; `peak_lr` is the peak.

    ``initial = peak/25``, ``final = initial/1e4``, cosine on both phases, warmup over
    ``w = round(pct_start · S)`` steps:

        i < w:  p = i/w,       lr = initial + (peak − initial)·(1 − cos(πp))/2
        i ≥ w:  p = (i−w)/(S−w), lr = final + (peak − final)·(1 + cos(πp))/2
    """
    if total_steps < 1:
        raise InvalidParamError(f"total_steps must be at least 1, got {total_steps}")
    if not 0 <= step < total_steps:
        raise InvalidParamError(f"step must be in 0..{total_steps - 1}, got {step}")
    if peak_lr <= 0:
        raise InvalidParamError(f"lr must be positive, got {peak_lr}")

    initial = peak_lr / div_factor
    final = initial / final_div_factor
    warmup = int(round(pct_start * total_steps))
    if step < warmup:
        p = step / warmup
        return initial + (peak_lr - initial) * (1.0 - math.cos(math.pi * p)) / 2.0
    p = (step - warmup) / (total_steps - warmup)
    return final + (peak_lr - final) * (1.0 + math.cos(math.pi * p)) / 2.0


def decay_param_groups(model: LexModel, weight_decay: float) -> list[dict[str, object]]:
    """Split parameters into the decayed matrices and everything else.

    Decayed: `qkv_w`, `proj_w`, `fc1_w`, `fc2_w`, and `head_w` when untied. NOT `embed`,
    NOT `pos`, NOT any bias, NOT LayerNorm gains — even though `embed` and `pos` are also
    2-D. Each group carries a `names` key so a test (and the UI) can read the split back.
    """
    decayed: list[tuple[str, torch.nn.Parameter]] = []
    plain: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if name.rsplit(".", 1)[-1] in DECAY_WEIGHTS:
            if param.ndim != 2:
                raise ComputeError(f"{name} is on the decay list but is not a matrix")
            decayed.append((name, param))
        else:
            plain.append((name, param))
    return [
        {
            "params": [p for _, p in decayed],
            "names": [n for n, _ in decayed],
            "weight_decay": float(weight_decay),
        },
        {
            "params": [p for _, p in plain],
            "names": [n for n, _ in plain],
            "weight_decay": 0.0,
        },
    ]


def batch_loss(model: LexModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over the block, `ignore_index = PAD_ID`."""
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.cfg.vocab_rows), targets.reshape(-1), ignore_index=PAD_ID
    )


def check_finite_loss(value: float, step: int) -> None:
    """Raise on a diverged run. Never clamp, never skip the step, never return a number."""
    if not math.isfinite(value):
        raise ComputeError(
            f"training diverged at step {step}: the loss is {value}. The learning rate is "
            "almost certainly too high for this configuration; lower it and re-run."
        )


@torch.no_grad()
def eval_loss(model: LexModel, split: np.ndarray, batch_size: int = 32) -> float:
    """Mean token cross-entropy (nats) over the fixed validation windows."""
    was_training = model.training
    model.eval()
    try:
        total, count = 0.0, 0
        for windows in iter_eval_windows(split, model.cfg.ctx, batch_size):
            targets = windows[:, 1:]
            n_tokens = int((targets != PAD_ID).sum())
            if n_tokens == 0:
                continue
            total += float(batch_loss(model, windows[:, :-1], targets)) * n_tokens
            count += n_tokens
    finally:
        model.train(was_training)
    if count == 0:
        raise ComputeError("eval_loss: the validation split holds no scoreable tokens")
    return total / count


@contextmanager
def _seeded_global_rng(seed: int):
    """Seed the global RNG (dropout masks read it) and restore it afterwards.

    Leaking a seeded global generator would make any later unseeded sampling depend on
    whether a training run happened first — the same trap the Geometry Lab documents.
    """
    previous = torch.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.set_rng_state(previous)


def train_lex(
    *,
    stream: np.ndarray | Sequence[int],
    config: LexConfig | None = None,
    model: LexModel | None = None,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    val_fraction: float = VAL_FRACTION,
    progress_cb: ProgressCb | None = None,
    progress_every: int = 1,
) -> TrainResult:
    """Train a `LexModel` on a token stream and return the model plus its trajectory.

    Pass `config` to train from scratch, or `model` to continue training an existing one —
    that second form is the fine-tune path (FR-619), and because the caller owns the model
    object, its vocabulary travels with it instead of being silently re-derived.

    `progress_cb` receives a `TrainProgress` every `progress_every` steps and always on the
    final step, which is what the SSE route streams. A caller that also wants generated
    samples mid-run constructs the model itself and samples from it inside the callback.
    """
    if (config is None) == (model is None):
        raise InvalidParamError("exactly one of config/model must be provided")
    steps = int(steps)
    if not 1 <= steps <= MAX_STEPS:
        raise InvalidParamError(f"steps must be in 1..{MAX_STEPS}, got {steps}")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise InvalidParamError(f"batch_size must be at least 1, got {batch_size}")
    if lr <= 0:
        raise InvalidParamError(f"lr must be positive, got {lr}")
    if weight_decay < 0:
        raise InvalidParamError(f"weight_decay must be non-negative, got {weight_decay}")
    if progress_every < 1:
        raise InvalidParamError(f"progress_every must be at least 1, got {progress_every}")

    stream = np.asarray(stream, dtype=np.int64)
    if model is None:
        assert config is not None
        model = LexModel(config, seed=seed)
    cfg = model.cfg
    if stream.size and (int(stream.max()) >= cfg.vocab_rows or int(stream.min()) < 0):
        raise InvalidParamError(
            f"token stream has ids outside 0..{cfg.vocab_rows - 1} — it was encoded with a "
            "different vocabulary than this model's"
        )
    train_split, val_split = split_stream(stream, cfg.ctx, val_fraction)

    groups = decay_param_groups(model, weight_decay)
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    result = TrainResult(model=model, steps=steps, seed=int(seed))
    started = time.perf_counter()
    with _seeded_global_rng(seed):
        batch_gen = torch.Generator().manual_seed(int(seed) + 1)
        model.train()
        for step in range(steps):
            step_lr = onecycle_lr(step, steps, lr)
            for group in opt.param_groups:
                group["lr"] = step_lr

            inputs, targets = sample_batch(train_split, cfg.ctx, batch_size, batch_gen)
            loss = batch_loss(model, inputs, targets)
            loss_value = float(loss.detach())
            check_finite_loss(loss_value, step + 1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            result.history.append({"step": step + 1, "loss": loss_value, "lr": step_lr})
            if step == 0:
                result.first_loss = loss_value
            result.final_loss = loss_value
            if progress_cb is not None and ((step + 1) % progress_every == 0 or step + 1 == steps):
                progress_cb(
                    TrainProgress(
                        step=step + 1,
                        total_steps=steps,
                        loss=loss_value,
                        lr=step_lr,
                        elapsed_s=time.perf_counter() - started,
                    )
                )
        model.eval()
        result.val_loss = eval_loss(model, val_split)
    result.elapsed_s = time.perf_counter() - started
    return result
