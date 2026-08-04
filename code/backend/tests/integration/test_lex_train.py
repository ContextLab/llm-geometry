"""Real training of the Lexicon Lab's model on the real shipped corpus.

Every run here trains an actual model on the committed public-domain text — no mocks, no
fixtures standing in for the corpus, no synthetic token streams except where the point is
to prove a guard fires. The runs are deliberately short (tens to a few hundred steps at
small `d`) so the suite stays quick while still measuring a genuine optimization.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from llm_geometry.errors import ComputeError, InvalidParamError
from llm_geometry.lex.config import (
    BOS_ID,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    EOS_ID,
    GENERATION_BANNED_IDS,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    UNK_ID,
)
from llm_geometry.lex.corpus import load_corpus_text
from llm_geometry.lex.dolch import DOLCH_ORDER
from llm_geometry.lex.generate import generate_ids, generate_text
from llm_geometry.lex.model import LexConfig, LexModel, model_from_weight_dict
from llm_geometry.lex.train import (
    DECAY_WEIGHTS,
    TrainProgress,
    check_finite_loss,
    decay_param_groups,
    onecycle_lr,
    split_stream,
    token_stream,
    train_lex,
)
from llm_geometry.lex.vocab import build_vocab, tokenize

#: Small enough to train several times in a test suite, real enough to learn something.
SMALL = dict(d_model=16, n_layers=1, n_heads=1, ctx=32)


@pytest.fixture(scope="module")
def corpus() -> str:
    return load_corpus_text()


@pytest.fixture(scope="module")
def full_vocab(corpus: str):
    return build_vocab("dolch", "full", corpus)


@pytest.fixture(scope="module")
def full_stream(corpus: str, full_vocab) -> np.ndarray:
    return token_stream(corpus, full_vocab)


# -- data preparation ------------------------------------------------------------------


def test_token_stream_is_a_real_encoding_of_the_real_corpus(full_stream, full_vocab) -> None:
    assert len(full_stream) > 15_000
    assert full_stream.dtype == np.int64
    assert full_stream.min() >= 0 and full_stream.max() < full_vocab.rows
    counts = np.bincount(full_stream, minlength=full_vocab.rows)
    # <eos> closes every non-blank line — exactly one per line, no more, no fewer.
    n_lines = sum(1 for line in load_corpus_text().splitlines() if tokenize(line))
    assert int(counts[EOS_ID]) == n_lines > 3_000
    assert full_stream[-1] == EOS_ID
    # It is also the commonest real id: no single budget word beats a line ending.
    assert int(np.argmax(counts[1:])) + 1 == EOS_ID
    # Out-of-budget words become <unk> rather than disappearing, and at the full Dolch
    # budget that is a third of the corpus — the measurable form of what a budget cannot
    # say (FR-604/FR-606), not something to hide.
    assert counts[UNK_ID] > counts[EOS_ID]
    assert int(counts[UNK_ID]) == sum(
        1 for word in tokenize(load_corpus_text()) if word not in set(full_vocab.words)
    )


def test_split_is_contiguous_ninety_five_five(full_stream) -> None:
    train, val = split_stream(full_stream, ctx=64)
    assert len(train) + len(val) == len(full_stream)
    assert len(val) == pytest.approx(0.05 * len(full_stream), rel=0.01)
    assert np.array_equal(train, full_stream[: len(train)])  # a prefix, never shuffled
    assert np.array_equal(val, full_stream[len(train) :])


def test_a_corpus_too_short_for_ctx_is_refused_with_the_numbers(full_vocab) -> None:
    with pytest.raises(InvalidParamError) as exc:
        split_stream(np.zeros(100, dtype=np.int64), ctx=64)
    message = exc.value.message
    assert "too short" in message and "ctx=64" in message and "65" in message


# -- the one-cycle schedule ------------------------------------------------------------


def test_onecycle_matches_the_closed_form_at_first_peak_and_last_step() -> None:
    steps, peak = 400, DEFAULT_LR
    initial = peak / ONECYCLE_DIV_FACTOR
    final = initial / ONECYCLE_FINAL_DIV_FACTOR
    warmup = int(round(ONECYCLE_PCT_START * steps))
    assert warmup == 120

    # First step: exactly lr/25, NOT lr. (The single most-missed detail of the recipe.)
    assert onecycle_lr(0, steps, peak) == pytest.approx(initial, rel=1e-12)
    assert initial == pytest.approx(1.2e-4, rel=1e-12)

    # Peak: exactly `lr`, at the end of warmup.
    assert onecycle_lr(warmup, steps, peak) == pytest.approx(peak, rel=1e-12)
    assert onecycle_lr(warmup, steps, peak) == max(
        onecycle_lr(i, steps, peak) for i in range(steps)
    )

    # Last step: the closed form's own value, just above `final`.
    p = (steps - 1 - warmup) / (steps - warmup)
    want_last = final + (peak - final) * (1.0 + math.cos(math.pi * p)) / 2.0
    assert onecycle_lr(steps - 1, steps, peak) == pytest.approx(want_last, rel=1e-12)
    assert final < onecycle_lr(steps - 1, steps, peak) < initial

    # Cosine both phases: strictly up to the peak, strictly down after it.
    warm = [onecycle_lr(i, steps, peak) for i in range(warmup + 1)]
    anneal = [onecycle_lr(i, steps, peak) for i in range(warmup, steps)]
    assert all(b > a for a, b in zip(warm, warm[1:]))
    assert all(b < a for a, b in zip(anneal, anneal[1:]))
    # Mid-warmup is the cosine midpoint, not the linear one.
    assert onecycle_lr(warmup // 2, steps, peak) == pytest.approx((initial + peak) / 2.0, rel=1e-9)


def test_onecycle_handles_degenerate_step_counts_and_refuses_nonsense() -> None:
    assert onecycle_lr(0, 1, 1e-3) == pytest.approx(1e-3)  # warmup rounds to 0 steps
    with pytest.raises(InvalidParamError):
        onecycle_lr(0, 0, 1e-3)
    with pytest.raises(InvalidParamError):
        onecycle_lr(5, 5, 1e-3)
    with pytest.raises(InvalidParamError):
        onecycle_lr(0, 10, 0.0)


def test_the_schedule_the_optimizer_actually_used_is_the_schedule(full_stream, full_vocab) -> None:
    ticks: list[TrainProgress] = []
    result = train_lex(
        stream=full_stream,
        config=LexConfig(vocab_rows=full_vocab.rows, **SMALL),
        steps=25,
        progress_cb=ticks.append,
    )
    assert [t.step for t in ticks] == list(range(1, 26))
    for tick in ticks:
        assert tick.lr == pytest.approx(onecycle_lr(tick.step - 1, 25, DEFAULT_LR))
        assert tick.total_steps == 25 and tick.elapsed_s >= 0.0
        assert math.isfinite(tick.loss)
    assert [h["lr"] for h in result.history] == [t.lr for t in ticks]


# -- weight decay ----------------------------------------------------------------------


def test_weight_decay_hits_exactly_the_intended_parameters(full_vocab) -> None:
    """FR-614: matrices only — not embed, not pos, not biases, not LayerNorm gains."""
    for tied in (True, False):
        model = LexModel(LexConfig(vocab_rows=full_vocab.rows, tied=tied, **SMALL))
        decayed, plain = decay_param_groups(model, DEFAULT_WEIGHT_DECAY)

        expected_decayed = ["blocks.0.qkv_w", "blocks.0.proj_w", "blocks.0.fc1_w", "blocks.0.fc2_w"]
        if not tied:
            expected_decayed.append("head_w")
        assert sorted(decayed["names"]) == sorted(expected_decayed)
        assert decayed["weight_decay"] == DEFAULT_WEIGHT_DECAY
        assert plain["weight_decay"] == 0.0

        # The exclusions are the point: state them by name, including the 2-D ones.
        assert "embed" in plain["names"] and "pos" in plain["names"]
        for name in plain["names"]:
            assert name.rsplit(".", 1)[-1] not in DECAY_WEIGHTS
        for gain in ("blocks.0.ln1_g", "blocks.0.ln2_g", "lnf_g"):
            assert gain in plain["names"]
        for bias in ("blocks.0.qkv_b", "blocks.0.proj_b", "blocks.0.fc1_b", "lnf_b"):
            assert bias in plain["names"]

        # Every parameter is in exactly one group, and nothing is lost.
        all_names = set(decayed["names"]) | set(plain["names"])
        assert all_names == set(n for n, _ in model.named_parameters())
        assert len(decayed["params"]) + len(plain["params"]) == len(list(model.parameters()))
        assert not set(decayed["names"]) & set(plain["names"])


def test_decay_actually_shrinks_only_the_decayed_matrices(full_stream, full_vocab) -> None:
    """A real AdamW step with zero gradients: decay is then the ONLY force acting.

    (Gradients must be present but zero — AdamW skips a parameter whose `.grad` is None,
    which would make every parameter look undecayed.)
    """
    cfg = LexConfig(vocab_rows=full_vocab.rows, tied=False, **SMALL)
    model = LexModel(cfg, seed=0)
    before = model.weight_dict()
    opt = torch.optim.AdamW(decay_param_groups(model, weight_decay=0.5), lr=0.1)
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    opt.step()
    after = model.weight_dict()

    # p ← p·(1 − lr·wd) = p·0.95 on the decayed matrices, exactly.
    for name in ("blocks.0.qkv_w", "blocks.0.proj_w", "blocks.0.fc1_w", "blocks.0.fc2_w", "head_w"):
        assert np.allclose(after[name], before[name] * 0.95, atol=1e-7), name
    # …and nothing at all anywhere else, including the other two 2-D matrices.
    for name in before:
        if name.rsplit(".", 1)[-1] not in DECAY_WEIGHTS:
            assert np.array_equal(after[name], before[name]), name


# -- a real run ------------------------------------------------------------------------


def test_a_real_short_run_on_the_real_corpus_reduces_loss(full_stream, full_vocab) -> None:
    cfg = LexConfig(vocab_rows=full_vocab.rows)  # the shipped defaults: d=64, L=2, H=2
    result = train_lex(stream=full_stream, config=cfg, steps=200, seed=0)

    uniform = math.log(full_vocab.rows)  # 5.76 nats: what an untrained model scores
    assert result.first_loss == pytest.approx(uniform, rel=0.05)
    assert len(result.history) == 200
    assert result.history[0]["step"] == 1 and result.history[-1]["step"] == 200

    early = float(np.mean([h["loss"] for h in result.history[:10]]))
    late = float(np.mean([h["loss"] for h in result.history[-10:]]))
    assert late < early - 1.5, f"loss barely moved: {early:.3f} → {late:.3f}"
    assert result.val_loss < uniform - 1.5
    assert result.model.n_parameters() == cfg.n_params
    assert result.elapsed_s > 0.0


def test_the_same_seed_gives_the_same_run(full_stream, full_vocab) -> None:
    cfg = LexConfig(vocab_rows=full_vocab.rows, **SMALL)
    runs = [train_lex(stream=full_stream, config=cfg, steps=30, seed=5) for _ in range(2)]
    assert [h["loss"] for h in runs[0].history] == [h["loss"] for h in runs[1].history]
    assert runs[0].val_loss == runs[1].val_loss
    for name, arr in runs[0].model.weight_dict().items():
        assert np.array_equal(arr, runs[1].model.weight_dict()[name]), name

    other = train_lex(stream=full_stream, config=cfg, steps=30, seed=6)
    assert [h["loss"] for h in other.history] != [h["loss"] for h in runs[0].history]


def test_a_diverged_run_raises_instead_of_returning_a_model(full_stream, full_vocab) -> None:
    """A learning rate seven orders of magnitude too high really does produce NaN here."""
    with pytest.raises(ComputeError) as exc:
        train_lex(
            stream=full_stream,
            config=LexConfig(vocab_rows=full_vocab.rows, **SMALL),
            steps=60,
            lr=1e4,
            batch_size=8,
        )
    assert "diverged" in exc.value.message and "learning rate" in exc.value.message

    # The guard itself, stated directly.
    check_finite_loss(3.2, step=1)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ComputeError):
            check_finite_loss(bad, step=7)


def test_training_refuses_incoherent_requests(full_stream, full_vocab) -> None:
    cfg = LexConfig(vocab_rows=full_vocab.rows, **SMALL)
    with pytest.raises(InvalidParamError):
        train_lex(stream=full_stream, steps=10)  # neither config nor model
    with pytest.raises(InvalidParamError):
        train_lex(stream=full_stream, config=cfg, model=LexModel(cfg), steps=10)
    with pytest.raises(InvalidParamError):
        train_lex(stream=full_stream, config=cfg, steps=0)
    with pytest.raises(InvalidParamError):
        train_lex(stream=full_stream, config=cfg, steps=10, lr=0.0)
    # A stream encoded with a wider vocabulary than the model has rows.
    with pytest.raises(InvalidParamError) as exc:
        train_lex(stream=np.full(5000, cfg.vocab_rows + 3, dtype=np.int64), config=cfg, steps=5)
    assert "different vocabulary" in exc.value.message


# -- fine-tuning (FR-619) --------------------------------------------------------------


def test_fine_tuning_continues_a_model_and_keeps_its_vocabulary(corpus: str, full_vocab) -> None:
    cfg = LexConfig(vocab_rows=full_vocab.rows, **SMALL)
    first_half = token_stream(corpus[: len(corpus) // 2], full_vocab)
    second_half = token_stream(corpus[len(corpus) // 2 :], full_vocab)

    base = train_lex(stream=first_half, config=cfg, steps=80, seed=0)
    before = base.model.weight_dict()

    tuned = train_lex(stream=second_half, model=base.model, steps=40, lr=1e-3, seed=1)
    assert tuned.model is base.model  # the caller's model, so its vocabulary travels
    assert tuned.model.cfg == cfg
    assert not np.array_equal(tuned.model.weight_dict()["embed"], before["embed"])
    # Fine-tuning starts from a trained model, so it does not restart at the uniform loss.
    assert tuned.first_loss < math.log(full_vocab.rows) - 0.5
    text = generate_text(tuned.model, full_vocab, "the little", temperature=0.8, seed=0)
    assert set(tokenize(text)) <= set(full_vocab.words)


# -- SC-602: in-budget by construction, at every budget --------------------------------


@pytest.mark.parametrize("budget", DOLCH_ORDER)
def test_generation_is_in_budget_at_every_dolch_budget(corpus: str, budget: str) -> None:
    """SC-602: zero out-of-budget words, verified programmatically at every budget size."""
    vocab = build_vocab("dolch", budget, corpus)
    stream = token_stream(corpus, vocab)
    cfg = LexConfig(vocab_rows=vocab.rows, **SMALL)
    result = train_lex(stream=stream, config=cfg, steps=60, batch_size=16, seed=0)
    in_budget = set(vocab.words)
    assert len(in_budget) == vocab.budget_size == vocab.rows - 4

    produced: set[str] = set()
    for seed in range(4):
        for temperature in (0.0, 0.9, 1.5):
            text = generate_text(
                result.model, vocab, "the", max_new_tokens=60, temperature=temperature, seed=seed
            )
            words = tokenize(text)
            offenders = [w for w in words if w not in in_budget]
            assert not offenders, f"{budget}: out-of-budget words {sorted(set(offenders))}"
            # The id-level guarantee, which holds even when the text is empty: a model
            # this small trained this briefly can lock greedy decoding onto <eos>, and
            # "only line breaks" is an honest thing for it to say.
            ids = generate_ids(
                result.model, [BOS_ID], max_new_tokens=60, temperature=temperature, seed=seed
            )
            assert len(ids) == 61
            assert not set(ids[1:]) & set(GENERATION_BANNED_IDS)
            assert all(0 <= i < vocab.rows for i in ids)
            if temperature > 0:
                assert words, f"{budget} produced nothing at T={temperature}"
                produced.update(words)
    assert produced <= in_budget
    # The model is really using the budget, not parroting one safe word.
    assert len(produced) > 5


# -- SC-607: save and reload reproduces generation exactly ------------------------------


def test_save_load_round_trip_reproduces_generation_exactly(full_stream, full_vocab) -> None:
    cfg = LexConfig(vocab_rows=full_vocab.rows, tied=False, **SMALL)
    result = train_lex(stream=full_stream, config=cfg, steps=80, seed=0)
    weights = result.model.weight_dict()

    # Through a JSON-shaped round trip, which is what a saved file is.
    as_lists = {name: arr.tolist() for name, arr in weights.items()}
    reloaded = model_from_weight_dict(
        LexConfig.from_dict(cfg.as_dict()),
        {name: np.asarray(values, dtype=np.float32) for name, values in as_lists.items()},
    )
    for name, arr in reloaded.weight_dict().items():
        assert np.array_equal(arr, weights[name]), name

    prompt_ids = full_vocab.encode(tokenize("little boy"))
    for temperature in (0.0, 0.9, 1.5):
        for seed in (0, 1):
            # Ids first, so the greedy path is covered even when it emits only <eos>.
            assert generate_ids(
                result.model, prompt_ids, temperature=temperature, seed=seed
            ) == generate_ids(reloaded, prompt_ids, temperature=temperature, seed=seed)
            before = generate_text(
                result.model, full_vocab, "little boy", temperature=temperature, seed=seed
            )
            after = generate_text(
                reloaded, full_vocab, "little boy", temperature=temperature, seed=seed
            )
            assert before == after
            if temperature > 0:
                assert before.strip()


def test_a_vocabulary_of_the_wrong_size_is_refused_not_silently_used(
    corpus: str, full_stream, full_vocab
) -> None:
    cfg = LexConfig(vocab_rows=full_vocab.rows, **SMALL)
    result = train_lex(stream=full_stream, config=cfg, steps=10, seed=0)
    other = build_vocab("dolch", "primer", corpus)
    with pytest.raises(InvalidParamError) as exc:
        generate_text(result.model, other, "the")
    assert str(other.rows) in exc.value.message and str(full_vocab.rows) in exc.value.message
