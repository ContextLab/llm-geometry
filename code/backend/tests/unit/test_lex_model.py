"""The Lexicon Lab's model, checked against the architecture contract itself.

No mocks anywhere: the forward pass is compared against an independent NumPy
transcription of `specs/006-lexicon-lab-tiny/architecture.md`, so a drift in either the
implementation or the contract shows up as a numeric disagreement rather than as a test
that quietly agrees with whatever the code does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from llm_geometry.errors import InvalidParamError
from llm_geometry.lex.config import (
    BOS_ID,
    GENERATION_BANNED_IDS,
    LAYER_NORM_EPS,
    PAD_ID,
    UNK_ID,
    param_count,
)
from llm_geometry.lex.generate import generate_ids
from llm_geometry.lex.model import INIT_STD, LexConfig, LexModel, model_from_weight_dict

#: (vocab_rows, d_model, n_layers, n_heads, ctx, tied). The first six are the
#: configurations `notes/agent-reports/006-source-model-arch.md` verified against the
#: source implementation; the last two are shipped defaults.
CONFIGS = [
    (320, 128, 4, 4, 128, False),
    (320, 128, 4, 4, 128, True),
    (320, 64, 2, 2, 64, False),
    (45, 16, 1, 1, 32, False),
    (45, 16, 1, 1, 32, True),
    (318, 64, 2, 2, 64, True),
    (318, 128, 4, 4, 128, False),
    (44, 32, 3, 2, 32, True),
]


def make(index: int = 0, **overrides) -> LexConfig:
    rows, d, layers, heads, ctx, tied = CONFIGS[index]
    base = {
        "vocab_rows": rows,
        "d_model": d,
        "n_layers": layers,
        "n_heads": heads,
        "ctx": ctx,
        "tied": tied,
    }
    return LexConfig(**{**base, **overrides})


# -- shape -----------------------------------------------------------------------------


@pytest.mark.parametrize("spec", CONFIGS)
def test_parameter_count_matches_the_closed_form(spec) -> None:
    rows, d, layers, heads, ctx, tied = spec
    cfg = LexConfig(vocab_rows=rows, d_model=d, n_layers=layers, n_heads=heads, ctx=ctx, tied=tied)
    model = LexModel(cfg)
    expected = param_count(vocab_rows=rows, d_model=d, n_layers=layers, ctx=ctx, tied=tied)
    assert (
        model.n_parameters() == expected == cfg.n_params
    ), f"{spec}: built {model.n_parameters()} parameters, formula says {expected}"


def test_the_two_known_source_configurations_are_reproduced_exactly() -> None:
    """Two absolute numbers measured from the source, not derived from our own formula."""
    assert LexModel(make(0)).n_parameters() == 891_648  # source defaults, |V|=320, untied
    assert LexModel(make(4)).n_parameters() == 4_544  # smallest sensible, tied


def test_indivisible_head_count_is_refused_with_a_clear_message() -> None:
    with pytest.raises(InvalidParamError) as exc:
        LexConfig(vocab_rows=44, d_model=64, n_heads=3)
    message = exc.value.message
    assert "d_model" in message and "n_heads" in message
    assert "64" in message and "3" in message


def test_config_validation_refuses_impossible_shapes() -> None:
    with pytest.raises(InvalidParamError):
        LexConfig(vocab_rows=4)  # the specials alone are not a vocabulary
    with pytest.raises(InvalidParamError):
        LexConfig(vocab_rows=44, d_model=0)
    with pytest.raises(InvalidParamError):
        LexConfig(vocab_rows=44, dropout=1.0)
    with pytest.raises(InvalidParamError):
        LexConfig.from_dict({"vocab_rows": 44, "d_mdoel": 16})
    with pytest.raises(InvalidParamError):
        LexConfig.from_dict({"d_model": 16})
    assert LexConfig.from_dict(make(3).as_dict()) == make(3)


# -- initialization --------------------------------------------------------------------


def test_the_mixed_init_is_reproduced_exactly() -> None:
    """qkv_w keeps PyTorch's xavier default; everything else is N(0, 0.02²)."""
    cfg = make(0)  # d=128, so the bound is sqrt(6/512) = 0.108253...
    model = LexModel(cfg, seed=0)
    weights = model.weight_dict()
    bound = math.sqrt(6.0 / (4 * cfg.d_model))
    assert bound == pytest.approx(0.1082532, abs=1e-6)

    qkv = weights["blocks.0.qkv_w"]
    assert qkv.min() >= -bound and qkv.max() <= bound
    # A uniform on (-b, b) has std b/sqrt(3) = 0.0625 — measurably NOT 0.02.
    assert float(qkv.std()) == pytest.approx(bound / math.sqrt(3.0), rel=0.03)
    assert abs(float(qkv.max()) - bound) < 0.002, "qkv_w does not fill the xavier bound"

    for name in ("embed", "pos", "blocks.0.proj_w", "blocks.0.fc1_w", "blocks.3.fc2_w"):
        assert float(weights[name].std()) == pytest.approx(INIT_STD, rel=0.05), name

    for name in ("blocks.0.ln1_g", "blocks.2.ln2_g", "lnf_g"):
        assert np.array_equal(weights[name], np.ones(cfg.d_model, dtype=np.float32))
    for name in ("blocks.0.ln1_b", "lnf_b", "blocks.1.qkv_b", "blocks.1.fc1_b"):
        assert not weights[name].any(), f"{name} should be zero-initialized"


def test_initialization_is_seeded_and_reproducible() -> None:
    a = LexModel(make(3), seed=7).weight_dict()
    b = LexModel(make(3), seed=7).weight_dict()
    c = LexModel(make(3), seed=8).weight_dict()
    for name, arr in a.items():
        assert np.array_equal(arr, b[name]), f"{name} differs across identical seeds"
    assert not np.array_equal(a["embed"], c["embed"])


# -- forward pass ----------------------------------------------------------------------


def _reference_forward(weights: dict[str, np.ndarray], cfg: LexConfig, ids: np.ndarray):
    """architecture.md's forward pass, transcribed independently in float64 NumPy."""

    def layer_norm(x, gain, bias):
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + LAYER_NORM_EPS) * gain + bias

    def gelu(x):  # exact erf form, via the standard normal CDF
        from scipy.special import erf

        return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))

    w = {k: v.astype(np.float64) for k, v in weights.items()}
    batch, seq = ids.shape
    d, heads, head_dim = cfg.d_model, cfg.n_heads, cfg.head_dim
    h = w["embed"][ids] + w["pos"][:seq][None, :, :]
    mask = np.triu(np.full((seq, seq), -np.inf), 1)

    for layer in range(cfg.n_layers):
        p = {k.split(".")[-1]: v for k, v in w.items() if k.startswith(f"blocks.{layer}.")}
        a = layer_norm(h, p["ln1_g"], p["ln1_b"])
        qkv = a @ p["qkv_w"].T + p["qkv_b"]
        q, k, v = qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]
        shape = (batch, seq, heads, head_dim)
        q = q.reshape(shape).transpose(0, 2, 1, 3)
        k = k.reshape(shape).transpose(0, 2, 1, 3)
        v = v.reshape(shape).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(head_dim) + mask
        scores = scores - scores.max(-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(-1, keepdims=True)
        o = (attn @ v).transpose(0, 2, 1, 3).reshape(batch, seq, d)
        h = h + (o @ p["proj_w"].T + p["proj_b"])

        m = layer_norm(h, p["ln2_g"], p["ln2_b"])
        m = gelu(m @ p["fc1_w"].T + p["fc1_b"])
        h = h + (m @ p["fc2_w"].T + p["fc2_b"])

    h = layer_norm(h, w["lnf_g"], w["lnf_b"])
    head = w["embed"] if cfg.tied else w["head_w"]
    return h @ head.T


@pytest.mark.parametrize("index", [3, 4, 5, 7])
def test_forward_matches_an_independent_transcription_of_the_contract(index: int) -> None:
    cfg = make(index)
    model = LexModel(cfg, seed=3).eval()
    rng = np.random.default_rng(index)
    ids = rng.integers(0, cfg.vocab_rows, size=(2, min(cfg.ctx, 11)))

    got = model(torch.from_numpy(ids)).detach().numpy().astype(np.float64)
    want = _reference_forward(model.weight_dict(), cfg, ids)
    assert got.shape == want.shape == (2, ids.shape[1], cfg.vocab_rows)
    assert np.abs(got - want).max() < 1e-5, f"max |Δ| = {np.abs(got - want).max():.2e}"


def test_gelu_is_the_exact_erf_form_not_the_tanh_approximation() -> None:
    """The two differ by ~1e-3 around |x| ≈ 2; the reference test above would catch it,
    but pin the distinction directly so the reason is legible."""
    x = torch.linspace(-4, 4, 81)
    exact = torch.nn.functional.gelu(x, approximate="none")
    tanh = torch.nn.functional.gelu(x, approximate="tanh")
    assert float((exact - tanh).abs().max()) > 1e-4
    from scipy.special import erf

    closed = 0.5 * x.numpy() * (1.0 + erf(x.numpy() / math.sqrt(2.0)))
    assert np.abs(exact.numpy() - closed).max() < 1e-6


def test_attention_is_causal() -> None:
    cfg = make(3)
    model = LexModel(cfg, seed=1).eval()
    ids = torch.randint(0, cfg.vocab_rows, (1, 9))
    base = model(ids).detach()
    changed = ids.clone()
    changed[0, 6] = (int(changed[0, 6]) + 1) % cfg.vocab_rows
    after = model(changed).detach()
    assert torch.allclose(base[:, :6], after[:, :6], atol=0)
    assert not torch.allclose(base[:, 6], after[:, 6])


def test_forward_refuses_out_of_range_input() -> None:
    cfg = make(3)
    model = LexModel(cfg)
    with pytest.raises(InvalidParamError):
        model(torch.zeros(1, cfg.ctx + 1, dtype=torch.long))
    with pytest.raises(InvalidParamError):
        model(torch.full((1, 3), cfg.vocab_rows, dtype=torch.long))
    with pytest.raises(InvalidParamError):
        model(torch.zeros(1, 2, 3, dtype=torch.long))


# -- tying -----------------------------------------------------------------------------


def test_tying_is_structural_and_cannot_be_lost_on_reload() -> None:
    """The source's probe.py dropped `tie` on reload and silently desynchronized."""
    tied = LexModel(make(5))
    untied = LexModel(make(2))
    assert "head_w" not in tied.weight_dict()
    assert "head_w" in untied.weight_dict()
    assert tied.readout is tied.embed
    assert untied.readout is untied.head_w
    assert not np.array_equal(untied.weight_dict()["head_w"], untied.weight_dict()["embed"])

    # A tied weight set cannot be loaded into an untied model, or the reverse.
    with pytest.raises(InvalidParamError) as exc:
        LexModel(make(2)).load_weight_dict(tied.weight_dict())
    assert "head_w" in exc.value.message


# -- save / load -----------------------------------------------------------------------


@pytest.mark.parametrize("index", [3, 5, 7])
def test_weight_dict_round_trips_to_exact_equality(index: int) -> None:
    cfg = make(index)
    trained = LexModel(cfg, seed=11)
    # Perturb so we are not just round-tripping a fresh init.
    with torch.no_grad():
        for param in trained.parameters():
            param.add_(torch.randn(param.shape) * 0.01)

    weights = trained.weight_dict()
    assert all(arr.dtype == np.float32 for arr in weights.values())
    reloaded = model_from_weight_dict(cfg, weights, seed=999)

    for name, arr in reloaded.weight_dict().items():
        assert np.array_equal(arr, weights[name]), f"{name} did not survive the round trip"

    ids = torch.randint(0, cfg.vocab_rows, (2, min(cfg.ctx, 8)))
    assert torch.equal(trained.eval()(ids), reloaded.eval()(ids))

    # Serializing through a list (what a JSON bundle does) is still exact for float32.
    as_json = {k: v.tolist() for k, v in weights.items()}
    from_json = model_from_weight_dict(
        cfg, {k: np.asarray(v, np.float32) for k, v in as_json.items()}
    )
    assert torch.equal(trained.eval()(ids), from_json.eval()(ids))


def test_bad_weight_dicts_are_refused_rather_than_partially_loaded() -> None:
    cfg = make(3)
    model = LexModel(cfg)
    good = model.weight_dict()

    with pytest.raises(InvalidParamError):
        LexModel(cfg).load_weight_dict({k: v for k, v in good.items() if k != "pos"})
    with pytest.raises(InvalidParamError):
        LexModel(cfg).load_weight_dict({**good, "surprise": np.zeros(3, np.float32)})
    with pytest.raises(InvalidParamError) as exc:
        LexModel(cfg).load_weight_dict({**good, "pos": np.zeros((3, 3), np.float32)})
    assert "shape" in exc.value.message
    with pytest.raises(InvalidParamError) as exc:
        LexModel(cfg).load_weight_dict({**good, "embed": np.full_like(good["embed"], np.nan)})
    assert "non-finite" in exc.value.message


# -- generation ------------------------------------------------------------------------


def test_generation_never_emits_a_banned_id() -> None:
    """FR-605: <unk>, <bos> and <pad> are masked, <eos> is not."""
    cfg = make(7)
    model = LexModel(cfg, seed=2)
    for seed in range(6):
        ids = generate_ids(model, [BOS_ID], max_new_tokens=60, temperature=1.5, seed=seed)
        assert len(ids) == 61
        for token_id in ids[1:]:
            assert token_id not in GENERATION_BANNED_IDS
            assert 0 <= token_id < cfg.vocab_rows
    assert set(GENERATION_BANNED_IDS) == {UNK_ID, BOS_ID, PAD_ID}


def test_temperature_zero_is_greedy_and_seed_independent() -> None:
    cfg = make(7)
    model = LexModel(cfg, seed=4)
    a = generate_ids(model, [BOS_ID], max_new_tokens=12, temperature=0.0, seed=0)
    b = generate_ids(model, [BOS_ID], max_new_tokens=12, temperature=0.0, seed=12345)
    assert a == b

    # It really is the argmax of the masked logits, step by step.
    mask = torch.zeros(cfg.vocab_rows)
    for banned in GENERATION_BANNED_IDS:
        mask[banned] = float("-inf")
    model.eval()
    with torch.no_grad():
        for step in range(1, len(a)):
            window = torch.tensor([a[max(0, step - cfg.ctx) : step]], dtype=torch.long)
            logits = model(window)[0, -1] + mask
            assert a[step] == int(torch.argmax(logits))


def test_sampling_is_seeded_and_reproducible() -> None:
    model = LexModel(make(7), seed=5)
    same = [
        generate_ids(model, [BOS_ID], max_new_tokens=20, temperature=1.0, seed=3) for _ in range(2)
    ]
    assert same[0] == same[1]
    other = generate_ids(model, [BOS_ID], max_new_tokens=20, temperature=1.0, seed=4)
    assert other != same[0]


def test_generation_rejects_impossible_requests() -> None:
    model = LexModel(make(7))
    with pytest.raises(InvalidParamError):
        generate_ids(model, [BOS_ID], max_new_tokens=0)
    with pytest.raises(InvalidParamError):
        generate_ids(model, [BOS_ID], max_new_tokens=10_000)
    with pytest.raises(InvalidParamError):
        generate_ids(model, [BOS_ID], temperature=-0.5)
    with pytest.raises(InvalidParamError):
        generate_ids(model, [99_999])
    with pytest.raises(InvalidParamError):
        generate_ids(model, [BOS_ID], banned_ids=list(range(model.cfg.vocab_rows)))
