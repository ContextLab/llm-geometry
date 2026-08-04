"""Real-arithmetic tests for the Lexicon Lab spectrum (FR-620..FR-623, SC-604).

Nothing here is mocked: every number is computed from a real matrix, and the Gram
optimization is checked against a direct SVD of the same matrix rather than against a
remembered constant. That is the point — architecture.md replaces the source project's
`torch.linalg.svdvals` with a `d x d` eigendecomposition for speed and for MPS
compatibility, and an optimization is only allowed if it is provably the same maths.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_geometry.errors import ComputeError, InvalidParamError
from llm_geometry.lex.config import D_MODEL_CHOICES, PCA_COMPONENTS
from llm_geometry.lex.dolch import dolch_sizes
from llm_geometry.lex.spectrum import (
    compare_to_baseline,
    effective_rank_of,
    random_baseline_spectrum,
    spectrum,
)

TOL = 1e-9


# -- reference implementation ---------------------------------------------------------


def _svd_reference(matrix: np.ndarray) -> dict[str, float | np.ndarray]:
    """The same statistics computed the slow, obvious way: a direct SVD of `Ac`.

    This is the oracle for `test_gram_agrees_with_svd`. It is deliberately written
    without reference to `spectrum()`'s internals.
    """
    a = np.asarray(matrix, dtype=np.float64)
    centred = a - a.mean(axis=0)
    sigma = np.linalg.svd(centred, compute_uv=False)
    # svd returns min(V, d) values; the spectrum reports d of them, zero-padded.
    padded = np.zeros(a.shape[1])
    padded[: sigma.size] = sigma
    lam = padded**2
    total = lam.sum()
    p = lam / total
    nz = p[p > 0]
    cumulative = np.cumsum(p)
    return {
        "singular_values": padded,
        "eigenvalues": lam,
        "explained_variance": p,
        "total_variance": float(total),
        "effective_rank": float(np.exp(-np.sum(nz * np.log(nz)))),
        "stable_rank": float(total / lam.max()),
        "participation_ratio": float(1.0 / np.sum(p**2)),
        "frac_var_top2": float(np.sort(p)[::-1][:2].sum()),
        "frac_var_top10": float(np.sort(p)[::-1][:10].sum()),
        "n_dims_for_90pct": int(np.searchsorted(cumulative, 0.9) + 1),
    }


# -- the two exactly-known spectra ----------------------------------------------------


@pytest.mark.parametrize("d", [3, 16, 64, 128])
def test_effective_rank_of_orthonormal_columns_equals_d(d: int) -> None:
    """A matrix with orthonormal, mean-zero columns has a flat spectrum: `eff rank == d`.

    Centring a random matrix puts every column in the hyperplane orthogonal to `1`, and
    QR preserves that span, so `Q` is both orthonormal and already column-mean-zero.
    Its Gram is the identity, every `p_i = 1/d`, and `exp(-sum p ln p) = d` exactly.
    """
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(4 * d, d))
    q, _ = np.linalg.qr(raw - raw.mean(axis=0))
    assert np.allclose(q.mean(axis=0), 0.0, atol=1e-12), "QR should preserve mean-zero columns"

    s = spectrum(q)
    assert abs(s.effective_rank - d) < TOL
    assert abs(s.stable_rank - d) < TOL
    assert abs(s.participation_ratio - d) < TOL
    assert abs(s.frac_var_top2 - 2.0 / d) < TOL
    assert abs(s.n_dims_for_90pct - int(np.ceil(0.9 * d))) <= 1


def test_effective_rank_of_rank_one_matrix_is_one() -> None:
    """`A = u v^T` has one nonzero eigenvalue, so every rank measure is exactly 1."""
    rng = np.random.default_rng(3)
    u = rng.normal(size=(200, 1))
    v = rng.normal(size=(1, 64))
    s = spectrum(u @ v)

    assert abs(s.effective_rank - 1.0) < TOL
    assert abs(s.stable_rank - 1.0) < TOL
    assert abs(s.participation_ratio - 1.0) < TOL
    assert abs(s.frac_var_top2 - 1.0) < TOL
    assert s.n_dims_for_90pct == 1
    # Only the leading eigenvalue survives; the rest are float dust below it.
    lam = np.array(s.eigenvalues)
    assert lam[0] > 0
    assert np.all(lam[1:] < lam[0] * 1e-12)


def test_a_constant_matrix_is_degenerate_not_rank_one() -> None:
    """Identical rows centre to exactly zero; the flag says so rather than reporting 0."""
    s = spectrum(np.tile(np.arange(32.0), (50, 1)))
    assert s.degenerate is True
    assert s.effective_rank == 0.0
    assert s.total_variance == 0.0
    assert len(s.pca_coords) == 50


# -- the Gram optimization is the same maths as an SVD --------------------------------


@pytest.mark.parametrize("shape", [(44, 128), (96, 64), (137, 128), (318, 128), (60, 16), (5, 32)])
def test_gram_agrees_with_svd(shape: tuple[int, int]) -> None:
    """`sqrt(eig(Ac^T Ac))` matches `svdvals(Ac)`, and so does every statistic, to 1e-9.

    One documented caveat, checked rather than waved at: forming the Gram matrix squares
    the condition number, so singular values inside the numerical *null* space (indices
    at or past `min(V-1, d)`) come back as dust of order `sqrt(eps) * sigma_max` instead
    of exact zeros. That is bounded below, it is the entire cost of the optimization, and
    it moves no reported statistic — every one of them still agrees to 1e-9.
    """
    rng = np.random.default_rng(sum(shape))
    matrix = rng.normal(scale=0.02, size=shape)
    s = spectrum(matrix)
    ref = _svd_reference(matrix)

    k = s.max_rank
    assert np.allclose(s.singular_values[:k], ref["singular_values"][:k], rtol=0, atol=TOL)
    null_space_dust = np.array(s.singular_values[k:])
    if null_space_dust.size:
        bound = 2.0 * np.sqrt(np.finfo(np.float64).eps) * ref["singular_values"][0]
        assert null_space_dust.max() <= bound, (null_space_dust.max(), bound)
    assert np.allclose(s.eigenvalues, ref["eigenvalues"], rtol=0, atol=TOL)
    assert np.allclose(s.explained_variance, ref["explained_variance"], rtol=0, atol=TOL)
    for name in (
        "effective_rank",
        "stable_rank",
        "participation_ratio",
        "frac_var_top2",
        "frac_var_top10",
    ):
        assert abs(getattr(s, name) - ref[name]) < TOL, name
    assert s.n_dims_for_90pct == ref["n_dims_for_90pct"]
    assert abs(s.total_variance - ref["total_variance"]) < TOL


def test_gram_agrees_with_svd_on_a_low_rank_matrix() -> None:
    """A deliberately rank-deficient matrix: the Gram route must not invent rank."""
    rng = np.random.default_rng(7)
    factors = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 64))
    s = spectrum(factors)
    ref = _svd_reference(factors)

    assert np.allclose(s.singular_values[:5], ref["singular_values"][:5], rtol=0, atol=TOL)
    dust = np.array(s.singular_values[5:]).max()
    assert dust <= 2.0 * np.sqrt(np.finfo(np.float64).eps) * ref["singular_values"][0]
    assert abs(s.effective_rank - ref["effective_rank"]) < TOL
    assert s.effective_rank < 5.0 + TOL
    assert np.all(np.array(s.eigenvalues) >= 0.0), "clamping must remove float-error negatives"


def test_effective_rank_of_helper_matches_the_full_computation() -> None:
    rng = np.random.default_rng(21)
    matrix = rng.normal(size=(150, 32))
    s = spectrum(matrix)
    assert abs(effective_rank_of(s.eigenvalues) - s.effective_rank) < TOL


# -- the ceiling ----------------------------------------------------------------------


@pytest.mark.parametrize("budget", sorted(dolch_sizes().items(), key=lambda kv: kv[1]))
@pytest.mark.parametrize("d_model", D_MODEL_CHOICES)
def test_effective_rank_never_exceeds_the_ceiling(budget: tuple[str, int], d_model: int) -> None:
    """`eff rank <= min(V-1, d)` at every shipped budget size and every `d_model`.

    Also checks the *algebraic* rank: centring removes exactly one degree of freedom, so
    the number of nonzero eigenvalues is `min(V-1, d)` on the nose.
    """
    _, size = budget
    rows = size + 4  # the four specials always ride along (FR-603)
    rng = np.random.default_rng(size * 1000 + d_model)
    s = spectrum(rng.normal(scale=0.02, size=(rows, d_model)))

    ceiling = min(rows - 1, d_model)
    assert s.max_rank == ceiling
    assert s.effective_rank <= ceiling + TOL
    assert s.participation_ratio <= ceiling + TOL
    assert s.stable_rank <= ceiling + TOL

    lam = np.array(s.eigenvalues)
    assert int((lam > lam[0] * 1e-10).sum()) == ceiling


def test_rank_one_matrix_respects_the_ceiling_at_every_budget() -> None:
    """The lower extreme: a rank-1 matrix is 1 at every budget, not `min(V-1, d)`."""
    rng = np.random.default_rng(5)
    for size in dolch_sizes().values():
        rows = size + 4
        s = spectrum(rng.normal(size=(rows, 1)) @ rng.normal(size=(1, 128)))
        assert abs(s.effective_rank - 1.0) < TOL
        assert s.max_rank == min(rows - 1, 128)


# -- PCA coordinates (FR-623) ---------------------------------------------------------


def test_pca_coords_are_a_projection_with_the_reported_variance() -> None:
    """Coordinates are `Ac @ E[:, :3]`: orthogonal columns whose sums of squares are λ."""
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(300, 64))
    s = spectrum(matrix)
    coords = np.array(s.pca_coords)

    assert coords.shape == (300, PCA_COMPONENTS)
    # Column i has squared norm λ_i and is orthogonal to the others.
    gram = coords.T @ coords
    for i in range(PCA_COMPONENTS):
        assert abs(gram[i, i] - s.eigenvalues[i]) < 1e-8
        for j in range(PCA_COMPONENTS):
            if i != j:
                assert abs(gram[i, j]) < 1e-8
    assert np.allclose(coords.mean(axis=0), 0.0, atol=1e-9), "a centred matrix projects to 0 mean"

    ratios = s.pca_explained_variance_ratio
    assert len(ratios) == PCA_COMPONENTS
    assert all(ratios[i] >= ratios[i + 1] for i in range(PCA_COMPONENTS - 1))
    assert abs(sum(ratios) - s.frac_var_top2 - ratios[2]) < TOL


def test_pca_sign_convention_is_deterministic() -> None:
    """The documented sign fix makes coordinates reproducible across eigensolvers."""
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(120, 32))
    first = np.array(spectrum(matrix).pca_coords)
    # Permuting rows changes eigenvector signs in many solvers; the convention must not
    # depend on that, so the projected cloud is the same set of points either way.
    order = rng.permutation(120)
    second = np.array(spectrum(matrix[order]).pca_coords)
    assert np.allclose(
        np.sort(first, axis=0), np.sort(second[np.argsort(order)], axis=0), atol=1e-8
    )
    for column in range(PCA_COMPONENTS):
        assert abs(first[:, column]).max() > 0


# -- input validation -----------------------------------------------------------------


def test_non_finite_matrix_is_refused() -> None:
    matrix = np.ones((10, 4))
    matrix[3, 2] = np.nan
    with pytest.raises(ComputeError):
        spectrum(matrix)


@pytest.mark.parametrize("bad", [np.ones(5), np.ones((2, 3, 4)), np.zeros((0, 4))])
def test_bad_shapes_are_refused(bad: np.ndarray) -> None:
    with pytest.raises(InvalidParamError):
        spectrum(bad)


def test_baseline_comparison_refuses_a_different_shape() -> None:
    rng = np.random.default_rng(2)
    a = spectrum(rng.normal(size=(100, 32)))
    b = spectrum(rng.normal(size=(200, 32)))
    with pytest.raises(InvalidParamError):
        compare_to_baseline(a, b)


# -- SC-604: the mechanical confound, measured on real untrained models ----------------

#: Measured on 2026-08-03 across five seeds per budget at `d_model=128`, from freshly
#: initialized (untrained) models. Values are shape-determined — the seed-to-seed spread
#: is under 0.5 — so the band below is tight enough to catch a real regression and loose
#: enough to survive a different RNG stream.
SC604_EXPECTED_EFFECTIVE_RANK = {
    "pre_primer": 36.2,
    "primer": 65.1,
    "first": 79.5,
    "service": 95.9,
    "full": 104.5,
}
SC604_BAND = 1.5


def _lex_config(vocab_rows: int, d_model: int = 128):
    from llm_geometry.lex.model import LexConfig

    return LexConfig(
        vocab_rows=vocab_rows,
        d_model=d_model,
        n_layers=1,
        n_heads=1,
        ctx=32,
        tied=True,
        dropout=0.0,
    )


def test_sc604_random_init_rank_rises_with_budget_and_saturates_the_ceiling() -> None:
    """SC-604: a model that has learned NOTHING still climbs in effective rank.

    This is the confound FR-622 exists to expose: taller embedding matrices have more
    independent rows, so rank rises with `|V|` on its own. The panel draws this curve
    behind the trained one so a staircase is never mistaken for learning.
    """
    sizes = dolch_sizes()
    measured: dict[str, float] = {}
    ceilings: dict[str, int] = {}
    algebraic: dict[str, int] = {}

    for name, size in sizes.items():
        rows = size + 4
        s = random_baseline_spectrum(_lex_config(rows), seed=0)
        assert s.rows == rows and s.d_model == 128
        measured[name] = s.effective_rank
        ceilings[name] = s.max_rank
        lam = np.array(s.eigenvalues)
        algebraic[name] = int((lam > lam[0] * 1e-10).sum())

    order = [n for n, _ in sorted(sizes.items(), key=lambda kv: kv[1])]

    # (a) it rises, strictly, at every step from 40 to 314.
    values = [measured[n] for n in order]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1)), measured

    # (b) it never crosses the ceiling, and the ceiling itself plateaus at d=128.
    for name in order:
        assert measured[name] < ceilings[name]
    assert [ceilings[n] for n in order] == [43, 95, 128, 128, 128]

    # (c) the ALGEBRAIC rank is the ceiling exactly — that is what plateaus.
    assert [algebraic[n] for n in order] == [43, 95, 128, 128, 128]

    # (d) the climb decelerates as the ceiling binds: the last step is a fraction of
    #     the first, which is the visible shape of a bound taking hold.
    steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    assert steps[-1] < steps[0] / 2

    # (e) the actual numbers.
    for name, expected in SC604_EXPECTED_EFFECTIVE_RANK.items():
        assert abs(measured[name] - expected) < SC604_BAND, (name, measured[name], expected)


def test_sc604_baseline_is_seed_stable() -> None:
    """Different seeds must give the same story, or the baseline is not a baseline."""
    rows = dolch_sizes()["full"] + 4
    ranks = [random_baseline_spectrum(_lex_config(rows), seed=s).effective_rank for s in (0, 1, 2)]
    assert max(ranks) - min(ranks) < 1.0, ranks


def test_baseline_comparison_reports_signed_deltas() -> None:
    """`compare_to_baseline` must be able to say "training LOWERED effective rank"."""
    rows = dolch_sizes()["primer"] + 4
    baseline = random_baseline_spectrum(_lex_config(rows), seed=0)
    rng = np.random.default_rng(0)
    concentrated = spectrum(rng.normal(size=(rows, 1)) @ rng.normal(size=(1, 128)))
    delta = compare_to_baseline(concentrated, baseline)
    assert delta["effective_rank_delta"] < 0
    assert delta["max_rank"] == baseline.max_rank
    assert 0.0 < delta["baseline_effective_rank_frac_of_ceiling"] <= 1.0
