"""Embedding geometry for the Lexicon Lab (FR-620..FR-623).

This module implements *exactly* the six steps of the "Spectrum" section of
`specs/006-lexicon-lab-tiny/architecture.md`, which is the contract the TypeScript
browser engine also implements (SC-605 holds the two to <= 1e-5):

1. **Column-mean-centre** ``Ac = A - mean(A, axis=0)``. Centring is why the maximum
   attainable rank is ``min(V-1, d)``: subtracting the column mean forces every row of
   ``Ac`` into the hyperplane orthogonal to the all-ones vector.
2. **Gram** ``G = Ac.T @ Ac``, shape ``(d, d)``, symmetric eigendecomposition. The
   eigenvalues of ``G`` *are* the squared singular values of ``Ac``, so there is no SVD
   anywhere in this file. Two reasons, both real: a ``d x d`` symmetric eigensolve is
   ~2 ms in a browser on the shapes this tab uses, and the source project crashes on
   this exact computation because ``torch.linalg.svdvals`` has no MPS kernel.
3. **Clamp** ``lambda_i <- max(lambda_i, 0)`` (negatives are float error only), sort
   descending, ``sigma_i = sqrt(lambda_i)``.
4. With ``p_i = lambda_i / sum(lambda)``: effective rank ``exp(-sum p ln p)``, stable
   rank ``sum(lambda)/lambda_1``, participation ratio ``1/sum(p^2)``, ``frac_var_top2``,
   ``frac_var_top10``, ``n_dims_for_90pct``.
5. The **ceiling** ``min(V-1, d)`` is reported alongside, never implied.
6. **PCA coordinates** ``Ac @ E[:, :3]`` for the top-3 eigenvectors of ``G``, with each
   component's explained-variance ratio. These are a *projection* and are labelled as
   such wherever they are drawn — unlike the Geometry Lab's sphere, which is native 3-D.

`random_baseline_spectrum` is the reason FR-622 exists: effective rank rises with ``|V|``
*for random matrices too*, purely because a taller matrix has more independent rows to
fill the ``min(V-1, d)`` ceiling with. Drawing an untrained model's spectrum next to a
trained one makes that mechanical confound visible instead of letting it masquerade as
learning (SC-604).

**Sign convention (an addition, documented because architecture.md is silent).**
Eigenvector signs are arbitrary — ``numpy.linalg.eigh`` and a browser eigensolver may
disagree on them, which would break the SC-605 golden test on `pca_coords` while every
statistic matched. We therefore fix each eigenvector's sign so its largest-magnitude
entry is positive (first index wins a tie). This changes nothing geometric: it is a
reflection of a coordinate axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..errors import ComputeError, InvalidParamError
from .config import DEFAULT_SEED, PCA_COMPONENTS

__all__ = [
    "Spectrum",
    "spectrum",
    "random_baseline_spectrum",
    "untrained_matrix",
    "model_matrix",
    "spectra_for_model",
    "compare_to_baseline",
    "effective_rank_of",
]


@dataclass(frozen=True)
class Spectrum:
    """Every statistic FR-620/FR-621/FR-622/FR-623 asks for, measured from one matrix.

    ``eigenvalues``/``singular_values``/``explained_variance`` are length ``d`` and sorted
    descending; entries beyond ``max_rank`` are exactly ``0.0`` up to float error.
    """

    rows: int
    """``V`` — the number of embedding rows the matrix has."""

    d_model: int
    """``d`` — the number of columns."""

    max_rank: int
    """``min(V-1, d)``: the ceiling centring imposes. FR-622 draws this."""

    eigenvalues: tuple[float, ...]
    singular_values: tuple[float, ...]
    explained_variance: tuple[float, ...]
    """``p_i = lambda_i / sum(lambda)``, so this sums to 1 (or is all-zero)."""

    total_variance: float
    """``sum(lambda_i)``, which is ``||Ac||_F^2``."""

    effective_rank: float
    stable_rank: float
    participation_ratio: float
    frac_var_top2: float
    frac_var_top10: float
    n_dims_for_90pct: int

    pca_coords: tuple[tuple[float, ...], ...]
    """``(V, 3)`` projection onto the leading eigenvectors. A PROJECTION (FR-623)."""

    pca_explained_variance_ratio: tuple[float, ...]

    degenerate: bool
    """True when the centred matrix is exactly zero (``V == 1``, or identical rows).

    Every ratio is then undefined, so the rank statistics are reported as ``0.0`` and
    this flag says why rather than letting a caller read ``0.0`` as a measurement.
    """

    def as_dict(self) -> dict[str, Any]:
        """Plain-JSON form; the routes round it to 6 significant digits."""
        return {
            "rows": self.rows,
            "d_model": self.d_model,
            "max_rank": self.max_rank,
            "eigenvalues": list(self.eigenvalues),
            "singular_values": list(self.singular_values),
            "explained_variance": list(self.explained_variance),
            "total_variance": self.total_variance,
            "effective_rank": self.effective_rank,
            "stable_rank": self.stable_rank,
            "participation_ratio": self.participation_ratio,
            "frac_var_top2": self.frac_var_top2,
            "frac_var_top10": self.frac_var_top10,
            "n_dims_for_90pct": self.n_dims_for_90pct,
            "pca_coords": [list(row) for row in self.pca_coords],
            "pca_explained_variance_ratio": list(self.pca_explained_variance_ratio),
            "degenerate": self.degenerate,
        }

    def summary(self) -> dict[str, Any]:
        """The scalar statistics only — what a baseline comparison needs."""
        return {
            "rows": self.rows,
            "d_model": self.d_model,
            "max_rank": self.max_rank,
            "effective_rank": self.effective_rank,
            "stable_rank": self.stable_rank,
            "participation_ratio": self.participation_ratio,
            "frac_var_top2": self.frac_var_top2,
            "frac_var_top10": self.frac_var_top10,
            "n_dims_for_90pct": self.n_dims_for_90pct,
            "total_variance": self.total_variance,
            "degenerate": self.degenerate,
        }


def spectrum(matrix: Any, *, n_components: int = PCA_COMPONENTS) -> Spectrum:
    """The spectrum of a ``(V, d)`` matrix, by the six steps above.

    ``matrix`` is anything ``numpy.asarray`` accepts with a 2-D float shape — a numpy
    array, a nested list, or a detached torch tensor. Computation is in float64
    regardless of the input dtype: the Gram matrix squares the condition number, and
    float32 there would cost about half the significant digits of every small eigenvalue.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2:
        raise InvalidParamError(f"spectrum needs a 2-D (V, d) matrix, got shape {a.shape}")
    if a.size == 0:
        raise InvalidParamError(f"spectrum needs a non-empty matrix, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ComputeError(
            "matrix contains non-finite values; refusing to report a spectrum of NaN/inf"
        )

    rows, d_model = a.shape
    max_rank = min(rows - 1, d_model)

    # 1. column-mean-centre
    centred = a - a.mean(axis=0, keepdims=True)

    # 2. d x d Gram + symmetric eigendecomposition (NO SVD)
    gram = centred.T @ centred
    # eigh returns ascending eigenvalues; reverse for descending.
    eigenvalues_asc, eigenvectors_asc = np.linalg.eigh(gram)
    eigenvalues = eigenvalues_asc[::-1].copy()
    eigenvectors = eigenvectors_asc[:, ::-1].copy()

    # 3. clamp float-error negatives, then sqrt
    np.maximum(eigenvalues, 0.0, out=eigenvalues)
    singular_values = np.sqrt(eigenvalues)

    total = float(eigenvalues.sum())
    n_keep = min(n_components, d_model)
    coords = _pca_coords(centred, eigenvectors, n_keep, n_components)

    if total <= 0.0:
        # Exactly-zero centred matrix: p_i is 0/0. Report zeros and flag it (see
        # `degenerate`) rather than inventing a rank for a matrix that has none.
        zeros = tuple(0.0 for _ in range(d_model))
        return Spectrum(
            rows=rows,
            d_model=d_model,
            max_rank=max_rank,
            eigenvalues=zeros,
            singular_values=zeros,
            explained_variance=zeros,
            total_variance=0.0,
            effective_rank=0.0,
            stable_rank=0.0,
            participation_ratio=0.0,
            frac_var_top2=0.0,
            frac_var_top10=0.0,
            n_dims_for_90pct=0,
            pca_coords=coords,
            pca_explained_variance_ratio=tuple(0.0 for _ in range(n_components)),
            degenerate=True,
        )

    # 4. the statistics
    p = eigenvalues / total
    positive = p[p > 0.0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    stable_rank = float(total / eigenvalues[0])
    participation_ratio = float(1.0 / np.sum(p * p))
    cumulative = np.cumsum(p)
    n_dims_for_90pct = int(np.searchsorted(cumulative, 0.9) + 1)
    n_dims_for_90pct = min(n_dims_for_90pct, d_model)

    ratios = [float(p[i]) if i < d_model else 0.0 for i in range(n_components)]

    return Spectrum(
        rows=rows,
        d_model=d_model,
        max_rank=max_rank,
        eigenvalues=tuple(float(v) for v in eigenvalues),
        singular_values=tuple(float(v) for v in singular_values),
        explained_variance=tuple(float(v) for v in p),
        total_variance=total,
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        participation_ratio=participation_ratio,
        frac_var_top2=float(p[:2].sum()),
        frac_var_top10=float(p[:10].sum()),
        n_dims_for_90pct=n_dims_for_90pct,
        pca_coords=coords,
        pca_explained_variance_ratio=tuple(ratios),
        degenerate=False,
    )


def _pca_coords(
    centred: np.ndarray,
    eigenvectors: np.ndarray,
    n_keep: int,
    n_components: int,
) -> tuple[tuple[float, ...], ...]:
    """``Ac @ E[:, :k]`` with the documented sign convention, right-padded with zeros.

    Padding only happens if ``d < n_components``, which the shipped ``D_MODEL_CHOICES``
    never trigger; it exists so the response shape is ``(V, n_components)`` unconditionally.
    """
    rows = centred.shape[0]
    basis = eigenvectors[:, :n_keep]
    if n_keep:
        # Sign convention: largest-magnitude entry of each eigenvector is positive.
        pivot = np.argmax(np.abs(basis), axis=0)
        signs = np.sign(basis[pivot, np.arange(n_keep)])
        signs[signs == 0.0] = 1.0
        basis = basis * signs
    projected = centred @ basis if n_keep else np.zeros((rows, 0))
    if n_keep < n_components:
        projected = np.hstack([projected, np.zeros((rows, n_components - n_keep))])
    return tuple(tuple(float(v) for v in row) for row in projected)


# -- the untrained baseline (FR-622 / SC-604) ---------------------------------------------


def untrained_matrix(
    config: Any, seed: int = DEFAULT_SEED, *, which: str = "embedding"
) -> np.ndarray:
    """The ``(V, d)`` matrix of a **freshly initialized, untrained** `LexModel`.

    The baseline must be the *same* initialization the trained model started from, so
    this builds a real model (``llm_geometry.lex.model.LexModel(config, seed=seed)``)
    rather than re-deriving ``N(0, 0.02²)`` here. A second copy of the init would be a
    second thing to keep in sync, and the first time the two drifted the baseline would
    quietly stop being a baseline.

    ``which`` is ``"embedding"`` (``model.embed``) or ``"readout"`` (``model.head_w``).
    A **tied** model has no separate readout and raises rather than returning the
    embedding under a second name — the source project's `--tie` bug was exactly that,
    logging one matrix as two spectra and inviting a comparison of a thing with itself.
    """
    from .model import LexModel  # local import: torch is not needed for the maths above

    return model_matrix(LexModel(config, seed=seed), which)


def model_matrix(model: Any, which: str = "embedding") -> np.ndarray:
    """Pull the ``(V, d)`` embedding or readout out of a `LexModel`, as float64.

    ``which="readout"`` on a **tied** model raises: a tied model has exactly one matrix
    and reporting it twice would dress a thing up as a comparison with itself.
    """
    if which == "embedding":
        param = model.embed
    elif which == "readout":
        if getattr(model.cfg, "tied", False):
            raise InvalidParamError(
                "this model is tied: its readout IS its embedding, so it has exactly one "
                "spectrum. Reporting it twice would dress one matrix up as a comparison."
            )
        param = model.head_w
    else:
        raise InvalidParamError(f'which must be "embedding" or "readout", got {which!r}')
    array = np.asarray(param.detach().cpu().numpy(), dtype=np.float64)
    if array.ndim != 2:
        raise ComputeError(f"{which} parameter has shape {array.shape}, expected (V, d)")
    return array


def random_baseline_spectrum(
    config: Any,
    seed: int = DEFAULT_SEED,
    *,
    which: str = "embedding",
    n_components: int = PCA_COMPONENTS,
) -> Spectrum:
    """The spectrum of an **untrained** model at the same shape (FR-622, SC-604).

    A trained model's effective rank means nothing on its own: a random ``(V, d)``
    Gaussian matrix already climbs toward ``min(V-1, d)`` as ``V`` grows, so a rank
    "staircase" against the vocabulary budget is *expected without any learning*. This
    is the control the panel draws behind the trained curve.
    """
    return spectrum(untrained_matrix(config, seed, which=which), n_components=n_components)


def compare_to_baseline(trained: Spectrum, baseline: Spectrum) -> dict[str, Any]:
    """The trained-vs-untrained deltas the spectrum panel states in words.

    ``effective_rank_delta`` is the honest headline number: trained minus random at the
    same shape. It can be negative — training often *concentrates* variance — and the
    panel says so rather than only celebrating increases.
    """
    if (trained.rows, trained.d_model) != (baseline.rows, baseline.d_model):
        raise InvalidParamError(
            f"baseline shape {(baseline.rows, baseline.d_model)} does not match "
            f"trained shape {(trained.rows, trained.d_model)}; a baseline at a "
            "different shape would not be a control"
        )
    return {
        "effective_rank_delta": trained.effective_rank - baseline.effective_rank,
        "stable_rank_delta": trained.stable_rank - baseline.stable_rank,
        "participation_ratio_delta": trained.participation_ratio - baseline.participation_ratio,
        "frac_var_top2_delta": trained.frac_var_top2 - baseline.frac_var_top2,
        "max_rank": trained.max_rank,
        "effective_rank_frac_of_ceiling": (
            trained.effective_rank / trained.max_rank if trained.max_rank else 0.0
        ),
        "baseline_effective_rank_frac_of_ceiling": (
            baseline.effective_rank / baseline.max_rank if baseline.max_rank else 0.0
        ),
    }


def spectra_for_model(
    model: Any, *, tied: bool, n_components: int = PCA_COMPONENTS
) -> dict[str, Spectrum]:
    """Every spectrum a model has: ``{"embedding": ...}``, plus ``"readout"`` when untied.

    A **tied** model reports exactly one spectrum, labelled tied. The source project logs
    E and U as two spectra for a tied model, which is the same matrix twice dressed up as
    a comparison; not repeating that is a stated requirement of this feature.
    """
    out = {"embedding": spectrum(model_matrix(model, "embedding"), n_components=n_components)}
    if not tied:
        out["readout"] = spectrum(model_matrix(model, "readout"), n_components=n_components)
    return out


def effective_rank_of(eigenvalues: Sequence[float]) -> float:
    """``exp(-sum p ln p)`` for an already-computed eigenvalue list (golden-test helper)."""
    lam = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(lam.sum())
    if total <= 0.0:
        return 0.0
    p = lam[lam > 0.0] / total
    return float(np.exp(-np.sum(p * np.log(p))))
