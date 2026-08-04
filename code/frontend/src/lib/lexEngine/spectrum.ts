/**
 * Embedding geometry: the six steps of `architecture.md` § Spectrum, in order.
 *
 * For a matrix `A` of shape (V, d) — the embedding, or the readout when untied:
 *
 *   1. column-mean-centre:  Ac = A - mean(A, axis=0)
 *   2. Gram:                G  = Ac^T Ac, shape (d, d); a SYMMETRIC eigendecomposition
 *                           gives lambda_i = sigma_i^2 directly
 *   3. clamp, sort descending, sigma_i = sqrt(lambda_i)
 *   4. with p_i = lambda_i / sum(lambda):
 *        effective_rank = exp(-sum p_i ln p_i)      (p_i = 0 contributes 0)
 *        stable_rank    = sum(lambda) / lambda_1
 *        participation  = 1 / sum(p_i^2)
 *        frac_var_top2, frac_var_top10, n_dims_for_90pct
 *   5. report the ceiling min(V-1, d) alongside — centring is why the ALGEBRAIC rank is
 *      bounded by V-1, and that bound is climbed even by random matrices (FR-622), so it
 *      is drawn rather than left implicit
 *   6. PCA coordinates Ac @ E[:, :3] with the explained-variance ratio of each component,
 *      eigenvector signs canonicalized. These are a PROJECTION and must be labelled as
 *      one (FR-623).
 *
 * **Effective rank is an entropy, not a count.** `min(|V|-1, d)` bounds it, but it
 * reaches that bound only when the spectrum is perfectly flat. Measured on untrained
 * models at d=128, the ALGEBRAIC rank hits the 128 ceiling exactly from the `first`
 * budget onward while the effective rank is still only ~104 at |V| = 314 and climbing
 * with decelerating increments. It does not plateau at the ceiling, and nothing in the
 * UI may say it does.
 *
 * **No SVD.** The d x d Gram route is what makes this ~2 ms in a browser, and it is also
 * what sidesteps the source project's crash (`torch.linalg.svdvals` has no MPS kernel).
 * d <= 128 here, so a cyclic Jacobi rotation sweep is both fast and simple, and unlike a
 * QR-iteration it needs no tridiagonalisation to get eigenvectors out.
 */

import { invalidParam } from "../geoEngine/errors";
import { sfc32 } from "./model";

/** Components kept for the PCA token cloud. */
export const PCA_COMPONENTS = 3;
/** Bars drawn in the spectrum plot (the leading singular values). */
export const SPECTRUM_DISPLAY_K = 48;

export interface Eigen {
  /** Eigenvalues, descending. */
  values: Float64Array;
  /** Eigenvectors as COLUMNS of a row-major (n, n) matrix, matching `values`. */
  vectors: Float64Array;
  /** Jacobi sweeps actually performed (diagnostic; convergence is typically 6-10). */
  sweeps: number;
}

/**
 * Cyclic Jacobi eigendecomposition of a symmetric (n, n) matrix.
 *
 * Each sweep annihilates every off-diagonal entry once with a Givens rotation chosen to
 * zero it; the off-diagonal Frobenius norm falls quadratically, so ~10 sweeps reach
 * double-precision accuracy. Eigenvalues come out on the diagonal and eigenvectors as
 * the accumulated rotation's columns. The input is not modified.
 */
export function jacobiEigen(matrix: ArrayLike<number>, n: number, maxSweeps = 100): Eigen {
  if (!Number.isInteger(n) || n < 1) throw invalidParam(`n must be a positive integer, got ${n}`);
  if (matrix.length !== n * n) throw invalidParam(`matrix has ${matrix.length} entries, expected ${n * n}`);

  const a = Float64Array.from(matrix);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;

  // Scale-free stopping rule: stop once the off-diagonal mass is at the rounding floor.
  let frob = 0;
  for (let i = 0; i < a.length; i++) frob += a[i] * a[i];
  const floor = Math.sqrt(frob) * 1e-18;

  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    }
    if (Math.sqrt(2 * off) <= floor) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (apq === 0) continue;
        const app = a[p * n + p];
        const aqq = a[q * n + q];
        // t = tan(theta), the numerically stable root of t^2 + 2*theta*t - 1 = 0.
        const theta = (aqq - app) / (2 * apq);
        const t =
          theta >= 0
            ? 1 / (theta + Math.sqrt(theta * theta + 1))
            : -1 / (-theta + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0;
        a[q * n + p] = 0;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          const np = c * akp - s * akq;
          const nq = s * akp + c * akq;
          a[k * n + p] = np;
          a[p * n + k] = np;
          a[k * n + q] = nq;
          a[q * n + k] = nq;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p];
          const vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  // Sort descending, permuting the eigenvector columns with the eigenvalues.
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j * n + j] - a[i * n + i]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const src = order[j];
    values[j] = a[src * n + src];
    for (let i = 0; i < n; i++) vectors[i * n + j] = v[i * n + src];
  }
  canonicalizeSigns(vectors, n);
  return { values, vectors, sweeps };
}

/**
 * Fix each eigenvector's sign, in place: an eigenvector is only defined up to sign, so
 * two correct implementations can return MIRROR-IMAGE point clouds while every scalar
 * statistic agrees. The convention (architecture.md § Spectrum step 6, and the Python
 * side): if a column's largest-magnitude entry is negative, negate the whole column.
 * Ties on magnitude go to the first such entry by index.
 */
export function canonicalizeSigns(vectors: Float64Array, n: number): void {
  for (let j = 0; j < n; j++) {
    let best = 0;
    let bestAbs = -1;
    for (let i = 0; i < n; i++) {
      const m = Math.abs(vectors[i * n + j]);
      if (m > bestAbs) {
        bestAbs = m;
        best = i;
      }
    }
    if (vectors[best * n + j] < 0) {
      for (let i = 0; i < n; i++) vectors[i * n + j] = -vectors[i * n + j];
    }
  }
}

// --- the spectrum --------------------------------------------------------------------

/** Step 1: subtract each column's mean. Returns a fresh (rows, cols) matrix. */
export function centreColumns(A: ArrayLike<number>, rows: number, cols: number): Float64Array {
  if (A.length !== rows * cols) throw invalidParam(`matrix has ${A.length} entries, expected ${rows * cols}`);
  const out = new Float64Array(rows * cols);
  const mean = new Float64Array(cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) mean[c] += A[r * cols + c];
  for (let c = 0; c < cols; c++) mean[c] /= rows;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = A[r * cols + c] - mean[c];
  return out;
}

/** Step 2: G = Ac^T Ac, shape (cols, cols), accumulated symmetrically. */
export function gram(Ac: Float64Array, rows: number, cols: number): Float64Array {
  const G = new Float64Array(cols * cols);
  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let s = 0;
      for (let r = 0; r < rows; r++) s += Ac[r * cols + i] * Ac[r * cols + j];
      G[i * cols + j] = s;
      G[j * cols + i] = s;
    }
  }
  return G;
}

export interface SpectrumStats {
  /** lambda_i, descending, negatives clamped to 0 (they are float error). */
  eigenvalues: number[];
  /**
   * sigma_i = sqrt(lambda_i). The Gram route returns the null space as dust of order
   * sqrt(eps) * sigma_max (~1e-9 relative) rather than exact zeros; that is expected and
   * moves no statistic below, so nothing here asserts an exact zero.
   */
  singularValues: number[];
  totalVariance: number;
  /**
   * exp(-sum p ln p) — an ENTROPY of the spectrum, not a count of nonzero directions. It
   * equals `ceiling` only for a perfectly flat spectrum, and in practice sits well below
   * it even when the algebraic rank is saturated.
   */
  effectiveRank: number;
  stableRank: number;
  participationRatio: number;
  fracVarTop2: number;
  fracVarTop10: number;
  nDimsFor90pct: number;
  /** min(rows - 1, cols) — the ceiling on the ALGEBRAIC rank after centring. */
  ceiling: number;
}

/** Steps 3 and 4, given raw eigenvalues of the Gram matrix. */
export function spectrumStats(eigenvalues: ArrayLike<number>, rows: number, cols: number): SpectrumStats {
  const lam: number[] = [];
  for (let i = 0; i < eigenvalues.length; i++) lam.push(Math.max(eigenvalues[i], 0));
  lam.sort((a, b) => b - a);

  const total = lam.reduce((s, x) => s + x, 0);
  const ceiling = Math.min(rows - 1, cols);
  const zero: SpectrumStats = {
    eigenvalues: lam,
    singularValues: lam.map(() => 0),
    totalVariance: 0,
    effectiveRank: 0,
    stableRank: 0,
    participationRatio: 0,
    fracVarTop2: 0,
    fracVarTop10: 0,
    nDimsFor90pct: 0,
    ceiling,
  };
  if (!(total > 0)) return zero;

  let entropy = 0;
  let sumP2 = 0;
  for (const l of lam) {
    const p = l / total;
    if (p > 0) entropy -= p * Math.log(p);
    sumP2 += p * p;
  }
  const head = (k: number): number => lam.slice(0, k).reduce((s, x) => s + x, 0) / total;
  let cum = 0;
  let nDims = lam.length;
  for (let i = 0; i < lam.length; i++) {
    cum += lam[i];
    if (cum >= 0.9 * total) {
      nDims = i + 1;
      break;
    }
  }

  return {
    eigenvalues: lam,
    singularValues: lam.map(Math.sqrt),
    totalVariance: total,
    effectiveRank: Math.exp(entropy),
    stableRank: total / lam[0],
    participationRatio: 1 / sumP2,
    fracVarTop2: head(2),
    fracVarTop10: head(10),
    nDimsFor90pct: nDims,
    ceiling,
  };
}

export interface SpectrumResult extends SpectrumStats {
  /** PCA coordinates, row-major (rows, components). A PROJECTION, not the geometry. */
  coords: Float64Array;
  components: number;
  /** lambda_i / sum(lambda) for each retained component. */
  explainedVarianceRatio: number[];
  sweeps: number;
}

/** All six steps for one matrix. `rows` is |V| + specials; `cols` is d_model. */
export function spectrum(
  A: ArrayLike<number>,
  rows: number,
  cols: number,
  components = PCA_COMPONENTS,
): SpectrumResult {
  if (rows < 2) throw invalidParam(`a spectrum needs at least 2 rows, got ${rows}`);
  const Ac = centreColumns(A, rows, cols);
  const G = gram(Ac, rows, cols);
  const { values, vectors, sweeps } = jacobiEigen(G, cols);
  const stats = spectrumStats(values, rows, cols);

  const k = Math.min(components, cols);
  const coords = new Float64Array(rows * k);
  for (let r = 0; r < rows; r++) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let c = 0; c < cols; c++) s += Ac[r * cols + c] * vectors[c * cols + j];
      coords[r * k + j] = s;
    }
  }
  const explainedVarianceRatio =
    stats.totalVariance > 0 ? stats.eigenvalues.slice(0, k).map((l) => l / stats.totalVariance) : new Array(k).fill(0);

  return { ...stats, coords, components: k, explainedVarianceRatio, sweeps };
}

/**
 * The untrained random-init baseline the panel draws beside the trained spectrum
 * (FR-622): a matrix of the same shape whose rows are drawn from the embedding's own
 * initializer, N(0, 0.02^2). Any rank the trained model has that this one does not is
 * the part learning contributed.
 */
export function randomBaselineSpectrum(rows: number, cols: number, seed = 0, std = 0.02): SpectrumResult {
  const rand = sfc32(seed);
  const A = new Float64Array(rows * cols);
  for (let i = 0; i < A.length; i += 2) {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1)) * std;
    A[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < A.length) A[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return spectrum(A, rows, cols);
}
