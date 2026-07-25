/**
 * Minimal dense-tensor helpers for the GeoTransformer port. No dependencies.
 *
 * Convention: matrices are row-major flat arrays (Float32Array for stored weights,
 * Float64Array for activations/accumulators). The backend computes in float32
 * (torch); we accumulate in float64, which stays well inside the port's <=1e-5
 * relative tolerance for d_model=3-sized reductions.
 */

export type Mat = Float32Array | Float64Array;

/** c = a(n,k) @ b(k,m) -> (n,m). */
export function matmul(a: Mat, b: Mat, n: number, k: number, m: number): Float64Array {
  const c = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const av = a[i * k + t];
      if (av === 0) continue;
      const rowB = t * m;
      const rowC = i * m;
      for (let j = 0; j < m; j++) c[rowC + j] += av * b[rowB + j];
    }
  }
  return c;
}

/** c = a(n,k) @ b(m,k)^T -> (n,m). (y = x @ W^T for row-major W of shape (m,k).) */
export function matmulNT(a: Mat, b: Mat, n: number, k: number, m: number): Float64Array {
  const c = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    const rowA = i * k;
    const rowC = i * m;
    for (let j = 0; j < m; j++) {
      const rowB = j * k;
      let s = 0;
      for (let t = 0; t < k; t++) s += a[rowA + t] * b[rowB + t];
      c[rowC + j] = s;
    }
  }
  return c;
}

/** c = a(n,k)^T @ b(n,m) -> (k,m). (Gradient accumulation: dW = x^T @ dy.) */
export function matmulTN(a: Mat, b: Mat, n: number, k: number, m: number): Float64Array {
  const c = new Float64Array(k * m);
  for (let i = 0; i < n; i++) {
    const rowA = i * k;
    const rowB = i * m;
    for (let t = 0; t < k; t++) {
      const av = a[rowA + t];
      if (av === 0) continue;
      const rowC = t * m;
      for (let j = 0; j < m; j++) c[rowC + j] += av * b[rowB + j];
    }
  }
  return c;
}

/** a += b (elementwise, in place). */
export function addInPlace(a: Float64Array, b: Mat): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

/** a += s * b (elementwise, in place). */
export function axpyInPlace(a: Float64Array, s: number, b: Mat): void {
  for (let i = 0; i < a.length; i++) a[i] += s * b[i];
}

export function dot(a: Mat, b: Mat, n: number, offA = 0, offB = 0): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[offA + i] * b[offB + i];
  return s;
}

/**
 * Causal row softmax in place over a (T,T) score matrix: row i attends to j <= i
 * (the backend adds an upper-triangular -inf mask before softmax); entries j > i
 * become exactly 0.
 */
export function softmaxRowsCausalInPlace(scores: Float64Array, T: number): void {
  for (let i = 0; i < T; i++) {
    const row = i * T;
    let max = -Infinity;
    for (let j = 0; j <= i; j++) if (scores[row + j] > max) max = scores[row + j];
    let sum = 0;
    for (let j = 0; j <= i; j++) {
      const e = Math.exp(scores[row + j] - max);
      scores[row + j] = e;
      sum += e;
    }
    for (let j = 0; j <= i; j++) scores[row + j] /= sum;
    for (let j = i + 1; j < T; j++) scores[row + j] = 0;
  }
}

/** Softmax of a vector, in place. */
export function softmaxInPlace(v: Float64Array): void {
  let max = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i] > max) max = v[i];
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    const e = Math.exp(v[i] - max);
    v[i] = e;
    sum += e;
  }
  for (let i = 0; i < v.length; i++) v[i] /= sum;
}

/** log(sum(exp(v))) computed stably. */
export function logSumExp(v: Mat, off = 0, n = v.length): number {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (v[off + i] > max) max = v[off + i];
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.exp(v[off + i] - max);
  return max + Math.log(sum);
}

/** Indices of v sorted by descending value (stable: ties keep the lower index). */
export function argsortDesc(v: Mat | number[]): number[] {
  const idx = Array.from({ length: v.length }, (_, i) => i);
  idx.sort((a, b) => {
    const d = (v[b] as number) - (v[a] as number);
    return d !== 0 ? d : a - b;
  });
  return idx;
}

/** Index of the first maximal entry (torch.argmax tie semantics). */
export function argmax(v: Mat, off = 0, n = v.length): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = v[off + i];
    if (x > bestVal) {
      bestVal = x;
      best = i;
    }
  }
  return best;
}

/** Flat row-major array -> nested number[][] (contract tensor encoding, unrounded). */
export function toNested2(data: Mat, rows: number, cols: number): number[][] {
  const out: number[][] = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row: number[] = new Array(cols);
    for (let j = 0; j < cols; j++) row[j] = data[i * cols + j];
    out[i] = row;
  }
  return out;
}

export function norm2(v: Mat, off = 0, n = v.length): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += v[off + i] * v[off + i];
  return Math.sqrt(s);
}

// ---------------------------------------------------------------------------
// erf / GELU (torch.nn.functional.gelu default = exact erf formulation)
// ---------------------------------------------------------------------------

// W. J. Cody's rational approximations (CALERF / netlib), |error| ~ 1e-16 relative
// on each branch — far below the float32 arithmetic the backend uses.
const ERF_A = [3.16112374387056560e0, 1.13864154151050156e2, 3.77485237685302021e2, 3.20937758913846947e3];
const ERF_A4 = 1.85777706184603153e-1;
const ERF_B = [2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3, 2.84423683343917062e3];
const ERF_C = [5.64188496988670089e-1, 8.88314979438837594e0, 6.61191906371416295e1, 2.98635138197400131e2,
  8.81952221241769090e2, 1.71204761263407058e3, 2.05107837782607147e3, 1.23033935479799725e3];
const ERF_C8 = 2.15311535474403846e-8;
const ERF_D = [1.57449261107098347e1, 1.17693950891312499e2, 5.37181101862009858e2, 1.62138957456669019e3,
  3.29079923573345963e3, 4.36261909014324716e3, 3.43936767414372164e3, 1.23033935480374942e3];
const ERF_P = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2,
  6.58749161529837803e-4];
const ERF_P5 = 1.63153871373020978e-2;
const ERF_Q = [2.56852019228982242e0, 1.87295284992346047e0, 5.27905102951428412e-1, 6.05183413124413191e-2,
  2.33520497626869185e-3];
const INV_SQRT_PI = 5.6418958354775628695e-1; // 1/sqrt(pi)

function erfcCore(x: number): number {
  // erfc(x) for x > 0.46875.
  if (x <= 4.0) {
    let num = ERF_C8 * x;
    let den = x;
    for (let i = 0; i < 7; i++) {
      num = (num + ERF_C[i]) * x;
      den = (den + ERF_D[i]) * x;
    }
    return (Math.exp(-x * x) * (num + ERF_C[7])) / (den + ERF_D[7]);
  }
  if (x >= 26.5) return 0;
  const z = 1 / (x * x);
  let num = ERF_P5 * z;
  let den = z;
  for (let i = 0; i < 4; i++) {
    num = (num + ERF_P[i]) * z;
    den = (den + ERF_Q[i]) * z;
  }
  let r = (z * (num + ERF_P[4])) / (den + ERF_Q[4]);
  r = (INV_SQRT_PI - r) / x;
  return Math.exp(-x * x) * r;
}

/** Double-precision error function. */
export function erf(x: number): number {
  const ax = Math.abs(x);
  if (ax <= 0.46875) {
    const z = x * x;
    let num = ERF_A4 * z;
    let den = z;
    for (let i = 0; i < 3; i++) {
      num = (num + ERF_A[i]) * z;
      den = (den + ERF_B[i]) * z;
    }
    return (x * (num + ERF_A[3])) / (den + ERF_B[3]);
  }
  const c = erfcCore(ax);
  return x > 0 ? 1 - c : c - 1;
}

const SQRT1_2 = 0.7071067811865476; // 1/sqrt(2)
const INV_SQRT_2PI = 0.3989422804014327; // 1/sqrt(2*pi)

/** Exact GELU: x * Phi(x) with Phi the standard normal CDF (torch default). */
export function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x * SQRT1_2));
}

/** d/dx gelu(x) = Phi(x) + x * phi(x). */
export function geluPrime(x: number): number {
  return 0.5 * (1 + erf(x * SQRT1_2)) + x * INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}

export const f32 = Math.fround;
