/**
 * Minimal dense-tensor helpers for the GeoTransformer port. No dependencies.
 *
 * Convention: matrices are row-major flat arrays (Float32Array for stored weights,
 * Float64Array for activations/accumulators). The backend computes in float32
 * (torch); we accumulate in float64, which stays well inside the port's <=1e-5
 * relative tolerance for d_model=3-sized reductions.
 */

export type Mat = Float32Array | Float64Array;

/**
 * Products below this many multiply-accumulates take the straightforward path.
 *
 * `matmul` and `matmulTN` are *reductions over a strided axis*, which is why the naive
 * loops for them run at ~0.6 GMAC/s while the contiguous-`k` dot-product kernel
 * (`matmulNT`) reaches ~3. Above the cut-off it is a clear win to materialize the
 * transpose and hand the work to that one good kernel; below it the transpose's
 * allocation dominates, which matters because the GeoTransformer multiplies
 * d_model = 3 matrices in a tight loop. Measured on the Lexicon Lab's five shapes
 * (2048x64x{64,192,256,318} and 2048x256x64): 0.63 -> 2.9 GMAC/s for `matmul`,
 * 0.61 -> 2.4 for `matmulTN`.
 *
 * Every kernel here is **bit-identical** to the naive triple loop it replaced: blocking
 * and unrolling only interleave independent accumulator chains, never reassociate one.
 * A 12-step Lexicon Lab training run hashes identically before and after, and the
 * golden parity tests depend on that property — keep it if you touch these.
 */
const VIA_TRANSPOSE_MIN_MACS = 1 << 16;

/** dst(cols,rows) = src(rows,cols)^T, cache-blocked. */
export function transposeInto(src: Mat, dst: Float64Array, rows: number, cols: number): Float64Array {
  const BS = 32;
  for (let i0 = 0; i0 < rows; i0 += BS) {
    const iEnd = Math.min(i0 + BS, rows);
    for (let j0 = 0; j0 < cols; j0 += BS) {
      const jEnd = Math.min(j0 + BS, cols);
      for (let i = i0; i < iEnd; i++) {
        const ro = i * cols;
        for (let j = j0; j < jEnd; j++) dst[j * rows + i] = src[ro + j];
      }
    }
  }
  return dst;
}

/** c = a(n,k) @ b(k,m) -> (n,m). */
export function matmul(a: Mat, b: Mat, n: number, k: number, m: number): Float64Array {
  if (n * k * m >= VIA_TRANSPOSE_MIN_MACS) {
    return matmulNT(a, transposeInto(b, new Float64Array(m * k), k, m), n, k, m);
  }
  const c = new Float64Array(n * m);
  const k8 = k - (k % 8);
  for (let i = 0; i < n; i++) {
    const rowA = i * k;
    const rowC = i * m;
    let t = 0;
    // Unrolled by 8 over the contraction axis: eight `a` scalars share one pass over
    // `c`, cutting its read-modify-write traffic 8x. Each c[j] still accumulates its
    // terms in ascending `t`, exactly as the rolled loop did.
    for (; t < k8; t += 8) {
      const a0 = a[rowA + t], a1 = a[rowA + t + 1], a2 = a[rowA + t + 2], a3 = a[rowA + t + 3];
      const a4 = a[rowA + t + 4], a5 = a[rowA + t + 5], a6 = a[rowA + t + 6], a7 = a[rowA + t + 7];
      const r0 = t * m, r1 = r0 + m, r2 = r1 + m, r3 = r2 + m;
      const r4 = r3 + m, r5 = r4 + m, r6 = r5 + m, r7 = r6 + m;
      for (let j = 0; j < m; j++) {
        let s = c[rowC + j];
        s += a0 * b[r0 + j];
        s += a1 * b[r1 + j];
        s += a2 * b[r2 + j];
        s += a3 * b[r3 + j];
        s += a4 * b[r4 + j];
        s += a5 * b[r5 + j];
        s += a6 * b[r6 + j];
        s += a7 * b[r7 + j];
        c[rowC + j] = s;
      }
    }
    for (; t < k; t++) {
      const av = a[rowA + t];
      if (av === 0) continue;
      const rowB = t * m;
      for (let j = 0; j < m; j++) c[rowC + j] += av * b[rowB + j];
    }
  }
  return c;
}

/**
 * c = a(n,k) @ b(m,k)^T -> (n,m). (y = x @ W^T for row-major W of shape (m,k).)
 *
 * The one kernel worth tuning: `k` runs contiguously in BOTH operands. A 4x4 block of
 * outputs is held in 16 accumulators, so each `b` element loaded serves four rows of `a`
 * and four independent dependency chains keep the FPU busy. ~3 GMAC/s vs 0.85 rolled.
 */
export function matmulNT(a: Mat, b: Mat, n: number, k: number, m: number): Float64Array {
  const c = new Float64Array(n * m);
  const n4 = n - (n % 4);
  const m4 = m - (m % 4);
  let i = 0;
  for (; i < n4; i += 4) {
    const A0 = i * k, A1 = A0 + k, A2 = A1 + k, A3 = A2 + k;
    const C0 = i * m, C1 = C0 + m, C2 = C1 + m, C3 = C2 + m;
    let j = 0;
    for (; j < m4; j += 4) {
      const r0 = j * k, r1 = r0 + k, r2 = r1 + k, r3 = r2 + k;
      let x0 = 0, x1 = 0, x2 = 0, x3 = 0;
      let y0 = 0, y1 = 0, y2 = 0, y3 = 0;
      let z0 = 0, z1 = 0, z2 = 0, z3 = 0;
      let w0 = 0, w1 = 0, w2 = 0, w3 = 0;
      for (let t = 0; t < k; t++) {
        const q0 = b[r0 + t], q1 = b[r1 + t], q2 = b[r2 + t], q3 = b[r3 + t];
        let av = a[A0 + t];
        x0 += av * q0; x1 += av * q1; x2 += av * q2; x3 += av * q3;
        av = a[A1 + t];
        y0 += av * q0; y1 += av * q1; y2 += av * q2; y3 += av * q3;
        av = a[A2 + t];
        z0 += av * q0; z1 += av * q1; z2 += av * q2; z3 += av * q3;
        av = a[A3 + t];
        w0 += av * q0; w1 += av * q1; w2 += av * q2; w3 += av * q3;
      }
      c[C0 + j] = x0; c[C0 + j + 1] = x1; c[C0 + j + 2] = x2; c[C0 + j + 3] = x3;
      c[C1 + j] = y0; c[C1 + j + 1] = y1; c[C1 + j + 2] = y2; c[C1 + j + 3] = y3;
      c[C2 + j] = z0; c[C2 + j + 1] = z1; c[C2 + j + 2] = z2; c[C2 + j + 3] = z3;
      c[C3 + j] = w0; c[C3 + j + 1] = w1; c[C3 + j + 2] = w2; c[C3 + j + 3] = w3;
    }
    for (; j < m; j++) {
      const rowB = j * k;
      let x = 0, y = 0, z = 0, w = 0;
      for (let t = 0; t < k; t++) {
        const q = b[rowB + t];
        x += a[A0 + t] * q; y += a[A1 + t] * q; z += a[A2 + t] * q; w += a[A3 + t] * q;
      }
      c[C0 + j] = x; c[C1 + j] = y; c[C2 + j] = z; c[C3 + j] = w;
    }
  }
  for (; i < n; i++) {
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
  if (n * k * m >= VIA_TRANSPOSE_MIN_MACS) {
    // a^T b = a^T(k,n) @ (b^T(m,n))^T — both transposes make `n` the contiguous
    // contraction axis, which is exactly what matmulNT wants.
    const aT = transposeInto(a, new Float64Array(k * n), n, k);
    const bT = transposeInto(b, new Float64Array(m * n), n, m);
    return matmulNT(aT, bT, k, n, m);
  }
  const c = new Float64Array(k * m);
  const k8 = k - (k % 8);
  for (let i = 0; i < n; i++) {
    const rowA = i * k;
    const rowB = i * m;
    let t = 0;
    for (; t < k8; t += 8) {
      const a0 = a[rowA + t], a1 = a[rowA + t + 1], a2 = a[rowA + t + 2], a3 = a[rowA + t + 3];
      const a4 = a[rowA + t + 4], a5 = a[rowA + t + 5], a6 = a[rowA + t + 6], a7 = a[rowA + t + 7];
      const r0 = t * m, r1 = r0 + m, r2 = r1 + m, r3 = r2 + m;
      const r4 = r3 + m, r5 = r4 + m, r6 = r5 + m, r7 = r6 + m;
      for (let j = 0; j < m; j++) {
        const bv = b[rowB + j];
        c[r0 + j] += a0 * bv;
        c[r1 + j] += a1 * bv;
        c[r2 + j] += a2 * bv;
        c[r3 + j] += a3 * bv;
        c[r4 + j] += a4 * bv;
        c[r5 + j] += a5 * bv;
        c[r6 + j] += a6 * bv;
        c[r7 + j] += a7 * bv;
      }
    }
    for (; t < k; t++) {
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
