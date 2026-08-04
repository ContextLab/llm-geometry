/**
 * The Lexicon Lab transformer, in the browser.
 *
 * A from-scratch TypeScript implementation of the model specified by
 * `specs/006-lexicon-lab-tiny/architecture.md` — which is the contract, not a summary.
 * The PyTorch backend implements the same file; a golden test holds the two to <=1e-5.
 * Everything below is that document, in code:
 *
 *   h = embed[x] + pos[:T]; dropout
 *   per layer: pre-LN -> packed-QKV multi-head causal attention (with bias) -> proj,
 *              residual with NO dropout on that branch;
 *              pre-LN -> d->4d exact-erf GELU -> 4d->d, dropout, residual
 *   final LN, then a BIAS-FREE readout (the embedding itself when tied)
 *
 * Two details are deliberate rather than accidental and must survive refactoring:
 *
 *   * the packed `qkv_w` gets **xavier-uniform** init, bound sqrt(6/(3d+d)), while every
 *     other matrix gets N(0, 0.02^2). That mixed init is what the source model actually
 *     is (its `_init` hook only matches nn.Linear/nn.Embedding, so the packed parameter
 *     keeps a library default) and reproducing it keeps us honest;
 *   * dropout sits on the embedding sum, on the ATTENTION WEIGHTS, and after the second
 *     MLP linear — *not* on the attention residual branch. It defaults to 0 here because
 *     a live demo people re-run should be deterministic.
 *
 * Convention (the geoEngine house style): weights are row-major Float32Array, matching
 * torch's float32 parameters; activations and gradients accumulate in Float64Array.
 */

import { computeError, invalidParam } from "../geoEngine/errors";
import { gelu, geluPrime, logSumExp, matmul, matmulNT, matmulTN } from "../geoEngine/tensor";
import { PAD_ID } from "./vocab";

const f32 = Math.fround;

// --- configuration (lex/config.py) ---------------------------------------------------

export const D_MODEL_CHOICES = [16, 32, 64, 128] as const;
export const N_LAYER_CHOICES = [1, 2, 3, 4] as const;
export const N_HEAD_CHOICES = [1, 2, 4] as const;
export const CTX_CHOICES = [32, 64, 128] as const;

export const DEFAULT_D_MODEL = 64;
export const DEFAULT_N_LAYERS = 2;
export const DEFAULT_N_HEADS = 2;
/**
 * 32, not 64 — see the note on `DEFAULT_CTX` in `lex/config.py`, which this mirrors.
 * Measured on the committed corpus at 400 steps: ctx 64 wins on TRAIN loss (2.247 vs
 * 2.406) and loses on HELD-OUT loss (2.294 vs 2.197), because a 64-token window over a
 * 19,050-token book of nursery rhymes spans several unrelated ones. It is also the
 * cheapest speed lever, since attention costs `batch x ctx^2`.
 */
export const DEFAULT_CTX = 32;
export const DEFAULT_TIED = true;
/** The source hard-codes 0.1 and does not expose it. A live demo wants determinism. */
export const DEFAULT_DROPOUT = 0;

export const LAYER_NORM_EPS = 1e-5;
export const MLP_RATIO = 4;

export interface LexConfig {
  /** Embedding rows: the budget size plus the four specials. */
  vocabRows: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  ctx: number;
  /** When true the readout weight IS the embedding (FR-610). */
  tied: boolean;
  dropout: number;
}

export function defaultConfig(vocabRows: number, over: Partial<LexConfig> = {}): LexConfig {
  // `??` rather than a spread: callers build `over` from optional UI fields, where an
  // explicit `undefined` means "not set" and a spread would blow away every default.
  // `vocabRows` is the argument, never an override — it follows from the vocabulary.
  return validateConfig({
    vocabRows,
    dModel: over.dModel ?? DEFAULT_D_MODEL,
    nLayers: over.nLayers ?? DEFAULT_N_LAYERS,
    nHeads: over.nHeads ?? DEFAULT_N_HEADS,
    ctx: over.ctx ?? DEFAULT_CTX,
    tied: over.tied ?? DEFAULT_TIED,
    dropout: over.dropout ?? DEFAULT_DROPOUT,
  });
}

export function validateConfig(cfg: LexConfig): LexConfig {
  const { vocabRows, dModel, nLayers, nHeads, ctx, dropout } = cfg;
  if (!(Number.isInteger(vocabRows) && vocabRows > 4)) {
    throw invalidParam(`vocabRows must be an integer > 4 (the specials), got ${vocabRows}`);
  }
  if (!(Number.isInteger(dModel) && dModel > 0)) throw invalidParam(`d_model must be a positive integer, got ${dModel}`);
  if (!(Number.isInteger(nLayers) && nLayers >= 1)) throw invalidParam(`n_layers must be >= 1, got ${nLayers}`);
  if (!(Number.isInteger(nHeads) && nHeads >= 1)) throw invalidParam(`n_heads must be >= 1, got ${nHeads}`);
  if (dModel % nHeads !== 0) {
    throw invalidParam(`d_model (${dModel}) must be divisible by n_heads (${nHeads})`);
  }
  if (!(Number.isInteger(ctx) && ctx >= 2)) throw invalidParam(`ctx must be an integer >= 2, got ${ctx}`);
  if (!(dropout >= 0 && dropout < 1)) throw invalidParam(`dropout must be in [0, 1), got ${dropout}`);
  return cfg;
}

/**
 * Exact parameter count, verified against the source implementation on 7 configurations:
 *
 *     N = (2 if untied else 1)*V*d + ctx*d + L*(12d^2 + 13d) + 2d
 *
 * The `12d^2 + 13d` per block is: packed QKV (3d^2 + 3d), attention output projection
 * (d^2 + d), MLP up (4d^2 + 4d), MLP down (4d^2 + d), and two LayerNorms (4d).
 */
export function paramCount(vocabRows: number, dModel: number, nLayers: number, ctx: number, tied: boolean): number {
  const embed = (tied ? 1 : 2) * vocabRows * dModel;
  return embed + ctx * dModel + nLayers * (12 * dModel * dModel + 13 * dModel) + 2 * dModel;
}

export const configParamCount = (cfg: LexConfig): number =>
  paramCount(cfg.vocabRows, cfg.dModel, cfg.nLayers, cfg.ctx, cfg.tied);

// --- weights -------------------------------------------------------------------------

export type WeightSet = Record<string, Float32Array>;

export const layerParamNames = [
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
] as const;

/** Every weight name, in a stable order. `head_w` exists only when untied. */
export function weightNames(cfg: LexConfig): string[] {
  const names = ["embed", "pos"];
  for (let l = 0; l < cfg.nLayers; l++) for (const p of layerParamNames) names.push(`layers.${l}.${p}`);
  names.push("lnf_g", "lnf_b");
  if (!cfg.tied) names.push("head_w");
  return names;
}

/** Element count of each weight tensor — the shape check a load must pass. */
export function weightSizes(cfg: LexConfig): Record<string, number> {
  const d = cfg.dModel;
  const sizes: Record<string, number> = {
    embed: cfg.vocabRows * d,
    pos: cfg.ctx * d,
    lnf_g: d,
    lnf_b: d,
  };
  if (!cfg.tied) sizes.head_w = cfg.vocabRows * d;
  for (let l = 0; l < cfg.nLayers; l++) {
    const p = `layers.${l}.`;
    sizes[`${p}ln1_g`] = d;
    sizes[`${p}ln1_b`] = d;
    sizes[`${p}qkv_w`] = 3 * d * d;
    sizes[`${p}qkv_b`] = 3 * d;
    sizes[`${p}proj_w`] = d * d;
    sizes[`${p}proj_b`] = d;
    sizes[`${p}ln2_g`] = d;
    sizes[`${p}ln2_b`] = d;
    sizes[`${p}fc1_w`] = MLP_RATIO * d * d;
    sizes[`${p}fc1_b`] = MLP_RATIO * d;
    sizes[`${p}fc2_w`] = d * MLP_RATIO * d;
    sizes[`${p}fc2_b`] = d;
  }
  return sizes;
}

/** The 2-D weight MATRICES — the only tensors AdamW decays (FR-614). */
export function decayedWeightNames(cfg: LexConfig): string[] {
  const names: string[] = [];
  for (let l = 0; l < cfg.nLayers; l++) {
    names.push(`layers.${l}.qkv_w`, `layers.${l}.proj_w`, `layers.${l}.fc1_w`, `layers.${l}.fc2_w`);
  }
  if (!cfg.tied) names.push("head_w");
  return names;
}

export function cloneWeights(ws: WeightSet): WeightSet {
  const out: WeightSet = {};
  for (const [k, v] of Object.entries(ws)) out[k] = Float32Array.from(v);
  return out;
}

/** Throws unless `ws` has exactly the tensors `cfg` calls for, at the right sizes. */
export function assertWeightsMatch(cfg: LexConfig, ws: WeightSet): void {
  const sizes = weightSizes(cfg);
  for (const name of weightNames(cfg)) {
    const got = ws[name];
    if (!got) throw invalidParam(`weights are missing ${name}`);
    if (got.length !== sizes[name]) {
      throw invalidParam(`weight ${name} has ${got.length} elements, expected ${sizes[name]}`);
    }
  }
}

/**
 * The tensors of `ws` that contain a NaN or an ±Infinity, in `weightNames` order.
 *
 * `Float32Array` stores anything of magnitude above ~3.4e38 as `Infinity`, so a single
 * cell edit of `1e40` — finite in JS, and therefore accepted by the weight lab's own
 * `Number.isFinite` check — silently makes the model non-finite, as does applying a
 * `×2` preset enough times. Every downstream number then becomes NaN, and NaN compares
 * false against everything, which is how a greedy argmax used to fall through to its
 * initialiser. Callers use this to REFUSE, matching the backend, rather than sanitise.
 */
export function nonFiniteWeightNames(cfg: LexConfig, ws: WeightSet): string[] {
  const bad: string[] = [];
  for (const name of weightNames(cfg)) {
    const t = ws[name];
    if (!t) continue; // shape problems are `assertWeightsMatch`'s to report
    for (let i = 0; i < t.length; i++) {
      if (!Number.isFinite(t[i])) {
        bad.push(name);
        break;
      }
    }
  }
  return bad;
}

// --- deterministic RNG ---------------------------------------------------------------

/**
 * sfc32: small, fast, deterministic PRNG — the same one the geoEngine uses.
 *
 * Whole-run equality with PyTorch is NOT claimed and cannot be: torch's Mersenne/Philox
 * streams are not portable to JS. The browser and Python runs are two independent runs
 * of the same recipe, which is exactly what architecture.md says.
 */
export function sfc32(seed: number): () => number {
  let a = 0x9e3779b9 ^ seed;
  let b = 0x243f6a88 + seed;
  let c = 0xb7e15162 ^ (seed << 13);
  let d = 1 + (seed >>> 3);
  return () => {
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

/** Box-Muller standard normals from a seeded uniform stream. */
function gaussians(rand: () => number, n: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

/** Fresh weights, matching architecture.md's initialization table exactly. */
export function initWeights(cfg: LexConfig, seed = 0): WeightSet {
  validateConfig(cfg);
  const rand = sfc32(seed);
  const d = cfg.dModel;
  const hidden = MLP_RATIO * d;

  const normal = (n: number, std: number): Float32Array => {
    const g = gaussians(rand, n);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = f32(g[i] * std);
    return out;
  };
  const uniform = (n: number, bound: number): Float32Array => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = f32((rand() * 2 - 1) * bound);
    return out;
  };
  const ones = (n: number): Float32Array => new Float32Array(n).fill(1);
  const zeros = (n: number): Float32Array => new Float32Array(n);

  const ws: WeightSet = {};
  ws.embed = normal(cfg.vocabRows * d, 0.02);
  ws.pos = normal(cfg.ctx * d, 0.02);
  for (let l = 0; l < cfg.nLayers; l++) {
    const p = `layers.${l}.`;
    ws[`${p}ln1_g`] = ones(d);
    ws[`${p}ln1_b`] = zeros(d);
    // The one xavier-uniform tensor. bound = sqrt(6 / (fan_in + fan_out)) = sqrt(6/(3d+d)).
    ws[`${p}qkv_w`] = uniform(3 * d * d, Math.sqrt(6 / (3 * d + d)));
    ws[`${p}qkv_b`] = zeros(3 * d);
    ws[`${p}proj_w`] = normal(d * d, 0.02);
    ws[`${p}proj_b`] = zeros(d);
    ws[`${p}ln2_g`] = ones(d);
    ws[`${p}ln2_b`] = zeros(d);
    ws[`${p}fc1_w`] = normal(hidden * d, 0.02);
    ws[`${p}fc1_b`] = zeros(hidden);
    ws[`${p}fc2_w`] = normal(d * hidden, 0.02);
    ws[`${p}fc2_b`] = zeros(d);
  }
  ws.lnf_g = ones(d);
  ws.lnf_b = zeros(d);
  if (!cfg.tied) ws.head_w = normal(cfg.vocabRows * d, 0.02);
  return ws;
}

// --- small kernels -------------------------------------------------------------------

/**
 * Row-wise LayerNorm. Writes the NORMALIZED rows (before the affine) into `xhat` and the
 * per-row mean/reciprocal-std into `mean`/`rstd`; the affine output goes to `out`.
 * Splitting it this way is what lets the backward pass run without a second forward.
 */
function layerNormForward(
  x: Float64Array,
  N: number,
  d: number,
  g: Float32Array,
  b: Float32Array,
  xhat: Float64Array,
  rstd: Float64Array,
  out: Float64Array,
): void {
  for (let n = 0; n < N; n++) {
    const o = n * d;
    let mu = 0;
    for (let c = 0; c < d; c++) mu += x[o + c];
    mu /= d;
    let va = 0;
    for (let c = 0; c < d; c++) {
      const z = x[o + c] - mu;
      va += z * z;
    }
    va /= d;
    const r = 1 / Math.sqrt(va + LAYER_NORM_EPS);
    rstd[n] = r;
    for (let c = 0; c < d; c++) {
      const z = (x[o + c] - mu) * r;
      xhat[o + c] = z;
      out[o + c] = g[c] * z + b[c];
    }
  }
}

/** Recompute a LayerNorm's affine output from its cached `xhat` (cheap; saves memory). */
function affine(xhat: Float64Array, N: number, d: number, g: Float32Array, b: Float32Array): Float64Array {
  const out = new Float64Array(N * d);
  for (let n = 0; n < N; n++) {
    const o = n * d;
    for (let c = 0; c < d; c++) out[o + c] = g[c] * xhat[o + c] + b[c];
  }
  return out;
}

/**
 * LayerNorm backward. Accumulates dg/db, returns dx.
 *   dxhat = dy * g
 *   dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
 */
function layerNormBackward(
  dy: Float64Array,
  xhat: Float64Array,
  rstd: Float64Array,
  N: number,
  d: number,
  g: Float32Array,
  dg: Float64Array,
  db: Float64Array,
): Float64Array {
  const dx = new Float64Array(N * d);
  for (let n = 0; n < N; n++) {
    const o = n * d;
    let sum = 0;
    let sumX = 0;
    for (let c = 0; c < d; c++) {
      const dyv = dy[o + c];
      dg[c] += dyv * xhat[o + c];
      db[c] += dyv;
      const dh = dyv * g[c];
      sum += dh;
      sumX += dh * xhat[o + c];
    }
    sum /= d;
    sumX /= d;
    const r = rstd[n];
    for (let c = 0; c < d; c++) {
      dx[o + c] = r * (dy[o + c] * g[c] - sum - xhat[o + c] * sumX);
    }
  }
  return dx;
}

/** y = x @ W^T + bias, for row-major W of shape (m, k). */
function linear(x: Float64Array, W: Float32Array, bias: Float32Array | null, n: number, k: number, m: number): Float64Array {
  const y = matmulNT(x, W, n, k, m);
  if (bias) {
    for (let i = 0; i < n; i++) {
      const o = i * m;
      for (let j = 0; j < m; j++) y[o + j] += bias[j];
    }
  }
  return y;
}

/** Causal row softmax over the (T,T) block at `off`; entries j > i become exactly 0. */
function softmaxCausalBlock(scores: Float64Array, off: number, T: number): void {
  for (let i = 0; i < T; i++) {
    const row = off + i * T;
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

// --- activations ---------------------------------------------------------------------

interface LayerCache {
  xhat1: Float64Array; // (N,d) pre-attention LayerNorm, normalized
  rstd1: Float64Array; // (N,)
  qkv: Float64Array; // (N,3d)
  attn: Float64Array; // (B*H, T, T) post-softmax, post-dropout
  attnPre: Float64Array | null; // the same, before dropout (only when dropout > 0)
  attnMask: Uint8Array | null;
  ctxMerged: Float64Array; // (N,d)
  hMid: Float64Array; // (N,d) after the attention residual
  xhat2: Float64Array; // (N,d) pre-MLP LayerNorm, normalized
  rstd2: Float64Array;
  fc1Pre: Float64Array; // (N,4d)
  act: Float64Array; // (N,4d) = gelu(fc1Pre)
  mlpMask: Uint8Array | null;
  hOut: Float64Array; // (N,d)
}

export interface Activations {
  B: number;
  T: number;
  ids: Int32Array; // (N,)
  /** The dropout probability this pass actually used (0 in eval mode). */
  dropoutP: number;
  embMask: Uint8Array | null;
  layers: LayerCache[];
  xhatF: Float64Array; // (N,d)
  rstdF: Float64Array;
  hF: Float64Array; // (N,d) final LayerNorm output (the readout input)
  logits: Float64Array; // (N,V)
}

// --- the model -----------------------------------------------------------------------

export interface ForwardOptions {
  /** > 0 puts the model in training mode; requires `rand`. Inverted dropout. */
  dropout?: number;
  rand?: () => number;
}

export class LexModel {
  readonly cfg: LexConfig;
  readonly weights: WeightSet;

  constructor(cfg: LexConfig, weights: WeightSet) {
    this.cfg = validateConfig(cfg);
    assertWeightsMatch(cfg, weights);
    this.weights = weights;
  }

  static fresh(cfg: LexConfig, seed = 0): LexModel {
    return new LexModel(cfg, initWeights(cfg, seed));
  }

  /** The readout matrix (V,d): the embedding itself when tied (FR-610). */
  get headWeight(): Float32Array {
    return this.cfg.tied ? this.weights.embed : this.weights.head_w;
  }

  private p(l: number, name: string): Float32Array {
    return this.weights[`layers.${l}.${name}`];
  }

  /**
   * Full forward pass over a batch of `B` sequences of length `T` (row-major ids).
   * Returns every activation the backward pass needs.
   */
  forward(ids: Int32Array, B: number, T: number, opts: ForwardOptions = {}): Activations {
    const { dModel: d, nHeads: H, ctx, vocabRows: V, nLayers: L } = this.cfg;
    if (T < 1 || T > ctx) throw invalidParam(`sequence length ${T} must be in 1..${ctx}`);
    if (ids.length !== B * T) throw invalidParam(`ids has ${ids.length} entries, expected B*T = ${B * T}`);
    const p = opts.dropout ?? 0;
    const rand = opts.rand;
    if (p > 0 && !rand) throw invalidParam("dropout > 0 requires a seeded rand() (determinism is a feature)");
    const keep = 1 - p;
    const scale = p > 0 ? 1 / keep : 1;
    const dh = d / H;
    const invSqrtDh = 1 / Math.sqrt(dh);
    const N = B * T;

    const drop = (buf: Float64Array): Uint8Array | null => {
      if (p <= 0) return null;
      const mask = new Uint8Array(buf.length);
      for (let i = 0; i < buf.length; i++) {
        const k = (rand as () => number)() < keep ? 1 : 0;
        mask[i] = k;
        buf[i] = k ? buf[i] * scale : 0;
      }
      return mask;
    };

    // h = embed[x] + pos[:T]
    const embed = this.weights.embed;
    const pos = this.weights.pos;
    let h = new Float64Array(N * d);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const n = b * T + t;
        const id = ids[n];
        if (id < 0 || id >= V) throw invalidParam(`token id ${id} is outside 0..${V - 1}`);
        const o = n * d;
        const e = id * d;
        const q = t * d;
        for (let c = 0; c < d; c++) h[o + c] = embed[e + c] + pos[q + c];
      }
    }
    const embMask = drop(h);

    const layers: LayerCache[] = [];
    for (let l = 0; l < L; l++) {
      const xhat1 = new Float64Array(N * d);
      const rstd1 = new Float64Array(N);
      const a1 = new Float64Array(N * d);
      layerNormForward(h, N, d, this.p(l, "ln1_g"), this.p(l, "ln1_b"), xhat1, rstd1, a1);

      const qkv = linear(a1, this.p(l, "qkv_w"), this.p(l, "qkv_b"), N, d, 3 * d);

      // Causal multi-head attention over the packed [q | k | v] rows.
      //
      // `qh`/`kh`/`vh` are (T, dh) contiguous copies of one head. They are pure data
      // movement — every multiply and every accumulation order below is unchanged — but
      // they matter: in the packed layout a head's slice is `dh` doubles out of every
      // `3d`, so walking `j` over the keys touches 256 useful bytes per 1536-byte row.
      // The triangular loops re-walk that stride T(T+1)/2 times per head; copying the
      // head out once into 8 KB of L1 instead measured 43 ms -> 12 ms for this block.
      const attn = new Float64Array(B * H * T * T);
      const ctxMerged = new Float64Array(N * d);
      const qh = new Float64Array(T * dh);
      const kh = new Float64Array(T * dh);
      const vh = new Float64Array(T * dh);
      for (let b = 0; b < B; b++) {
        for (let hh = 0; hh < H; hh++) {
          const block = ((b * H + hh) * T) * T;
          const qOff = hh * dh;
          const kOff = d + hh * dh;
          for (let i = 0; i < T; i++) {
            const src = (b * T + i) * 3 * d;
            const dst = i * dh;
            for (let c = 0; c < dh; c++) {
              qh[dst + c] = qkv[src + qOff + c];
              kh[dst + c] = qkv[src + kOff + c];
            }
          }
          // Four keys per pass. A single q.k is `dh` dependent FP adds, so the rolled
          // loop retires one multiply-accumulate every four cycles no matter how well
          // it caches; four independent accumulators fill the pipeline instead. Each
          // s_n still sums over `c` in ascending order, so every score is bit-identical.
          for (let i = 0; i < T; i++) {
            const qi = i * dh;
            const row = block + i * T;
            const jEnd = i + 1;
            let j = 0;
            for (; j + 4 <= jEnd; j += 4) {
              const k0 = j * dh, k1 = k0 + dh, k2 = k1 + dh, k3 = k2 + dh;
              let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
              for (let c = 0; c < dh; c++) {
                const qv = qh[qi + c];
                s0 += qv * kh[k0 + c];
                s1 += qv * kh[k1 + c];
                s2 += qv * kh[k2 + c];
                s3 += qv * kh[k3 + c];
              }
              attn[row + j] = s0 * invSqrtDh;
              attn[row + j + 1] = s1 * invSqrtDh;
              attn[row + j + 2] = s2 * invSqrtDh;
              attn[row + j + 3] = s3 * invSqrtDh;
            }
            for (; j < jEnd; j++) {
              const kj = j * dh;
              let s = 0;
              for (let c = 0; c < dh; c++) s += qh[qi + c] * kh[kj + c];
              attn[row + j] = s * invSqrtDh;
            }
          }
          softmaxCausalBlock(attn, block, T);
        }
      }
      // Dropout on the ATTENTION WEIGHTS (the source's unusual placement).
      const attnPre = p > 0 ? Float64Array.from(attn) : null;
      const attnMask = drop(attn);
      for (let b = 0; b < B; b++) {
        for (let hh = 0; hh < H; hh++) {
          const block = ((b * H + hh) * T) * T;
          const vOff = 2 * d + hh * dh;
          for (let j = 0; j < T; j++) {
            const src = (b * T + j) * 3 * d + vOff;
            const dst = j * dh;
            for (let c = 0; c < dh; c++) vh[dst + c] = qkv[src + c];
          }
          for (let i = 0; i < T; i++) {
            const out = (b * T + i) * d + hh * dh;
            const row = block + i * T;
            for (let j = 0; j <= i; j++) {
              const a = attn[row + j];
              if (a === 0) continue;
              const vj = j * dh;
              for (let c = 0; c < dh; c++) ctxMerged[out + c] += a * vh[vj + c];
            }
          }
        }
      }

      const attnOut = linear(ctxMerged, this.p(l, "proj_w"), this.p(l, "proj_b"), N, d, d);
      const hMid = new Float64Array(N * d);
      for (let i = 0; i < N * d; i++) hMid[i] = h[i] + attnOut[i]; // NO dropout here

      const xhat2 = new Float64Array(N * d);
      const rstd2 = new Float64Array(N);
      const a2 = new Float64Array(N * d);
      layerNormForward(hMid, N, d, this.p(l, "ln2_g"), this.p(l, "ln2_b"), xhat2, rstd2, a2);

      const hidden = MLP_RATIO * d;
      const fc1Pre = linear(a2, this.p(l, "fc1_w"), this.p(l, "fc1_b"), N, d, hidden);
      const act = new Float64Array(N * hidden);
      for (let i = 0; i < act.length; i++) act[i] = gelu(fc1Pre[i]);
      const m = linear(act, this.p(l, "fc2_w"), this.p(l, "fc2_b"), N, hidden, d);
      const mlpMask = drop(m);
      const hOut = new Float64Array(N * d);
      for (let i = 0; i < N * d; i++) hOut[i] = hMid[i] + m[i];

      layers.push({
        xhat1,
        rstd1,
        qkv,
        attn,
        attnPre,
        attnMask,
        ctxMerged,
        hMid,
        xhat2,
        rstd2,
        fc1Pre,
        act,
        mlpMask,
        hOut,
      });
      h = hOut;
    }

    const xhatF = new Float64Array(N * d);
    const rstdF = new Float64Array(N);
    const hF = new Float64Array(N * d);
    layerNormForward(h, N, d, this.weights.lnf_g, this.weights.lnf_b, xhatF, rstdF, hF);
    const logits = matmulNT(hF, this.headWeight, N, d, V); // NO bias

    return { B, T, ids, dropoutP: p, embMask, layers, xhatF, rstdF, hF, logits };
  }

  /**
   * Logits for the LAST position of one sequence (generation's inner loop).
   * Runs a plain forward with B = 1; at ctx <= 128 and tens of new tokens the quadratic
   * re-computation is cheaper than the bookkeeping a KV cache would need, and it shares
   * exactly one code path with training.
   */
  lastLogits(ids: number[]): Float64Array {
    const T = ids.length;
    const acts = this.forward(Int32Array.from(ids), 1, T, {});
    const V = this.cfg.vocabRows;
    return acts.logits.slice((T - 1) * V, T * V);
  }

  /** Total number of scalar parameters actually allocated. Must equal paramCount(). */
  get nParams(): number {
    let n = 0;
    for (const name of weightNames(this.cfg)) n += this.weights[name].length;
    return n;
  }
}

// --- loss and gradients --------------------------------------------------------------

export type GradSet = Record<string, Float64Array>;

export function zeroGrads(cfg: LexConfig): GradSet {
  const sizes = weightSizes(cfg);
  const grads: GradSet = {};
  for (const name of weightNames(cfg)) grads[name] = new Float64Array(sizes[name]);
  return grads;
}

export interface LossResult {
  /** Sum of -log p(target) over non-pad targets. */
  ceSum: number;
  /** Number of non-pad targets. */
  nValid: number;
  /** d(loss)/d(logits), already divided by `nValid`. Zero on ignored rows. */
  dLogits: Float64Array;
}

/**
 * Mean cross-entropy against `targets`, with `ignore_index = PAD_ID`.
 *
 * architecture.md writes the objective as "logits[:, :-1] against x[:, 1:]" over a
 * `ctx+1`-token window. That is exactly this function with the window's first `ctx`
 * tokens as the model input and its last `ctx` tokens as `targets` — the same
 * supervision, without asking the positional table for a `ctx+1`-th row.
 *
 * `logits` is (N, V) and `targets` is (N,), N = B*T.
 */
export function crossEntropy(logits: Float64Array, targets: Int32Array, N: number, V: number): LossResult {
  const dLogits = new Float64Array(N * V);
  let ceSum = 0;
  let nValid = 0;
  // Two passes: the first counts, so dLogits can be scaled by 1/nValid in the second
  // (torch's cross_entropy averages over every non-ignored target of the whole batch).
  for (let n = 0; n < N; n++) if (targets[n] !== PAD_ID) nValid++;
  if (nValid === 0) throw computeError("this batch has no non-pad targets to learn from");
  for (let n = 0; n < N; n++) {
    const target = targets[n];
    if (target === PAD_ID) continue;
    const off = n * V;
    const lse = logSumExp(logits, off, V);
    ceSum += lse - logits[off + target];
    for (let w = 0; w < V; w++) dLogits[off + w] = Math.exp(logits[off + w] - lse) / nValid;
    dLogits[off + target] -= 1 / nValid;
  }
  return { ceSum, nValid, dLogits };
}

/**
 * Backward pass. Accumulates into `grads` (never zeroes them, so a caller can sum over
 * micro-batches) and returns nothing — the loss came from `crossEntropy`.
 */
export function backward(model: LexModel, acts: Activations, dLogits: Float64Array, grads: GradSet): void {
  const { dModel: d, nHeads: H, vocabRows: V, nLayers: L, tied } = model.cfg;
  const { B, T, ids } = acts;
  const N = B * T;
  const dh = d / H;
  const invSqrtDh = 1 / Math.sqrt(dh);
  const hidden = MLP_RATIO * d;
  const w = model.weights;
  const dropScale = acts.dropoutP > 0 ? 1 / (1 - acts.dropoutP) : 1;

  // Readout: logits = hF @ head_w^T (no bias).
  const dHeadName = tied ? "embed" : "head_w";
  addInto(grads[dHeadName], matmulTN(dLogits, acts.hF, N, V, d));
  let dh_ = matmul(dLogits, model.headWeight, N, V, d); // (N,d)

  // Final LayerNorm.
  dh_ = layerNormBackward(dh_, acts.xhatF, acts.rstdF, N, d, w.lnf_g, grads.lnf_g, grads.lnf_b);

  for (let l = L - 1; l >= 0; l--) {
    const la = acts.layers[l];
    const pre = `layers.${l}.`;
    const fc2W = w[`${pre}fc2_w`];
    const fc1W = w[`${pre}fc1_w`];
    const projW = w[`${pre}proj_w`];
    const qkvW = w[`${pre}qkv_w`];

    // --- MLP branch: hOut = hMid + dropout(fc2(gelu(fc1(LN2(hMid))))) ---
    const dm = new Float64Array(N * d);
    if (la.mlpMask) {
      for (let i = 0; i < dm.length; i++) dm[i] = la.mlpMask[i] ? dh_[i] * dropScale : 0;
    } else {
      dm.set(dh_);
    }
    colSum(grads[`${pre}fc2_b`], dm, N, d);
    addInto(grads[`${pre}fc2_w`], matmulTN(dm, la.act, N, d, hidden));
    const dAct = matmul(dm, fc2W, N, d, hidden); // (N,4d)
    const dPre = new Float64Array(N * hidden);
    for (let i = 0; i < dPre.length; i++) dPre[i] = dAct[i] * geluPrime(la.fc1Pre[i]);
    colSum(grads[`${pre}fc1_b`], dPre, N, hidden);
    addInto(grads[`${pre}fc1_w`], matmulTN(dPre, affine(la.xhat2, N, d, w[`${pre}ln2_g`], w[`${pre}ln2_b`]), N, hidden, d));
    const dA2 = matmul(dPre, fc1W, N, hidden, d); // (N,d)
    const dMlpIn = layerNormBackward(dA2, la.xhat2, la.rstd2, N, d, w[`${pre}ln2_g`], grads[`${pre}ln2_g`], grads[`${pre}ln2_b`]);
    const dHMid = new Float64Array(N * d);
    for (let i = 0; i < dHMid.length; i++) dHMid[i] = dh_[i] + dMlpIn[i]; // + residual

    // --- attention branch: hMid = hIn + proj(merge(A @ V)) ---
    colSum(grads[`${pre}proj_b`], dHMid, N, d);
    addInto(grads[`${pre}proj_w`], matmulTN(dHMid, la.ctxMerged, N, d, d));
    const dCtx = matmul(dHMid, projW, N, d, d); // (N,d)

    const dQkv = new Float64Array(N * 3 * d);
    const qkvA = la.qkv; // the packed [q | k | v] activations of this layer
    const attnUsed = la.attn; // post-dropout weights are what multiplied V
    const attnPre = la.attnPre ?? la.attn; // pre-dropout weights are the softmax output
    // Head-contiguous scratch, for the same cache reason as the forward pass: reads come
    // from `qh`/`kh`/`vh` and writes land in `dQh`/`dKh`/`dVh`, which are scattered back
    // into the packed `dQkv` once per head. Each of those slots is written by exactly
    // this (b, head) iteration, so the scatter is a copy and the accumulation order
    // inside each buffer is exactly the order the packed loops used.
    const qh = new Float64Array(T * dh);
    const kh = new Float64Array(T * dh);
    const vh = new Float64Array(T * dh);
    const dQh = new Float64Array(T * dh);
    const dKh = new Float64Array(T * dh);
    const dVh = new Float64Array(T * dh);
    const dA = new Float64Array(T * T);
    const dScores = new Float64Array(T * T);
    for (let b = 0; b < B; b++) {
      for (let hh = 0; hh < H; hh++) {
        const block = ((b * H + hh) * T) * T;
        const qOff = hh * dh;
        const kOff = d + hh * dh;
        const vOff = 2 * d + hh * dh;
        for (let i = 0; i < T; i++) {
          const src = (b * T + i) * 3 * d;
          const dst = i * dh;
          for (let c = 0; c < dh; c++) {
            qh[dst + c] = qkvA[src + qOff + c];
            kh[dst + c] = qkvA[src + kOff + c];
            vh[dst + c] = qkvA[src + vOff + c];
          }
        }
        dQh.fill(0);
        dKh.fill(0);
        dVh.fill(0);
        dA.fill(0);
        dScores.fill(0);
        // dA and dV from ctx = A @ V. dA's four-at-a-time pass is the same pipeline fix
        // as the forward scores; dV keeps its own rolled loop because its `c` iterations
        // are already independent of one another, and staying inside the ascending-`i`
        // outer loop is what keeps its accumulation order intact.
        for (let i = 0; i < T; i++) {
          const dci = (b * T + i) * d + hh * dh;
          const row = block + i * T;
          const jEnd = i + 1;
          let j = 0;
          for (; j + 4 <= jEnd; j += 4) {
            const v0 = j * dh, v1 = v0 + dh, v2 = v1 + dh, v3 = v2 + dh;
            let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (let c = 0; c < dh; c++) {
              const gv = dCtx[dci + c];
              s0 += gv * vh[v0 + c];
              s1 += gv * vh[v1 + c];
              s2 += gv * vh[v2 + c];
              s3 += gv * vh[v3 + c];
            }
            const o = i * T + j;
            dA[o] = s0; dA[o + 1] = s1; dA[o + 2] = s2; dA[o + 3] = s3;
          }
          for (; j < jEnd; j++) {
            const vj = j * dh;
            let s = 0;
            for (let c = 0; c < dh; c++) s += dCtx[dci + c] * vh[vj + c];
            dA[i * T + j] = s;
          }
          for (let jj = 0; jj < jEnd; jj++) {
            const a = attnUsed[row + jj];
            if (a === 0) continue;
            const vj = jj * dh;
            for (let c = 0; c < dh; c++) dVh[vj + c] += a * dCtx[dci + c];
          }
        }
        // Attention-weight dropout backward, then softmax backward per causal row.
        if (la.attnMask) {
          for (let i = 0; i < T; i++) {
            for (let j = 0; j <= i; j++) {
              const idx = block + i * T + j;
              dA[i * T + j] = la.attnMask[idx] ? dA[i * T + j] * dropScale : 0;
            }
          }
        }
        for (let i = 0; i < T; i++) {
          let s = 0;
          for (let j = 0; j <= i; j++) s += attnPre[block + i * T + j] * dA[i * T + j];
          for (let j = 0; j <= i; j++) {
            dScores[i * T + j] = attnPre[block + i * T + j] * (dA[i * T + j] - s) * invSqrtDh;
          }
        }
        for (let i = 0; i < T; i++) {
          const qi = i * dh;
          for (let j = 0; j <= i; j++) {
            const g = dScores[i * T + j];
            if (g === 0) continue;
            const kj = j * dh;
            for (let c = 0; c < dh; c++) {
              dQh[qi + c] += g * kh[kj + c];
              dKh[kj + c] += g * qh[qi + c];
            }
          }
        }
        for (let i = 0; i < T; i++) {
          const dst = (b * T + i) * 3 * d;
          const srcH = i * dh;
          for (let c = 0; c < dh; c++) {
            dQkv[dst + qOff + c] += dQh[srcH + c];
            dQkv[dst + kOff + c] += dKh[srcH + c];
            dQkv[dst + vOff + c] += dVh[srcH + c];
          }
        }
      }
    }

    colSum(grads[`${pre}qkv_b`], dQkv, N, 3 * d);
    addInto(grads[`${pre}qkv_w`], matmulTN(dQkv, affine(la.xhat1, N, d, w[`${pre}ln1_g`], w[`${pre}ln1_b`]), N, 3 * d, d));
    const dA1 = matmul(dQkv, qkvW, N, 3 * d, d); // (N,d)
    const dAttnIn = layerNormBackward(dA1, la.xhat1, la.rstd1, N, d, w[`${pre}ln1_g`], grads[`${pre}ln1_g`], grads[`${pre}ln1_b`]);
    const dHIn = new Float64Array(N * d);
    for (let i = 0; i < dHIn.length; i++) dHIn[i] = dHMid[i] + dAttnIn[i]; // + residual
    dh_ = dHIn;
  }

  // Input: h0 = embed[ids] + pos[:T], then dropout.
  if (acts.embMask) {
    for (let i = 0; i < dh_.length; i++) dh_[i] = acts.embMask[i] ? dh_[i] * dropScale : 0;
  }
  const dEmbed = grads.embed;
  const dPos = grads.pos;
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const n = b * T + t;
      const o = n * d;
      const e = ids[n] * d;
      const q = t * d;
      for (let c = 0; c < d; c++) {
        dEmbed[e + c] += dh_[o + c];
        dPos[q + c] += dh_[o + c];
      }
    }
  }
}

function addInto(target: Float64Array, src: Float64Array): void {
  for (let i = 0; i < target.length; i++) target[i] += src[i];
}

function colSum(target: Float64Array, src: Float64Array, n: number, m: number): void {
  for (let i = 0; i < n; i++) {
    const o = i * m;
    for (let j = 0; j < m; j++) target[j] += src[o + j];
  }
}
