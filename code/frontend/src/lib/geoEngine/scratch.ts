/**
 * From-scratch training in the browser (feature 004, FR-420/FR-421).
 *
 * Mirrors `llm_geometry/geo/{scratch,train}.py`: build a fresh vocabulary from the
 * supplied text, initialize weights the same way the PyTorch model does, then run Adam
 * over cross-entropy + the Wang & Isola uniformity term, renormalizing the embeddings
 * onto S² after every step.
 *
 * WHAT IS AND IS NOT MIRRORED — worth being precise, because "mirrors the backend" is
 * easy to over-claim:
 *   * The objective, optimizer, hyperparameters, gradient clipping, the S² projection,
 *     and the vocabulary construction are the same.
 *   * The random INITIALIZATION is not bit-identical. numpy/torch's Mersenne/Philox
 *     streams cannot be reproduced in JS, so a browser run and a Python run starting
 *     from "the same seed" are different draws. They are two independent training runs
 *     of the same recipe, not the same run.
 *
 * Because whole runs cannot be compared, the cross-language pin is one training STEP
 * from a SUPPLIED initialization with SUPPLIED repulsion sample indices:
 * `scratchTrainStep` below takes both, so `tests/unit/geoScratchGolden.test.ts` can
 * check the Adam update and the Wang & Isola uniformity gradient against numbers the
 * real Python backend produced (`tests/fixtures/geo/scratch_step.json`, written by
 * `tests/fixtures/geo/generate.py`). Before that golden existed this comment claimed
 * it did, and those two pieces of arithmetic had no cross-language coverage at all.
 */

import { computeError, invalidParam } from "./errors";
import { accumulateWindowGrads as accumulate, evalLoss, makeWindows, sfc32 } from "./finetune";
import {
  CONTEXT_WINDOW,
  D_MODEL,
  EOS_ID,
  GeoModel,
  MLP_HIDDEN,
  N_LAYERS,
  PAD_ID,
  VOCAB_SIZE,
  VOCAB_WORDS,
  cloneWeightSet,
  weightNames,
  type WeightSet,
} from "./model";
import { splitWords } from "./tokenizer";

// train.py / config.py
export const TRAIN_LR = 2e-2; // Adam
export const TRAIN_BATCH_SIZE = 64;
export const TRAIN_WINDOW_STRIDE = 10;
export const REPULSION_WEIGHT = 0.3;
export const REPULSION_SAMPLE = 256;
export const REPULSION_T = 2.0;
export const SCRATCH_DEFAULT_EPOCHS = 12;
export const SCRATCH_MAX_EPOCHS = 60;

/**
 * How far below `ln(VOCAB_SIZE)` a run's final cross-entropy has to land before we are
 * willing to call it "learned". A model that predicts the uniform distribution scores
 * exactly `ln(VOCAB_SIZE)` nats, so half a nat below it means the next-token
 * distribution is at least e^0.5 ≈ 1.65× more concentrated than uniform — a low bar on
 * purpose, because this flag exists to catch runs that never left the baseline, not to
 * grade good ones. Mirrors geo/scratch.SCRATCH_LEARNED_MARGIN.
 *
 * NOTE: the checkpoint's two "non-degeneracy" gates — coverage_uniformity and
 * field_directional_entropy — CANNOT stand in for this. They guard against COLLAPSE,
 * and a model that learned nothing scores BETTER on both than the real checkpoint,
 * because near-random embeddings are maximally dispersed and maximally
 * multi-directional (measured: entropy 3.28 vs 2.81, coverage 0.988 vs 0.900).
 */
export const SCRATCH_LEARNED_MARGIN = 0.5;

/** Cross-entropy (nats) of the uniform next-token distribution: the learn-nothing floor. */
export function uniformBaselineLoss(): number {
  return Math.log(VOCAB_SIZE);
}

export interface CorpusStats {
  n_tokens: number;
  n_distinct: number;
  vocab_words_required: number;
}

/** Token / distinct-type counts — the same numbers GET /api/geo/corpus_stats reports. */
export function corpusStats(text: string): CorpusStats {
  const words = splitWords(text);
  return {
    n_tokens: words.length,
    n_distinct: new Set(words).size,
    vocab_words_required: VOCAB_WORDS,
  };
}

/**
 * The top-VOCAB_WORDS token types of `text` by frequency, ties broken alphabetically.
 * Byte-for-byte the same rule as GeoTokenizer.from_corpus_text, so a vocabulary built
 * here and one built by the backend from the same text are identical.
 */
export function buildVocabWords(text: string): string[] {
  const counts = new Map<string, number>();
  for (const w of splitWords(text)) counts.set(w, (counts.get(w) ?? 0) + 1);
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  if (ranked.length < VOCAB_WORDS) {
    throw invalidParam(
      `This text has only ${ranked.length} distinct word types, and the model's ` +
        `vocabulary is ${VOCAB_WORDS} words wide — training it would leave most of the ` +
        "vocabulary undefined. Paste more text (a few pages of prose), or pick a larger " +
        "HuggingFace dataset.",
    );
  }
  return ranked.slice(0, VOCAB_WORDS).map(([w]) => w);
}

/** Box–Muller standard normals from the engine's existing seeded uniform stream. */
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

/** Fresh weights, matching GeoTransformer.reset_parameters's distributions. */
export function initWeights(seed = 0): WeightSet {
  const rand = sfc32(seed);
  const ws: WeightSet = {};
  const draw = (n: number, scale: number): Float32Array => {
    const g = gaussians(rand, n);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = g[i] * scale;
    return out;
  };

  // Token embeddings: random unit vectors on S².
  const emb = draw(VOCAB_SIZE * D_MODEL, 1);
  for (let v = 0; v < VOCAB_SIZE; v++) {
    const o = v * D_MODEL;
    const norm = Math.hypot(emb[o], emb[o + 1], emb[o + 2]) || 1;
    emb[o] /= norm;
    emb[o + 1] /= norm;
    emb[o + 2] /= norm;
  }
  ws.embedding = emb;
  // Small positional offsets so token geometry dominates at init.
  ws.pos_embedding = draw(CONTEXT_WINDOW * D_MODEL, 0.02);

  for (let l = 0; l < N_LAYERS; l++) {
    ws[`layers.${l}.W_Q`] = draw(D_MODEL * D_MODEL, 0.4);
    ws[`layers.${l}.W_K`] = draw(D_MODEL * D_MODEL, 0.4);
    ws[`layers.${l}.W_V`] = draw(D_MODEL * D_MODEL, 0.4);
    // W_O and W_out start small: the norm-free residual stream stays near the sphere.
    ws[`layers.${l}.W_O`] = draw(D_MODEL * D_MODEL, 0.1);
    ws[`layers.${l}.W_in`] = draw(D_MODEL * MLP_HIDDEN, 1 / Math.sqrt(D_MODEL));
    ws[`layers.${l}.b_in`] = new Float32Array(MLP_HIDDEN);
    ws[`layers.${l}.W_out`] = draw(MLP_HIDDEN * D_MODEL, 0.1 / Math.sqrt(MLP_HIDDEN));
    ws[`layers.${l}.b_out`] = new Float32Array(D_MODEL);
  }
  return ws;
}

/** The REPULSION_SAMPLE embedding rows one step repels (mirrors torch.randint). */
export function sampleUniformityIndices(rand: () => number): Int32Array {
  const m = Math.min(REPULSION_SAMPLE, VOCAB_SIZE);
  const idx = new Int32Array(m);
  for (let i = 0; i < m; i++) idx[i] = Math.min(VOCAB_SIZE - 1, Math.floor(rand() * VOCAB_SIZE));
  return idx;
}

/**
 * Wang & Isola uniformity over the GIVEN embedding rows, and its gradient w.r.t. them.
 * L = log( mean_{i<j} exp(−t·‖eᵢ−eⱼ‖²) ).
 *
 * The indices are a parameter, not drawn here, so one step is reproducible from data
 * alone — torch's RNG stream is not portable but a list of indices is, which is what
 * lets `tests/unit/geoScratchGolden.test.ts` pin this gradient against the real Python
 * backend (`geo/train.uniformity_loss`).
 */
export function uniformityLossAndGrad(
  embedding: Float32Array,
  idx: Int32Array,
  gradEmbedding: Float64Array,
): number {
  const m = idx.length;
  const pairs = (m * (m - 1)) / 2;
  if (pairs === 0) return 0;
  const w = new Float64Array(pairs);
  let sum = 0;
  let p = 0;
  for (let a = 0; a < m; a++) {
    const ia = idx[a] * D_MODEL;
    for (let b = a + 1; b < m; b++) {
      const ib = idx[b] * D_MODEL;
      const dx = embedding[ia] - embedding[ib];
      const dy = embedding[ia + 1] - embedding[ib + 1];
      const dz = embedding[ia + 2] - embedding[ib + 2];
      const e = Math.exp(-REPULSION_T * (dx * dx + dy * dy + dz * dz));
      w[p++] = e;
      sum += e;
    }
  }
  if (sum <= 0) return 0;

  // dL/d(sq_dist_ab) = (−t · w_ab) / Σw   →   dL/de_a = that × 2(e_a − e_b)
  p = 0;
  for (let a = 0; a < m; a++) {
    const ia = idx[a] * D_MODEL;
    for (let b = a + 1; b < m; b++) {
      const ib = idx[b] * D_MODEL;
      const coeff = ((-REPULSION_T * w[p++]) / sum) * 2;
      for (let c = 0; c < D_MODEL; c++) {
        const d = embedding[ia + c] - embedding[ib + c];
        gradEmbedding[ia + c] += coeff * d;
        gradEmbedding[ib + c] -= coeff * d;
      }
    }
  }
  return Math.log(sum / pairs);
}

export type ProgressCb = (fraction: number, message: string) => void;

export interface ScratchRunOptions {
  /** Untruncated token ids of the corpus (encoded with ITS OWN fresh vocabulary). */
  tokenIds: number[];
  epochs?: number;
  seed?: number;
  onProgress?: ProgressCb;
}

export interface ScratchRunResult {
  weights: WeightSet;
  finalLoss: number;
  epochs: number;
  nWindows: number;
}

/** Adam moments, one pair of buffers per weight tensor. */
export interface AdamState {
  m: Record<string, Float64Array>;
  v: Record<string, Float64Array>;
  t: number;
}

const ADAM_B1 = 0.9;
const ADAM_B2 = 0.999;
const ADAM_EPS = 1e-8;

/** Zeroed Adam moments + gradient buffers shaped for `ws`. */
export function newAdamState(ws: WeightSet): { adam: AdamState; grads: Record<string, Float64Array> } {
  const adam: AdamState = { m: {}, v: {}, t: 0 };
  const grads: Record<string, Float64Array> = {};
  for (const name of weightNames()) {
    adam.m[name] = new Float64Array(ws[name].length);
    adam.v[name] = new Float64Array(ws[name].length);
    grads[name] = new Float64Array(ws[name].length);
  }
  return { adam, grads };
}

export interface ScratchStepResult {
  /** Mean per-token cross-entropy over the batch. */
  ce: number;
  /** The Wang & Isola uniformity term (before its REPULSION_WEIGHT scaling). */
  uniformity: number;
  /** Global gradient norm before clipping. */
  gradNorm: number;
}

/**
 * ONE training step: forward+backward over `batch`, the uniformity term over
 * `sampleIdx`, global-norm clipping, an Adam update, and re-projection onto S².
 * Mutates `ws`, `adam` and `grads` in place.
 *
 * Factored out of `runScratchTrain` (which calls it for every batch, so the two cannot
 * drift) precisely so a single step from a FIXED weight set and FIXED sample indices
 * can be pinned against the Python backend's `geo/train.train_batch_step` — the pin
 * `scratch.ts` used to claim existed and did not.
 */
export function scratchTrainStep(
  model: GeoModel,
  ws: WeightSet,
  adam: AdamState,
  grads: Record<string, Float64Array>,
  batch: Int32Array[],
  sampleIdx: Int32Array,
  lr: number = TRAIN_LR,
  repulsionWeight: number = REPULSION_WEIGHT,
): ScratchStepResult {
  const names = weightNames();
  for (const name of names) grads[name].fill(0);
  let nValid = 0;
  for (const window of batch) {
    for (let t = 1; t < window.length; t++) if (window[t] !== PAD_ID) nValid++;
  }
  if (nValid === 0) throw invalidParam("training batch has no non-padding target tokens");

  let ceSum = 0;
  for (const window of batch) ceSum += accumulate(model, window, nValid, grads);
  const ce = ceSum / nValid;

  // Repulsion term: gradient goes straight into the embedding gradient buffer.
  const scaled = new Float64Array(ws.embedding.length);
  const uniformity = uniformityLossAndGrad(ws.embedding, sampleIdx, scaled);
  for (let i = 0; i < scaled.length; i++) grads.embedding[i] += repulsionWeight * scaled[i];

  // Global-norm clip at 1.0 — the same guard the backend uses against the
  // platform-dependent gradient blow-ups that once diverged only on Linux.
  let sq = 0;
  for (const name of names) {
    const g = grads[name];
    for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
  }
  const gradNorm = Math.sqrt(sq);
  const clip = gradNorm > 1 ? 1 / gradNorm : 1;

  adam.t++;
  const bc1 = 1 - Math.pow(ADAM_B1, adam.t);
  const bc2 = 1 - Math.pow(ADAM_B2, adam.t);
  for (const name of names) {
    const g = grads[name];
    const m = adam.m[name];
    const v = adam.v[name];
    const w = ws[name];
    for (let i = 0; i < g.length; i++) {
      const gi = g[i] * clip;
      m[i] = ADAM_B1 * m[i] + (1 - ADAM_B1) * gi;
      v[i] = ADAM_B2 * v[i] + (1 - ADAM_B2) * gi * gi;
      w[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + ADAM_EPS);
    }
  }

  // Keep the embeddings on S² (FR-103) — the whole point of this model.
  const emb = ws.embedding;
  for (let v0 = 0; v0 < VOCAB_SIZE; v0++) {
    const o = v0 * D_MODEL;
    const en = Math.hypot(emb[o], emb[o + 1], emb[o + 2]);
    if (en > 1e-12) {
      emb[o] /= en;
      emb[o + 1] /= en;
      emb[o + 2] /= en;
    }
  }
  return { ce, uniformity, gradNorm };
}

export function runScratchTrain(opts: ScratchRunOptions): ScratchRunResult {
  const epochs = Math.trunc(opts.epochs ?? SCRATCH_DEFAULT_EPOCHS);
  const seed = Math.trunc(opts.seed ?? 0);
  if (!(epochs >= 1 && epochs <= SCRATCH_MAX_EPOCHS)) {
    throw invalidParam(`epochs must be in 1..${SCRATCH_MAX_EPOCHS}, got ${opts.epochs}`);
  }
  if (opts.tokenIds.length < CONTEXT_WINDOW) {
    throw invalidParam(
      `training text is too short after tokenization (${opts.tokenIds.length} tokens; ` +
        `at least ${CONTEXT_WINDOW} are needed for one training window)`,
    );
  }

  const windows = makeWindows([...opts.tokenIds, EOS_ID], CONTEXT_WINDOW, TRAIN_WINDOW_STRIDE);
  const ws = cloneWeightSet(initWeights(seed));
  const model = new GeoModel(ws);

  const rand = sfc32(seed + 1);
  const { adam, grads } = newAdamState(ws);

  const n = windows.length;
  const batchSize = Math.min(TRAIN_BATCH_SIZE, n);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const order = shuffled(n, rand);
    let epochCe = 0;
    let batches = 0;
    for (let start = 0; start < n; start += batchSize) {
      const batch: Int32Array[] = [];
      for (let i = start; i < Math.min(start + batchSize, n); i++) batch.push(windows[order[i]]);
      if (batch.length === 0) continue;
      // Drawn here, not inside the step, so the step is reproducible from data alone
      // (the cross-language golden supplies these indices explicitly). The PRNG
      // consumption order is one shuffle per epoch then one sample per batch.
      const sampleIdx = sampleUniformityIndices(rand);
      const { ce } = scratchTrainStep(model, ws, adam, grads, batch, sampleIdx);
      if (!Number.isFinite(ce)) {
        throw computeError(`Training diverged at epoch ${epoch + 1} (non-finite loss)`);
      }
      epochCe += ce;
      batches++;
    }
    opts.onProgress?.(
      (epoch + 1) / epochs,
      `epoch ${epoch + 1}/${epochs} · loss ${(epochCe / Math.max(batches, 1)).toFixed(2)}`,
    );
  }

  const finalLoss = evalLoss(model, windows);
  if (!Number.isFinite(finalLoss)) throw computeError("training produced a non-finite loss");
  return { weights: ws, finalLoss, epochs, nWindows: n };
}

function shuffled(n: number, rand: () => number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}
