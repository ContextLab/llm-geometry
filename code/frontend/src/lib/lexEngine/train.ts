/**
 * Real from-scratch training in the browser, to the recipe in `architecture.md` §
 * Training. Nothing here is simulated: every loss is a forward pass through
 * `LexModel`, every step an AdamW update of real gradients.
 *
 *   * loss        mean cross-entropy of `logits[:, :-1]` vs `x[:, 1:]`, ignore `<pad>`
 *   * optimizer   AdamW, betas (0.9, 0.999), eps 1e-8, DECOUPLED decay applied ONLY to
 *                 the 2-D weight matrices (qkv_w, proj_w, fc1_w, fc2_w, and head_w when
 *                 untied) — not embeddings, not positions, not biases, not LayerNorm
 *                 gains. This deliberately DIFFERS from the source, which decays every
 *                 parameter; we follow the standard convention and say so in the UI.
 *   * schedule    one-cycle with `lr` as the PEAK: initial = lr/25, final = initial/1e4,
 *                 pct_start = 0.3, cosine in both phases
 *   * clipping    global L2 norm 1.0, before the step
 *   * batching    contiguous 95/5 train/val split; each batch draws B start offsets
 *                 uniformly WITH REPLACEMENT from the training span; a window is ctx+1
 *                 tokens so the last input token has a target
 *
 * Determinism: one seeded RNG drives initialization, batch selection and dropout.
 * Bit-equality with PyTorch is **not** claimed and is not achievable — platform BLAS and
 * non-portable RNG streams make a browser run and a Python run two independent runs of
 * the same recipe. The golden test pins a fixed-weights forward/backward instead.
 */

import { computeError, invalidParam } from "../geoEngine/errors";
import {
  LexModel,
  backward,
  cloneWeights,
  crossEntropy,
  decayedWeightNames,
  initWeights,
  nonFiniteWeightNames,
  sfc32,
  weightNames,
  zeroGrads,
  type LexConfig,
  type WeightSet,
} from "./model";
import { EOS_ID, LexVocab, splitLines, tokenize } from "./vocab";

// --- constants (lex/config.py) -------------------------------------------------------

export const DEFAULT_STEPS = 400;
export const MAX_STEPS = 3000;
/** The PEAK of the one-cycle schedule, not a constant learning rate. */
export const DEFAULT_LR = 3e-3;
/**
 * 16, not 32 — see the note on `DEFAULT_BATCH` in `lex/config.py`, which this mirrors.
 * With `DEFAULT_CTX = 32` a default 400-step run is ~47 s of compute rather than ~193 s.
 * Held-out loss after 400 steps goes 2.294 -> 2.333; `steps = 500` recovers it (2.258)
 * at ~56 s for anyone who moves the slider.
 */
export const DEFAULT_BATCH = 16;
export const DEFAULT_WEIGHT_DECAY = 0.01;
export const GRAD_CLIP_NORM = 1.0;
export const ONECYCLE_PCT_START = 0.3;
export const ONECYCLE_DIV_FACTOR = 25.0;
export const ONECYCLE_FINAL_DIV_FACTOR = 1e4;
export const VAL_FRACTION = 0.05;
export const DEFAULT_SEED = 0;
/** Emit a sample every N steps while training so the user watches text improve. */
export const DEFAULT_SAMPLE_EVERY = 50;

export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;
export const ADAM_EPS = 1e-8;

// --- the learning-rate schedule ------------------------------------------------------

/** Python's `round`: half-to-even, so `w` matches the backend at every step count. */
export function pyRound(x: number): number {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * One-cycle learning rate at 0-indexed step `i` of `steps`, with `peak` as the maximum.
 * Warms up cosine-wise from `peak/25` over the first 30% of the run, then anneals
 * cosine-wise to `peak/25/1e4`.
 */
export function oneCycleLr(i: number, steps: number, peak: number): number {
  const initial = peak / ONECYCLE_DIV_FACTOR;
  const final = initial / ONECYCLE_FINAL_DIV_FACTOR;
  const w = pyRound(ONECYCLE_PCT_START * steps);
  if (i < w) {
    const p = i / w;
    return initial + ((peak - initial) * (1 - Math.cos(Math.PI * p))) / 2;
  }
  const denom = steps - w;
  const p = denom > 0 ? (i - w) / denom : 1;
  return final + ((peak - final) * (1 + Math.cos(Math.PI * p))) / 2;
}

// --- the token stream ----------------------------------------------------------------

/**
 * Encode a corpus as ONE id stream, with `<eos>` closing every line that has a word.
 * The TypeScript port of `lex/train.py::token_stream`, and the contract's definition
 * (`architecture.md` § Training, "Token stream construction"):
 *
 *   > Split the text into lines. For each line with at least one word token, emit that
 *   > line's ids **followed by `<eos>`**. Blank lines emit nothing. Nothing is prepended.
 *
 * This is the ONLY way training data may be built. It used to be `vocab.encodeText`, a
 * flat tokenize with no `<eos>` at all, so the browser model never saw a line boundary
 * and could not learn verse — while the UI claimed both runtimes ran the same recipe.
 * On the shipped corpus at the `full` Dolch budget this yields exactly 19,071 ids of
 * which 3,071 are `<eos>`, the same numbers Python produces (pinned by a test).
 *
 * Nursery rhymes are line-shaped: the line, not the paragraph, is the unit the model can
 * plausibly learn to finish, which is also why generation renders `<eos>` as a line break.
 */
export function tokenStream(text: string, vocab: LexVocab): number[] {
  const ids: number[] = [];
  for (const line of splitLines(text)) {
    const lineIds = vocab.encode(tokenize(line));
    if (lineIds.length > 0) {
      for (const id of lineIds) ids.push(id);
      ids.push(EOS_ID);
    }
  }
  return ids;
}

// --- batching ------------------------------------------------------------------------

export interface TokenSplit {
  train: Int32Array;
  val: Int32Array;
}

/** Contiguous 95/5 split of the token stream. The val span is never trained on. */
export function splitTokens(tokens: ArrayLike<number>, valFraction = VAL_FRACTION): TokenSplit {
  const n = tokens.length;
  const nTrain = Math.floor(n * (1 - valFraction));
  return {
    train: Int32Array.from({ length: nTrain }, (_, i) => tokens[i]),
    val: Int32Array.from({ length: n - nTrain }, (_, i) => tokens[nTrain + i]),
  };
}

export interface Batch {
  /** (B*T) model inputs — the window's first `ctx` tokens. */
  ids: Int32Array;
  /** (B*T) targets — the window's last `ctx` tokens. */
  targets: Int32Array;
  B: number;
  T: number;
}

/**
 * `B` windows of `ctx+1` tokens, start offsets drawn uniformly WITH REPLACEMENT.
 * Sampling with replacement is the source's scheme and is what makes a "step" a fixed
 * unit of work independent of corpus length.
 */
export function sampleBatch(stream: Int32Array, ctx: number, B: number, rand: () => number): Batch {
  const span = ctx + 1;
  const maxStart = stream.length - span;
  if (maxStart < 0) {
    throw invalidParam(
      `the training text is too short: ${stream.length} tokens after tokenization, but one ` +
        `window at ctx = ${ctx} needs ${span}. Paste more text or lower ctx.`,
    );
  }
  const ids = new Int32Array(B * ctx);
  const targets = new Int32Array(B * ctx);
  for (let b = 0; b < B; b++) {
    const start = Math.min(maxStart, Math.floor(rand() * (maxStart + 1)));
    for (let t = 0; t < ctx; t++) {
      ids[b * ctx + t] = stream[start + t];
      targets[b * ctx + t] = stream[start + t + 1];
    }
  }
  return { ids, targets, B, T: ctx };
}

/** Deterministic non-overlapping windows — what validation and `evalLoss` walk. */
export function tilingBatches(stream: Int32Array, ctx: number, maxWindows = Infinity): Batch[] {
  const span = ctx + 1;
  const out: Batch[] = [];
  for (let s = 0; s + span <= stream.length && out.length < maxWindows; s += ctx) {
    const ids = new Int32Array(ctx);
    const targets = new Int32Array(ctx);
    for (let t = 0; t < ctx; t++) {
      ids[t] = stream[s + t];
      targets[t] = stream[s + t + 1];
    }
    out.push({ ids, targets, B: 1, T: ctx });
  }
  return out;
}

/** Mean token cross-entropy over deterministic windows, dropout off. */
export function evalLoss(model: LexModel, stream: Int32Array, maxWindows = 16): number {
  const batches = tilingBatches(stream, model.cfg.ctx, maxWindows);
  if (batches.length === 0) return NaN;
  let sum = 0;
  let count = 0;
  const V = model.cfg.vocabRows;
  for (const batch of batches) {
    const acts = model.forward(batch.ids, batch.B, batch.T, {});
    // crossEntropy divides by nValid; multiply back to accumulate a true token mean.
    const { ceSum, nValid } = crossEntropy(acts.logits, batch.targets, batch.B * batch.T, V);
    sum += ceSum;
    count += nValid;
  }
  return count > 0 ? sum / count : NaN;
}

// --- the optimizer -------------------------------------------------------------------

interface AdamState {
  m: Record<string, Float64Array>;
  v: Record<string, Float64Array>;
  t: number;
}

function newAdamState(cfg: LexConfig, ws: WeightSet): AdamState {
  const state: AdamState = { m: {}, v: {}, t: 0 };
  for (const name of weightNames(cfg)) {
    state.m[name] = new Float64Array(ws[name].length);
    state.v[name] = new Float64Array(ws[name].length);
  }
  return state;
}

// --- the run -------------------------------------------------------------------------

export interface TrainPoint {
  step: number;
  loss: number;
  lr: number;
  gradNorm: number;
  /** Present only on the steps validation ran. */
  valLoss?: number;
}

export type ProgressCb = (fraction: number, point: TrainPoint) => void;
export type SampleCb = (step: number, model: LexModel) => void;

export interface TrainOptions {
  cfg: LexConfig;
  /** The corpus, already encoded with the model's vocabulary. */
  tokens: ArrayLike<number>;
  steps?: number;
  lr?: number;
  batchSize?: number;
  weightDecay?: number;
  seed?: number;
  /** Continue from these weights instead of a fresh init (fine-tuning, FR-619). */
  initialWeights?: WeightSet;
  /** Validate + report every N steps (0 disables). */
  evalEvery?: number;
  sampleEvery?: number;
  onProgress?: ProgressCb;
  onSample?: SampleCb;
}

export interface TrainResult {
  weights: WeightSet;
  history: TrainPoint[];
  /** Mean CE on the held-out 5% after the last step. */
  valLoss: number;
  /** Mean CE on the training span before the first step — the "did it learn?" baseline. */
  initialTrainLoss: number;
  finalTrainLoss: number;
  steps: number;
  nTokens: number;
}

export function runTraining(opts: TrainOptions): TrainResult {
  const cfg = opts.cfg;
  const steps = Math.trunc(opts.steps ?? DEFAULT_STEPS);
  const peakLr = opts.lr ?? DEFAULT_LR;
  const batchSize = Math.trunc(opts.batchSize ?? DEFAULT_BATCH);
  const weightDecay = opts.weightDecay ?? DEFAULT_WEIGHT_DECAY;
  const seed = Math.trunc(opts.seed ?? DEFAULT_SEED);
  const evalEvery = Math.trunc(opts.evalEvery ?? Math.max(1, Math.floor(steps / 10)));
  const sampleEvery = Math.trunc(opts.sampleEvery ?? DEFAULT_SAMPLE_EVERY);

  if (!(steps >= 1 && steps <= MAX_STEPS)) throw invalidParam(`steps must be in 1..${MAX_STEPS}, got ${opts.steps}`);
  if (!(peakLr > 0 && Number.isFinite(peakLr))) throw invalidParam(`lr must be a finite number > 0, got ${opts.lr}`);
  if (!(batchSize >= 1)) throw invalidParam(`batch_size must be >= 1, got ${opts.batchSize}`);
  if (!(weightDecay >= 0)) throw invalidParam(`weight_decay must be >= 0, got ${opts.weightDecay}`);

  // A fine-tune whose base is already broken must fail HERE, naming the cause, rather
  // than a hundred lines later as "training diverged; try a lower learning rate".
  if (opts.initialWeights) {
    const bad = nonFiniteWeightNames(cfg, opts.initialWeights);
    if (bad.length > 0) {
      throw computeError(
        `cannot train from these weights: ${bad.join(", ")} contain non-finite values (NaN or ±Infinity), ` +
          "most likely from a weight edit that pushed them out of float32 range or from a diverged run. " +
          "Reset the edited tensor, or retrain from scratch.",
      );
    }
  }

  const { train, val } = splitTokens(opts.tokens);
  // Never mutate the caller's tensors: a fine-tune must leave its base model intact.
  const model = new LexModel(cfg, opts.initialWeights ? cloneWeights(opts.initialWeights) : initWeights(cfg, seed));
  const ws = model.weights;
  const names = weightNames(cfg);
  const decayed = new Set(decayedWeightNames(cfg));
  const adam = newAdamState(cfg, ws);
  const grads = zeroGrads(cfg);
  // A separate stream for data and dropout, so changing `steps` does not change the init.
  const rand = sfc32(seed + 1);

  const initialTrainLoss = evalLoss(model, train);
  const history: TrainPoint[] = [];
  const V = cfg.vocabRows;

  for (let step = 0; step < steps; step++) {
    const batch = sampleBatch(train, cfg.ctx, batchSize, rand);
    for (const name of names) grads[name].fill(0);

    const acts = model.forward(batch.ids, batch.B, batch.T, { dropout: cfg.dropout, rand });
    const { ceSum, nValid, dLogits } = crossEntropy(acts.logits, batch.targets, batch.B * batch.T, V);
    const loss = ceSum / nValid;
    if (!Number.isFinite(loss)) {
      throw computeError(`training diverged at step ${step + 1} (non-finite loss); try a lower learning rate`);
    }
    backward(model, acts, dLogits, grads);

    // Global-norm clipping at 1.0, before the step (torch clip_grad_norm_ semantics).
    let sq = 0;
    for (const name of names) {
      const g = grads[name];
      for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    }
    const gradNorm = Math.sqrt(sq);
    const clip = Math.min(1, GRAD_CLIP_NORM / (gradNorm + 1e-6));

    const lr = oneCycleLr(step, steps, peakLr);
    adam.t++;
    const bc1 = 1 - Math.pow(ADAM_BETA1, adam.t);
    const bc2 = 1 - Math.pow(ADAM_BETA2, adam.t);
    for (const name of names) {
      const p = ws[name];
      const g = grads[name];
      const m = adam.m[name];
      const v = adam.v[name];
      const decay = decayed.has(name) ? lr * weightDecay : 0;
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * clip;
        m[i] = ADAM_BETA1 * m[i] + (1 - ADAM_BETA1) * gi;
        v[i] = ADAM_BETA2 * v[i] + (1 - ADAM_BETA2) * gi * gi;
        // AdamW: decoupled decay first, then the adaptive step.
        const decayed_ = decay > 0 ? p[i] - decay * p[i] : p[i];
        p[i] = Math.fround(decayed_ - (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + ADAM_EPS));
      }
    }

    const point: TrainPoint = { step: step + 1, loss, lr, gradNorm };
    if (evalEvery > 0 && ((step + 1) % evalEvery === 0 || step === steps - 1) && val.length > cfg.ctx) {
      point.valLoss = evalLoss(model, val);
    }
    history.push(point);
    opts.onProgress?.((step + 1) / steps, point);
    if (sampleEvery > 0 && (step + 1) % sampleEvery === 0) opts.onSample?.(step + 1, model);
  }

  const finalTrainLoss = evalLoss(model, train);
  const valLoss = val.length > cfg.ctx ? evalLoss(model, val) : NaN;
  if (!Number.isFinite(finalTrainLoss)) throw computeError("training produced a non-finite loss; refusing to save");

  return {
    weights: ws,
    history,
    valLoss,
    initialTrainLoss,
    finalTrainLoss,
    steps,
    nTokens: opts.tokens.length,
  };
}

/** Guard against a corpus too small for even one window, before any work is done. */
export function assertTrainable(nTokens: number, ctx: number): void {
  const nTrain = Math.floor(nTokens * (1 - VAL_FRACTION));
  if (nTrain < ctx + 1) {
    throw invalidParam(
      `this text has ${nTokens} in-budget tokens, of which ${nTrain} are the training ` +
        `split — one window at ctx = ${ctx} needs ${ctx + 1}. Paste more text or lower ctx.`,
    );
  }
}
