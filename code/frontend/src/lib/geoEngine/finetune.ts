/**
 * Real SGD fine-tuning of the GeoTransformer — TypeScript port of
 * code/backend/src/llm_geometry/geo/finetune.py (+ make_windows / eval_loss from
 * geo/train.py), with a hand-derived backward pass through the whole model
 * (embedding lookup + tied unembedding, learned positions, unscaled attention,
 * exact-erf GELU MLP, pre-residual no-norm blocks).
 *
 * Matches the backend exactly in: window building (window 50, stride 25, <eos>
 * appended, short streams padded with <pad>), the cross-entropy objective
 * (ignore_index = pad, mean over non-pad targets across the whole batch),
 * gradient clipping (global L2 norm, max 1.0, torch's coef = 1/(norm+1e-6)),
 * plain SGD steps, and per-step embedding row renormalization (S^2 invariant,
 * clamp_min 1e-12).
 *
 * DOCUMENTED DIVERGENCE (RNG): the backend shuffles batch order with
 * torch.randperm(generator seeded seed+1), whose Mersenne-twister-derived stream
 * is not portable to JS. Static-mode fine-tuning uses its own deterministic PRNG
 * (sfc32 seeded from seed+1) for the Fisher-Yates shuffle. Consequently the loss
 * TRAJECTORY differs in detail from the backend run; golden tests assert the
 * trajectory's properties instead of bit-equality: loss_before matches the
 * backend (it is RNG-free), loss_after < loss_before, and loss_after lands within
 * 15% of the backend's for the same text/steps/lr. The minted weights_token is a
 * content hash of the resulting weights (self-consistent, like every token).
 */

import { computeError, invalidParam } from "./errors";
import {
  CONTEXT_WINDOW,
  D_MODEL,
  EOS_ID,
  FINETUNE_MAX_STEPS,
  GeoModel,
  MLP_HIDDEN,
  N_LAYERS,
  PAD_ID,
  VOCAB_SIZE,
  cloneWeightSet,
  weightNames,
  type WeightSet,
} from "./model";
import { geluPrime, logSumExp, matmul, matmulNT, matmulTN } from "./tensor";

const f32 = Math.fround;

// --- data (geo/train.py) -----------------------------------------------------------

/**
 * (N, window+1) sliding windows: columns [:-1] are inputs, [1:] are targets.
 * A stream shorter than window+1 becomes a single <pad>-padded window.
 */
export function makeWindows(stream: number[], window = CONTEXT_WINDOW, stride = 1): Int32Array[] {
  const span = window + 1;
  if (stream.length < span) {
    const padded = new Int32Array(span).fill(PAD_ID);
    padded.set(stream);
    return [padded];
  }
  const windows: Int32Array[] = [];
  for (let s = 0; s + span <= stream.length; s += stride) {
    windows.push(Int32Array.from(stream.slice(s, s + span)));
  }
  return windows;
}

// --- losses ------------------------------------------------------------------------

/**
 * Sum of -log p(target) over non-pad targets of one window + the non-pad count.
 * logits: (T,V) for inputs window[0..T-1]; targets are window[1..T].
 */
function windowCeSum(logits: Float64Array, window: Int32Array, T: number): { sum: number; n: number } {
  let sum = 0;
  let n = 0;
  for (let t = 0; t < T; t++) {
    const target = window[t + 1];
    if (target === PAD_ID) continue;
    const lse = logSumExp(logits, t * VOCAB_SIZE, VOCAB_SIZE);
    sum += lse - logits[t * VOCAB_SIZE + target];
    n++;
  }
  return { sum, n };
}

/** Mean token cross-entropy (nats) over all windows (geo/train.py eval_loss). */
export function evalLoss(model: GeoModel, windows: Int32Array[]): number {
  let total = 0;
  let count = 0;
  for (const window of windows) {
    const T = window.length - 1;
    const ids = Array.from(window.subarray(0, T));
    const acts = model.forwardSeq(ids);
    const logits = model.readout(acts.layers[N_LAYERS - 1].hiddenOut, T);
    const { sum, n } = windowCeSum(logits, window, T);
    total += sum;
    count += n;
  }
  if (count === 0) throw computeError("eval_loss: no non-pad tokens to evaluate");
  return total / count;
}

// --- deterministic shuffle RNG -----------------------------------------------------

/** sfc32: small, fast, deterministic PRNG (see the RNG divergence note above). */
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

function shuffledOrder(n: number, rand: () => number): Int32Array {
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  return order;
}

// --- gradients ---------------------------------------------------------------------

type GradSet = Record<string, Float64Array>;

function zeroGrads(): GradSet {
  const grads: GradSet = {};
  for (const name of weightNames()) grads[name] = new Float64Array(0);
  return grads;
}

function grad(grads: GradSet, name: string, size: number): Float64Array {
  if (grads[name].length !== size) grads[name] = new Float64Array(size);
  return grads[name];
}

/**
 * Accumulate gradients for one window into `grads`, given `nValid` (the number of
 * non-pad targets across the WHOLE batch — torch's cross_entropy averages over
 * all of them at once). Returns this window's summed CE (for the batch loss).
 */
export function accumulateWindowGrads(model: GeoModel, window: Int32Array, nValid: number, grads: GradSet): number {
  const T = window.length - 1;
  const ids = Array.from(window.subarray(0, T));
  const acts = model.forwardSeq(ids);
  const E = model.embedding;
  const finalH = acts.layers[N_LAYERS - 1].hiddenOut; // (T,3)
  const logits = model.readout(finalH, T); // (T,V)

  // dlogits = (softmax - onehot) / nValid on non-pad target rows; 0 on pad rows.
  const dLogits = new Float64Array(T * VOCAB_SIZE);
  let ceSum = 0;
  for (let t = 0; t < T; t++) {
    const target = window[t + 1];
    if (target === PAD_ID) continue;
    const off = t * VOCAB_SIZE;
    const lse = logSumExp(logits, off, VOCAB_SIZE);
    ceSum += lse - logits[off + target];
    for (let w = 0; w < VOCAB_SIZE; w++) {
      dLogits[off + w] = Math.exp(logits[off + w] - lse) / nValid;
    }
    dLogits[off + target] -= 1 / nValid;
  }

  // Tied unembedding: logits = h E^T  =>  dh = dlogits @ E ; dE += dlogits^T @ h.
  const dH = matmul(dLogits, E, T, VOCAB_SIZE, D_MODEL);
  const dE = grad(grads, "embedding", VOCAB_SIZE * D_MODEL);
  for (let t = 0; t < T; t++) {
    const off = t * VOCAB_SIZE;
    const h0 = finalH[t * D_MODEL];
    const h1 = finalH[t * D_MODEL + 1];
    const h2 = finalH[t * D_MODEL + 2];
    if (window[t + 1] === PAD_ID) continue; // dLogits row is all zero
    for (let w = 0; w < VOCAB_SIZE; w++) {
      const g = dLogits[off + w];
      if (g === 0) continue;
      dE[w * D_MODEL] += g * h0;
      dE[w * D_MODEL + 1] += g * h1;
      dE[w * D_MODEL + 2] += g * h2;
    }
  }

  // Backward through the layers.
  let dh = dH; // gradient wrt the layer's hidden_out
  for (let l = N_LAYERS - 1; l >= 0; l--) {
    const la = acts.layers[l];
    const WOut = model.layerParam(l, "W_out");
    const WIn = model.layerParam(l, "W_in");
    const WO = model.layerParam(l, "W_O");
    const WQ = model.layerParam(l, "W_Q");
    const WK = model.layerParam(l, "W_K");
    const WV = model.layerParam(l, "W_V");

    // MLP: h2 = h1 + gelu(h1 W_in + b_in) W_out + b_out
    const dMlpOut = dh; // (T,3)
    const dbOut = grad(grads, `layers.${l}.b_out`, D_MODEL);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < D_MODEL; c++) dbOut[c] += dMlpOut[t * D_MODEL + c];
    }
    const dWOut = matmulTN(la.mlpAct, dMlpOut, T, MLP_HIDDEN, D_MODEL); // (12,3)
    accumInto(grad(grads, `layers.${l}.W_out`, MLP_HIDDEN * D_MODEL), dWOut);
    const dAct = matmulNT(dMlpOut, WOut, T, D_MODEL, MLP_HIDDEN); // (T,12)
    const dPre = new Float64Array(T * MLP_HIDDEN);
    for (let i = 0; i < dPre.length; i++) dPre[i] = dAct[i] * geluPrime(la.mlpPre[i]);
    const dbIn = grad(grads, `layers.${l}.b_in`, MLP_HIDDEN);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < MLP_HIDDEN; j++) dbIn[j] += dPre[t * MLP_HIDDEN + j];
    }
    const dWIn = matmulTN(la.h1, dPre, T, D_MODEL, MLP_HIDDEN); // (3,12)
    accumInto(grad(grads, `layers.${l}.W_in`, D_MODEL * MLP_HIDDEN), dWIn);
    const dH1 = matmulNT(dPre, WIn, T, MLP_HIDDEN, D_MODEL); // dPre @ W_in^T (T,3)
    for (let i = 0; i < dH1.length; i++) dH1[i] += dh[i]; // + residual path

    // Attention: h1 = hIn + (attn @ v) @ W_O^T
    const dAttnOut = dH1; // (T,3)
    // attn_out = c @ W_O^T  =>  dW_O[a,b] = sum_i dAttnOut[i,a] * c[i,b] = (dAttnOut^T @ c)[a,b]
    const dWO = matmulTN(dAttnOut, la.attnCtx, T, D_MODEL, D_MODEL);
    accumInto(grad(grads, `layers.${l}.W_O`, D_MODEL * D_MODEL), dWO);
    const dCtx = matmul(dAttnOut, WO, T, D_MODEL, D_MODEL); // (T,3)
    const dAttn = matmulNT(dCtx, la.v, T, D_MODEL, T); // (T,T)
    const dV = matmulTN(la.attention, dCtx, T, T, D_MODEL); // attn^T @ dCtx (T,3)
    // Softmax backward per causal row: ds = a * (da - sum_j a_j da_j).
    const dScores = new Float64Array(T * T);
    for (let i = 0; i < T; i++) {
      let s = 0;
      for (let j = 0; j <= i; j++) s += la.attention[i * T + j] * dAttn[i * T + j];
      for (let j = 0; j <= i; j++) {
        dScores[i * T + j] = la.attention[i * T + j] * (dAttn[i * T + j] - s);
      }
    }
    const dQ = matmul(dScores, la.k, T, T, D_MODEL); // (T,3)
    const dK = matmulTN(dScores, la.q, T, T, D_MODEL); // dScores^T @ q (T,3)

    // Projections: q = hIn W_Q^T => dW_Q += dq^T hIn ; dhIn += dq @ W_Q. Same K,V.
    const dHIn = dH1; // reuse: residual path grad wrt hIn equals dH1, add projections below
    addProjGrads(grads, `layers.${l}.W_Q`, dQ, la.hiddenIn, WQ, dHIn, T);
    addProjGrads(grads, `layers.${l}.W_K`, dK, la.hiddenIn, WK, dHIn, T);
    addProjGrads(grads, `layers.${l}.W_V`, dV, la.hiddenIn, WV, dHIn, T);
    dh = dHIn;
  }

  // Input: h0 = E[ids] + pos[:T]
  const dPos = grad(grads, "pos_embedding", CONTEXT_WINDOW * D_MODEL);
  for (let t = 0; t < T; t++) {
    const id = ids[t];
    for (let c = 0; c < D_MODEL; c++) {
      dE[id * D_MODEL + c] += dh[t * D_MODEL + c];
      dPos[t * D_MODEL + c] += dh[t * D_MODEL + c];
    }
  }
  return ceSum;
}

function accumInto(target: Float64Array, src: Float64Array): void {
  for (let i = 0; i < target.length; i++) target[i] += src[i];
}

/** For y = x W^T: dW += dy^T @ x and dx += dy @ W. */
function addProjGrads(
  grads: GradSet,
  name: string,
  dY: Float64Array,
  x: Float64Array,
  W: Float32Array,
  dX: Float64Array,
  T: number,
): void {
  const dW = grad(grads, name, D_MODEL * D_MODEL);
  const contrib = matmulTN(dY, x, T, D_MODEL, D_MODEL); // dY^T @ x -> dW[a,b]
  accumInto(dW, contrib);
  const dxContrib = matmul(dY, W, T, D_MODEL, D_MODEL); // dY @ W
  for (let i = 0; i < dX.length; i++) dX[i] += dxContrib[i];
}

// --- the fine-tune loop ------------------------------------------------------------

export type ProgressCb = (fraction: number, message: string) => void;

export interface FinetuneRunOptions {
  baseWeights: WeightSet;
  /** Untruncated token ids of the fine-tuning text (tokenizer.encodeStream). */
  tokenIds: number[];
  steps: number;
  lr: number;
  seed?: number;
  onProgress?: ProgressCb;
}

export interface FinetuneRunResult {
  weights: WeightSet;
  lossBefore: number;
  lossAfter: number;
}

/**
 * SGD fine-tune from a copy of `baseWeights` (never mutated). Mirrors
 * geo/finetune.py: window stride 25 with <eos> appended, batch min(32, n),
 * reshuffle when the cursor runs out, grad clip at global norm 1.0, per-step
 * embedding renormalization, loud failure on non-finite loss.
 */
export function runFinetune(opts: FinetuneRunOptions): FinetuneRunResult {
  const { baseWeights, tokenIds, onProgress } = opts;
  const steps = Math.trunc(opts.steps);
  const lr = opts.lr;
  const seed = Math.trunc(opts.seed ?? 0);
  if (!(steps >= 1 && steps <= FINETUNE_MAX_STEPS)) {
    throw invalidParam(`steps must be in 1..${FINETUNE_MAX_STEPS}, got ${opts.steps}`);
  }
  if (!(lr > 0)) throw invalidParam(`lr must be > 0, got ${opts.lr}`);
  if (tokenIds.length < 2) {
    throw invalidParam("fine-tuning text is too short after tokenization (need at least 2 tokens)");
  }

  const windows = makeWindows([...tokenIds, EOS_ID], CONTEXT_WINDOW, 25);
  const ws = cloneWeightSet(baseWeights);
  const model = new GeoModel(ws);
  const lossBefore = evalLoss(model, windows);

  const rand = sfc32(seed + 1);
  const n = windows.length;
  const batchSize = Math.min(32, n);
  let order = shuffledOrder(n, rand);
  let cursor = 0;
  const grads = zeroGrads();
  const names = weightNames();

  for (let step = 0; step < steps; step++) {
    if (cursor + batchSize > n) {
      order = shuffledOrder(n, rand);
      cursor = 0;
    }
    const batch: Int32Array[] = [];
    for (let i = 0; i < batchSize; i++) batch.push(windows[order[cursor + i]]);
    cursor += batchSize;

    // zero grads
    for (const name of names) grads[name].fill(0);
    // count non-pad targets across the batch (torch CE averages over all at once)
    let nValid = 0;
    for (const window of batch) {
      for (let t = 1; t < window.length; t++) if (window[t] !== PAD_ID) nValid++;
    }
    if (nValid === 0) {
      // Unreachable with >=2 real tokens + <eos>; the backend would surface a nan.
      throw computeError("fine-tuning batch contained no non-pad targets");
    }
    let ceSum = 0;
    for (const window of batch) ceSum += accumulateWindowGrads(model, window, nValid, grads);
    const loss = ceSum / nValid;

    // Global-norm gradient clipping at 1.0 (torch clip_grad_norm_ semantics).
    let sq = 0;
    for (const name of names) {
      const g = grads[name];
      for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    }
    const totalNorm = Math.sqrt(sq);
    const clipCoef = Math.min(1, 1.0 / (totalNorm + 1e-6));

    // SGD step (params stay float32 like torch).
    for (const name of names) {
      const p = ws[name];
      const g = grads[name];
      for (let i = 0; i < p.length; i++) p[i] = f32(p[i] - lr * clipCoef * g[i]);
    }
    // Preserve the S^2 embedding invariant (row renorm, clamp_min 1e-12).
    const E = ws["embedding"];
    for (let r = 0; r < VOCAB_SIZE; r++) {
      const x = E[r * D_MODEL];
      const y = E[r * D_MODEL + 1];
      const z = E[r * D_MODEL + 2];
      const norm = Math.max(Math.sqrt(x * x + y * y + z * z), 1e-12);
      E[r * D_MODEL] = f32(x / norm);
      E[r * D_MODEL + 1] = f32(y / norm);
      E[r * D_MODEL + 2] = f32(z / norm);
    }

    if (!Number.isFinite(loss)) {
      throw computeError(
        `fine-tuning diverged (non-finite loss at step ${step + 1}); try fewer steps or a lower learning rate`,
      );
    }
    if (onProgress && (step + 1) % 10 === 0) {
      onProgress((step + 1) / steps, `step ${step + 1}/${steps} · loss ${loss.toFixed(2)}`);
    }
  }

  const lossAfter = evalLoss(model, windows);
  if (!Number.isFinite(lossBefore) || !Number.isFinite(lossAfter)) {
    throw computeError("fine-tuning produced a non-finite loss; refusing to save");
  }
  return { weights: ws, lossBefore, lossAfter };
}
