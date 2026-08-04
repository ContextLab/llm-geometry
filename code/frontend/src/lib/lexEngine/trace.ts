/**
 * A traced forward pass for the Lexicon Lab (US-7).
 *
 * `LexModel.forward` already computes and keeps every intermediate the backward pass
 * needs — attention weights, the residual stream after each branch, the final LayerNorm
 * output and the logits. This module does not re-implement any of it. It calls the SAME
 * `forward`, in eval mode, and re-presents what came back as a sequence of stages you can
 * step through. That is deliberate and load-bearing:
 *
 *   * the trace cannot drift from the model, because it IS the model's own activations;
 *   * `trace.logits` is the array `forward` returned, so the traced logits and the normal
 *     logits are equal to the bit, not merely to a tolerance (a unit test asserts this at
 *     1e-9 against an independently-run `forward`);
 *   * nothing here is illustrative. With no corpus and therefore no model there is no
 *     trace at all, and an untrained model's trace is labelled as an untrained model's.
 *
 * ## The logit lens, and its honest caveat
 *
 * At each stage we read the residual stream out through the model's **final** LayerNorm
 * and its readout matrix. At the last layer's output that is exactly what the model does,
 * so the readout there is the model's real distribution. At every EARLIER stage it is an
 * approximation: those hidden states never pass through the remaining layers, and
 * `lnf_g`/`lnf_b` were fitted to the statistics of the final residual stream, not to the
 * statistics of layer 0's. `LensReadout.exact` marks which case a stage is, and the UI
 * must not claim an intermediate readout is what the model would predict.
 *
 * Distributions here are a plain softmax over the real logits: no temperature, and none
 * of generation's `<unk>`/`<bos>`/`<pad>` masking. That masking belongs to the sampler,
 * not to the model, and hiding those rows here would misreport what the model believes.
 */

import { invalidParam } from "../geoEngine/errors";
import { LAYER_NORM_EPS, LexModel } from "./model";
import { BOS_ID, LexVocab, SPECIAL_TOKENS, tokenize, UNK_ID } from "./vocab";

/** How many next-token candidates a stage reports. */
export const DEFAULT_TRACE_TOPK = 8;

export type StageKind = "embed" | "attention" | "mlp" | "readout";

export interface TraceToken {
  position: number;
  id: number;
  /** The word as the tokenizer produced it, or the special's own name. */
  word: string;
  /** False when the budget cannot express this word — it entered the model as `<unk>`. */
  inBudget: boolean;
  /** True for the prepended `<bos>` (and for any other reserved row). */
  special: boolean;
}

export interface LensReadout {
  ids: number[];
  words: string[];
  probs: number[];
  /**
   * True only where the readout is applied exactly where the model applies it — the last
   * layer's output and the readout stage. Everywhere else this is the logit lens: an
   * approximation, because the state has not been through the remaining layers and the
   * final LayerNorm was not fitted to it.
   */
  exact: boolean;
}

export interface TraceStage {
  index: number;
  kind: StageKind;
  /** The layer this stage belongs to, or null for the embedding and the readout. */
  layer: number | null;
  label: string;
  /** One sentence naming the tensor this stage stopped at. */
  detail: string;
  /** L2 norm of the residual stream at every position, at this point in the pass. */
  residualNorm: number[];
  /** `nHeads` causal `T × T` matrices — attention stages only. Rows attend to columns. */
  attention: number[][][] | null;
  /** Next-token candidates read out from the LAST position at this point in the pass. */
  lens: LensReadout;
}

export interface LexTrace {
  tokens: TraceToken[];
  /** Sequence length actually run (after the context window is applied). */
  T: number;
  nLayers: number;
  nHeads: number;
  dModel: number;
  vocabRows: number;
  stages: TraceStage[];
  /** The model's real logits, `(T, V)` row-major — `forward`'s own array. */
  logits: Float64Array;
  /** Prompt tokens the budget could not express; they entered the model as `<unk>`. */
  unkCount: number;
  /** Those words, distinct, in order of first appearance. */
  unkWords: string[];
  /** True when the prompt was longer than `ctx` and its earliest tokens were dropped. */
  truncated: boolean;
  droppedTokens: number;
}

export interface TraceOptions {
  /** Free text, tokenized with the model's own vocabulary. */
  prompt?: string;
  /** Candidates per stage. */
  topK?: number;
}

const SPECIALS = new Set<string>(SPECIAL_TOKENS);

/** Row-wise LayerNorm with the model's affine parameters (eps matches `model.ts`). */
function layerNormRow(h: Float64Array, off: number, d: number, g: Float32Array, b: Float32Array): Float64Array {
  let mu = 0;
  for (let c = 0; c < d; c++) mu += h[off + c];
  mu /= d;
  let va = 0;
  for (let c = 0; c < d; c++) {
    const z = h[off + c] - mu;
    va += z * z;
  }
  va /= d;
  const r = 1 / Math.sqrt(va + LAYER_NORM_EPS);
  const out = new Float64Array(d);
  for (let c = 0; c < d; c++) out[c] = g[c] * ((h[off + c] - mu) * r) + b[c];
  return out;
}

/** L2 norm of every position's hidden state. */
export function residualNorms(h: Float64Array, T: number, d: number): number[] {
  const out: number[] = [];
  for (let t = 0; t < T; t++) {
    let s = 0;
    const o = t * d;
    for (let c = 0; c < d; c++) s += h[o + c] * h[o + c];
    out.push(Math.sqrt(s));
  }
  return out;
}

/** softmax over `logits`, then the `k` largest — the panel's per-stage readout. */
export function topKFromLogits(
  logits: Float64Array,
  vocab: LexVocab,
  k: number,
  exact: boolean,
): LensReadout {
  const V = logits.length;
  let max = -Infinity;
  for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
  const probs = new Float64Array(V);
  let sum = 0;
  for (let i = 0; i < V; i++) {
    const e = Math.exp(logits[i] - max);
    probs[i] = e;
    sum += e;
  }
  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b);
  const take = order.slice(0, Math.max(1, Math.min(k, V)));
  return {
    ids: take,
    words: vocab.decode(take),
    probs: take.map((i) => probs[i] / sum),
    exact,
  };
}

/**
 * The traced forward pass.
 *
 * Runs the real model in eval mode (no dropout — a trace you can step through must be the
 * same trace twice) and returns its intermediates, stage by stage:
 *
 *   embed → (layer 0: attention, MLP) → … → (layer L−1: attention, MLP) → readout
 */
export function traceForward(model: LexModel, vocab: LexVocab, opts: TraceOptions = {}): LexTrace {
  if (vocab.rows !== model.cfg.vocabRows) {
    throw invalidParam(
      `this vocabulary has ${vocab.rows} rows but the model has ${model.cfg.vocabRows} — ` +
        "a model can only be traced with the vocabulary its ids mean something in",
    );
  }
  const topK = Math.trunc(opts.topK ?? DEFAULT_TRACE_TOPK);
  if (!(topK >= 1)) throw invalidParam(`topK must be >= 1, got ${opts.topK}`);

  const { dModel: d, nHeads: H, nLayers: L, ctx, vocabRows: V } = model.cfg;

  // A prompt is tokenized like any other text, so out-of-budget words become <unk> —
  // counted and named here so the UI can show it rather than swallow it (FR-604).
  const words = tokenize(opts.prompt ?? "");
  const unkWords: string[] = [];
  for (const w of words) if (!vocab.has(w) && !unkWords.includes(w)) unkWords.push(w);
  const unkCount = words.reduce((n, w) => (vocab.has(w) ? n : n + 1), 0);

  // `<bos>` leads, exactly as generation does — the trace must show the sequence the
  // model is really given, not a tidied-up version of the prompt.
  const full: { id: number; word: string }[] = [{ id: BOS_ID, word: SPECIAL_TOKENS[BOS_ID] }];
  for (const w of words) full.push({ id: vocab.stoi(w), word: w });

  const dropped = Math.max(0, full.length - ctx);
  const kept = full.slice(dropped);
  const T = kept.length;
  const tokens: TraceToken[] = kept.map((t, i) => ({
    position: i,
    id: t.id,
    word: t.word,
    inBudget: t.id !== UNK_ID,
    special: SPECIALS.has(t.word),
  }));

  const ids = Int32Array.from(kept.map((t) => t.id));
  // Eval mode: no dropout, so stepping the same prompt twice gives the same trace.
  const acts = model.forward(ids, 1, T, {});

  const gF = model.weights.lnf_g;
  const bF = model.weights.lnf_b;
  const head = model.headWeight;

  /** The logit lens: final LayerNorm + readout applied to the LAST position of `h`. */
  const lensOf = (h: Float64Array, exact: boolean): LensReadout => {
    const hn = layerNormRow(h, (T - 1) * d, d, gF, bF);
    const logits = new Float64Array(V);
    for (let w = 0; w < V; w++) {
      let s = 0;
      const o = w * d;
      for (let c = 0; c < d; c++) s += hn[c] * head[o + c];
      logits[w] = s;
    }
    return topKFromLogits(logits, vocab, topK, exact);
  };

  const stages: TraceStage[] = [];

  // Stage 0 — the input embedding. `forward` overwrites its own `h`, so recompute it:
  // with dropout off this is exactly the tensor the first layer was handed.
  const h0 = new Float64Array(T * d);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    const e = ids[t] * d;
    const q = t * d;
    for (let c = 0; c < d; c++) h0[o + c] = model.weights.embed[e + c] + model.weights.pos[q + c];
  }
  stages.push({
    index: 0,
    kind: "embed",
    layer: null,
    label: "embed + position",
    detail: "h = embed[x] + pos[:T] — before any layer has run",
    residualNorm: residualNorms(h0, T, d),
    attention: null,
    lens: lensOf(h0, false),
  });

  for (let l = 0; l < L; l++) {
    const la = acts.layers[l];

    // Attention weights for this layer, one T×T matrix per head (B = 1).
    const attention: number[][][] = [];
    for (let hh = 0; hh < H; hh++) {
      const block = hh * T * T;
      const m: number[][] = [];
      for (let i = 0; i < T; i++) {
        const row: number[] = [];
        for (let j = 0; j < T; j++) row.push(la.attn[block + i * T + j]);
        m.push(row);
      }
      attention.push(m);
    }

    stages.push({
      index: stages.length,
      kind: "attention",
      layer: l,
      label: `layer ${l} · attention`,
      detail: "h ← h + (A v) W_projᵀ + b_proj — the attention residual, no dropout on this branch",
      residualNorm: residualNorms(la.hMid, T, d),
      attention,
      lens: lensOf(la.hMid, false),
    });

    // The last layer's output IS the final LayerNorm's input, so the lens is exact there.
    const lastLayer = l === L - 1;
    stages.push({
      index: stages.length,
      kind: "mlp",
      layer: l,
      label: `layer ${l} · MLP`,
      detail: `h ← h + gelu(m W₁ᵀ + b₁) W₂ᵀ + b₂ — the layer's output (d → ${4 * d} → d)`,
      residualNorm: residualNorms(la.hOut, T, d),
      attention: null,
      lens: lensOf(la.hOut, lastLayer),
    });
  }

  // The readout the model actually performs — taken from `forward`'s own logits, so this
  // stage cannot disagree with the model even by a rounding step.
  stages.push({
    index: stages.length,
    kind: "readout",
    layer: null,
    label: "final LayerNorm → readout",
    detail: "logits = layernorm(h) · head_wᵀ (no bias) — the model's real next-token distribution",
    residualNorm: residualNorms(acts.hF, T, d),
    attention: null,
    lens: topKFromLogits(acts.logits.slice((T - 1) * V, T * V), vocab, topK, true),
  });

  return {
    tokens,
    T,
    nLayers: L,
    nHeads: H,
    dModel: d,
    vocabRows: V,
    stages,
    logits: acts.logits,
    unkCount,
    unkWords,
    truncated: dropped > 0,
    droppedTokens: dropped,
  };
}
