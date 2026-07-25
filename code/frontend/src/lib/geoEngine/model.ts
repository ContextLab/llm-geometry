/**
 * The GeoTransformer forward pass — exact TypeScript port of
 * code/backend/src/llm_geometry/geo/model.py (+ the frozen constants from
 * geo/config.py).
 *
 * Architecture (frozen contract): d_model=3, 4 layers, 1 head, mlp_hidden=12,
 * vocab 1003, context window 50, tied unembedding (logits = h @ E^T), learned
 * absolute positional embeddings, NO layer norm, UNSCALED attention scores
 * <k_j, q_i> (no 1/sqrt(d)), pre-residual blocks:
 *   h += (softmax(q k^T + causal) @ v) @ W_O^T ;  h += gelu(h W_in + b_in) W_out + b_out
 *
 * Weights are stored as float32 (like the backend); activations accumulate in
 * float64, which keeps every traced quantity within <=1e-5 relative of torch's
 * float32 results at these tiny dimensions.
 *
 * Checkpoint JSON (exported by the backend, e.g. static-data/geo/checkpoint.json):
 * an object carrying (a) the full state_dict as nested lists under `state_dict`
 * (also accepted: `weights`, `weight_set`, `params`, or tensors at the top level),
 * (b) optionally `config` with the frozen architecture constants (validated), and
 * (c) optionally `meta` with {checkpoint_id, final_loss, coverage_uniformity,
 * field_directional_entropy, seed}. Every expected tensor must be present with the
 * exact shape below or loading throws a ComputeError naming the offender.
 */

import { computeError, invalidParam } from "./errors";
import { gelu, matmul, matmulNT, softmaxRowsCausalInPlace, toNested2 } from "./tensor";

// --- Fixed architecture (frozen contract; geo/config.py) ---------------------------
export const D_MODEL = 3;
export const N_LAYERS = 4;
export const N_HEADS = 1;
export const MLP_HIDDEN = 12;
export const VOCAB_WORDS = 1000;
export const VOCAB_SIZE = 1003;
export const CONTEXT_WINDOW = 50;
export const SEED = 0;

export const UNK_ID = 0;
export const EOS_ID = 1;
export const PAD_ID = 2;
export const UNK_TOKEN = "<unk>";
export const EOS_TOKEN = "<eos>";
export const PAD_TOKEN = "<pad>";
export const SPECIAL_TOKENS = { unk: UNK_ID, eos: EOS_ID, pad: PAD_ID } as const;

export const CORPUS_ID = "gutenberg-11-alice-in-wonderland";

// Fine-tuning limits (frozen contract).
export const FINETUNE_MAX_STEPS = 500;
export const FINETUNE_DEFAULT_STEPS = 100;
export const FINETUNE_DEFAULT_LR = 1e-2;

// The matrices addressable through the weights API (embedding ignores `layer`).
export const EDITABLE_MATRICES = ["W_Q", "W_K", "W_V", "W_O", "embedding"] as const;
export type EditableMatrix = (typeof EDITABLE_MATRICES)[number];

// --- Weight-set plumbing -----------------------------------------------------------

/** A complete parameter dict, keyed like the backend's named_parameters(). */
export type WeightSet = Record<string, Float32Array>;

const LAYER_PARAM_SHAPES: Record<string, number[]> = {
  W_Q: [D_MODEL, D_MODEL],
  W_K: [D_MODEL, D_MODEL],
  W_V: [D_MODEL, D_MODEL],
  W_O: [D_MODEL, D_MODEL],
  W_in: [D_MODEL, MLP_HIDDEN],
  b_in: [MLP_HIDDEN],
  W_out: [MLP_HIDDEN, D_MODEL],
  b_out: [D_MODEL],
};

/** name -> shape for every parameter (backend GeoTransformer.weight_names order). */
export const WEIGHT_SHAPES: ReadonlyMap<string, number[]> = (() => {
  const shapes = new Map<string, number[]>();
  shapes.set("embedding", [VOCAB_SIZE, D_MODEL]);
  shapes.set("pos_embedding", [CONTEXT_WINDOW, D_MODEL]);
  for (let i = 0; i < N_LAYERS; i++) {
    for (const name of ["W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out"]) {
      shapes.set(`layers.${i}.${name}`, LAYER_PARAM_SHAPES[name]);
    }
  }
  return shapes;
})();

export function weightNames(): string[] {
  return [...WEIGHT_SHAPES.keys()];
}

export function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Throws unless `ws` has exactly the expected tensors with the expected sizes. */
export function validateWeightSet(ws: WeightSet): void {
  const missing = weightNames().filter((n) => !(n in ws));
  const extra = Object.keys(ws).filter((n) => !WEIGHT_SHAPES.has(n));
  if (missing.length > 0 || extra.length > 0) {
    throw invalidParam(
      `Weight set mismatch (missing: ${missing.length ? missing.join(", ") : "none"}, ` +
        `extra: ${extra.length ? extra.join(", ") : "none"})`,
    );
  }
  for (const [name, shape] of WEIGHT_SHAPES) {
    if (ws[name].length !== numel(shape)) {
      throw invalidParam(
        `Weight '${name}' has ${ws[name].length} values, expected shape (${shape.join(", ")})`,
      );
    }
  }
}

export function cloneWeightSet(ws: WeightSet): WeightSet {
  const out: WeightSet = {};
  for (const [name, arr] of Object.entries(ws)) out[name] = new Float32Array(arr);
  return out;
}

// --- Checkpoint loading ------------------------------------------------------------

export interface CheckpointMeta {
  checkpoint_id: string | null;
  final_loss: number | null;
  coverage_uniformity: number | null;
  field_directional_entropy: number | null;
  seed: number;
  corpus: string;
}

export interface LoadedCheckpoint {
  weights: WeightSet;
  meta: CheckpointMeta;
}

function flattenTensor(name: string, value: unknown, shape: number[]): Float32Array {
  const out = new Float32Array(numel(shape));
  let idx = 0;
  const fill = (v: unknown, depth: number): void => {
    if (depth === shape.length) {
      if (typeof v !== "number" || !Number.isFinite(v)) {
        throw computeError(`checkpoint tensor '${name}' contains a non-finite or non-numeric entry`);
      }
      out[idx++] = Math.fround(v);
      return;
    }
    if (!Array.isArray(v) || v.length !== shape[depth]) {
      throw computeError(
        `checkpoint tensor '${name}' has the wrong shape: expected (${shape.join(", ")})`,
      );
    }
    for (const item of v) fill(item, depth + 1);
  };
  fill(value, 0);
  return out;
}

const FROZEN_CONFIG: Record<string, number | boolean> = {
  d_model: D_MODEL,
  n_layers: N_LAYERS,
  n_heads: N_HEADS,
  mlp_hidden: MLP_HIDDEN,
  vocab_size: VOCAB_SIZE,
  context_window: CONTEXT_WINDOW,
  tied_unembedding: true,
};

/** Defensive checkpoint.json loader (schema documented at the top of this file). */
export function loadCheckpoint(json: unknown): LoadedCheckpoint {
  if (typeof json === "string") json = JSON.parse(json);
  if (json === null || typeof json !== "object" || Array.isArray(json)) {
    throw computeError("checkpoint.json: expected a JSON object");
  }
  const obj = json as Record<string, unknown>;

  // Locate the state dict.
  let sd: Record<string, unknown> | null = null;
  for (const key of ["state_dict", "weights", "weight_set", "params"]) {
    const cand = obj[key];
    if (cand !== null && typeof cand === "object" && !Array.isArray(cand)) {
      sd = cand as Record<string, unknown>;
      break;
    }
  }
  if (sd === null && "embedding" in obj) sd = obj; // tensors at the top level
  if (sd === null) {
    throw computeError(
      "checkpoint.json: could not find the state dict (looked for keys " +
        "state_dict/weights/weight_set/params, or top-level tensors)",
    );
  }

  const weights: WeightSet = {};
  const missing: string[] = [];
  for (const [name, shape] of WEIGHT_SHAPES) {
    let value = sd[name];
    if (value !== undefined && value !== null && typeof value === "object" && !Array.isArray(value)) {
      // tolerate {"values": [...]} / {"data": [...]} wrappers
      const wrapped = value as Record<string, unknown>;
      value = wrapped.values ?? wrapped.data;
    }
    if (value === undefined || value === null) {
      missing.push(name);
      continue;
    }
    weights[name] = flattenTensor(name, value, shape);
  }
  if (missing.length > 0) {
    throw computeError(
      `checkpoint.json: missing tensor(s) ${missing.join(", ")} — the static export must ` +
        "contain the full GeoTransformer state_dict as nested lists",
    );
  }

  // Config sanity: if the export carries the architecture, it must match the frozen contract.
  const config = (obj.config ?? obj.model ?? null) as Record<string, unknown> | null;
  if (config !== null && typeof config === "object") {
    for (const [key, expected] of Object.entries(FROZEN_CONFIG)) {
      if (key in config && config[key] !== expected) {
        throw computeError(
          `checkpoint.json: config.${key} = ${String(config[key])} does not match the frozen ` +
            `contract value ${String(expected)}`,
        );
      }
    }
  }

  const metaSrc = (obj.meta ?? obj.metrics ?? obj.checkpoint ?? obj) as Record<string, unknown>;
  const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const meta: CheckpointMeta = {
    checkpoint_id: typeof metaSrc.checkpoint_id === "string" ? metaSrc.checkpoint_id : null,
    final_loss: num(metaSrc.final_loss),
    coverage_uniformity: num(metaSrc.coverage_uniformity),
    field_directional_entropy: num(metaSrc.field_directional_entropy),
    seed: num(metaSrc.seed) ?? SEED,
    corpus: typeof config?.corpus === "string" ? (config.corpus as string) : CORPUS_ID,
  };

  return { weights, meta };
}

// --- Layer param resolution --------------------------------------------------------

export type LayerParam = number | "full";

/** Map the contract's `layer` param (0..3 or "full") to a hidden-state index. */
export function resolveLayer(layer: LayerParam | string): number {
  if (layer === "full") return N_LAYERS - 1;
  const idx = typeof layer === "number" ? layer : Number(layer);
  if (!Number.isInteger(idx) || idx < 0 || idx >= N_LAYERS) {
    throw invalidParam(`layer must be 0..${N_LAYERS - 1} or "full", got ${JSON.stringify(layer)}`);
  }
  return idx;
}

// --- The model ---------------------------------------------------------------------

/** Per-layer activations from one forward pass (flat row-major Float64Arrays). */
export interface LayerActs {
  q: Float64Array; // (T,3)
  k: Float64Array; // (T,3)
  v: Float64Array; // (T,3)
  attention: Float64Array; // (T,T) row-stochastic, causal
  hiddenIn: Float64Array; // (T,3)
  attnCtx: Float64Array; // (T,3) attn @ v (pre-W_O; needed for backward)
  attnOut: Float64Array; // (T,3)
  h1: Float64Array; // (T,3) post-attention residual (input to the MLP)
  mlpPre: Float64Array; // (T,12) pre-GELU
  mlpAct: Float64Array; // (T,12) gelu(mlpPre)
  mlpOut: Float64Array; // (T,3)
  hiddenOut: Float64Array; // (T,3)
}

export interface SeqActs {
  T: number;
  ids: number[];
  tokenEmbeddings: Float64Array; // (T,3) E[ids] (no positional offset)
  layers: LayerActs[]; // length = layers actually run
}

export interface GeoTraceArrays {
  embeddings: number[][];
  layers: {
    layer: number;
    attention: number[][];
    q: number[][];
    k: number[][];
    v: number[][];
    hidden_in: number[][];
    attn_out: number[][];
    mlp_out: number[][];
    hidden_out: number[][];
  }[];
  probs: Float64Array; // (V,) softmax over the final position's logits
}

export class GeoModel {
  readonly ws: WeightSet;

  constructor(ws: WeightSet) {
    validateWeightSet(ws);
    this.ws = ws;
  }

  get embedding(): Float32Array {
    return this.ws["embedding"];
  }

  layerParam(layer: number, name: string): Float32Array {
    return this.ws[`layers.${layer}.${name}`];
  }

  /**
   * One causal forward pass over a single sequence (ids length 1..50), running
   * layers 0..upToLayer-1 (all four by default). Mirrors GeoTransformer._run.
   */
  forwardSeq(ids: number[], upToLayer: number = N_LAYERS): SeqActs {
    const T = ids.length;
    if (T < 1 || T > CONTEXT_WINDOW) {
      throw invalidParam(`sequence length must be in 1..${CONTEXT_WINDOW}, got ${T}`);
    }
    const E = this.embedding;
    const pos = this.ws["pos_embedding"];

    const tok = new Float64Array(T * D_MODEL);
    const h = new Float64Array(T * D_MODEL);
    for (let t = 0; t < T; t++) {
      const id = ids[t];
      if (!Number.isInteger(id) || id < 0 || id >= VOCAB_SIZE) {
        throw invalidParam(`token id ${id} at position ${t} is out of range 0..${VOCAB_SIZE - 1}`);
      }
      for (let c = 0; c < D_MODEL; c++) {
        tok[t * D_MODEL + c] = E[id * D_MODEL + c];
        h[t * D_MODEL + c] = E[id * D_MODEL + c] + pos[t * D_MODEL + c];
      }
    }

    const layers: LayerActs[] = [];
    for (let l = 0; l < upToLayer; l++) {
      const hiddenIn = new Float64Array(h);
      const q = matmulNT(h, this.layerParam(l, "W_Q"), T, D_MODEL, D_MODEL);
      const k = matmulNT(h, this.layerParam(l, "W_K"), T, D_MODEL, D_MODEL);
      const v = matmulNT(h, this.layerParam(l, "W_V"), T, D_MODEL, D_MODEL);
      // Unscaled scores <k_j, q_i> with the causal mask, then row softmax.
      const attention = matmulNT(q, k, T, D_MODEL, T);
      softmaxRowsCausalInPlace(attention, T);
      const attnCtx = matmul(attention, v, T, T, D_MODEL);
      const attnOut = matmulNT(attnCtx, this.layerParam(l, "W_O"), T, D_MODEL, D_MODEL);
      for (let i = 0; i < h.length; i++) h[i] += attnOut[i];
      const h1 = new Float64Array(h);
      const mlpPre = matmul(h, this.layerParam(l, "W_in"), T, D_MODEL, MLP_HIDDEN);
      const bIn = this.layerParam(l, "b_in");
      for (let t = 0; t < T; t++) {
        for (let j = 0; j < MLP_HIDDEN; j++) mlpPre[t * MLP_HIDDEN + j] += bIn[j];
      }
      const mlpAct = new Float64Array(T * MLP_HIDDEN);
      for (let i = 0; i < mlpPre.length; i++) mlpAct[i] = gelu(mlpPre[i]);
      const mlpOut = matmul(mlpAct, this.layerParam(l, "W_out"), T, MLP_HIDDEN, D_MODEL);
      const bOut = this.layerParam(l, "b_out");
      for (let t = 0; t < T; t++) {
        for (let c = 0; c < D_MODEL; c++) mlpOut[t * D_MODEL + c] += bOut[c];
      }
      for (let i = 0; i < h.length; i++) h[i] += mlpOut[i];
      const hiddenOut = new Float64Array(h);
      layers.push({ q, k, v, attention, hiddenIn, attnCtx, attnOut, h1, mlpPre, mlpAct, mlpOut, hiddenOut });
    }
    return { T, ids: [...ids], tokenEmbeddings: tok, layers };
  }

  /** Tied unembedding: logits = h @ E^T ((T,3) -> (T,V)); logit-lens readout. */
  readout(h: Float64Array, T: number): Float64Array {
    return matmulNT(h, this.embedding, T, D_MODEL, VOCAB_SIZE);
  }

  /** Logits for a single 3-vector (one position). */
  readoutOne(h3: Float64Array, off = 0): Float64Array {
    const E = this.embedding;
    const logits = new Float64Array(VOCAB_SIZE);
    for (let w = 0; w < VOCAB_SIZE; w++) {
      logits[w] =
        h3[off] * E[w * D_MODEL] + h3[off + 1] * E[w * D_MODEL + 1] + h3[off + 2] * E[w * D_MODEL + 2];
    }
    return logits;
  }

  /** Everything the /api/geo/trace contract needs (backend forward_trace). */
  forwardTrace(ids: number[]): GeoTraceArrays {
    if (ids.length === 0) throw invalidParam("forward_trace requires at least one token");
    const run = this.forwardSeq(ids);
    const T = run.T;
    const layers = run.layers.map((tr, i) => ({
      layer: i,
      attention: toNested2(tr.attention, T, T),
      q: toNested2(tr.q, T, D_MODEL),
      k: toNested2(tr.k, T, D_MODEL),
      v: toNested2(tr.v, T, D_MODEL),
      hidden_in: toNested2(tr.hiddenIn, T, D_MODEL),
      attn_out: toNested2(tr.attnOut, T, D_MODEL),
      mlp_out: toNested2(tr.mlpOut, T, D_MODEL),
      hidden_out: toNested2(tr.hiddenOut, T, D_MODEL),
    }));
    const finalHidden = run.layers[N_LAYERS - 1].hiddenOut;
    const logits = this.readoutOne(finalHidden, (T - 1) * D_MODEL);
    // softmax in place -> probs
    const probs = logits;
    let max = -Infinity;
    for (let i = 0; i < probs.length; i++) if (probs[i] > max) max = probs[i];
    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
      probs[i] = Math.exp(probs[i] - max);
      sum += probs[i];
    }
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
    return { embeddings: toNested2(run.tokenEmbeddings, T, D_MODEL), layers, probs };
  }
}
