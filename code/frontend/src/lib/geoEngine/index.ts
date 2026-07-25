/**
 * GeoEngine — the in-browser GeoTransformer backend for feature 003's static
 * build. Exposes the same operations (and the same contract-shaped results) as
 * the frozen /api/geo/* routes, computed natively in TypeScript from the
 * backend-exported checkpoint + vocab. A future staticClient wraps this class to
 * implement the dataClient interface with the Python backend absent.
 *
 * Construction:
 *   const engine = GeoEngine.fromAssets(checkpointJson, vocabJson);
 * where checkpointJson is the parsed static-data/geo/checkpoint.json (schema
 * documented in model.ts loadCheckpoint) and vocabJson the parsed vocab.json
 * (tokenizer.py to_json()). Both are validated on load; missing tensors or a
 * checkpoint whose content hash disagrees with its recorded checkpoint_id throw.
 *
 * Statefulness: minted weight sets (POST /weights, fine-tunes) live in an
 * in-memory Map keyed by their content-hash token, with the same per-matrix
 * source sidecars the backend keeps. Persistence across page reloads (e.g.
 * sessionStorage) is the staticClient's concern, via export/import of raw weight
 * sets if needed.
 */

import type {
  GeoFinetuneResult,
  GeoSpec,
  GeoTokenizeResult,
  GeoTrace,
  GeoVectorFieldData,
  GeoVectorFieldParams,
  GeoWeightsData,
  GeoWeightsParams,
  GeoWeightsPostBody,
  GeoWeightsPostResult,
} from "../dataClient";
import { GeoEngineError, computeError, invalidParam, notFound } from "./errors";

/** Serialized minted weight set for external persistence (sessionStorage). */
export interface ExportedWeightSet {
  weights: Record<string, string>; // tensor name -> base64 of float32-LE bytes
  sources: Record<string, string>;
  setSource: string;
}

function b64FromF32(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let s = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}

function f32FromB64(b64: string): Float32Array {
  const s = atob(b64);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
import { forceField, nextNextField } from "./fields";
import { runFinetune, type ProgressCb } from "./finetune";
import {
  CONTEXT_WINDOW,
  D_MODEL,
  EDITABLE_MATRICES,
  FINETUNE_DEFAULT_LR,
  FINETUNE_DEFAULT_STEPS,
  GeoModel,
  MLP_HIDDEN,
  N_HEADS,
  N_LAYERS,
  SPECIAL_TOKENS,
  VOCAB_SIZE,
  loadCheckpoint,
  type CheckpointMeta,
  type WeightSet,
} from "./model";
import { sha256Hex, utf8Bytes } from "./hash";
import { toNested2 } from "./tensor";
import { GeoTokenizer, encodedTokens } from "./tokenizer";
import {
  buildWeightSet,
  weightsToken,
  type PresetContext,
  type PresetFixtures,
  type WeightEditInput,
} from "./weights";
import bundledPresetFixtures from "./presetFixtures.json";

export { GeoEngineError } from "./errors";
export type { WeightSet } from "./model";
export { weightsToken } from "./weights";
export { runFinetune } from "./finetune";
export type { FinetuneWorkerRequest, FinetuneWorkerResponse } from "./finetuneWorker";

const WEIGHTS_KEY = (layer: number | null, matrix: string): string =>
  matrix === "embedding" ? "embedding" : `layers.${layer}.${matrix}`;

export interface GeoEngineFinetuneBody {
  text?: string | null;
  hf_dataset?: string | null;
  steps?: number;
  lr?: number;
  seed?: number;
  base?: string;
  onProgress?: ProgressCb;
}

export class GeoEngine {
  readonly tokenizer: GeoTokenizer;
  readonly canonical: WeightSet;
  readonly canonicalToken: string;
  readonly meta: CheckpointMeta;

  private readonly fixtures: PresetFixtures | null;
  private readonly weightSets = new Map<string, WeightSet>();
  /** token -> matrixKey -> "edited" | "preset:<name>" (absent key => "learned"). */
  private readonly sourceMaps = new Map<string, Record<string, string>>();
  /** token -> set-level provenance ("learned" | "edited" | "finetuned"). */
  private readonly setSources = new Map<string, string>();
  private readonly models = new Map<string, GeoModel>();
  private readonly finetuneCache = new Map<string, GeoFinetuneResult>();

  private constructor(
    checkpoint: { weights: WeightSet; meta: CheckpointMeta },
    tokenizer: GeoTokenizer,
    fixtures: PresetFixtures | null,
  ) {
    this.tokenizer = tokenizer;
    this.canonical = checkpoint.weights;
    this.meta = checkpoint.meta;
    this.fixtures = fixtures;
    this.canonicalToken = weightsToken(this.canonical);
    if (checkpoint.meta.checkpoint_id !== null && checkpoint.meta.checkpoint_id !== this.canonicalToken) {
      throw computeError(
        `checkpoint integrity failure: content hash ${this.canonicalToken} does not match the ` +
          `recorded checkpoint_id ${checkpoint.meta.checkpoint_id} — the exported weights and the ` +
          "TS hash must agree bit-for-bit",
      );
    }
    this.weightSets.set(this.canonicalToken, this.canonical);
    this.setSources.set(this.canonicalToken, "learned");
  }

  /**
   * Build an engine from parsed JSON assets. `presetFixtures` may override the
   * bundled seeded-preset matrices (e.g. if the static export ships more seeds).
   */
  static fromAssets(checkpointJson: unknown, vocabJson: unknown, presetFixtures?: unknown): GeoEngine {
    const checkpoint = loadCheckpoint(checkpointJson);
    const tokenizer = GeoTokenizer.fromVocabJson(vocabJson);
    const fixtures = (presetFixtures ?? bundledPresetFixtures) as PresetFixtures;
    return new GeoEngine(checkpoint, tokenizer, fixtures);
  }

  // --- GET /api/geo/spec -----------------------------------------------------------

  spec(): GeoSpec {
    return {
      model: {
        d_model: D_MODEL,
        n_layers: N_LAYERS,
        n_heads: N_HEADS,
        mlp_hidden: MLP_HIDDEN,
        vocab_size: VOCAB_SIZE,
        context_window: CONTEXT_WINDOW,
        tied_unembedding: true,
        corpus: this.meta.corpus,
        seed: this.meta.seed,
      },
      special_tokens: { ...SPECIAL_TOKENS },
      checkpoint: {
        status: "ready",
        checkpoint_id: this.canonicalToken,
        final_loss: this.meta.final_loss,
        coverage_uniformity: this.meta.coverage_uniformity,
        field_directional_entropy: this.meta.field_directional_entropy,
        job_id: null,
      },
    };
  }

  // --- GET /api/geo/tokenize -------------------------------------------------------

  tokenize(text: string): GeoTokenizeResult {
    const enc = this.tokenizer.encode(text);
    return { tokens: encodedTokens(enc), n_unk: enc.n_unk, truncated: enc.truncated };
  }

  // --- GET /api/geo/trace ----------------------------------------------------------

  trace(prompt: string, weightsTokenParam?: string): GeoTrace {
    const enc = this.tokenizer.encode(prompt);
    if (enc.ids.length === 0) throw invalidParam("prompt is empty after tokenization");
    const model = this.modelFor(weightsTokenParam ?? "learned");
    const tr = model.forwardTrace(enc.ids);
    const probs = tr.probs;
    // Descending argsort of the next-token distribution; top-10 readout.
    const top: number[] = [];
    {
      const idx = Array.from({ length: probs.length }, (_, i) => i);
      idx.sort((a, b) => (probs[b] !== probs[a] ? probs[b] - probs[a] : a - b));
      for (let i = 0; i < 10; i++) top.push(idx[i]);
    }
    const nextId = top[0];
    const text = (id: number): string => this.tokenizer.idToText.get(id) as string;
    return {
      tokens: encodedTokens(enc),
      embeddings: tr.embeddings,
      layers: tr.layers,
      probs: Array.from(probs),
      logits_topk: {
        ids: top,
        texts: top.map(text),
        probs: top.map((i) => probs[i]),
      },
      next_token: { id: nextId, text: text(nextId) },
    };
  }

  // --- GET /api/geo/vector_field ---------------------------------------------------

  vectorField(params: GeoVectorFieldParams): GeoVectorFieldData {
    const { mode, layer, prompt } = params;
    if (mode !== "next_next" && mode !== "force") {
      throw invalidParam(`mode must be "next_next" or "force", got ${JSON.stringify(mode)}`);
    }
    const promptIds = this.tokenizer.encode(prompt).ids;
    const model = this.modelFor(params.weights_token ?? "learned");
    const field =
      mode === "next_next"
        ? nextNextField(model, promptIds, layer, params.temperature ?? 0, params.top_m ?? 1)
        : forceField(model, promptIds, layer, params.antisymmetrize ?? false);
    return field as GeoVectorFieldData;
  }

  // --- GET /api/geo/weights --------------------------------------------------------

  getWeights(params: GeoWeightsParams): GeoWeightsData {
    const { matrix } = params;
    if (!(EDITABLE_MATRICES as readonly string[]).includes(matrix)) {
      throw invalidParam(`matrix must be one of ${EDITABLE_MATRICES.join(", ")}, got ${JSON.stringify(matrix)}`);
    }
    const ws = this.resolveWeightSet(params.weights_token ?? "learned");
    let name: string;
    let srcLayer: number | null;
    if (matrix === "embedding") {
      name = "embedding";
      srcLayer = null;
    } else {
      const layer = params.layer;
      if (layer === undefined || !Number.isInteger(layer) || layer < 0 || layer >= N_LAYERS) {
        throw invalidParam(
          `layer must be an int in 0..${N_LAYERS - 1} for ${matrix}, got ${JSON.stringify(params.layer)}`,
        );
      }
      name = `layers.${layer}.${matrix}`;
      srcLayer = layer;
    }
    const values = ws[name];
    const rows = matrix === "embedding" ? VOCAB_SIZE : D_MODEL;
    const cols = D_MODEL;
    return {
      values: toNested2(values, rows, cols),
      shape: [rows, cols],
      source: this.matrixSource(params.weights_token ?? null, srcLayer, matrix) as GeoWeightsData["source"],
    };
  }

  // --- minted-set persistence (red-team static finding #3) -------------------------
  // The engine's store is in-memory; the static client persists minted sets across
  // reloads via these hooks. Import validates the content hash so a corrupted or
  // stale payload can never impersonate a token.

  exportWeightSet(token: string): ExportedWeightSet {
    const ws = this.weightSets.get(token);
    if (!ws) throw notFound(`weights_token '${token}' is unknown (nothing to export)`);
    const weights: Record<string, string> = {};
    for (const [name, arr] of Object.entries(ws)) weights[name] = b64FromF32(arr);
    return {
      weights,
      sources: { ...(this.sourceMaps.get(token) ?? {}) },
      setSource: this.setSources.get(token) ?? "edited",
    };
  }

  importWeightSet(token: string, payload: ExportedWeightSet): boolean {
    if (this.weightSets.has(token)) return true;
    const ws: WeightSet = {};
    try {
      for (const [name, b64] of Object.entries(payload.weights)) ws[name] = f32FromB64(b64);
    } catch {
      return false;
    }
    if (weightsToken(ws) !== token) return false; // hash mismatch — refuse
    this.weightSets.set(token, ws);
    this.sourceMaps.set(token, { ...payload.sources });
    this.setSources.set(token, payload.setSource);
    return true;
  }

  listMintedTokens(): string[] {
    return [...this.weightSets.keys()];
  }

  // --- POST /api/geo/weights -------------------------------------------------------

  postWeights(body: GeoWeightsPostBody): GeoWeightsPostResult {
    const base = body.base ?? "learned";
    const baseWs = this.resolveWeightSet(base);
    const ctx = this.presetContext();
    const { ws, summaries } = buildWeightSet(baseWs, body.edits as WeightEditInput[], ctx);
    const token = weightsToken(ws);
    this.weightSets.set(token, ws);
    this.setSources.set(token, "edited");
    // Per-matrix provenance: inherit the base's sidecar, overlay these edits.
    const sources: Record<string, string> = { ...this.sourceMapFor(base) };
    for (const s of summaries) sources[WEIGHTS_KEY(s.layer, s.matrix)] = s.source;
    this.sourceMaps.set(token, sources);
    return {
      weights_token: token,
      edited: summaries as GeoWeightsPostResult["edited"],
    };
  }

  // --- POST /api/geo/finetune ------------------------------------------------------

  /**
   * Synchronous static-mode fine-tune (run it inside finetuneWorker.ts to keep the
   * UI thread free — see registerFinetuneResult for wiring worker output back in).
   * Only the `text` source is available in the static build; hf_dataset requires
   * the full stack's streaming access and throws an InvalidParamError here.
   */
  finetune(body: GeoEngineFinetuneBody): GeoFinetuneResult {
    if (body.hf_dataset != null) {
      throw invalidParam(
        "hf_dataset fine-tuning is not available in the static build (it needs the full " +
          "stack's HuggingFace streaming); provide `text` instead",
      );
    }
    const text = body.text;
    if (text == null) throw invalidParam("exactly one of text/hf_dataset must be provided");
    if (text.trim().length === 0) throw invalidParam("fine-tuning text is empty");
    const steps = Math.trunc(body.steps ?? FINETUNE_DEFAULT_STEPS);
    const lr = body.lr ?? FINETUNE_DEFAULT_LR;
    const seed = Math.trunc(body.seed ?? 0);
    const base = body.base ?? "learned";
    const baseWs = this.resolveWeightSet(base);
    const baseToken = weightsToken(baseWs);

    // Content-derived cache key (identical requests are instant cache hits, like
    // the backend's 200 responses).
    const cacheKey = sha256Hex(
      utf8Bytes(JSON.stringify({ baseToken, text, steps, lr, seed })),
    );
    const cached = this.finetuneCache.get(cacheKey);
    if (cached !== undefined) return { ...cached };

    const tokenIds = this.tokenizer.encodeStream(text);
    const result = runFinetune({
      baseWeights: baseWs,
      tokenIds,
      steps,
      lr,
      seed,
      onProgress: body.onProgress,
    });
    const token = this.registerFinetunedWeights(result.weights);
    const out: GeoFinetuneResult = {
      ready: true,
      weights_token: token,
      loss_before: result.lossBefore,
      loss_after: result.lossAfter,
    };
    this.finetuneCache.set(cacheKey, out);
    return { ...out };
  }

  /**
   * Register a fine-tuned weight set (e.g. received from finetuneWorker.ts) and
   * mint its content-hash token. Every matrix of a fine-tuned set reports source
   * "edited" (the backend's closed-enum mapping for fine-tuning).
   */
  registerFinetunedWeights(weights: WeightSet): string {
    const token = weightsToken(weights);
    this.weightSets.set(token, weights);
    this.setSources.set(token, "finetuned");
    return token;
  }

  // --- internals -------------------------------------------------------------------

  private presetContext(): PresetContext {
    return { canonical: this.canonical, fixtures: this.fixtures };
  }

  private resolveWeightSet(tokenOrLearned: string): WeightSet {
    if (tokenOrLearned === "learned") return this.canonical;
    const ws = this.weightSets.get(tokenOrLearned);
    if (ws === undefined) {
      throw notFound(
        `weights_token '${tokenOrLearned}' is unknown (never minted here, or evicted); ` +
          "re-submit the edit to mint it again",
      );
    }
    return ws;
  }

  private modelFor(tokenOrLearned: string): GeoModel {
    const key = tokenOrLearned === "learned" ? this.canonicalToken : tokenOrLearned;
    let model = this.models.get(key);
    if (model === undefined) {
      model = new GeoModel(this.resolveWeightSet(tokenOrLearned));
      this.models.set(key, model);
    }
    return model;
  }

  private sourceMapFor(tokenOrLearned: string): Record<string, string> {
    if (tokenOrLearned === "learned" || tokenOrLearned === this.canonicalToken) return {};
    const sidecar = this.sourceMaps.get(tokenOrLearned);
    if (sidecar !== undefined) return sidecar;
    // No sidecar (a fine-tuned token): every matrix is "edited" (backend mapping).
    if (this.setSources.get(tokenOrLearned) === "finetuned") {
      const all: Record<string, string> = { embedding: "edited" };
      for (let l = 0; l < N_LAYERS; l++) {
        for (const m of ["W_Q", "W_K", "W_V", "W_O"]) all[`layers.${l}.${m}`] = "edited";
      }
      return all;
    }
    return {};
  }

  /** Backend geo/jobs.py matrix_source: "learned" | "edited" | "preset:<name>". */
  private matrixSource(token: string | null, layer: number | null, matrix: string): string {
    if (token === null || token === "learned" || token === this.canonicalToken) return "learned";
    const map = this.sourceMapFor(token);
    return map[WEIGHTS_KEY(layer, matrix)] ?? "learned";
  }
}
