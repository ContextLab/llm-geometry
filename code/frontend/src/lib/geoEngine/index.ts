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
/** The portable model-file format — the same one GET /api/geo/model emits. */
export const BUNDLE_FORMAT = "llm-geometry/geo-model";
export const BUNDLE_VERSION = 2; // v1 had no vocabulary integrity check

export interface GeoModelBundle {
  format: string;
  version: number;
  weights_token: string;
  config: Record<string, number>;
  vocab: string;
  /** SHA-256 of `vocab`. The weights hash cannot cover the word list. */
  vocab_sha256: string;
  weights: Record<string, { shape: number[]; data: string }>;
}

/** Mirrors geo/bundle.vocab_digest. */
function vocabDigest(vocabJson: string): string {
  return sha256Hex(utf8Bytes(vocabJson));
}

export interface ExportedWeightSet {
  weights: Record<string, string>; // tensor name -> base64 of float32-LE bytes
  sources: Record<string, string>;
  setSource: string;
  /**
   * The word list this set's token ids mean, for the sets that HAVE one of their own:
   * `scratch`, `imported`, and anything DERIVED from those by fine-tuning or a weight
   * edit. Absent only for sets descended from the canonical checkpoint, which really do
   * keep the shipped vocabulary — for those, absence is the correct answer, not a gap.
   *
   * Omitting it for a set that owns a vocabulary is not a lossy shortcut, it is a
   * corruption: the engine would fall back to the canonical tokenizer and `exportBundle`
   * would then write a file pairing YOUR weights with Alice in Wonderland's words, under
   * a `vocab_sha256` computed over that wrong list — internally consistent, so no
   * integrity check could catch it. That is precisely the failure the three digests
   * exist to prevent, committed by the writer. `restorePersistedSets` therefore drops a
   * payload that lacks a vocabulary it needs rather than restoring it half-right.
   */
  vocabWords?: string[];
  /**
   * Whether this set's ids mean its own words — recorded explicitly, because it is a
   * property of the DERIVATION CHAIN, not of `setSource`. A fine-tune of the shipped
   * model keeps the canonical vocabulary; a fine-tune of a scratch model does not, and
   * both have `setSource === "finetuned"`.
   */
  ownsVocab?: boolean;
}

/**
 * Weight-set kinds that ALWAYS own a vocabulary of their own, whatever they came from.
 * Everything else (`edited`, `finetuned`) owns one exactly when its base did, which is
 * why ownership is tracked per token in `ownsVocab` rather than inferred from a kind.
 * Retained for reading `ExportedWeightSet` payloads that predate the `ownsVocab` field.
 */
const SET_SOURCES_WITH_OWN_VOCAB: ReadonlySet<string> = new Set(["scratch", "imported"]);

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
  EOS_ID,
  EOS_TOKEN,
  FINETUNE_DEFAULT_LR,
  FINETUNE_DEFAULT_STEPS,
  FINETUNE_MAX_UNK_RATE,
  GeoModel,
  MLP_HIDDEN,
  N_HEADS,
  N_LAYERS,
  PAD_ID,
  PAD_TOKEN,
  SPECIAL_TOKENS,
  UNK_ID,
  UNK_TOKEN,
  VOCAB_SIZE,
  WEIGHT_SHAPES,
  loadCheckpoint,
  validateWeightSet,
  type CheckpointMeta,
  type WeightSet,
} from "./model";
import { sha256Hex, utf8Bytes } from "./hash";
import { toNested2 } from "./tensor";
import { GeoTokenizer, canonicalVocabJson, encodedTokens } from "./tokenizer";
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
  /**
   * token -> the vocabulary THAT model's ids mean (feature 004). Only models trained
   * from scratch appear here; everything derived from the canonical checkpoint shares
   * the canonical tokenizer. Reading a scratch model's ids with the canonical
   * vocabulary would mislabel every token on screen.
   */
  private readonly vocabs = new Map<string, GeoTokenizer>();
  /**
   * Tokens whose ids mean their OWN words. Distinct from `vocabs`: this is the CLAIM,
   * `vocabs` is the payload. A token in here with nothing in `vocabs` is a set we
   * cannot describe, and `exportBundle` refuses to write a file for it rather than
   * silently substituting the shipped word list.
   */
  private readonly ownsVocab = new Set<string>();
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

  /** The vocabulary that gives `token`'s ids meaning (mirrors geo/tokenizer.tokenizer_for). */
  tokenizerFor(token?: string | null): GeoTokenizer {
    if (!token || token === "learned") return this.tokenizer;
    return this.vocabs.get(token) ?? this.tokenizer;
  }

  // --- GET /api/geo/tokenize -------------------------------------------------------

  tokenize(text: string, weightsTokenParam?: string): GeoTokenizeResult {
    const enc = this.tokenizerFor(weightsTokenParam).encode(text);
    return { tokens: encodedTokens(enc), n_unk: enc.n_unk, truncated: enc.truncated };
  }

  // --- GET /api/geo/trace ----------------------------------------------------------

  trace(prompt: string, weightsTokenParam?: string): GeoTrace {
    const tok = this.tokenizerFor(weightsTokenParam);
    const enc = tok.encode(prompt);
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
    const text = (id: number): string => tok.idToText.get(id) as string;
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
    const promptIds = this.tokenizerFor(params.weights_token).encode(prompt).ids;
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
    const vocab = this.vocabs.get(token);
    return {
      weights,
      sources: { ...(this.sourceMaps.get(token) ?? {}) },
      setSource: this.setSources.get(token) ?? "edited",
      ownsVocab: this.ownsVocab.has(token),
      // The word list travels WITH the weights, exactly as the backend's
      // `save_weight_set(..., vocab_json=…)` stores it beside them: a scratch or
      // imported model's ids mean its own words — and so do a fine-tune or an edit of
      // one — so weights alone do not describe it.
      ...(vocab ? { vocabWords: [...vocab.words] } : {}),
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
    const words = payload.vocabWords;
    // `ownsVocab` is authoritative when present. Payloads written before it existed
    // are read the only way they can be: by their kind, which is why the storage key
    // is versioned — a pre-fix `finetuned` payload derived from a scratch model would
    // otherwise restore claiming the canonical vocabulary is correct for it.
    const ownsVocab = payload.ownsVocab ?? SET_SOURCES_WITH_OWN_VOCAB.has(payload.setSource);
    if (words !== undefined && (!Array.isArray(words) || words.some((w) => typeof w !== "string"))) {
      return false;
    }
    // A set whose ids mean its own words is not restorable without them. Restoring the
    // weights alone would leave `tokenizerFor` falling back to the canonical vocabulary
    // and every later read — the sphere's labels, a trace, and above all a SAVED model
    // file — would silently describe the wrong words. Refuse, and let the caller drop it.
    if (ownsVocab && words === undefined) return false;
    this.weightSets.set(token, ws);
    this.sourceMaps.set(token, { ...payload.sources });
    this.setSources.set(token, payload.setSource);
    if (ownsVocab) this.ownsVocab.add(token);
    if (words !== undefined) this.vocabs.set(token, new GeoTokenizer(words));
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
    // Editing a matrix changes the numbers, not what the ids MEAN — so the base's
    // vocabulary comes with it. Dropping it here silently reverted an edited scratch
    // model to the shipped word list, and a file saved from it verified, because the
    // writer computed `vocab_sha256` over the substituted list.
    this.inheritVocab(base, token);
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

    // The BASE model's own vocabulary, not the canonical one (issue #6): a scratch
    // model's fine-tuning corpus encoded with the shipped Alice words became a stream
    // of <unk>, and the resulting loss drop was still labelled "on your text".
    const enc = this.tokenizerFor(base).encode(text, { truncate: false });
    const tokenIds = enc.ids;
    if (tokenIds.length < 2) {
      throw invalidParam("fine-tuning text is too short after tokenization (need at least 2 tokens)");
    }
    const unkRate = enc.n_unk / tokenIds.length;
    if (unkRate > FINETUNE_MAX_UNK_RATE) {
      throw invalidParam(
        `${enc.n_unk} of ${tokenIds.length} tokens (${Math.round(unkRate * 100)}%) in this ` +
          "text are outside the active model's vocabulary, so fine-tuning on it would mostly " +
          "teach the model to emit <unk> and the loss would say nothing about your words. Use " +
          "'Train a new model' to build a vocabulary from this text instead.",
      );
    }
    const result = runFinetune({
      baseWeights: baseWs,
      tokenIds,
      steps,
      lr,
      seed,
      onProgress: body.onProgress,
    });
    const token = this.registerFinetunedWeights(result.weights, base);
    const out: GeoFinetuneResult = {
      ready: true,
      weights_token: token,
      loss_before: result.lossBefore,
      loss_after: result.lossAfter,
      n_tokens: tokenIds.length,
      n_unk: enc.n_unk,
      unk_rate: unkRate,
    };
    this.finetuneCache.set(cacheKey, out);
    return { ...out };
  }

  /**
   * Register a fine-tuned weight set (e.g. received from finetuneWorker.ts) and
   * mint its content-hash token. Every matrix of a fine-tuned set reports source
   * "edited" (the backend's closed-enum mapping for fine-tuning).
   *
   * `base` is REQUIRED because a fine-tune inherits the base model's vocabulary: a
   * fine-tune of the shipped checkpoint means Alice's words, a fine-tune of a
   * scratch-trained model means that model's words, and the two are indistinguishable
   * from the weights alone.
   */
  registerFinetunedWeights(weights: WeightSet, base: string): string {
    const token = weightsToken(weights);
    this.weightSets.set(token, weights);
    this.setSources.set(token, "finetuned");
    this.inheritVocab(base, token);
    return token;
  }

  // --- from-scratch models + portable bundles (feature 004) ------------------------

  /**
   * Register a model trained from scratch, together with the vocabulary its ids mean.
   * Unlike a fine-tune, this is a different model, not a variation of the canonical
   * one — so the vocabulary is stored with it and every read path uses it.
   */
  registerScratchModel(weights: WeightSet, vocabWords: string[]): string {
    const token = weightsToken(weights);
    this.weightSets.set(token, weights);
    this.setSources.set(token, "scratch");
    this.vocabs.set(token, new GeoTokenizer(vocabWords));
    this.ownsVocab.add(token);
    return token;
  }

  /**
   * Carry `base`'s vocabulary (and its ownership claim) onto a set derived from it.
   * The mirror of the backend's `weights.inherited_vocab`.
   */
  private inheritVocab(base: string, derived: string): void {
    if (base === "learned" || base === this.canonicalToken) return;
    if (!this.ownsVocab.has(base)) return;
    this.ownsVocab.add(derived);
    const vocab = this.vocabs.get(base);
    // Deliberately no `else`: a base that claims a vocabulary but has none is already
    // unwritable, and `exportBundle` refuses BOTH it and this derived set. Substituting
    // the canonical word list here is the exact corruption the refusal exists to stop.
    if (vocab) this.vocabs.set(derived, vocab);
  }

  /** The portable bundle for a model — the same shape as GET /api/geo/model. */
  exportBundle(token?: string | null): GeoModelBundle {
    const resolved = token && token !== "learned" ? token : this.canonicalToken;
    const ws = this.resolveWeightSet(resolved);
    // `tokenizerFor` falls back to the canonical vocabulary, which is RIGHT for a set
    // descended from the shipped checkpoint and CATASTROPHIC for one whose ids mean its
    // own words: the file would carry your weights under Alice in Wonderland's word
    // list, with a `vocab_sha256` computed over that list, so no reader could ever
    // detect it. Writing such a file is refused.
    if (this.ownsVocab.has(resolved) && !this.vocabs.has(resolved)) {
      throw notFound(
        `weights_token '${resolved}' has no vocabulary in this session, and its ids mean ` +
          "its own words rather than the shipped model's — saving it now would pair these " +
          "weights with the wrong word list. Load the model file again (or retrain) so its " +
          "vocabulary is present.",
      );
    }
    const vocabJson = canonicalVocabJson(this.tokenizerFor(resolved).words);
    const weights: GeoModelBundle["weights"] = {};
    for (const name of Object.keys(ws).sort()) {
      weights[name] = { shape: [...(WEIGHT_SHAPES.get(name) ?? [ws[name].length])], data: b64FromF32(ws[name]) };
    }
    return {
      format: BUNDLE_FORMAT,
      version: BUNDLE_VERSION,
      weights_token: weightsToken(ws),
      config: {
        d_model: D_MODEL,
        n_layers: N_LAYERS,
        n_heads: N_HEADS,
        mlp_hidden: MLP_HIDDEN,
        vocab_size: VOCAB_SIZE,
        context_window: CONTEXT_WINDOW,
      },
      vocab: vocabJson,
      vocab_sha256: vocabDigest(vocabJson),
      weights,
    };
  }

  /**
   * Validate and load a bundle; returns its token. Mirrors geo/bundle.py including
   * the hash check — a bundle whose weights do not re-hash to its declared token is
   * REFUSED, because loading it would pair the wrong vocabulary with those weights
   * and silently mislabel everything.
   */
  importBundle(bundle: unknown): { weights_token: string; vocab_size: number } {
    if (bundle === null || typeof bundle !== "object" || Array.isArray(bundle)) {
      throw invalidParam("a model file must be a JSON object");
    }
    const b = bundle as Record<string, unknown>;
    if (b.format !== BUNDLE_FORMAT) {
      throw invalidParam(
        `not a Geometry Lab model file (format=${JSON.stringify(b.format)}, expected ` +
          `${JSON.stringify(BUNDLE_FORMAT)})`,
      );
    }
    if (b.version !== BUNDLE_VERSION) {
      throw invalidParam(
        `model file version ${JSON.stringify(b.version)} is not supported ` +
          `(this build reads version ${BUNDLE_VERSION}). Version 1 files carried no ` +
          "vocabulary integrity check; re-export the model to get a v2 file.",
      );
    }
    const cfg = b.config as Record<string, unknown> | undefined;
    if (!cfg || typeof cfg !== "object") throw invalidParam("model file is missing its `config` block");
    const expected: Record<string, number> = {
      d_model: D_MODEL,
      n_layers: N_LAYERS,
      n_heads: N_HEADS,
      mlp_hidden: MLP_HIDDEN,
      vocab_size: VOCAB_SIZE,
      context_window: CONTEXT_WINDOW,
    };
    for (const [field, want] of Object.entries(expected)) {
      if (cfg[field] !== want) {
        throw invalidParam(
          `model file was built for ${field}=${JSON.stringify(cfg[field])}, but this build's ` +
            `GeoTransformer is ${field}=${want} — they are different architectures, so the ` +
            "weights cannot be loaded",
        );
      }
    }

    const rawWeights = b.weights as Record<string, { shape?: number[]; data?: string }> | undefined;
    if (!rawWeights || typeof rawWeights !== "object" || Object.keys(rawWeights).length === 0) {
      throw invalidParam("model file carries no weights");
    }
    const ws: WeightSet = {};
    for (const [name, payload] of Object.entries(rawWeights)) {
      if (!payload || typeof payload.data !== "string" || !Array.isArray(payload.shape)) {
        throw invalidParam(`weight ${JSON.stringify(name)} is malformed (need shape + data)`);
      }
      const arr = f32FromB64(payload.data);
      const want = payload.shape.reduce((a, d) => a * d, 1);
      if (arr.length !== want) {
        throw invalidParam(
          `weight ${JSON.stringify(name)} has ${arr.length} values but shape ` +
            `[${payload.shape.join(", ")}] needs ${want}`,
        );
      }
      ws[name] = arr;
    }
    validateWeightSet(ws); // shapes + completeness, loudly

    // Integrity is MANDATORY, not opt-in: treating a missing token as "nothing to
    // check" let a file with tampered weights load just by deleting the field.
    if (typeof b.weights_token !== "string" || !b.weights_token) {
      throw invalidParam(
        "model file has no `weights_token`, so its contents cannot be verified — " +
          "refusing to load it. Re-export the model to get a valid file.",
      );
    }
    const actual = weightsToken(ws);
    if (b.weights_token !== actual) {
      throw invalidParam(
        `this model file is corrupt: its weights hash to ${actual} but it declares ` +
          `${b.weights_token}. Loading it would pair the wrong vocabulary with these ` +
          "weights, so it is refused.",
      );
    }

    if (typeof b.vocab !== "string") throw invalidParam("model file is missing its `vocab` block");
    // The weights hash says nothing about the word list, so the vocabulary carries its
    // own digest — otherwise genuine weights plus an invented vocabulary load cleanly
    // and mislabel every token on screen.
    if (typeof b.vocab_sha256 !== "string" || !b.vocab_sha256) {
      throw invalidParam(
        "model file has no `vocab_sha256`, so its vocabulary cannot be verified — " +
          "refusing to load it. Re-export the model to get a valid file.",
      );
    }
    const actualVocab = vocabDigest(b.vocab);
    if (b.vocab_sha256 !== actualVocab) {
      throw invalidParam(
        `this model file is corrupt: its vocabulary hashes to ${actualVocab.slice(0, 16)}… ` +
          `but it declares ${String(b.vocab_sha256).slice(0, 16)}…. Loading it would label ` +
          "every token with the wrong word, so it is refused.",
      );
    }
    const tokenizer = GeoTokenizer.fromVocabJson(b.vocab);

    this.weightSets.set(actual, ws);
    this.setSources.set(actual, "imported");
    this.vocabs.set(actual, tokenizer);
    // An imported model always owns its word list: the file carried one, and that is
    // what the ids in these weights mean.
    this.ownsVocab.add(actual);
    return { weights_token: actual, vocab_size: VOCAB_SIZE };
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
