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
import { asFloat, asInt } from "./params";

/** Serialized minted weight set for external persistence (sessionStorage). */
/** The portable model-file format — the same one GET /api/geo/model emits. */
export const BUNDLE_FORMAT = "llm-geometry/geo-model";
/**
 * v1 had no vocabulary integrity check; v2 named a model by a hash of its WEIGHTS ALONE.
 * `weightsToken` now hashes the word list too, which changes what the file's
 * `weights_token` FIELD means, so the format moved with it. Leaving it at 2 made every
 * pre-change file with its own word list fail the re-hash and be reported as "this model
 * file is corrupt" — an accusation against an intact file. Mirrors `geo/bundle.py`.
 */
export const BUNDLE_VERSION = 3;
/** The version whose `weights_token` covers the weights only — READ, not refused. */
export const LEGACY_WEIGHTS_ONLY_BUNDLE_VERSION = 2;

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
 * Why a payload written before `ownsVocab` existed cannot be restored. Undecidable, not
 * merely old: a `finetuned` payload derived from a scratch model and one derived from the
 * shipped checkpoint are byte-identical in that format.
 */
const LEGACY_OWNERSHIP_REASON =
  "it was written by an earlier version of this page, which did not record whether a " +
  "model's ids mean its own words — and in that format a fine-tune of your own model and " +
  "a fine-tune of the shipped one are indistinguishable, so restoring it could label every " +
  "token with the wrong word";

/**
 * Why a payload written under the weights-only identity cannot be restored. Detected, not
 * assumed: such a payload hashes to `weightsToken(ws)` — the token this build would give
 * the same weights with NO word list — while declaring a word list of its own.
 */
const LEGACY_IDENTITY_REASON =
  "it was written by an earlier version of this page, when a model was named by its " +
  "weights alone; a model is now named by its weights AND the words its ids mean, and the " +
  "older format cannot show which word list is really its own — that ambiguity is exactly " +
  "how one model came to be saved under another model's words";

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

  /** The shipped word list in its canonical serialization — the hash's "no own vocabulary". */
  private readonly canonicalVocab: string;
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
  /**
   * token -> why a persisted payload for it was REFUSED, so the refusal can be explained
   * later instead of the model just being absent.
   *
   * `importWeightSet` returning false is how a stale or corrupted payload is dropped, and
   * the static client then deletes it from sessionStorage. For a payload written by an
   * older build that is exactly right except that the identity rule moved beneath it,
   * "deleted, no message" destroys a model the user trained and tells them nothing; the
   * next call would have said "unknown (never minted here, or evicted)", which is not what
   * happened either. The reason is kept here and raised by `resolveWeightSet`, which is
   * where the app reaches for the model.
   */
  private readonly refusedSets = new Map<string, string>();
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
    this.canonicalVocab = canonicalVocabJson(tokenizer.words);
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

  // --- model identity ---------------------------------------------------------------
  // A model's identity is its weights AND the word list its ids mean. See
  // `weightsToken` and the backend's `weights.weights_token`.

  /**
   * The vocabulary bytes that take part in a token's hash, or null when the model reads
   * under the shipped word list. A list identical to the canonical one is NOT an own
   * vocabulary: there is nothing that could be substituted for it, and treating it as
   * owned would give the canonical checkpoint two different tokens depending on whether
   * it arrived as the checkpoint or through a file. Mirrors `weights.own_vocab_json`.
   */
  private ownVocabJson(words: readonly string[] | null | undefined): string | null {
    if (!words) return null;
    const json = canonicalVocabJson([...words]);
    return json === this.canonicalVocab ? null : json;
  }

  /** The content-hash token of (`ws`, the word list `words`) — the model's identity. */
  private tokenFor(ws: WeightSet, words: readonly string[] | null | undefined): string {
    return weightsToken(ws, this.ownVocabJson(words));
  }

  /**
   * The word list a set DERIVED from `base` inherits, or null. A base that claims to own
   * a vocabulary but has none returns null and the claim is carried separately — such a
   * set is unwritable, and `exportBundle` refuses it rather than substituting.
   */
  private ownedWordsFor(base: string): readonly string[] | null {
    if (base === "learned" || base === this.canonicalToken) return null;
    if (!this.ownsVocab.has(base)) return null;
    return this.vocabs.get(base)?.words ?? null;
  }

  /**
   * The vocabulary that gives `token`'s ids meaning (mirrors geo/tokenizer.tokenizer_for).
   *
   * A token this engine does not hold THROWS; it does not fall back to the shipped word
   * list. The fallback made `tokenize()` answer with plausible tokens for a model that is
   * gone — the static build LRU-drops persisted sets (`MINTED_SETS_CAP`) while the active
   * token is persisted separately, so a token routinely outlives its payload — while
   * `trace()` on the identical token threw `NotFoundError` from `resolveWeightSet`. The
   * token strip then showed Alice in Wonderland's words for the user's own model, and the
   * canonical-vocabulary probe agreed with them, so the tab reported the vocabulary
   * verified. Same rule, same reason, as the backend's `tokenizer_for`.
   */
  tokenizerFor(token?: string | null): GeoTokenizer {
    if (!token || token === "learned") return this.tokenizer;
    const vocab = this.vocabs.get(token);
    if (vocab !== undefined) return vocab;
    // Not "no own vocabulary" unless the set is actually here: resolveWeightSet throws
    // the one refusal (including the pre-identity-format explanation) if it is not.
    this.resolveWeightSet(token);
    // A set that CLAIMS its own word list and has none is the state `exportBundle`
    // refuses, calling the substitution catastrophic. Reading it here returned the
    // shipped tokenizer — so `tokenize`, `trace`, `vectorField` and the fine-tune probe
    // answered with Alice in Wonderland's words while the writer refused the identical
    // state, and the tab's verification probe (which compares against exactly that
    // vocabulary) then reported it VERIFIED. Same refusal, same words.
    if (this.ownsVocab.has(token)) {
      throw invalidParam(
        `weights_token '${token}' has no vocabulary in this session, and its ids mean its ` +
          "own words rather than the shipped model's — reading them under the shipped word " +
          "list would label every token wrongly. Load the model file again (or retrain) so " +
          "its vocabulary is present.",
      );
    }
    return this.tokenizer;
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
    // Null-prototype for the same reason as `importBundle`: sessionStorage is a file the
    // user's machine hands us, and `ws["__proto__"] = …` on a `{}` sets the prototype
    // instead of adding a key, dropping a declared tensor with nothing thrown.
    const ws: WeightSet = Object.create(null);
    try {
      for (const [name, b64] of Object.entries(payload.weights)) ws[name] = f32FromB64(b64);
    } catch {
      return this.refuseSet(token, "its weights could not be decoded");
    }
    const words = payload.vocabWords;
    if (words !== undefined && (!Array.isArray(words) || words.some((w) => typeof w !== "string"))) {
      return this.refuseSet(token, "its stored word list is not a list of words");
    }
    const declaredOwns = payload.ownsVocab;
    if (declaredOwns !== undefined && typeof declaredOwns !== "boolean") {
      return this.refuseSet(token, "its `ownsVocab` flag is not a boolean");
    }
    // A payload that carries NEITHER an `ownsVocab` flag NOR a word list is the shape
    // this engine wrote before ownership was recorded, and it is genuinely undecidable:
    // a `finetuned` payload derived from a scratch model and one derived from the shipped
    // checkpoint are byte-identical, and reading it as "does not own one" restores the
    // scratch model under Alice in Wonderland's words — the exact corruption `ownsVocab`
    // exists to stop. It is REFUSED here rather than made unreachable by a storage-key
    // rename: a key rename hides the payloads this build happens to know about, and does
    // nothing about one copied between profiles, restored from a backup, or handed to
    // this method by any other caller.
    if (declaredOwns === undefined && words === undefined) {
      return this.refuseSet(token, LEGACY_OWNERSHIP_REASON);
    }
    const ownsVocab = declaredOwns ?? true;
    // Claims and payload must agree: "owns nothing" with a word list attached, or "owns
    // a word list" with none, is a payload we cannot describe either way.
    if (ownsVocab !== (words !== undefined)) {
      return this.refuseSet(
        token,
        `it claims ownsVocab=${ownsVocab} but ${words === undefined ? "carries no word list" : "carries one"}`,
      );
    }
    // And the claim is CHECKED, not believed: the token covers the word list, so a
    // payload cannot pair one model's weights with another's words and still hash right.
    // `weightsToken` REFUSES a tensor name it does not know, and this method's contract is
    // to refuse a payload by returning false — never to throw. It is called from
    // `restorePersistedSets` while the engine is booting, so a throw here would take the
    // whole tab down over one bad sessionStorage entry.
    let hashed: string;
    try {
      hashed = this.tokenFor(ws, words ?? null);
    } catch (e) {
      return this.refuseSet(token, `its weights could not be hashed (${(e as Error).message})`);
    }
    if (hashed !== token) {
      // One of those mismatches is not tampering: a payload written when the token
      // covered the WEIGHTS ALONE hashes to `weightsToken(ws, null)`, and that is
      // decidable, so it gets its own explanation rather than "corrupt".
      return this.refuseSet(
        token,
        words !== undefined && weightsToken(ws) === token
          ? LEGACY_IDENTITY_REASON
          : "its contents do not hash to the token it is stored under",
      );
    }
    this.refusedSets.delete(token); // a good payload for it arrived after all
    if (this.weightSets.has(token)) return true;
    this.weightSets.set(token, ws);
    this.sourceMaps.set(token, { ...payload.sources });
    this.setSources.set(token, payload.setSource);
    // `ownVocabJson` is null when the list IS the shipped one — such a set reads under
    // the canonical tokenizer and owns nothing of its own, exactly as the backend
    // records it (`save_weight_set` normalizes the same case).
    if (words !== undefined && this.ownVocabJson(words) !== null) {
      this.vocabs.set(token, new GeoTokenizer(words));
      this.ownsVocab.add(token);
    }
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
    // Identity includes the base's word list, which the edit inherits: an edited scratch
    // model is not the same model as an edited canonical one that happens to share bytes.
    const token = this.tokenFor(ws, this.ownedWordsFor(base));
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
    // `asInt`/`asFloat`, not `Math.trunc` and `??`: this is the reference implementation
    // the Python backend is golden-tested against, so it may not accept `steps: 7.5` as 7
    // (the backend answers a typed 400 saying it is "not rounded or truncated") nor
    // `lr: Infinity`, which passes `lr > 0` and makes every parameter NaN by step 1.
    const steps = asInt(body.steps, "steps", FINETUNE_DEFAULT_STEPS);
    const lr = asFloat(body.lr, "lr", FINETUNE_DEFAULT_LR);
    const seed = asInt(body.seed, "seed", 0);
    const base = body.base ?? "learned";
    const baseWs = this.resolveWeightSet(base);
    // The base's IDENTITY (vocabulary included), not a re-hash of its weights: two
    // scratch models with the same numbers and different words must not share a
    // fine-tune cache entry.
    const baseToken = base === "learned" ? this.canonicalToken : base;

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
    // `>=`, mirroring the backend: a stream that is EXACTLY 90 % <unk> is refused. It
    // used to be `>`, so 900 of 1000 unknown tokens was accepted and reported as a clean
    // loss drop while 901 was refused with a message rounding to the same "(90%)".
    if (unkRate >= FINETUNE_MAX_UNK_RATE) {
      throw invalidParam(
        `${enc.n_unk} of ${tokenIds.length} tokens (${(unkRate * 100).toFixed(1)}%, the ` +
          `limit is ${Math.round(FINETUNE_MAX_UNK_RATE * 100)}%) in this text are outside ` +
          "the active model's vocabulary, so fine-tuning on it would mostly teach the " +
          "model to emit <unk> and the loss would say nothing about your words. Use " +
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
    const token = this.tokenFor(weights, this.ownedWordsFor(base));
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
    // The word list is part of the token, so two scratch runs that landed on identical
    // weights with different words get different tokens. They used to collide, and this
    // map overwrote — last write wins — where the Python store kept the first, so the two
    // stacks disagreed about WHICH of the two models had been corrupted.
    const token = this.tokenFor(weights, vocabWords);
    this.weightSets.set(token, weights);
    this.setSources.set(token, "scratch");
    if (this.ownVocabJson(vocabWords) !== null) {
      this.vocabs.set(token, new GeoTokenizer(vocabWords));
      this.ownsVocab.add(token);
    }
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
      // `invalidParam`, not `notFound`: the model IS here, and the frozen contract states
      // this case explicitly — "Where a model's vocabulary cannot be recovered, this
      // endpoint returns 400 InvalidParamError rather than substituting the shipped one"
      // (`specs/002-interactive-model-explorer/contracts/api.md`, GET /api/geo/model).
      // The backend raises `InvalidParamError` here; this threw a 404-shaped error, so one
      // user action produced two different statuses depending on which build served it.
      throw invalidParam(
        `weights_token '${resolved}' has no vocabulary in this session, and its ids mean ` +
          "its own words rather than the shipped model's — saving it now would pair these " +
          "weights with the wrong word list. Load the model file again (or retrain) so its " +
          "vocabulary is present.",
      );
    }
    const words = this.tokenizerFor(resolved).words;
    const vocabJson = canonicalVocabJson(words);
    const weights: GeoModelBundle["weights"] = {};
    for (const name of Object.keys(ws).sort()) {
      weights[name] = { shape: [...(WEIGHT_SHAPES.get(name) ?? [ws[name].length])], data: b64FromF32(ws[name]) };
    }
    // The file's identity covers its word list, so this re-hash must reproduce the token
    // the engine minted. If it does not, the session and the file disagree about which
    // model this is — say so rather than writing a file that names the wrong one.
    const fileToken = this.tokenFor(ws, words);
    if (resolved !== this.canonicalToken && fileToken !== resolved) {
      throw computeError(
        `weights_token '${resolved}' does not match a re-hash of its own weights and ` +
          `vocabulary (${fileToken}) — the stored model is inconsistent, so saving it ` +
          "would produce a file that names the wrong model. Retrain or reload it.",
      );
    }
    return {
      format: BUNDLE_FORMAT,
      version: BUNDLE_VERSION,
      weights_token: fileToken,
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
    const version = b.version;
    if (version !== BUNDLE_VERSION && version !== LEGACY_WEIGHTS_ONLY_BUNDLE_VERSION) {
      throw invalidParam(
        `model file version ${JSON.stringify(version)} is not supported ` +
          `(this build reads versions ${LEGACY_WEIGHTS_ONLY_BUNDLE_VERSION} and ` +
          `${BUNDLE_VERSION}). Version 1 files carried no vocabulary integrity check; ` +
          `re-export the model to get a v${BUNDLE_VERSION} file.`,
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
    // NULL-PROTOTYPE: `ws["__proto__"] = arr` on a `{}` sets the object's PROTOTYPE rather
    // than adding a key, so a file declaring a tensor by that name lost it silently —
    // `Object.keys` never reports it, `validateWeightSet` sees nothing extra, and the
    // re-hash below is computed over the tensors that survived. The file loads as if it
    // held exactly the tensors its config implies while carrying one nobody looked at.
    const ws: WeightSet = Object.create(null);
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
    // The STRICT parser: a file carries one vocabulary format, and the permissive asset
    // loader accepted shapes the backend refused with a 500 (see fromModelVocabJson).
    const tokenizer = GeoTokenizer.fromModelVocabJson(b.vocab);

    // Integrity is MANDATORY, not opt-in: treating a missing token as "nothing to
    // check" let a file with tampered weights load just by deleting the field.
    //
    // The hash covers the WORD LIST as well as the weights, so this also catches the
    // attack the vocabulary digest cannot: swapping the word list AND recomputing
    // `vocab_sha256` over the substitute leaves a file that is internally consistent but
    // no longer hashes to the model it names. It runs after the vocabulary is validated
    // because the vocabulary is one of its inputs.
    if (typeof b.weights_token !== "string" || !b.weights_token) {
      throw invalidParam(
        "model file has no `weights_token`, so its contents cannot be verified — " +
          "refusing to load it. Re-export the model to get a valid file.",
      );
    }
    const own = this.ownVocabJson(tokenizer.words);
    const actual = this.tokenFor(ws, tokenizer.words);
    const legacy = weightsToken(ws, null);
    if (version === LEGACY_WEIGHTS_ONLY_BUNDLE_VERSION) {
      // MIGRATION, not a refusal — the mirror of `geo/bundle.py`. A v2 file names itself
      // by a hash of its weights alone, so it is checked against that hash and the current
      // identity is re-derived from the (weights, word list) pair it carries. Refusing
      // would strand an intact file and buy nothing: the binding a v3 token gives is
      // absent from EVERY v2 file, including the ones that load today only because their
      // word list is the shipped one and so takes no part in either hash. What this
      // format cannot prove — that these words are the words these weights were trained
      // with — it never could; that is what the bump records, not something introduced
      // by reading it.
      if (b.weights_token !== legacy) {
        throw invalidParam(
          `this model file is corrupt: its weights hash to ${legacy} but it declares ` +
            `${b.weights_token}. Loading it would pair the wrong vocabulary with these ` +
            "weights, so it is refused.",
        );
      }
    } else if (b.weights_token !== actual) {
      // Deliberately NOT special-cased when `b.weights_token === legacy`: a file carrying
      // weights, an own word list and a weights-only token is what a version-2 writer
      // produced AND what swapping a version-3 file's word list produces. The two are
      // indistinguishable, so a file that DECLARES version 3 is held to version 3.
      throw invalidParam(
        `this model file is corrupt: its weights and vocabulary hash to ${actual} but it ` +
          `declares ${b.weights_token}. Loading it would pair the wrong vocabulary with ` +
          "these weights, so it is refused.",
      );
    }

    this.weightSets.set(actual, ws);
    this.setSources.set(actual, "imported");
    // An imported model owns the word list the file carried — unless that list IS the
    // shipped one, in which case there is nothing of its own to own and the model keeps
    // the token it would have had anyway.
    if (own !== null) {
      this.vocabs.set(actual, tokenizer);
      this.ownsVocab.add(actual);
    }
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
      // A model that WAS here and was refused on restore is not the same event as an
      // evicted one, and the remedies differ. Say which happened (`refusedSets`).
      const refused = this.refusedSets.get(tokenOrLearned);
      if (refused !== undefined) {
        throw notFound(
          `the model '${tokenOrLearned}' was saved in this tab but could not be restored: ` +
            `${refused}. It is left unloaded rather than loaded under words that may not be ` +
            "its own. A model file you saved still opens normally; otherwise train it again.",
        );
      }
      throw notFound(
        `weights_token '${tokenOrLearned}' is unknown (never minted here, or evicted); ` +
          "re-submit the edit to mint it again",
      );
    }
    return ws;
  }

  /** Record WHY a persisted payload was refused, and report the refusal. */
  private refuseSet(token: string, reason: string): false {
    this.refusedSets.set(token, reason);
    return false;
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
