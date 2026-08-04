/**
 * The portable Lexicon Lab model file (US-8) — ONE wire format, written and read by the
 * browser AND by the Python backend.
 *
 * There is exactly one format tag, `llm-geometry/lex-model`, and it belongs to the
 * contract in `specs/006-lexicon-lab-tiny/contracts/api-lex.md` (`GET|POST
 * /api/lex/model`). This module implements THAT payload, not a browser dialect of it:
 *
 *   * the config block is snake_case (`vocab_rows`, `d_model`, …), as the backend emits;
 *   * the vocabulary is an object — `{source, budget, words, specials}`;
 *   * tensors are named with the PYTHON model's names (`blocks.N.*`), because the backend
 *     and `staticClient/lex.ts` already agree on those. The browser engine calls the same
 *     tensors `layers.N.*`, so the translation happens HERE, at the file boundary, and
 *     the engine's internal names are left alone.
 *
 * One tag with two meanings is the failure this avoids: a file written by one side and
 * silently misread — or refused for the wrong reason — by the other.
 *
 * ## Integrity: three digests, all mandatory, all fatal
 *
 * A model file carries weights AND the vocabulary that gives its token ids meaning. In
 * this model the vocabulary IS the budget, so a file pairing genuine weights with a
 * different word list would relabel every token on screen while looking healthy.
 *
 *   * `model_token`  — the backend's JOINT hash over config + word list + weights
 *                      (`routes_lex.py::_model_token`), reproduced byte-for-byte.
 *   * `weights_token`— the weights alone, the same construction the Geometry Lab and this
 *                      tab's Weight Lab use to address an edited weight set.
 *   * `vocab_sha256` — the vocabulary alone.
 *
 * The joint hash proves the halves belong together; the two half-hashes say WHICH half is
 * wrong, and let a weight set be recognised across vocabularies. **Missing is treated
 * exactly like wrong.** Feature 004 shipped a Geometry Lab loader that skipped the check
 * when the field was absent, so tampered weights loaded cleanly the moment you DELETED
 * the field rather than edited it. `tests/unit/lexBundle.test.ts` runs that attack.
 *
 * ## `metrics` is outside every digest, and is therefore a CLAIM
 *
 * `metrics` is excluded from all three hashes, exactly as the backend's `_model_token`
 * excludes it, so that a file can be re-labelled without becoming a different model.
 *
 * This comment used to justify that by calling `metrics` "the one block that cannot
 * mislabel a token". That was false as shipped, and red-team finding F1 is the proof: the
 * Lexicon Lab reads `metrics.provenance` to decide whether the weights on screen were ever
 * trained, and every provenance-conditioned sentence in the tab follows the answer. The
 * block cannot mislabel a *token id* — the vocabulary is inside `model_token`, which is a
 * genuinely narrower statement than the one that stood here.
 *
 * So the rule is a division of labour, stated where a reader of a bundle can act on it:
 *
 *   * the three digests establish that these weights and this word list are the ones the
 *     file was written with, and a mismatch or an absence is fatal;
 *   * `metrics` establishes nothing. It round-trips verbatim — a forged `final_loss` of
 *     `1e-05` beside a random initialization survives an export/import cycle intact — so
 *     every surface that repeats it must attribute it to the file rather than present it
 *     as checked. `viz/lex/ModelFile.svelte` does that on load; `viz/lex/provenance.ts`
 *     documents what may and may not be concluded from it.
 *
 * Bringing `metrics` inside a digest was the alternative, and is rejected: the backend's
 * `_model_token` is the contract's hash (`api-lex.md`), a bundle written by either stack
 * must verify in the other, and a token that changed when a note was edited would make the
 * cache key depend on prose. The honest fix is the attribution, not a fourth hash.
 */

import { invalidParam } from "../geoEngine/errors";
import { sha256Hex, utf8Bytes } from "../geoEngine/hash";
import {
  MLP_RATIO,
  assertWeightsMatch,
  validateConfig,
  weightNames,
  type LexConfig,
  type WeightSet,
} from "./model";
import { BUDGET_SOURCES, SPECIAL_TOKENS, type BudgetSource } from "./vocab";

/** The contract's tag and version (`api-lex.md`, `routes_lex.py::BUNDLE_FORMAT`). */
export const LEX_BUNDLE_FORMAT = "llm-geometry/lex-model";
export const LEX_BUNDLE_VERSION = 1;
/** File suffix the save button writes, and the only one the loader advertises. */
export const LEX_BUNDLE_SUFFIX = ".llmlex.json";

/** The bundle exactly as `GET /api/lex/model` returns it, plus the two half-digests. */
export interface LexModelBundle {
  format: string;
  version: number;
  model_token: string;
  weights_token: string;
  vocab_sha256: string;
  config: LexWireConfig;
  vocab: { source: string; budget: string; words: string[]; specials: string[] };
  metrics: Record<string, unknown>;
  weights: Record<string, { shape: number[]; data: string }>;
}

export interface LexWireConfig {
  vocab_rows: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  ctx: number;
  tied: boolean;
  dropout: number;
}

export interface LexBundleInput {
  config: LexConfig;
  /** Engine-named weights (`layers.N.*`); translated to wire names on the way out. */
  weights: WeightSet;
  /** Budget words WITHOUT the specials, exactly as `LexVocab.words` holds them. */
  vocabWords: readonly string[];
  budgetSource: BudgetSource;
  budgetName: string;
  /** Free provenance. Carried through, never trusted, never hashed. */
  metrics?: Record<string, unknown>;
}

export interface LexBundleLoad {
  config: LexConfig;
  /** Engine-named weights, ready for `new LexModel(config, weights)`. */
  weights: WeightSet;
  vocabWords: string[];
  budgetSource: BudgetSource;
  budgetName: string;
  modelToken: string;
  weightsToken: string;
  vocabSha256: string;
  metrics: Record<string, unknown>;
}

// --- the engine ⇄ wire name bridge ----------------------------------------------------

/**
 * `layers.N.*` (browser engine) ⇄ `blocks.N.*` (PyTorch model, and therefore the wire).
 * The two layouts are the same row-major float32 buffers; `tests/unit/staticLex.test.ts`
 * loads a bundle in the wire names and reproduces PyTorch's logits to ≤1e-5.
 */
export function toWireName(name: string): string {
  return name.startsWith("layers.") ? `blocks.${name.slice("layers.".length)}` : name;
}

export function toEngineName(name: string): string {
  return name.startsWith("blocks.") ? `layers.${name.slice("blocks.".length)}` : name;
}

/**
 * Tensor shapes by WIRE name — what `repr(arr.shape)` hashes on the Python side, and
 * what the file declares. The key SET is also the tied-flag check: a tied model has no
 * `head_w`, so a tied bundle carrying one (or an untied one missing it) fails the
 * exact-key comparison in `readWeights` rather than loading as a different model. The
 * source project's `probe.py` dropped `tie` on reload and did exactly that.
 */
export function lexWireShapes(cfg: LexConfig): Record<string, number[]> {
  const d = cfg.dModel;
  const hidden = MLP_RATIO * d;
  const shapes: Record<string, number[]> = {
    embed: [cfg.vocabRows, d],
    pos: [cfg.ctx, d],
    lnf_g: [d],
    lnf_b: [d],
  };
  for (let l = 0; l < cfg.nLayers; l++) {
    const p = `blocks.${l}.`;
    shapes[`${p}ln1_g`] = [d];
    shapes[`${p}ln1_b`] = [d];
    shapes[`${p}qkv_w`] = [3 * d, d];
    shapes[`${p}qkv_b`] = [3 * d];
    shapes[`${p}proj_w`] = [d, d];
    shapes[`${p}proj_b`] = [d];
    shapes[`${p}ln2_g`] = [d];
    shapes[`${p}ln2_b`] = [d];
    shapes[`${p}fc1_w`] = [hidden, d];
    shapes[`${p}fc1_b`] = [hidden];
    shapes[`${p}fc2_w`] = [d, hidden];
    shapes[`${p}fc2_b`] = [d];
  }
  if (!cfg.tied) shapes.head_w = [cfg.vocabRows, d];
  return shapes;
}

/** The same shapes under the ENGINE's names, for anything that inspects tensors in-app. */
export function lexEngineShapes(cfg: LexConfig): Record<string, number[]> {
  const wire = lexWireShapes(cfg);
  const out: Record<string, number[]> = {};
  for (const name of weightNames(cfg)) out[name] = wire[toWireName(name)];
  return out;
}

/** The engine's config as the wire (and the backend) spells it. */
export function toWireConfig(cfg: LexConfig): LexWireConfig {
  return {
    vocab_rows: cfg.vocabRows,
    d_model: cfg.dModel,
    n_layers: cfg.nLayers,
    n_heads: cfg.nHeads,
    ctx: cfg.ctx,
    tied: cfg.tied,
    dropout: cfg.dropout,
  };
}

// --- digests ---------------------------------------------------------------------------

/** Python `repr(tuple)`: `(3, 3)` / `(12,)`. */
function reprShape(shape: number[]): string {
  return shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
}

/** `repr()` of a Python float: an integral value keeps its `.0`. */
function pyFloat(x: number): string {
  if (!Number.isFinite(x)) throw invalidParam(`dropout must be finite, got ${x}`);
  const s = String(x);
  if (s.includes("e") || s.includes("E")) {
    throw invalidParam(`dropout ${x} is too small to hash the way the backend does`);
  }
  return Number.isInteger(x) ? `${s}.0` : s;
}

/**
 * `cache/keys.py::_canonical` — `json.dumps(sort_keys=True, separators=(",", ":"),
 * ensure_ascii=True)`. Written out by hand for the two shapes that need it, because the
 * bytes must match Python's exactly. Every string involved (special tokens, budget names,
 * and word tokens, which the tokenizer restricts to `[A-Za-z'-]`) is ASCII, so
 * `ensure_ascii` and `JSON.stringify` cannot disagree.
 */
export function lexCanonicalConfig(cfg: LexWireConfig): string {
  return (
    `{"ctx":${cfg.ctx},"d_model":${cfg.d_model},"dropout":${pyFloat(cfg.dropout)},` +
    `"n_heads":${cfg.n_heads},"n_layers":${cfg.n_layers},"tied":${cfg.tied ? "true" : "false"},` +
    `"vocab_rows":${cfg.vocab_rows}}`
  );
}

/** The exact bytes `vocab_sha256` covers: the vocab block, canonical-JSON encoded. */
export function lexCanonicalVocab(
  words: readonly string[],
  source: string,
  budgetName: string,
): string {
  return (
    `{"budget":${JSON.stringify(budgetName)},"source":${JSON.stringify(source)},` +
    `"specials":${JSON.stringify([...SPECIAL_TOKENS])},"words":${JSON.stringify([...words])}}`
  );
}

function f32Bytes(values: ArrayLike<number>): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const dv = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) dv.setFloat32(i * 4, values[i], true); // LE
  return bytes;
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const c of chunks) total += c.length;
  const all = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    all.set(c, off);
    off += c.length;
  }
  return all;
}

/** `[utf8(name) ‖ utf8(repr(shape)) ‖ float32-LE]` for every tensor, name-sorted. */
function weightChunks(
  wire: Record<string, ArrayLike<number>>,
  shapes: Record<string, number[]>,
): Uint8Array[] {
  const chunks: Uint8Array[] = [];
  for (const name of Object.keys(wire).sort()) {
    const shape = shapes[name];
    if (shape === undefined) throw invalidParam(`unknown weight name '${name}'`);
    chunks.push(utf8Bytes(name), utf8Bytes(reprShape(shape)), f32Bytes(wire[name]));
  }
  return chunks;
}

/**
 * `routes_lex.py::_model_token`, reproduced byte-for-byte: sha256 over the canonical
 * config, then the canonical word list, then every weight; first 32 hex characters.
 *
 * The vocabulary is inside this hash on purpose (contract): two models with identical
 * weights but different word lists are different models. Pinned against a token the real
 * Python produced in `tests/unit/lexBundle.test.ts`.
 */
export function lexModelToken(
  wire: Record<string, ArrayLike<number>>,
  shapes: Record<string, number[]>,
  config: LexWireConfig,
  words: readonly string[],
): string {
  const chunks: Uint8Array[] = [
    utf8Bytes(lexCanonicalConfig(config)),
    utf8Bytes(JSON.stringify([...words])),
    ...weightChunks(wire, shapes),
  ];
  return sha256Hex(concatBytes(chunks)).slice(0, 32);
}

/**
 * The weights ALONE, addressed by content — the same construction `geoEngine`'s
 * `weightsToken` uses, so "content-hash addressed" means one thing across this project.
 * This is the id the Weight Lab mints for an edited weight set, and it is what tells you
 * two files hold the same weights under different word lists.
 */
export function lexWeightsToken(
  wire: Record<string, ArrayLike<number>>,
  shapes: Record<string, number[]>,
): string {
  return sha256Hex(concatBytes(weightChunks(wire, shapes))).slice(0, 32);
}

/** Convenience: the weights token of an ENGINE-named weight set. */
export function lexWeightsTokenOf(cfg: LexConfig, ws: WeightSet): string {
  return lexWeightsToken(toWire(cfg, ws), lexWireShapes(cfg));
}

/** sha256 of the canonical vocabulary block. The weights hash cannot cover a word list. */
export function lexVocabDigest(canonicalVocabJson: string): string {
  return sha256Hex(utf8Bytes(canonicalVocabJson));
}

/** Engine-named weights → wire-named, with the exact tensor set the config calls for. */
function toWire(cfg: LexConfig, ws: WeightSet): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  for (const name of weightNames(cfg)) out[toWireName(name)] = ws[name];
  return out;
}

// --- base64 ----------------------------------------------------------------------------

const B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/** Pure JS, so a bundle encodes identically in a window, a worker and Node. */
function toBase64(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = bytes[i + 1];
    const b2 = bytes[i + 2];
    out += B64_ALPHABET[b0 >> 2];
    out += B64_ALPHABET[((b0 & 3) << 4) | ((b1 ?? 0) >> 4)];
    out += i + 1 < bytes.length ? B64_ALPHABET[((b1 & 15) << 2) | ((b2 ?? 0) >> 6)] : "=";
    out += i + 2 < bytes.length ? B64_ALPHABET[b2 & 63] : "=";
  }
  return out;
}

function fromBase64(data: string, what: string): Uint8Array {
  const clean = data.replace(/[\n\r\t ]/g, "");
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(clean) || clean.length % 4 !== 0) {
    throw invalidParam(`weight ${JSON.stringify(what)} is not valid base64`);
  }
  const pad = clean.endsWith("==") ? 2 : clean.endsWith("=") ? 1 : 0;
  const out = new Uint8Array((clean.length / 4) * 3 - pad);
  let o = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const n =
      (B64_ALPHABET.indexOf(clean[i]) << 18) |
      (B64_ALPHABET.indexOf(clean[i + 1]) << 12) |
      ((clean[i + 2] === "=" ? 0 : B64_ALPHABET.indexOf(clean[i + 2])) << 6) |
      (clean[i + 3] === "=" ? 0 : B64_ALPHABET.indexOf(clean[i + 3]));
    if (o < out.length) out[o++] = (n >> 16) & 255;
    if (o < out.length) out[o++] = (n >> 8) & 255;
    if (o < out.length) out[o++] = n & 255;
  }
  return out;
}

function decodeF32(data: string, shape: number[], what: string): Float32Array {
  const raw = fromBase64(data, what);
  const expected = shape.reduce((a, b) => a * b, 1) * 4;
  if (raw.length !== expected) {
    throw invalidParam(
      `weight ${JSON.stringify(what)} is ${raw.length} bytes but shape [${shape.join(", ")}] ` +
        `needs ${expected}`,
    );
  }
  const out = new Float32Array(raw.length / 4);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

// --- export -----------------------------------------------------------------------------

/** Build the portable bundle. All three digests are computed here, from real contents. */
export function exportLexBundle(input: LexBundleInput): LexModelBundle {
  const config = validateConfig({ ...input.config });
  assertWeightsMatch(config, input.weights);
  if (input.vocabWords.length + SPECIAL_TOKENS.length !== config.vocabRows) {
    throw invalidParam(
      `this vocabulary has ${input.vocabWords.length} words plus ${SPECIAL_TOKENS.length} ` +
        `specials, but the model has ${config.vocabRows} embedding rows — refusing to write ` +
        "a file whose word list does not fit its own weights",
    );
  }
  if (!(BUDGET_SOURCES as readonly string[]).includes(input.budgetSource)) {
    throw invalidParam(`unknown budget source ${JSON.stringify(input.budgetSource)}`);
  }

  const wireConfig = toWireConfig(config);
  const shapes = lexWireShapes(config);
  const wire = toWire(config, input.weights);
  const words = [...input.vocabWords];
  const canonicalVocab = lexCanonicalVocab(words, input.budgetSource, input.budgetName);

  const weights: LexModelBundle["weights"] = {};
  for (const name of Object.keys(shapes).sort()) {
    weights[name] = { shape: [...shapes[name]], data: toBase64(f32Bytes(wire[name])) };
  }

  return {
    format: LEX_BUNDLE_FORMAT,
    version: LEX_BUNDLE_VERSION,
    model_token: lexModelToken(wire, shapes, wireConfig, words),
    weights_token: lexWeightsToken(wire, shapes),
    vocab_sha256: lexVocabDigest(canonicalVocab),
    config: wireConfig,
    vocab: { source: input.budgetSource, budget: input.budgetName, words, specials: [...SPECIAL_TOKENS] },
    metrics: { ...(input.metrics ?? {}) },
    weights,
  };
}

// --- import -----------------------------------------------------------------------------

function readConfig(raw: unknown): LexConfig {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw invalidParam("model file is missing its `config` object");
  }
  const c = raw as Record<string, unknown>;
  const num = (field: string): number => {
    const v = c[field];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw invalidParam(`model file's config.${field} is ${JSON.stringify(v)}, expected a number`);
    }
    return v;
  };
  if (typeof c.tied !== "boolean") {
    throw invalidParam(
      `model file's config.tied is ${JSON.stringify(c.tied)}, expected true or false`,
    );
  }
  // validateConfig raises on anything outside the documented choices (FR-611), so an
  // architecture this build cannot construct is refused rather than half-built.
  return validateConfig({
    vocabRows: num("vocab_rows"),
    dModel: num("d_model"),
    nLayers: num("n_layers"),
    nHeads: num("n_heads"),
    ctx: num("ctx"),
    tied: c.tied,
    dropout: num("dropout"),
  });
}

function readVocab(
  cfg: LexConfig,
  raw: unknown,
): { words: string[]; source: BudgetSource; budgetName: string } {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw invalidParam("model file is missing its `vocab` object");
  }
  const v = raw as Record<string, unknown>;
  const words = v.words;
  if (!Array.isArray(words) || words.some((w) => typeof w !== "string")) {
    throw invalidParam("model file's vocab.words must be a list of strings");
  }
  // The specials occupy rows 0..3 and give every id below 4 its meaning; reordering or
  // renaming them would shift every token id in the model by a silent offset.
  const specials = v.specials;
  if (
    !Array.isArray(specials) ||
    specials.length !== SPECIAL_TOKENS.length ||
    specials.some((s, i) => s !== SPECIAL_TOKENS[i])
  ) {
    throw invalidParam(
      `model file's special tokens are ${JSON.stringify(specials)}, but this build reserves ` +
        `rows 0..${SPECIAL_TOKENS.length - 1} for ${JSON.stringify([...SPECIAL_TOKENS])} — ` +
        "loading it would shift every token id",
    );
  }
  if (!(BUDGET_SOURCES as readonly unknown[]).includes(v.source)) {
    throw invalidParam(
      `model file's vocab.source is ${JSON.stringify(v.source)}, expected one of ` +
        `${BUDGET_SOURCES.join(", ")}`,
    );
  }
  if (typeof v.budget !== "string" || !v.budget) {
    throw invalidParam("model file's vocab.budget is missing");
  }
  // The digests are independent: each proves its own half is intact, and neither proves
  // the halves belong together. This is the join, and it is not optional.
  if (words.length + SPECIAL_TOKENS.length !== cfg.vocabRows) {
    throw invalidParam(
      `model file's vocabulary has ${words.length} words (+${SPECIAL_TOKENS.length} specials = ` +
        `${words.length + SPECIAL_TOKENS.length} rows) but its config declares ` +
        `${cfg.vocabRows} embedding rows; the weights and the word list do not describe ` +
        "one model, so it is refused",
    );
  }
  return { words: words as string[], source: v.source as BudgetSource, budgetName: v.budget };
}

function readWeights(cfg: LexConfig, raw: unknown): Record<string, Float32Array> {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw invalidParam("model file is missing its `weights` object");
  }
  const entries = raw as Record<string, { shape?: unknown; data?: unknown }>;
  const shapes = lexWireShapes(cfg);
  const wire: Record<string, Float32Array> = {};
  for (const [name, payload] of Object.entries(entries)) {
    const shape = shapes[name];
    if (shape === undefined) {
      throw invalidParam(
        `model file carries a weight ${JSON.stringify(name)} that a ` +
          `${cfg.tied ? "tied" : "untied"} ${cfg.nLayers}-layer model has no slot for — ` +
          "refusing to load a file this build does not fully understand",
      );
    }
    if (
      payload === null ||
      typeof payload !== "object" ||
      typeof payload.data !== "string" ||
      !Array.isArray(payload.shape)
    ) {
      throw invalidParam(`weight ${JSON.stringify(name)} must be an object with shape and data`);
    }
    const declared = payload.shape as unknown[];
    if (declared.length !== shape.length || declared.some((d, i) => d !== shape[i])) {
      throw invalidParam(
        `weight ${JSON.stringify(name)} declares shape [${declared.join(", ")}] but this ` +
          `config needs [${shape.join(", ")}]`,
      );
    }
    wire[name] = decodeF32(payload.data, shape, name);
  }
  // Exact key equality is also the TIED check: a tied bundle carrying `head_w`, or an
  // untied one missing it, is refused rather than reloaded as a different model.
  const missing = Object.keys(shapes).filter((n) => !(n in wire));
  if (missing.length > 0) {
    throw invalidParam(
      `model file is missing ${missing.join(", ")} — a ${cfg.tied ? "tied" : "untied"} model ` +
        "needs exactly the tensors its config implies",
    );
  }
  return wire;
}

/**
 * Validate a bundle and return the model it describes, in the engine's own names.
 *
 * Every failure is fatal and named. Nothing here falls back, repairs, or loads "the part
 * that verified" — a partially trusted model file is a model file that lies quietly.
 */
export function importLexBundle(bundle: unknown): LexBundleLoad {
  if (bundle === null || typeof bundle !== "object" || Array.isArray(bundle)) {
    throw invalidParam("a model file must be a JSON object");
  }
  const b = bundle as Record<string, unknown>;
  if (b.format !== LEX_BUNDLE_FORMAT) {
    throw invalidParam(
      `not a Lexicon Lab model bundle: format is ${JSON.stringify(b.format)}, expected ` +
        `${JSON.stringify(LEX_BUNDLE_FORMAT)}`,
    );
  }
  if (b.version !== LEX_BUNDLE_VERSION) {
    throw invalidParam(
      `bundle version ${JSON.stringify(b.version)} is not supported (this build reads ` +
        `version ${LEX_BUNDLE_VERSION})`,
    );
  }

  const config = readConfig(b.config);
  const vocab = readVocab(config, b.vocab);
  const wire = readWeights(config, b.weights);
  const shapes = lexWireShapes(config);
  const wireConfig = toWireConfig(config);

  // --- integrity. MANDATORY, not opt-in: treating a missing digest as "nothing to
  // check" is exactly the hole that let a tampered file load by DELETING a field.
  const digest = (field: string, hex: number): string => {
    const v = b[field];
    if (typeof v !== "string" || !new RegExp(`^[0-9a-f]{${hex}}$`).test(v)) {
      throw invalidParam(
        `model file has no usable \`${field}\`, so its contents cannot be verified — ` +
          "refusing to load it. Re-export the model to get a valid file.",
      );
    }
    return v;
  };

  const declaredModel = digest("model_token", 32);
  const actualModel = lexModelToken(wire, shapes, wireConfig, vocab.words);
  if (declaredModel !== actualModel) {
    throw invalidParam(
      `bundle declares model_token ${declaredModel} but its own contents hash to ` +
        `${actualModel}; refusing to load a file whose weights and label disagree`,
    );
  }

  const declaredWeights = digest("weights_token", 32);
  const actualWeights = lexWeightsToken(wire, shapes);
  if (declaredWeights !== actualWeights) {
    throw invalidParam(
      `this model file is corrupt: its weights hash to ${actualWeights} but it declares ` +
        `${declaredWeights}. Loading it would pair the wrong vocabulary with these weights, ` +
        "so it is refused.",
    );
  }

  const declaredVocab = digest("vocab_sha256", 64);
  const actualVocab = lexVocabDigest(
    lexCanonicalVocab(vocab.words, vocab.source, vocab.budgetName),
  );
  if (declaredVocab !== actualVocab) {
    throw invalidParam(
      `this model file is corrupt: its vocabulary hashes to ${actualVocab.slice(0, 16)}… but ` +
        `it declares ${declaredVocab.slice(0, 16)}…. Loading it would label every token with ` +
        "the wrong word, so it is refused.",
    );
  }

  const weights: WeightSet = {};
  for (const [name, values] of Object.entries(wire)) weights[toEngineName(name)] = values;
  assertWeightsMatch(config, weights);

  return {
    config,
    weights,
    vocabWords: vocab.words,
    budgetSource: vocab.source,
    budgetName: vocab.budgetName,
    modelToken: actualModel,
    weightsToken: actualWeights,
    vocabSha256: actualVocab,
    metrics:
      b.metrics !== null && typeof b.metrics === "object" && !Array.isArray(b.metrics)
        ? { ...(b.metrics as Record<string, unknown>) }
        : {},
  };
}
