/**
 * Lexicon Lab in static mode (feature 006) — `/api/lex/*` with no Python behind it.
 *
 * This tab is the easiest of the three to serve statically and the hardest to fake, and
 * the reason is the same: **it trains in the browser in both modes**. `src/lib/lexEngine/`
 * is the model, the budgets, the training recipe, the sampler and the spectrum — the same
 * ones `llm_geometry.lex` implements — so every number below is computed here, live, from
 * real weights. Nothing on this path is a stored answer pretending to be a computation.
 *
 * Per method, data is either:
 *   LIVE        — computed by lexEngine in this browser (budgets, coverage, training,
 *                 generation, spectrum, bundles), or
 *   PRECOMPUTED — the build-time export of a read-only backend response
 *                 (`static-data/lex/spec.json`, `budgets.json`), or the shipped corpus
 *                 TEXT (`static-data/lex/corpus.json`), which is a committed file on the
 *                 backend's disk and the one thing a browser cannot derive, or
 *   REFUSED     — a typed `StaticModeError` naming what IS available (FR-203).
 *
 * Two deliberate divergences from the live backend, both loud rather than silent:
 *
 *   1. `POST /generate` with `stop_at_eos: false` (the backend's default) is REFUSED.
 *      The browser engine's sampler always stops at `<eos>`; continuing past it and
 *      rendering it as a line break is something it cannot do, and synthesising the
 *      multi-line form by re-prompting would produce different tokens under a different
 *      RNG stream — a fabricated answer wearing the right shape. This client therefore
 *      defaults `stop_at_eos` to `true` and refuses an explicit `false`.
 *   2. A `model_token` is a handle into a store. The backend's is a persistent on-disk
 *      cache; this one is this page's memory. An unknown token is a `NotFoundError`
 *      exactly as the contract says, with a message that explains where models live here.
 *
 * NOTE (import cycle): like every module under staticClient/, `ApiError` may only be
 * touched inside functions — see ./errors.
 */

import {
  DEFAULT_BATCH,
  DEFAULT_BUDGET,
  DEFAULT_BUDGET_SOURCE,
  DEFAULT_CTX,
  DEFAULT_DROPOUT,
  DEFAULT_D_MODEL,
  DEFAULT_LR,
  DEFAULT_MAX_NEW_TOKENS,
  DEFAULT_N_HEADS,
  DEFAULT_N_LAYERS,
  DEFAULT_SAMPLE_EVERY,
  DEFAULT_SEED,
  DEFAULT_STEPS,
  DEFAULT_TEMPERATURE,
  DEFAULT_TIED,
  DEFAULT_WEIGHT_DECAY,
  DOLCH_ORDER,
  D_MODEL_CHOICES,
  CTX_CHOICES,
  GRAD_CLIP_NORM,
  LAYER_NORM_EPS,
  LexModel,
  LexVocab,
  MAX_NEW_TOKENS,
  MAX_STEPS,
  MLP_RATIO,
  N_HEAD_CHOICES,
  N_LAYER_CHOICES,
  ONECYCLE_DIV_FACTOR,
  ONECYCLE_FINAL_DIV_FACTOR,
  ONECYCLE_PCT_START,
  PCA_COMPONENTS,
  SPECIAL_TOKENS,
  SPECTRUM_DISPLAY_K,
  VAL_FRACTION,
  buildVocab,
  dolchSizes,
  generate,
  hasWord,
  paramCount,
  randomBaselineSpectrum,
  runTrainingJob,
  spectrum,
  splitLines,
  tokenize,
  validateConfig,
  type BudgetSource,
  type Coverage,
  type LexConfig,
  type LexTrainRequest,
  type LexTrainResponse,
  type SpectrumResult,
  type WeightSet,
} from "../lexEngine";
import {
  SWAP_INCONSISTENT_REFUSAL,
  noMappedVocabularyRefusal,
} from "../lexEngine/vacancyRefusals";
import {
  buildVacancyMap,
  mapVocabWords,
  typeCounts,
  vacancyDomain,
  vacancyParams as vacancyParamsWithDefaults,
  vacancyStats,
  vacateText,
  type MintStrategy,
  type VacancyMap,
  type VacancyParams,
  type VacancyStats,
} from "../lexEngine/vacancy";
import { sha256Hex, utf8Bytes } from "../geoEngine/hash";
import type { StaticAssets } from "./assets";
import { computeError, invalidParamError, notFoundError, staticModeError, toApiError } from "./errors";
import { fetchDatasetText } from "./hfDatasets";
import type { LocalJobRegistry, ProgressFn } from "./jobs";

// --- contract payload shapes (specs/006-lexicon-lab-tiny/contracts/api-lex.md) ---------

export interface LexCorpusAsset {
  format: string;
  title: string;
  year: number;
  gutenberg_id: number;
  /** sha256 of the COMMITTED file, header and licence footer included. */
  sha256: string;
  bytes: number;
  /** sha256 of the trimmed body carried in `text` — an integrity check on the download. */
  body_sha256: string;
  body_bytes: number;
  n_tokens: number;
  n_distinct: number;
  n_lines: number;
  n_chars: number;
  text: string;
}

export interface LexSpec {
  corpus: {
    title: string;
    year: number;
    gutenberg_id: number;
    sha256: string;
    bytes: number;
    n_tokens: number;
    n_distinct: number;
    n_lines: number;
    n_chars: number;
  };
  budget_sources: string[];
  budgets: { name: string; size: number; rows: number }[];
  special_tokens: Record<string, number>;
  generation_banned_ids: number[];
  model: {
    d_model_choices: number[];
    n_layer_choices: number[];
    n_head_choices: number[];
    ctx_choices: number[];
    mlp_ratio: number;
    layer_norm_eps: number;
    defaults: {
      d_model: number;
      n_layers: number;
      n_heads: number;
      ctx: number;
      tied: boolean;
      dropout: number;
      budget_source: string;
      budget: string;
    };
  };
  training: {
    max_steps: number;
    grad_clip_norm: number;
    val_fraction: number;
    onecycle: { pct_start: number; div_factor: number; final_div_factor: number };
    defaults: {
      steps: number;
      lr: number;
      batch_size: number;
      weight_decay: number;
      seed: number;
      sample_every: number;
    };
  };
  generation: {
    max_new_tokens_limit: number;
    defaults: { temperature: number; max_new_tokens: number; seed: number };
  };
  spectrum: { pca_components: number; display_k: number };
}

export interface LexShapeParams {
  d_model?: number;
  n_layers?: number;
  n_heads?: number;
  ctx?: number;
  tied?: boolean;
  dropout?: number;
}

export interface LexBudgetRow {
  source: string;
  budget: string;
  size: number;
  rows: number;
  coverage: Coverage;
  param_count: number;
}

export interface LexBudgetsResult {
  source: string;
  corpus: { title: string; n_tokens: number; n_distinct: number; n_lines: number; n_chars: number };
  model: { d_model: number; n_layers: number; n_heads: number; ctx: number; tied: boolean; dropout: number };
  budgets: LexBudgetRow[];
}

export interface LexCorpusBody {
  text?: string;
  hf_dataset?: string;
  hf_split?: string;
  max_samples?: number;
}

export interface LexCoverageBody extends LexCorpusBody {
  source?: string;
  budget?: string;
  size?: number;
}

export interface LexCoverageResult {
  source: string;
  budget: string;
  size: number;
  rows: number;
  coverage: Coverage;
  corpus: { n_tokens: number; n_distinct: number; n_lines: number; n_chars: number };
  oov_sample: { word: string; count: number }[];
  words: string[];
}

/**
 * Contract §7.1's knobs, on the wire. `snake_case` because that is this API's convention;
 * the transform's own TypeScript surface is camelCase in both stacks (§5.8), and
 * `vacancyParamsFrom` is the single place the two spellings meet.
 */
export interface LexVacancyParamsBody {
  p?: number;
  seed?: number;
  consistent?: boolean;
  match_prosody?: boolean;
  reveal_after?: number;
  keep?: readonly string[];
  /**
   * §8.3's minting strategy. Typed as `string`, not `MintStrategy`, because this is a WIRE
   * field: it arrives from a caller who may send anything, and the point of parsing it is
   * to refuse what is not one of the two rather than to assume it away.
   */
  mint?: string;
}

export interface LexVacancyBody extends LexCoverageBody, LexVacancyParamsBody {
  preview_chars?: number;
}

/** §10's statistics, camelCase in both stacks because §10 names them that way. */
export interface LexVacancyStats {
  domainTypesTotal: number;
  domainTypesEligible: number;
  domainTypesVacated: number;
  corpusTypesTotal: number;
  corpusTypesEligible: number;
  corpusTypesVacated: number;
  stemsTotal: number;
  stemsVacated: number;
  tokensTotal: number;
  tokensVacated: number;
  meanSyllablesBefore: number;
  meanSyllablesAfter: number;
  meanAnapestBefore: number;
  meanAnapestAfter: number;
  stressFromTableBefore: number;
  stressFromTableAfter: number;
  stressFromMintedBefore: number;
  stressFromMintedAfter: number;
  stressFromRuleBefore: number;
  stressFromRuleAfter: number;
  bijective: boolean;
  imageSize: number;
  remintRounds: number;
}

export interface LexVacancyResult {
  p: number;
  seed: number;
  consistent: boolean;
  match_prosody: boolean;
  reveal_after: number;
  keep: string[];
  /**
   * Which minting strategy actually ran. Echoed for the same reason the five knobs above
   * are: a caller must be able to see WHICH transform produced `vacated_sha256`, not infer
   * it from what it asked for.
   */
  mint: MintStrategy;
  /** Which of §7.2's two rules produced `words`. */
  vocabulary_rule: "mapped" | "rebuilt";
  words: string[];
  budget: { source: string; budget: string; size: number; rows: number; coverage: Coverage };
  corpus: { n_tokens: number; n_distinct: number; n_lines: number; n_chars: number };
  vacancy_stats: LexVacancyStats;
  bijective: boolean;
  remint_rounds: number;
  preview: string;
  original_preview: string;
  preview_chars: number;
  truncated: boolean;
  vacated_chars: number;
  vacated_sha256: string;
  original_chars: number;
  original_sha256: string;
}

export interface LexTrainBody extends LexCoverageBody, LexShapeParams {
  steps?: number;
  lr?: number;
  batch_size?: number;
  weight_decay?: number;
  seed?: number;
  sample_every?: number;
  base?: string;
  /** Feature 007, optional and additive: train on the VACATED corpus (§7.2). */
  vacancy?: LexVacancyParamsBody;
}

export interface LexTrainRecord {
  model_token: string;
  first_loss: number;
  final_loss: number;
  val_loss: number;
  steps: number;
  seed: number;
  elapsed_s: number;
  n_tokens: number;
  vocab_size: number;
  vocab_rows: number;
  param_count: number;
  sample: string;
  history: { step: number; loss: number; lr: number }[];
}

export type LexTrainResult =
  | ({ ready: true } & LexTrainRecord)
  | { ready: false; job_id: string };

export interface LexSpectrumParams {
  model_token: string;
  matrix?: string;
  baseline?: boolean;
  baseline_seed?: number;
}

export interface LexSpectrumBlock {
  rows: number;
  d_model: number;
  max_rank: number;
  eigenvalues: number[];
  singular_values: number[];
  explained_variance: number[];
  total_variance: number;
  effective_rank: number;
  stable_rank: number;
  participation_ratio: number;
  frac_var_top2: number;
  frac_var_top10: number;
  n_dims_for_90pct: number;
  pca_coords: number[][];
  pca_explained_variance_ratio: number[];
  degenerate: boolean;
}

export type LexSpectrumSummary = Omit<
  LexSpectrumBlock,
  "eigenvalues" | "singular_values" | "explained_variance" | "pca_coords" | "pca_explained_variance_ratio"
>;

export interface LexSpectrumResult {
  model_token: string;
  matrix: string;
  tied: boolean;
  projection: "pca";
  display_k: number;
  tokens: string[];
  spectrum: LexSpectrumBlock;
  baseline?: LexSpectrumSummary;
  comparison?: Record<string, number>;
}

export interface LexGenerateBody {
  model_token: string;
  prompt?: string;
  temperature?: number;
  max_new_tokens?: number;
  seed?: number;
  stop_at_eos?: boolean;
}

export interface LexGenerateResult {
  model_token: string;
  prompt: string;
  text: string;
  words: string[];
  n_words: number;
  out_of_budget: string[];
  prompt_tokens: { text: string; id: number; unk: boolean }[];
  temperature: number;
  seed: number;
  vocab_size: number;
  final_loss: number | null;
}

export interface LexModelBundle {
  format: string;
  version: number;
  model_token: string;
  config: {
    vocab_rows: number;
    d_model: number;
    n_layers: number;
    n_heads: number;
    ctx: number;
    tied: boolean;
    dropout: number;
  };
  vocab: { source: string; budget: string; words: string[]; specials: string[] };
  metrics: Record<string, unknown>;
  weights: Record<string, { shape: number[]; data: string }>;
}

export interface LexImportResult {
  model_token: string;
  config: LexModelBundle["config"];
  vocab_size: number;
  vocab_rows: number;
  param_count: number;
}

/** The `/api/lex/*` surface, identical in shape to what the live backend serves. */
export interface LexClient {
  lexSpec(): Promise<LexSpec>;
  lexBudgets(params?: LexShapeParams & { source?: string }): Promise<LexBudgetsResult>;
  lexCoverage(body?: LexCoverageBody): Promise<LexCoverageResult>;
  lexVacancy(body?: LexVacancyBody): Promise<LexVacancyResult>;
  lexTrain(body?: LexTrainBody): Promise<LexTrainResult>;
  lexSpectrum(params: LexSpectrumParams): Promise<LexSpectrumResult>;
  lexGenerate(body: LexGenerateBody): Promise<LexGenerateResult>;
  lexExportModel(model_token: string): Promise<LexModelBundle>;
  lexImportModel(bundle: unknown): Promise<LexImportResult>;
}

export const LEX_BUNDLE_FORMAT = "llm-geometry/lex-model";
export const LEX_BUNDLE_VERSION = 1;

/** The most frequent out-of-budget types returned by /coverage. A sample, labelled so. */
const OOV_SAMPLE_SIZE = 24;

/**
 * Characters of vacated text returned inline by default, and the ceiling a caller may
 * ask for. These MUST equal `routes_lex.py`'s `VACANCY_PREVIEW_CHARS` /
 * `VACANCY_PREVIEW_MAX`: the API-parity fixture is generated with the default and
 * compared here, so a drift in the default fails a test rather than quietly serving two
 * different excerpts in the two modes.
 */
export const VACANCY_PREVIEW_CHARS = 2000;
export const VACANCY_PREVIEW_MAX = 20000;

/**
 * Contract §7.1's knobs out of a wire body, snake_case to camelCase, defaults filled by
 * the engine's own `vacancyParams` so this cannot drift from them.
 *
 * Written as a spread over the engine's defaults rather than as an object literal on
 * purpose: the transform's parameter set is still growing (§8.3's swap mint), and a
 * literal here would silently drop whatever is added next.
 *
 * That comment used to be aspirational: `mint` was in the engine's params, in the UI and
 * in the transform, and NOT read here — so this function answered `mint="swap"` with nonce
 * output. Every knob the engine declares must be read out of the body below, and an
 * unrecognised value must be REFUSED, because a control silently replaced by its default
 * is indistinguishable from a control that worked.
 */
function vacancyParamsFrom(body: LexVacancyParamsBody): VacancyParams {
  const keep = body.keep;
  if (typeof keep === "string") {
    throw invalidParamError(
      `keep must be a list of words, not a string (got ${JSON.stringify(keep)}); ` +
        "a string would be read letter by letter",
    );
  }
  if (keep != null && !Array.isArray(keep)) {
    throw invalidParamError(`keep must be a list of words, got ${JSON.stringify(keep)}`);
  }
  const p = asFloat(body.p, "p", 0);
  if (!(Number.isFinite(p) && p >= 0 && p <= 1)) {
    throw invalidParamError(`p must lie in [0, 1], got ${JSON.stringify(body.p)}`);
  }
  const revealAfter = asInt(body.reveal_after, "reveal_after", 0);
  if (revealAfter < 0) throw invalidParamError(`reveal_after must be >= 0, got ${revealAfter}`);
  const mint = body.mint === undefined ? "nonce" : body.mint;
  // `Object.hasOwn`, not `in`: `in` walks the prototype chain, so `"constructor"`,
  // `"toString"`, `"valueOf"`, `"hasOwnProperty"` and `"__proto__"` all passed this check
  // on their way to being cast to `MintStrategy`. The backend answers those six with the
  // same typed 400 as `"bogus"`; this stack answered them with an untyped `Error` thrown
  // out of `buildVacancyMap`, which is the divergence this check exists to prevent.
  if (!(typeof mint === "string" && Object.hasOwn(MINT_STRATEGIES, mint))) {
    throw invalidParamError(
      `mint must be one of ${Object.keys(MINT_STRATEGIES).join(", ")}, got ` +
        `${JSON.stringify(body.mint)}`,
    );
  }
  const consistent = asBool(body.consistent, "consistent", true);
  // Mirrors `VacancyParams.__post_init__`, which refuses the same pair, so the two stacks
  // answer this request with the same typed error rather than one 400 and one 200. (The
  // engine checks it again at map-build time; this is the wire boundary's copy, in the
  // same place and for the same reason as the `p` and `reveal_after` range checks above.)
  //
  // The sentence itself comes from the engine (`SWAP_INCONSISTENT_REFUSAL`) rather than
  // being retyped here: this copy said `1680` / `8202` for a full release after the
  // transform rewrite moved the counts to `1676` / `8125`, so the wire boundary the
  // deployed site runs disagreed with the engine in the same bundle.
  if (mint === "swap" && !consistent) {
    throw invalidParamError(SWAP_INCONSISTENT_REFUSAL);
  }
  return vacancyParamsWithDefaults({
    p,
    seed: asInt(body.seed, "seed", 0),
    consistent,
    matchProsody: asBool(body.match_prosody, "match_prosody", true),
    revealAfter,
    keep: keep == null ? [] : keep.map(String),
    mint: mint as MintStrategy,
  });
}

/**
 * The runtime spelling of `MintStrategy`, which is a compile-time union and therefore
 * cannot be iterated. Declared as a `Record<MintStrategy, true>` so that adding a third
 * strategy to the union fails to compile HERE — the alternative, an array cast to the
 * union, would keep compiling while quietly refusing the new strategy on the wire.
 */
const MINT_STRATEGIES: Record<MintStrategy, true> = { nonce: true, swap: true };

/**
 * The map for these parameters, with the frequency counts `mint = "swap"` needs.
 *
 * `buildVacancyMap` throws rather than inventing a ranking when they are missing, so this
 * is not defensive: it is the one place both call sites (`lexVacancy` and `lexTrain`) get
 * the same corpus's counts, which is what makes `swap` the same transform in both.
 */
function buildMapFor(text: string, params: VacancyParams): VacancyMap {
  const tokens = tokenize(text);
  return buildVacancyMap(
    vacancyDomain(tokens),
    params,
    params.mint === "swap" ? typeCounts(tokens) : undefined,
  );
}

/**
 * The backend runs the whole response through `jsonable_6sig`, so the counts pass through
 * as integers and every measured fraction is rounded to 6 significant digits. Doing the
 * same here is what lets the two payloads be compared field for field.
 */
function roundVacancyStats(stats: VacancyStats): LexVacancyStats {
  return {
    ...stats,
    meanSyllablesBefore: sig6(stats.meanSyllablesBefore, "meanSyllablesBefore"),
    meanSyllablesAfter: sig6(stats.meanSyllablesAfter, "meanSyllablesAfter"),
    meanAnapestBefore: sig6(stats.meanAnapestBefore, "meanAnapestBefore"),
    meanAnapestAfter: sig6(stats.meanAnapestAfter, "meanAnapestAfter"),
    stressFromTableBefore: sig6(stats.stressFromTableBefore, "stressFromTableBefore"),
    stressFromTableAfter: sig6(stats.stressFromTableAfter, "stressFromTableAfter"),
    stressFromMintedBefore: sig6(stats.stressFromMintedBefore, "stressFromMintedBefore"),
    stressFromMintedAfter: sig6(stats.stressFromMintedAfter, "stressFromMintedAfter"),
    stressFromRuleBefore: sig6(stats.stressFromRuleBefore, "stressFromRuleBefore"),
    stressFromRuleAfter: sig6(stats.stressFromRuleAfter, "stressFromRuleAfter"),
  };
}

// --- transport parity ------------------------------------------------------------------

/**
 * The backend rounds every float in a response to 6 significant digits
 * (`api/encoding.py::jsonable_6sig`) and refuses non-finite values. Applying the same
 * rule here keeps a static payload byte-comparable with a live one, and keeps a NaN from
 * ever reaching a caller as a number.
 */
function sig6(x: number, what: string): number {
  if (!Number.isFinite(x)) {
    throw computeError(`non-finite value in ${what} — refusing to serve it as a number`);
  }
  return Number(x.toPrecision(6));
}

function sig6All(xs: ArrayLike<number>, what: string): number[] {
  const out: number[] = [];
  for (let i = 0; i < xs.length; i++) out.push(sig6(xs[i], what));
  return out;
}

// --- weight-name bridge ----------------------------------------------------------------

/**
 * The saved-bundle wire format uses the PYTHON model's parameter names (`blocks.N.*`);
 * the browser engine names the same tensors `layers.N.*`. Translating here — rather than
 * shipping the engine's names — is what lets a file saved in this build load into the
 * full stack and vice versa, which is the entire point of US-8. The two layouts are the
 * same row-major float32 buffers (verified against the real PyTorch model in
 * tests/unit/staticLex.test.ts, which loads a bundle in these names and reproduces
 * PyTorch's logits).
 */
function toWireName(name: string): string {
  return name.startsWith("layers.") ? `blocks.${name.slice("layers.".length)}` : name;
}

function toEngineName(name: string): string {
  return name.startsWith("blocks.") ? `layers.${name.slice("blocks.".length)}` : name;
}

/** Tensor shapes by WIRE name, derived from the config (what `repr(arr.shape)` hashes). */
function wireShapes(cfg: LexConfig): Record<string, number[]> {
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

// --- model_token (the backend's content hash, reproduced exactly) -----------------------

/** Python `repr(tuple)`: `(3, 3)` / `(12,)`. */
function reprShape(shape: number[]): string {
  return shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
}

/** `repr()` of a Python float: an integral value keeps its `.0`. */
function pyFloat(x: number): string {
  if (!Number.isFinite(x)) throw invalidParamError(`dropout must be finite, got ${x}`);
  const s = String(x);
  if (s.includes("e") || s.includes("E")) {
    throw invalidParamError(`dropout ${x} is too small to hash the way the backend does`);
  }
  return Number.isInteger(x) ? `${s}.0` : s;
}

/** `cache/keys.py::_canonical` of a `LexConfig.as_dict()` — sorted keys, no whitespace. */
function canonicalConfig(cfg: LexModelBundle["config"]): string {
  return (
    `{"ctx":${cfg.ctx},"d_model":${cfg.d_model},"dropout":${pyFloat(cfg.dropout)},` +
    `"n_heads":${cfg.n_heads},"n_layers":${cfg.n_layers},"tied":${cfg.tied ? "true" : "false"},` +
    `"vocab_rows":${cfg.vocab_rows}}`
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

/**
 * `routes_lex.py::_model_token`, reproduced byte-for-byte: sha256 over the canonical
 * config, then the canonical word list, then each weight (name-sorted) as
 * `utf8(name) ‖ utf8(repr(shape)) ‖ float32-LE bytes`; first 32 hex characters.
 *
 * The vocabulary is inside the hash on purpose (contract): two models with identical
 * weights but different word lists are different models, and one token for both would
 * let a cache hit serve the wrong labels. Reproducing the hash exactly is what lets a
 * bundle written by the full stack pass this build's `model_token` check instead of
 * being refused as a file whose weights and label disagree.
 */
export function lexModelToken(
  weights: Record<string, ArrayLike<number>>,
  shapes: Record<string, number[]>,
  config: LexModelBundle["config"],
  words: readonly string[],
): string {
  const chunks: Uint8Array[] = [
    utf8Bytes(canonicalConfig(config)),
    utf8Bytes(JSON.stringify([...words])),
  ];
  for (const name of Object.keys(weights).sort()) {
    const shape = shapes[name];
    if (shape === undefined) throw invalidParamError(`unknown weight name '${name}'`);
    chunks.push(utf8Bytes(name), utf8Bytes(reprShape(shape)), f32Bytes(weights[name]));
  }
  return sha256Hex(concatBytes(chunks)).slice(0, 32);
}

// --- base64 (float32-LE payloads in the portable bundle) --------------------------------

const B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/** Pure-JS base64, so the bundle encodes identically in a window, a worker and Node. */
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

function fromBase64(data: string): Uint8Array {
  const clean = data.replace(/[\n\r\t ]/g, "");
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(clean) || clean.length % 4 !== 0) {
    throw invalidParamError("weight payload is not valid base64");
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

function decodeF32(data: string, shape: number[]): Float32Array {
  const raw = fromBase64(data);
  const expected = shape.reduce((a, b) => a * b, 1) * 4;
  if (raw.length !== expected) {
    throw invalidParamError(
      `weight payload is ${raw.length} bytes but shape [${shape}] needs ${expected}`,
    );
  }
  const out = new Float32Array(raw.length / 4);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

// --- parameter coercion (the backend's error envelope, verbatim) ------------------------

function asInt(value: unknown, name: string, fallback: number): number {
  if (value === undefined || value === null) return fallback;
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n) || !Number.isInteger(n)) {
    throw invalidParamError(`${name} must be an integer, got ${JSON.stringify(value)}`);
  }
  return n;
}

function asFloat(value: unknown, name: string, fallback: number): number {
  if (value === undefined || value === null) return fallback;
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) {
    throw invalidParamError(`${name} must be a number, got ${JSON.stringify(value)}`);
  }
  return n;
}

function asBool(value: unknown, name: string, fallback: boolean): boolean {
  if (value === undefined || value === null) return fallback;
  if (typeof value === "boolean") return value;
  if (typeof value === "string" && ["true", "false", "1", "0"].includes(value.toLowerCase())) {
    return value.toLowerCase() === "true" || value === "1";
  }
  throw invalidParamError(`${name} must be a boolean, got ${JSON.stringify(value)}`);
}

function oneOf(value: unknown, choices: readonly number[], name: string, fallback: number): number {
  const n = asInt(value, name, fallback);
  if (!choices.includes(n)) {
    throw invalidParamError(`${name} must be one of [${choices.join(", ")}], got ${n}`);
  }
  return n;
}

/** `routes_lex.py::_model_config_from` — the shape controls, validated the same way. */
function configFrom(params: LexShapeParams, vocabRows: number): LexConfig {
  const dModel = oneOf(params.d_model, D_MODEL_CHOICES, "d_model", DEFAULT_D_MODEL);
  const nLayers = oneOf(params.n_layers, N_LAYER_CHOICES, "n_layers", DEFAULT_N_LAYERS);
  const nHeads = oneOf(params.n_heads, N_HEAD_CHOICES, "n_heads", DEFAULT_N_HEADS);
  const ctx = oneOf(params.ctx, CTX_CHOICES, "ctx", DEFAULT_CTX);
  if (dModel % nHeads !== 0) {
    throw invalidParamError(
      "d_model must be divisible by n_heads so every head gets an equal slice: " +
        `d_model=${dModel}, n_heads=${nHeads}`,
    );
  }
  const dropout = asFloat(params.dropout, "dropout", DEFAULT_DROPOUT);
  if (!(dropout >= 0 && dropout < 1)) {
    throw invalidParamError(`dropout must be in [0, 1), got ${dropout}`);
  }
  return { vocabRows, dModel, nLayers, nHeads, ctx, tied: asBool(params.tied, "tied", DEFAULT_TIED), dropout };
}

function asConfigDict(cfg: LexConfig): LexModelBundle["config"] {
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

// --- corpus statistics (the backend's `_corpus_stats`, same definitions) ----------------

interface CorpusStats {
  n_tokens: number;
  n_distinct: number;
  n_lines: number;
  n_chars: number;
}

function corpusStats(text: string): CorpusStats {
  const toks = tokenize(text);
  return {
    n_tokens: toks.length,
    n_distinct: new Set(toks).size,
    // Python: `[line for line in text.splitlines() if line.strip()]` — non-blank lines,
    // which is a different count from coverage's "lines containing a word token".
    n_lines: splitLines(text).filter((line) => line.trim() !== "").length,
    n_chars: text.length,
  };
}

interface StoredModel {
  token: string;
  model: LexModel;
  vocab: LexVocab;
  cfg: LexConfig;
  metrics: Record<string, unknown>;
}

/** The Lexicon Lab's `/api/lex/*` surface, computed in the browser. */
export class LexSection {
  private corpusPromise: Promise<LexCorpusAsset> | null = null;
  private readonly models = new Map<string, StoredModel>();
  private readonly trainCache = new Map<string, LexTrainRecord>();

  constructor(
    private readonly assets: StaticAssets,
    private readonly jobs: LocalJobRegistry,
  ) {}

  // --- assets ---------------------------------------------------------------------

  /**
   * The shipped corpus TEXT. This is the one thing on this tab that a browser cannot
   * derive: the book is a committed file in the backend's package data, exported by
   * `scripts/export_static_assets.py`. A build without it is refused rather than run on
   * substitute text, because every number on the page is measured against this string.
   */
  corpus(): Promise<LexCorpusAsset> {
    if (!this.corpusPromise) {
      this.corpusPromise = (async () => {
        let asset: LexCorpusAsset;
        try {
          asset = await this.assets.json<LexCorpusAsset>("lex/corpus.json");
        } catch (e) {
          const err = toApiError(e);
          if (err.type === "NotFoundError") {
            throw staticModeError(
              "This build did not export the Lexicon Lab corpus (static-data/lex/" +
                "corpus.json). The corpus is a committed public-domain file the backend " +
                "serves; the browser cannot reconstruct it, and measuring budgets against " +
                "substitute text would make every number on this tab a fiction. Rebuild " +
                "with `python scripts/export_static_assets.py`, or run the full stack.",
            );
          }
          throw err;
        }
        if (typeof asset?.text !== "string" || asset.text.length === 0) {
          throw computeError("static-data/lex/corpus.json carries no corpus text");
        }
        if (asset.body_sha256) {
          const actual = sha256Hex(utf8Bytes(asset.text));
          if (actual !== asset.body_sha256) {
            throw computeError(
              `the exported corpus hashes to ${actual.slice(0, 16)}… but declares ` +
                `${String(asset.body_sha256).slice(0, 16)}… — refusing to measure budgets ` +
                "against text that is not the text the export recorded",
            );
          }
        }
        return asset;
      })().catch((e) => {
        this.corpusPromise = null;
        throw toApiError(e);
      });
    }
    return this.corpusPromise;
  }

  /**
   * GET /api/lex/spec — served verbatim from the build-time export of the real route,
   * then CROSS-CHECKED against the browser engine's own constants. The export and the
   * engine are two independent transcriptions of `lex/config.py`; if they ever disagree
   * the page would document one model and run another, so this fails loudly instead.
   */
  async lexSpec(): Promise<LexSpec> {
    const spec = await this.assets.json<LexSpec>("lex/spec.json").catch((e) => {
      const err = toApiError(e);
      if (err.type === "NotFoundError") {
        throw staticModeError(
          "This build did not export the Lexicon Lab spec (static-data/lex/spec.json). " +
            "Rebuild with `python scripts/export_static_assets.py`, or run the full stack.",
        );
      }
      throw err;
    });

    const sizes = dolchSizes();
    const mismatches: string[] = [];
    const check = (path: string, exported: unknown, engine: unknown): void => {
      if (JSON.stringify(exported) !== JSON.stringify(engine)) {
        mismatches.push(`${path}: export ${JSON.stringify(exported)} vs engine ${JSON.stringify(engine)}`);
      }
    };
    check("model.d_model_choices", spec.model.d_model_choices, [...D_MODEL_CHOICES]);
    check("model.n_layer_choices", spec.model.n_layer_choices, [...N_LAYER_CHOICES]);
    check("model.n_head_choices", spec.model.n_head_choices, [...N_HEAD_CHOICES]);
    check("model.ctx_choices", spec.model.ctx_choices, [...CTX_CHOICES]);
    check("model.mlp_ratio", spec.model.mlp_ratio, MLP_RATIO);
    check("model.layer_norm_eps", spec.model.layer_norm_eps, LAYER_NORM_EPS);
    check("model.defaults.d_model", spec.model.defaults.d_model, DEFAULT_D_MODEL);
    check("model.defaults.n_layers", spec.model.defaults.n_layers, DEFAULT_N_LAYERS);
    check("model.defaults.n_heads", spec.model.defaults.n_heads, DEFAULT_N_HEADS);
    check("model.defaults.ctx", spec.model.defaults.ctx, DEFAULT_CTX);
    check("model.defaults.tied", spec.model.defaults.tied, DEFAULT_TIED);
    check("model.defaults.dropout", spec.model.defaults.dropout, DEFAULT_DROPOUT);
    check("model.defaults.budget_source", spec.model.defaults.budget_source, DEFAULT_BUDGET_SOURCE);
    check("model.defaults.budget", spec.model.defaults.budget, DEFAULT_BUDGET);
    check("training.max_steps", spec.training.max_steps, MAX_STEPS);
    check("training.grad_clip_norm", spec.training.grad_clip_norm, GRAD_CLIP_NORM);
    check("training.val_fraction", spec.training.val_fraction, VAL_FRACTION);
    check("training.onecycle.pct_start", spec.training.onecycle.pct_start, ONECYCLE_PCT_START);
    check("training.onecycle.div_factor", spec.training.onecycle.div_factor, ONECYCLE_DIV_FACTOR);
    check(
      "training.onecycle.final_div_factor",
      spec.training.onecycle.final_div_factor,
      ONECYCLE_FINAL_DIV_FACTOR,
    );
    check("training.defaults.steps", spec.training.defaults.steps, DEFAULT_STEPS);
    check("training.defaults.lr", spec.training.defaults.lr, DEFAULT_LR);
    check("training.defaults.batch_size", spec.training.defaults.batch_size, DEFAULT_BATCH);
    check("training.defaults.weight_decay", spec.training.defaults.weight_decay, DEFAULT_WEIGHT_DECAY);
    check("training.defaults.seed", spec.training.defaults.seed, DEFAULT_SEED);
    check("training.defaults.sample_every", spec.training.defaults.sample_every, DEFAULT_SAMPLE_EVERY);
    check("generation.max_new_tokens_limit", spec.generation.max_new_tokens_limit, MAX_NEW_TOKENS);
    check("generation.defaults.temperature", spec.generation.defaults.temperature, DEFAULT_TEMPERATURE);
    check(
      "generation.defaults.max_new_tokens",
      spec.generation.defaults.max_new_tokens,
      DEFAULT_MAX_NEW_TOKENS,
    );
    check("spectrum.pca_components", spec.spectrum.pca_components, PCA_COMPONENTS);
    check("spectrum.display_k", spec.spectrum.display_k, SPECTRUM_DISPLAY_K);
    check(
      "budgets",
      spec.budgets,
      DOLCH_ORDER.map((name) => ({ name, size: sizes[name], rows: sizes[name] + SPECIAL_TOKENS.length })),
    );
    if (mismatches.length > 0) {
      throw computeError(
        "the exported Lexicon spec disagrees with the browser engine — " +
          "this build would document one model and run another: " +
          mismatches.join("; "),
      );
    }
    return spec;
  }

  // --- budgets + coverage -----------------------------------------------------------

  /** GET /api/lex/budgets — every budget measured against the shipped corpus, live. */
  async lexBudgets(params: LexShapeParams & { source?: string } = {}): Promise<LexBudgetsResult> {
    const asset = await this.corpus();
    const source = this.assertSource(params.source ?? DEFAULT_BUDGET_SOURCE);
    const cfg = configFrom(params, 0);
    const stats = corpusStats(asset.text);
    const budgets = DOLCH_ORDER.map((name) => {
      const vocab = buildVocab(source, name, asset.text);
      return this.budgetRow(vocab, asset.text, cfg);
    });
    return {
      source,
      corpus: { title: asset.title, ...stats },
      model: {
        d_model: cfg.dModel,
        n_layers: cfg.nLayers,
        n_heads: cfg.nHeads,
        ctx: cfg.ctx,
        tied: cfg.tied,
        dropout: cfg.dropout,
      },
      budgets,
    };
  }

  /** POST /api/lex/coverage — one budget against one corpus (the shipped one by default). */
  async lexCoverage(body: LexCoverageBody = {}): Promise<LexCoverageResult> {
    const text = await this.textSource(body);
    const vocab = await this.resolveBudget(body, text);
    const inBudget = new Set(vocab.words);
    const counts = new Map<string, number>();
    for (const t of tokenize(text)) {
      if (!inBudget.has(t)) counts.set(t, (counts.get(t) ?? 0) + 1);
    }
    const oov_sample = [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .slice(0, OOV_SAMPLE_SIZE)
      .map(([word, count]) => ({ word, count }));
    return {
      source: vocab.source,
      budget: vocab.budgetName,
      size: vocab.budgetSize,
      rows: vocab.rows,
      coverage: this.roundCoverage(vocab.coverage(text)),
      corpus: corpusStats(text),
      oov_sample,
      words: [...vocab.words],
    };
  }

  // --- the vacancy transform (feature 007) -------------------------------------------

  /**
   * POST /api/lex/vacancy — computed here, LIVE, not refused.
   *
   * The Lexicon Lab is browser-side in both modes, and the transform is pure string work
   * over a corpus this build already ships, so there is nothing here a static page cannot
   * do for real. `lexEngine/vacancy.ts` and `llm_geometry/lex/vacancy.py` implement the
   * same normative document, and `tests/unit/staticVacancy.test.ts` pins this method's
   * whole response against what the real FastAPI route returned for the same request —
   * the statistics AND the sha256 of the entire vacated corpus, which is the parity this
   * feature rests on.
   *
   * The response is an excerpt plus a digest for the same reason the backend's is: the
   * corpus is ~86 kB and a `p` sweep would otherwise move megabytes to show a screenful.
   */
  async lexVacancy(body: LexVacancyBody = {}): Promise<LexVacancyResult> {
    const original = await this.textSource(body);
    const params = vacancyParamsFrom(body);

    const previewChars = asInt(body.preview_chars, "preview_chars", VACANCY_PREVIEW_CHARS);
    if (!(previewChars >= 0 && previewChars <= VACANCY_PREVIEW_MAX)) {
      throw invalidParamError(
        `preview_chars must be in 0..${VACANCY_PREVIEW_MAX}, got ${previewChars}. ` +
          "The whole vacated corpus is never returned; `vacated_sha256` identifies it, " +
          "and /api/lex/train vacates in place so the text never needs a round trip.",
      );
    }

    const vmap = buildMapFor(original, params);
    const vacated = vacateText(original, vmap, params);
    const { vocab, rule } = this.vacancyVocab(body, params, vmap, original, vacated);
    const stats = vacancyStats(original, vacated, vmap, params);

    return {
      p: sig6(params.p, "p"),
      seed: params.seed,
      consistent: params.consistent,
      match_prosody: params.matchProsody,
      reveal_after: params.revealAfter,
      keep: [...params.keep].map(String).sort(),
      mint: params.mint,
      vocabulary_rule: rule,
      words: [...vocab.words],
      budget: {
        source: vocab.source,
        budget: vocab.budgetName,
        size: vocab.budgetSize,
        rows: vocab.rows,
        coverage: this.roundCoverage(vocab.coverage(vacated)),
      },
      corpus: corpusStats(vacated),
      vacancy_stats: roundVacancyStats(stats),
      bijective: stats.bijective,
      remint_rounds: stats.remintRounds,
      preview: vacated.slice(0, previewChars),
      original_preview: original.slice(0, previewChars),
      preview_chars: previewChars,
      truncated: vacated.length > previewChars,
      vacated_chars: vacated.length,
      vacated_sha256: sha256Hex(utf8Bytes(vacated)),
      original_chars: original.length,
      original_sha256: sha256Hex(utf8Bytes(original)),
    };
  }

  /**
   * §7.2's two rules, and which one applies.
   *
   * **Mapped** (`consistent`, `revealAfter = 0`): resolve the budget against the ENGLISH
   * corpus, then push its word list through the same `transformWord`, preserving order.
   * The map is injective, so every word keeps the id its pre-image had — that is the
   * invariance theorem of §7.3, and it is why a mapped run's loss is bit-identical.
   *
   * **Rebuilt** (everything else): a source type no longer has a single image, so the
   * budget is rebuilt from the vacated corpus by the tab's normal rule and coverage
   * collapses. The collapse is not a defect, it is the measurement (FR-715).
   */
  private vacancyVocab(
    body: LexCoverageBody,
    params: VacancyParams,
    vmap: VacancyMap,
    original: string,
    vacated: string,
  ): { vocab: LexVocab; rule: "mapped" | "rebuilt" } {
    if (!params.consistent || params.revealAfter !== 0) {
      return { vocab: this.resolveBudgetSync(body, vacated), rule: "rebuilt" };
    }
    const english = this.resolveBudgetSync(body, original);
    // §5.2a, at the wire boundary and with the contract's TYPE on it. The engine refuses
    // this too, but with a plain `Error` — so the same request was a typed 400 from the
    // backend and an unlabelled crash here, which the parity fixture's
    // `swap-at-intermediate-p` reject case caught. The condition is the engine's own
    // (`injectiveAtEveryP` is false exactly for a swap map), and the sentence is the
    // engine's own; only the envelope is added.
    if (!vmap.injectiveAtEveryP && params.p > 0 && params.p < 1) {
      throw invalidParamError(noMappedVocabularyRefusal(params.mint, params.p));
    }
    const mapped = mapVocabWords([...english.words], vmap, params);
    return {
      vocab: new LexVocab(mapped, english.source, english.budgetName),
      rule: "mapped",
    };
  }

  // --- training ---------------------------------------------------------------------

  /**
   * POST /api/lex/train — real from-scratch training (or a fine-tune from `base`) in the
   * trainWorker, reported through the same `{ready:false, job_id}` → subscribeProgress
   * protocol as the backend's SSE, with the contract's message format. Identical requests
   * are 200-style cache hits, exactly like the backend's content-hash single-flight.
   */
  async lexTrain(body: LexTrainBody = {}): Promise<LexTrainResult> {
    const originalText = await this.textSource(body);

    // Feature 007, optional and additive: absent, everything below is exactly what it was
    // before the transform existed. Present, the model trains on the VACATED corpus under
    // the vocabulary §7.2 assigns it. Vacating here rather than in the caller matches the
    // backend, where `/api/lex/vacancy` deliberately returns only an excerpt.
    let text = originalText;
    let vacParams: VacancyParams | null = null;
    let vmap: VacancyMap | null = null;
    if (body.vacancy != null) {
      if (typeof body.vacancy !== "object" || Array.isArray(body.vacancy)) {
        throw invalidParamError(
          "vacancy must be an object of the transform's parameters " +
            "(p, seed, consistent, match_prosody, reveal_after, keep, mint), got " +
            JSON.stringify(body.vacancy),
        );
      }
      vacParams = vacancyParamsFrom(body.vacancy);
      vmap = buildMapFor(originalText, vacParams);
      text = vacateText(originalText, vmap, vacParams);
    }

    const steps = asInt(body.steps, "steps", DEFAULT_STEPS);
    if (!(steps >= 1 && steps <= MAX_STEPS)) {
      throw invalidParamError(`steps must be in 1..${MAX_STEPS}, got ${steps}`);
    }
    const lr = asFloat(body.lr, "lr", DEFAULT_LR);
    if (!(lr > 0)) throw invalidParamError(`lr must be > 0, got ${lr}`);
    const batchSize = asInt(body.batch_size, "batch_size", DEFAULT_BATCH);
    if (batchSize < 1) throw invalidParamError(`batch_size must be at least 1, got ${batchSize}`);
    const weightDecay = asFloat(body.weight_decay, "weight_decay", DEFAULT_WEIGHT_DECAY);
    if (weightDecay < 0) throw invalidParamError(`weight_decay must be >= 0, got ${weightDecay}`);
    const seed = asInt(body.seed, "seed", DEFAULT_SEED);
    const sampleEvery = asInt(body.sample_every, "sample_every", DEFAULT_SAMPLE_EVERY);
    if (sampleEvery < 1) throw invalidParamError(`sample_every must be at least 1, got ${sampleEvery}`);

    let cfg: LexConfig;
    let vocab: LexVocab;
    let initialWeights: WeightSet | undefined;
    if (body.base != null) {
      const base = this.requireModel(String(body.base));
      for (const control of ["d_model", "n_layers", "n_heads", "ctx", "tied", "source", "budget", "size"] as const) {
        if ((body as Record<string, unknown>)[control] !== undefined) {
          throw invalidParamError(
            `${control} cannot be set when fine-tuning from \`base\`: a fine-tune keeps the ` +
              "base model's shape and vocabulary, which is the whole point of it",
          );
        }
      }
      cfg = base.cfg;
      vocab = base.vocab;
      initialWeights = base.model.weights;
    } else if (vacParams !== null && vmap !== null) {
      vocab = this.vacancyVocab(body, vacParams, vmap, originalText, text).vocab;
      cfg = configFrom(body, vocab.rows);
    } else {
      vocab = await this.resolveBudget(body, text);
      cfg = configFrom(body, vocab.rows);
    }

    const cacheKey = JSON.stringify({
      text,
      cfg,
      words: vocab.words,
      source: vocab.source,
      // Redundant with (text, words) today, and in the key anyway so that a knob added to
      // the transform later cannot land on a cache entry made before it existed.
      vacancy: vacParams,
      base: body.base ?? null,
      steps,
      lr,
      batchSize,
      weightDecay,
      seed,
    });
    const cached = this.trainCache.get(cacheKey);
    if (cached) return { ready: true, ...cached };

    const request: LexTrainRequest = {
      text,
      budgetSource: vocab.source,
      budget: vocab.budgetName,
      vocabWords: [...vocab.words],
      initialWeights,
      model: {
        dModel: cfg.dModel,
        nLayers: cfg.nLayers,
        nHeads: cfg.nHeads,
        ctx: cfg.ctx,
        tied: cfg.tied,
        dropout: cfg.dropout,
      },
      steps,
      lr,
      batchSize,
      weightDecay,
      seed,
      sampleEvery,
    };

    const jobId = this.jobs.create(cacheKey, async (report) => {
      const startedAt = Date.now();
      const done = await this.runTrainAsync(request, steps, report);
      const elapsed = (Date.now() - startedAt) / 1000;
      const trainedVocab = new LexVocab(
        done.vocabWords,
        done.budgetSource === "frequency" ? "frequency" : "dolch",
        done.budgetName,
      );
      const model = new LexModel(done.config, done.weights);
      const record: LexTrainRecord = {
        model_token: this.storeModel(model, trainedVocab, {
          first_loss: done.initialTrainLoss,
          final_loss: done.finalTrainLoss,
          val_loss: done.valLoss,
          steps,
          seed,
          elapsed_s: elapsed,
          base: body.base ?? null,
          // What these weights ARE, in the block that travels with them. A bundle carrying
          // losses and no provenance leaves the reader that opens it later with no way to
          // find out — and the Lexicon Lab's loader now says so rather than assuming.
          provenance: "trained",
          trained: true,
          edited: false,
        }),
        first_loss: sig6(done.initialTrainLoss, "first_loss"),
        final_loss: sig6(done.finalTrainLoss, "final_loss"),
        val_loss: sig6(done.valLoss, "val_loss"),
        steps,
        seed,
        elapsed_s: sig6(elapsed, "elapsed_s"),
        n_tokens: done.nTokens,
        vocab_size: trainedVocab.budgetSize,
        vocab_rows: trainedVocab.rows,
        param_count: model.nParams,
        sample: done.sample.text,
        history: done.history.map((p) => ({
          step: p.step,
          loss: sig6(p.loss, "history.loss"),
          lr: sig6(p.lr, "history.lr"),
        })),
      };
      this.trainCache.set(cacheKey, record);
      // The SSE `done` event carries the result WITHOUT history (routes_lex.py), which
      // only the 200 cache-hit body includes.
      const { history: _history, ...event } = record;
      return event as unknown as Record<string, unknown>;
    });
    return { ready: false, job_id: jobId };
  }

  // --- spectrum ---------------------------------------------------------------------

  /** GET /api/lex/spectrum — the geometry of a trained model's embedding, computed here. */
  async lexSpectrum(params: LexSpectrumParams): Promise<LexSpectrumResult> {
    const matrix = params.matrix ?? "embedding";
    if (matrix !== "embedding" && matrix !== "readout") {
      throw invalidParamError(`matrix must be "embedding" or "readout", got ${JSON.stringify(matrix)}`);
    }
    const entry = this.requireModel(params.model_token);
    const { cfg } = entry;
    if (matrix === "readout" && cfg.tied) {
      throw invalidParamError(
        "this model is tied: its readout IS its embedding, so it has exactly one spectrum. " +
          'Request matrix="embedding" and label it tied.',
      );
    }
    const source =
      matrix === "readout" ? entry.model.headWeight : entry.model.weights.embed;
    const trained = spectrum(source, cfg.vocabRows, cfg.dModel);
    const payload: LexSpectrumResult = {
      model_token: entry.token,
      matrix,
      tied: cfg.tied,
      projection: "pca",
      display_k: SPECTRUM_DISPLAY_K,
      tokens: [...entry.vocab.itos],
      spectrum: this.spectrumBlock(trained, cfg),
    };
    if (params.baseline ?? true) {
      const seed = asInt(params.baseline_seed, "baseline_seed", DEFAULT_SEED);
      const untrained = randomBaselineSpectrum(cfg.vocabRows, cfg.dModel, seed);
      const base = this.spectrumBlock(untrained, cfg);
      const {
        eigenvalues: _e,
        singular_values: _s,
        explained_variance: _v,
        pca_coords: _c,
        pca_explained_variance_ratio: _r,
        ...summary
      } = base;
      payload.baseline = summary;
      payload.comparison = {
        effective_rank_delta: sig6(trained.effectiveRank - untrained.effectiveRank, "comparison"),
        stable_rank_delta: sig6(trained.stableRank - untrained.stableRank, "comparison"),
        participation_ratio_delta: sig6(
          trained.participationRatio - untrained.participationRatio,
          "comparison",
        ),
        frac_var_top2_delta: sig6(trained.fracVarTop2 - untrained.fracVarTop2, "comparison"),
        max_rank: trained.ceiling,
        effective_rank_frac_of_ceiling: sig6(
          trained.ceiling ? trained.effectiveRank / trained.ceiling : 0,
          "comparison",
        ),
        baseline_effective_rank_frac_of_ceiling: sig6(
          untrained.ceiling ? untrained.effectiveRank / untrained.ceiling : 0,
          "comparison",
        ),
      };
    }
    return payload;
  }

  // --- generation -------------------------------------------------------------------

  /**
   * POST /api/lex/generate. Every generated word is in budget BY CONSTRUCTION — the
   * vocabulary is the budget and the special ids are masked — and, as the contract
   * requires, that guarantee is checked rather than assumed: a non-empty `out_of_budget`
   * is a loud failure, never a quietly filtered string.
   */
  async lexGenerate(body: LexGenerateBody): Promise<LexGenerateResult> {
    if (!body?.model_token) throw invalidParamError("model_token is required");
    const entry = this.requireModel(String(body.model_token));
    const temperature = asFloat(body.temperature, "temperature", DEFAULT_TEMPERATURE);
    if (temperature < 0) throw invalidParamError(`temperature must be >= 0, got ${temperature}`);
    const maxNew = asInt(body.max_new_tokens, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS);
    if (!(maxNew >= 1 && maxNew <= MAX_NEW_TOKENS)) {
      throw invalidParamError(`max_new_tokens must be in 1..${MAX_NEW_TOKENS}, got ${maxNew}`);
    }
    const seed = asInt(body.seed, "seed", DEFAULT_SEED);
    // `stop_at_eos` used to be refused here: the browser sampler could only stop at
    // `<eos>`, so the backend's default (run on, render `<eos>` as a line break) was
    // unreachable and refusing was the honest answer. The engine now implements both,
    // so both runtimes mean the same thing by the same name and no refusal is needed.
    const stopAtEos = asBool(body.stop_at_eos, "stop_at_eos", false);

    const prompt = String(body.prompt ?? "");
    const result = generate(entry.model, entry.vocab, {
      prompt,
      temperature,
      maxNewTokens: maxNew,
      seed,
      stopAtEos,
    });
    const inBudget = new Set(entry.vocab.words);
    const outOfBudget = [...new Set(result.words.filter((w) => !inBudget.has(w)))].sort();
    if (outOfBudget.length > 0) {
      throw computeError(
        `generated text contains out-of-budget words ${JSON.stringify(outOfBudget.slice(0, 10))} — ` +
          "the in-budget-by-construction guarantee is broken",
      );
    }
    const finalLoss = entry.metrics.final_loss;
    return {
      model_token: entry.token,
      prompt,
      text: result.text,
      words: result.words,
      n_words: result.words.length,
      out_of_budget: outOfBudget,
      prompt_tokens: tokenize(prompt).map((word) => ({
        text: word,
        id: entry.vocab.stoi(word),
        unk: !inBudget.has(word),
      })),
      temperature,
      seed,
      vocab_size: entry.vocab.budgetSize,
      final_loss: typeof finalLoss === "number" ? sig6(finalLoss, "final_loss") : null,
    };
  }

  // --- portable bundles (US-8) --------------------------------------------------------

  /** GET /api/lex/model — the whole model as one self-describing bundle. */
  async lexExportModel(model_token: string): Promise<LexModelBundle> {
    const entry = this.requireModel(model_token);
    const config = asConfigDict(entry.cfg);
    const shapes = wireShapes(entry.cfg);
    const wire: Record<string, Float32Array> = {};
    for (const [name, array] of Object.entries(entry.model.weights)) {
      wire[toWireName(name)] = array;
    }
    const weights: LexModelBundle["weights"] = {};
    for (const [name, array] of Object.entries(wire)) {
      weights[name] = { shape: shapes[name], data: toBase64(f32Bytes(array)) };
    }
    return {
      format: LEX_BUNDLE_FORMAT,
      version: LEX_BUNDLE_VERSION,
      model_token: entry.token,
      config,
      vocab: {
        source: entry.vocab.source,
        budget: entry.vocab.budgetName,
        words: [...entry.vocab.words],
        specials: [...SPECIAL_TOKENS],
      },
      metrics: { ...entry.metrics },
      weights,
    };
  }

  /** POST /api/lex/model — load a bundle. Validation is strict and loud, as specified. */
  async lexImportModel(bundle: unknown): Promise<LexImportResult> {
    const payload = bundle as Partial<LexModelBundle> | null;
    if (!payload || typeof payload !== "object") {
      throw invalidParamError("a Lexicon Lab model bundle must be a JSON object");
    }
    if (payload.format !== LEX_BUNDLE_FORMAT) {
      throw invalidParamError(
        `not a Lexicon Lab model bundle: format is ${JSON.stringify(payload.format)}, ` +
          `expected ${JSON.stringify(LEX_BUNDLE_FORMAT)}`,
      );
    }
    if (asInt(payload.version, "version", 0) !== LEX_BUNDLE_VERSION) {
      throw invalidParamError(
        `bundle version ${JSON.stringify(payload.version)} is not supported ` +
          `(this build reads version ${LEX_BUNDLE_VERSION})`,
      );
    }
    for (const field of ["config", "vocab", "weights"] as const) {
      if (!payload[field] || typeof payload[field] !== "object") {
        throw invalidParamError(`bundle is missing its ${JSON.stringify(field)} object`);
      }
    }
    const declaredConfig = payload.config as LexModelBundle["config"];
    // Loading is validated the way `LexConfig.from_dict` validates it — the *enumerated*
    // shape choices constrain what this build will TRAIN, not what it can read back. A
    // bundle written by another build at a shape this menu no longer offers is still a
    // real model, and refusing it would lose the file rather than describe it.
    let cfg: LexConfig;
    try {
      cfg = validateConfig({
        vocabRows: asInt(declaredConfig.vocab_rows, "config.vocab_rows", -1),
        dModel: asInt(declaredConfig.d_model, "config.d_model", -1),
        nLayers: asInt(declaredConfig.n_layers, "config.n_layers", -1),
        nHeads: asInt(declaredConfig.n_heads, "config.n_heads", -1),
        ctx: asInt(declaredConfig.ctx, "config.ctx", -1),
        tied: asBool(declaredConfig.tied, "config.tied", DEFAULT_TIED),
        dropout: asFloat(declaredConfig.dropout, "config.dropout", DEFAULT_DROPOUT),
      });
    } catch (e) {
      throw toApiError(e);
    }
    const words = payload.vocab?.words;
    if (!Array.isArray(words) || !words.every((w) => typeof w === "string")) {
      throw invalidParamError("bundle vocab.words must be a list of strings");
    }
    const vocab = new LexVocab(
      words,
      payload.vocab?.source === "frequency" ? "frequency" : "dolch",
      String(payload.vocab?.budget ?? "custom"),
    );
    if (vocab.rows !== cfg.vocabRows) {
      throw invalidParamError(
        `bundle vocabulary has ${vocab.rows} rows but its config declares ${cfg.vocabRows}; ` +
          "the weights and the word list do not describe one model",
      );
    }

    const shapes = wireShapes(cfg);
    const supplied = payload.weights as LexModelBundle["weights"];
    const expected = Object.keys(shapes).sort();
    const got = Object.keys(supplied).sort();
    // `Object.hasOwn` on both sides: with `in`, a bundle carrying a weight called
    // `constructor` (or `toString`, or any other `Object.prototype` key) was not reported
    // as extra, so the file loaded as if it had exactly the tensors its config implies
    // while carrying one nobody had looked at.
    const missing = expected.filter((n) => !Object.hasOwn(supplied, n));
    const extra = got.filter((n) => !Object.hasOwn(shapes, n));
    if (missing.length > 0 || extra.length > 0) {
      throw invalidParamError(
        `weight set mismatch (missing: ${missing.length ? missing.join(", ") : "none"}; ` +
          `extra: ${extra.length ? extra.join(", ") : "none"}); a tied model has no head_w ` +
          "and an untied one requires it",
      );
    }
    const wire: Record<string, Float32Array> = {};
    const engineWeights: WeightSet = {};
    for (const name of expected) {
      const entry = supplied[name];
      if (!entry || typeof entry !== "object" || !("shape" in entry) || !("data" in entry)) {
        throw invalidParamError(`weight ${JSON.stringify(name)} must be an object with shape and data`);
      }
      const shape = (entry.shape as unknown[]).map((v, i) => asInt(v, `${name}.shape[${i}]`, -1));
      if (JSON.stringify(shape) !== JSON.stringify(shapes[name])) {
        throw invalidParamError(
          `weight ${JSON.stringify(name)} has shape [${shape}], expected [${shapes[name]}]`,
        );
      }
      const array = decodeF32(String(entry.data), shape);
      wire[name] = array;
      engineWeights[toEngineName(name)] = array;
    }

    const recomputed = lexModelToken(wire, shapes, asConfigDict(cfg), vocab.words);
    if (payload.model_token != null && String(payload.model_token) !== recomputed) {
      throw invalidParamError(
        `bundle declares model_token ${JSON.stringify(payload.model_token)} but its own ` +
          `contents hash to ${JSON.stringify(recomputed)}; refusing to load a file whose ` +
          "weights and label disagree",
      );
    }

    let model: LexModel;
    try {
      model = new LexModel(cfg, engineWeights);
    } catch (e) {
      throw toApiError(e);
    }
    const metrics =
      payload.metrics && typeof payload.metrics === "object" ? { ...payload.metrics } : {};
    return {
      model_token: this.storeModel(model, vocab, metrics),
      config: asConfigDict(cfg),
      vocab_size: vocab.budgetSize,
      vocab_rows: vocab.rows,
      param_count: model.nParams,
    };
  }

  // --- internals ----------------------------------------------------------------------

  private assertSource(source: string): BudgetSource {
    if (source !== "dolch" && source !== "frequency") {
      throw invalidParamError(`source must be one of ["dolch", "frequency"], got ${JSON.stringify(source)}`);
    }
    return source;
  }

  private roundCoverage(cov: Coverage): Coverage {
    return {
      ...cov,
      token_coverage: sig6(cov.token_coverage, "token_coverage"),
      unk_rate: sig6(cov.unk_rate, "unk_rate"),
    };
  }

  private budgetRow(vocab: LexVocab, text: string, cfg: LexConfig): LexBudgetRow {
    return {
      source: vocab.source,
      budget: vocab.budgetName,
      size: vocab.budgetSize,
      rows: vocab.rows,
      coverage: this.roundCoverage(vocab.coverage(text)),
      param_count: paramCount(vocab.rows, cfg.dModel, cfg.nLayers, cfg.ctx, cfg.tied),
    };
  }

  /** The corpus a request measures/trains against: pasted text, an HF dataset, or ours. */
  private async textSource(body: LexCorpusBody): Promise<string> {
    if (body.text != null && body.hf_dataset != null) {
      throw invalidParamError("provide at most one of text / hf_dataset, not both");
    }
    if (body.hf_dataset != null && String(body.hf_dataset).trim() !== "") {
      // Real rows from the Hub's public, CORS-enabled dataset-viewer — the same source
      // the Geometry Lab's static build uses, so this is genuine data, not a stand-in.
      const pulled = await fetchDatasetText(String(body.hf_dataset), {
        maxSamples: asInt(body.max_samples, "max_samples", 2000),
      });
      return pulled.text;
    }
    if (body.text != null) {
      const resolved = String(body.text);
      if (tokenize(resolved).length === 0) {
        throw invalidParamError("the supplied text has no word tokens to train on");
      }
      return resolved;
    }
    return (await this.corpus()).text;
  }

  private async resolveBudget(body: LexCoverageBody, text: string): Promise<LexVocab> {
    return this.resolveBudgetSync(body, text);
  }

  /**
   * The same resolution, without the promise. `resolveBudget` is `async` for its callers'
   * convenience and never awaits anything; the vacancy path resolves a budget twice
   * against two different texts (§7.2) and reads better without the ceremony.
   */
  private resolveBudgetSync(body: LexCoverageBody, text: string): LexVocab {
    const source = this.assertSource(body.source ?? DEFAULT_BUDGET_SOURCE);
    const budget = String(body.budget ?? DEFAULT_BUDGET);
    if (!(DOLCH_ORDER as readonly string[]).includes(budget)) {
      throw invalidParamError(
        `budget must be one of [${DOLCH_ORDER.join(", ")}], got ${JSON.stringify(budget)}`,
      );
    }
    let size: number | undefined;
    if (body.size != null) {
      if (source !== "frequency") {
        throw invalidParamError(
          'size applies only to source="frequency"; a Dolch budget IS its list, so its ' +
            "size is measured from the data, not chosen",
        );
      }
      size = asInt(body.size, "size", 0);
      if (size < 1) throw invalidParamError(`size must be at least 1, got ${size}`);
    }
    const vocab = buildVocab(source, budget, text, size);
    if (vocab.budgetSize < 1) {
      throw invalidParamError(
        "the resolved budget is empty — this corpus has no word tokens to draw one from",
      );
    }
    return vocab;
  }

  private spectrumBlock(result: SpectrumResult, cfg: LexConfig): LexSpectrumBlock {
    const k = result.components;
    const coords: number[][] = [];
    for (let r = 0; r < cfg.vocabRows; r++) {
      const row: number[] = [];
      for (let j = 0; j < k; j++) row.push(sig6(result.coords[r * k + j], "pca_coords"));
      coords.push(row);
    }
    const total = result.totalVariance;
    return {
      rows: cfg.vocabRows,
      d_model: cfg.dModel,
      max_rank: result.ceiling,
      eigenvalues: sig6All(result.eigenvalues, "eigenvalues"),
      singular_values: sig6All(result.singularValues, "singular_values"),
      explained_variance: result.eigenvalues.map((l) =>
        sig6(total > 0 ? l / total : 0, "explained_variance"),
      ),
      total_variance: sig6(total, "total_variance"),
      effective_rank: sig6(result.effectiveRank, "effective_rank"),
      stable_rank: sig6(result.stableRank, "stable_rank"),
      participation_ratio: sig6(result.participationRatio, "participation_ratio"),
      frac_var_top2: sig6(result.fracVarTop2, "frac_var_top2"),
      frac_var_top10: sig6(result.fracVarTop10, "frac_var_top10"),
      n_dims_for_90pct: result.nDimsFor90pct,
      pca_coords: coords,
      pca_explained_variance_ratio: sig6All(
        result.explainedVarianceRatio,
        "pca_explained_variance_ratio",
      ),
      // The backend's flag: an exactly-zero centred matrix has no rank to report, and
      // this says why rather than letting a caller read the zeros as a measurement.
      degenerate: !(total > 0),
    };
  }

  private storeModel(
    model: LexModel,
    vocab: LexVocab,
    metrics: Record<string, unknown>,
  ): string {
    const cfg = model.cfg;
    const shapes = wireShapes(cfg);
    const wire: Record<string, Float32Array> = {};
    for (const [name, array] of Object.entries(model.weights)) wire[toWireName(name)] = array;
    const token = lexModelToken(wire, shapes, asConfigDict(cfg), vocab.words);
    if (!this.models.has(token)) {
      this.models.set(token, { token, model, vocab, cfg, metrics });
    }
    return token;
  }

  /**
   * A stored model, or the contract's 404. The backend's store is a persistent on-disk
   * cache; this one is this page's memory, so the message says where a token has to have
   * come from rather than implying the model might turn up later.
   */
  private requireModel(token: string): StoredModel {
    const entry = this.models.get(String(token));
    if (!entry) {
      const known = [...this.models.keys()];
      throw notFoundError(
        `no Lexicon model with token ${JSON.stringify(token)}. In this build models live ` +
          "in THIS page's memory (there is no server cache to look in, and a reload " +
          "clears it): train one, or load a saved bundle. " +
          (known.length
            ? `Currently loaded: ${known.join(", ")}.`
            : "No model has been trained or loaded yet."),
      );
    }
    return entry;
  }

  /**
   * Real training in the lexEngine worker; the same code runs synchronously where
   * Workers do not exist (vitest/jsdom), as the Geometry Lab's static path does.
   */
  private runTrainAsync(
    request: LexTrainRequest,
    totalSteps: number,
    report: ProgressFn,
  ): Promise<Extract<LexTrainResponse, { type: "done" }>> {
    // FR-618: a sample rides along on every progress message until the next one replaces
    // it, exactly as the backend's SSE does — a sample that only appeared on the single
    // tick that produced it would essentially never be seen.
    let sample = "";
    const onMessage = (
      message: LexTrainResponse,
      resolve: (v: Extract<LexTrainResponse, { type: "done" }>) => void,
      reject: (e: unknown) => void,
    ): void => {
      if (message.type === "sample") {
        sample = message.text.split(/\s+/).filter(Boolean).join(" ");
        return;
      }
      if (message.type === "progress") {
        const { point } = message;
        let text =
          `step ${point.step}/${totalSteps} · loss ${point.loss.toFixed(3)} · ` +
          `lr ${point.lr.toExponential(2)}`;
        if (sample) text += ` · ${sample}`;
        report(message.fraction, text);
        return;
      }
      if (message.type === "done") resolve(message);
      else reject(toApiError(Object.assign(new Error(message.message), { type: message.errorType })));
    };

    if (typeof Worker === "undefined") {
      return new Promise((resolve, reject) => {
        try {
          runTrainingJob(request, (message) => onMessage(message, resolve, reject));
        } catch (e) {
          reject(toApiError(e));
        }
      });
    }
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("../lexEngine/trainWorker.ts", import.meta.url), {
        type: "module",
      });
      const settle = <T>(fn: (v: T) => void) => (value: T): void => {
        worker.terminate();
        fn(value);
      };
      worker.onmessage = (ev: MessageEvent<LexTrainResponse>) =>
        onMessage(ev.data, settle(resolve), settle(reject));
      worker.onerror = (ev) => {
        worker.terminate();
        reject(computeError(`training worker failed: ${ev.message || "unknown error"}`));
      };
      worker.postMessage(request);
    });
  }
}

/** Bind a `LexSection` to the flat `LexClient` surface the data layer exposes. */
export function lexClientFrom(section: LexSection): LexClient {
  return {
    lexSpec: () => section.lexSpec(),
    lexBudgets: (params) => section.lexBudgets(params),
    lexCoverage: (body) => section.lexCoverage(body),
    lexVacancy: (body) => section.lexVacancy(body),
    lexTrain: (body) => section.lexTrain(body),
    lexSpectrum: (params) => section.lexSpectrum(params),
    lexGenerate: (body) => section.lexGenerate(body),
    lexExportModel: (token) => section.lexExportModel(token),
    lexImportModel: (bundle) => section.lexImportModel(bundle),
  };
}
