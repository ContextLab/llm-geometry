/**
 * Architecture Explorer in static mode — live where the browser allows, honest
 * where it doesn't:
 *  - graph: precomputed JSON per curated model (real traced forward pass);
 *  - traces: precomputed for the labeled example prompts only (ONNX exports
 *    expose no attentions/hidden states) → other prompts get StaticModeError;
 *  - weights: EXACT windows via safetensors HTTP Range reads at the pinned
 *    revision; over-budget windows serve the precomputed uint8 overview tiles;
 *  - tokenize + generate: LIVE via transformers.js (lazy-loaded runtime).
 */

import type {
  ArchGenerateBody,
  ArchGenerateResult,
  ArchGraph,
  ArchTrace,
  ArchTraceParams,
  ArchVacancyDifference,
  ArchVacancyRefusal,
  ArchVacancyScoreBody,
  ArchVacancyScoreResult,
  ArchVacancyStats,
  ArchWeightsData,
  ArchWeightsParams,
  TokenizeResult,
} from "../dataClient";
import { WORD_RE, tokenize } from "../lexEngine";
import {
  buildVacancyMap,
  typeCounts,
  vacancyDomain,
  vacancyParams,
  vacateText,
} from "../lexEngine/vacancy";
import type { StaticAssets, StaticIndexModel } from "./assets";
import { preservedTokenIndices, tokenByteSpans, wordSpans, type WordSpan } from "./byteSpans";
import { computeError, invalidParamError, notFoundError, staticModeError } from "./errors";
import {
  IDLE_GENERATION_INFO,
  type ArchRuntime,
  type RuntimeGenerationInfo,
  type RuntimeLoader,
  type RuntimeScoredText,
} from "./runtimeTypes";
import { SafetensorsFile, asMatrixShape } from "./safetensors";

const DEFAULT_MAX_CELLS = 4096; // ARCH_WEIGHTS_MAX_CELLS (backend config)
const EXACT_CELLS_HARD_CAP = 65536; // bound the number/size of range reads
const TRACE_EXPORT_MAX_CONTEXT = 64; // the exporter ran the backend default

/** Community ONNX exports for the curated models (repos verified on the Hub; all ship
 * the `model_quantized.onnx` the runtime's q8 ladder loads — see transformersRuntime). */
const ONNX_REPOS: Record<string, string> = {
  "HuggingFaceTB/SmolLM2-135M-Instruct": "onnx-community/SmolLM2-135M-Instruct-ONNX",
  "HuggingFaceTB/SmolLM2-360M-Instruct": "onnx-community/SmolLM2-360M-Instruct-ONNX",
  gpt2: "onnx-community/gpt2-ONNX",
  "Qwen/Qwen2.5-0.5B-Instruct": "onnx-community/Qwen2.5-0.5B-Instruct",
};

interface ArchMeta {
  model_id: string;
  revision: string;
  safetensors_url: string;
}

interface TileEntry {
  param: string;
  shape: number[]; // matrixized [R, C] (the backend's full-window response)
  grid_shape: number[];
  downsampled: boolean;
  method: string;
  stats: { min: number; max: number; mean: number; std: number };
  offset: number;
  nbytes: number;
  vmin: number;
  vmax: number;
}

interface TilesManifest {
  model_id: string;
  revision: string;
  dtype: string; // "uint8"
  bin: string; // "tiles.bin"
  encoding: string;
  tiles: TileEntry[];
}

export interface TraceIndexEntry {
  n: number;
  label: string;
  prompt: string;
  file: string;
  system_prompt?: string;
}

interface TracesIndex {
  model_id: string;
  traces: TraceIndexEntry[];
}

/** Dequantize one uint8 overview tile: value = vmin + (u8/255)·(vmax−vmin). */
export function dequantizeTile(
  bytes: Uint8Array,
  gridRows: number,
  gridCols: number,
  vmin: number,
  vmax: number,
): number[][] {
  if (bytes.byteLength < gridRows * gridCols) {
    throw computeError(
      `tile has ${bytes.byteLength} bytes for a ${gridRows}×${gridCols} grid`,
    );
  }
  const scale = (vmax - vmin) / 255;
  const out: number[][] = [];
  for (let r = 0; r < gridRows; r++) {
    const row = new Array<number>(gridCols);
    for (let c = 0; c < gridCols; c++) row[c] = vmin + bytes[r * gridCols + c] * scale;
    out.push(row);
  }
  return out;
}

// --- the vacancy instrument, pretrained arm (contract §8) --------------------------------

/** The three variants, in the order the panel reads them. MIRROR of `VARIANTS` (Python). */
const VACANCY_VARIANTS = ["english", "swap", "nonce"] as const;
type VacancyVariant = (typeof VACANCY_VARIANTS)[number];

/** MIRROR of `MAX_PASSAGES` — each passage costs three real forward passes. */
const VACANCY_MAX_PASSAGES = 12;
/** MIRROR of `DEFAULT_PASSAGE_WORDS` / `DEFAULT_PASSAGE_COUNT` (the measured shape). */
const VACANCY_PASSAGE_WORDS = 250;
const VACANCY_PASSAGE_COUNT = 6;
/**
 * MIRROR of `FRONT_MATTER_FRACTION`. The shipped book opens with a title page and an
 * alphabetical index of first lines — measured to end in block 11 of 63 — and a passage
 * cut from that is a column of titles, not English. The digest fixture pins the two
 * stacks to the same cut.
 */
const VACANCY_FRONT_MATTER_FRACTION = 0.2;

/**
 * Dtypes whose error against float32 has actually been MEASURED on this contrast
 * (§8.3a). q8 is bounded at ≤ 0.054 nats on the pooled difference; q4f16 is not on this
 * list because it does not compute — it returns input-independent logits — and no other
 * dtype has been measured. Anything absent is refused, never extrapolated onto.
 */
const VACANCY_MEASURED_DTYPES: readonly string[] = ["q8"];

/**
 * The uncertainty stated beside a pooled difference in static mode.
 *
 * DERIVED FROM TWO MEASUREMENTS, not chosen:
 *
 *  1. The independent six-passage study of §8.3a (2 models × 2 seeds, its own passage
 *     cut and its own swap implementation) bounded the pooled q8-vs-fp32 discrepancy at
 *     **≤ 0.054 nats** on every contrast, and recommended stating ~2× that.
 *  2. This build's own configuration, measured across the two stacks on the SHIPPED
 *     default passage set with gpt2 — identical tokenization (2754/2858/3810 tokens,
 *     847 preserved) in both, so the only difference is the dtype:
 *
 *         swap  − english :  fp32 0.7166   q8 0.6440   |Δ| = 0.073
 *         nonce − english :  fp32 0.9892   q8 0.8790   |Δ| = 0.110
 *
 * The second exceeds 0.1, so a stated ±0.1 would have been a bound its own first
 * comparison broke. 0.2 is ~2× the largest discrepancy actually observed here, in the
 * configuration that actually ships. Never quoted to more than one decimal place: that
 * is all either measurement supports. If this constant changes, re-run BOTH stacks on
 * the default set before changing it — the number is a measurement, not a margin.
 */
const VACANCY_Q8_UNCERTAINTY_NATS = 0.2;

/**
 * Preserved tokens that must be pooled before a q8 number may be shown. The bound above
 * was measured on ~700 preserved tokens per condition; a single 250-word passage carries
 * ~120, and at that size q8 was wrong by up to 115 % of the passage's own delta.
 */
const VACANCY_MIN_POOLED_PRESERVED = 700;

/** MIRROR of `TINY_ARM` (Python) — the other half of the 2×2, restated for the panel. */
const VACANCY_TINY_ARM = {
  delta_nats: 0,
  exact: true,
  label: "the same measurement on a model with no locations",
  note:
    "For the from-scratch word-level GeoTransformer of the Lexicon Lab, the vacancy " +
    "transform is a pure relabelling of the vocabulary: with consistent=true and " +
    "revealAfter=0 the token id stream is element-for-element identical, so the training " +
    "loss is bit-identical and a word's FORM is worth exactly 0. That is not a rounding " +
    "— it is an identity, and it is asserted in that tab, not assumed.",
};

/** MIRROR of `UNKNOWN_FORM_NOTE` (Python) — the residual, stated even where the number
 * itself is refused: a reader must not have to earn the caveat by being shown a value. */
const VACANCY_UNKNOWN_FORM_NOTE =
  "Nonce forms fragment into more subword tokens than real words do, so this difference " +
  "is the cost of an unknown form TOGETHER WITH the cost of a stranger, longer context. " +
  "The two are not separable without a tokenizer-level control, so treat this as an " +
  "UPPER BOUND on what a word's location was worth — never as pure location.";

/** MIRROR of `CONFOUND_NOTE` (Python) — §8.4, stated wherever a delta is. */
const VACANCY_CONFOUND =
  "A vacated passage genuinely has higher entropy, so every prediction inside it gets " +
  "worse — the scaffolding included. A positive difference is therefore expected, not a " +
  "surprise: its MAGNITUDE is the result, and it is only interpretable against the tiny " +
  "arm's exact zero.";

/**
 * The passage and its two vacated twins (§8.3), from the SAME TypeScript transform the
 * Lexicon Lab runs — so the nonce a stem gets here is the nonce it gets there, for the
 * same seed, and the golden fixture that pins TS against Python covers this too.
 */
export function vacancyVariantTexts(
  passage: string,
  opts: { p: number; seed: number; matchProsody: boolean; keep: readonly string[] },
): Record<VacancyVariant, string> {
  const tokens = tokenize(passage);
  const domain = vacancyDomain(tokens);
  const counts = typeCounts(tokens);
  const out = { english: passage } as Record<VacancyVariant, string>;
  for (const mint of ["swap", "nonce"] as const) {
    // consistent / revealAfter are fixed at the invariance theorem's condition: the tiny
    // arm's exact zero holds only there, and this number is only interpretable beside it.
    const params = vacancyParams({
      p: opts.p,
      seed: opts.seed,
      consistent: true,
      matchProsody: opts.matchProsody,
      revealAfter: 0,
      keep: [...opts.keep],
      mint,
    });
    out[mint] = vacateText(passage, buildVacancyMap(domain, params, counts), params);
  }
  return out;
}

/**
 * Word spans of the English passage, and the word indices PRESERVED in EVERY variant.
 *
 * "Preserved" is character identity across all three, which is stronger than "closed
 * class": it also covers eligible stems the `u(stem) < p` decision spared. Restricting
 * all three NLLs to the same word set is what makes them comparable.
 */
export function preservedWordIndices(texts: Record<VacancyVariant, string>): {
  words: WordSpan[];
  preserved: Set<number>;
} {
  const per = {} as Record<VacancyVariant, WordSpan[]>;
  for (const name of VACANCY_VARIANTS) per[name] = wordSpans(texts[name], WORD_RE);
  const counts = VACANCY_VARIANTS.map((n) => per[n].length);
  if (new Set(counts).size !== 1) {
    throw computeError(
      "the variants do not have the same number of words, so preserved words cannot be " +
        `aligned: ${VACANCY_VARIANTS.map((n, i) => `${n}=${counts[i]}`).join(", ")}`,
    );
  }
  const preserved = new Set<number>();
  for (const w of per.english) {
    if (VACANCY_VARIANTS.every((n) => per[n][w.index].word === w.word)) preserved.add(w.index);
  }
  return { words: per.english, preserved };
}

function mean(values: readonly number[]): number {
  if (values.length === 0) {
    throw computeError("no tokens to average — the passage has no scored positions");
  }
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

/**
 * §8.1's fields pooled over passages at the TOKEN level — token-weighted, never a mean of
 * means, so a passage with twice the tokens carries twice the weight and the pooled
 * figure really is the mean surprisal of a preserved token.
 *
 * The absolutes are `null` here by design: this function only ever runs in the quantized
 * static build, where they are refused (§8.3a). The counts are not — a token count is
 * exact at any dtype, and it is what shows the reader that nonce forms fragment.
 */
export function pooledStats(
  scored: readonly RuntimeScoredText[],
  preserved: readonly number[][],
): ArchVacancyStats {
  let nTokens = 0;
  let nChars = 0;
  let nPreserved = 0;
  for (let i = 0; i < scored.length; i++) {
    nTokens += scored[i].nll.length - 1; // position 0 has no prediction
    nChars += scored[i].nChars;
    nPreserved += preserved[i].length;
  }
  return {
    nllPreserved: null,
    nllAll: null,
    bitsPerChar: null,
    nTokens,
    nPreservedTokens: nPreserved,
    nChars,
  };
}

/**
 * `mean(nll_b − nll_a)` over preserved tokens, PAIRED, with its standard error.
 *
 * The pairing is exact, not an approximation: preserved words are character-identical
 * across variants and the curated models' pretokenizers never merge across a word
 * boundary, so each preserved word yields the same pieces in every variant. Pairing
 * removes the between-token variance — the same function word is compared with itself in
 * the other condition — which is what makes a standard error on a ~0.1 nat effect worth
 * printing. A length mismatch means the pairing assumption broke, and it refuses.
 */
export function pairedDifference(
  a: readonly RuntimeScoredText[],
  aPreserved: readonly number[][],
  b: readonly RuntimeScoredText[],
  bPreserved: readonly number[][],
): { nats: number; se: number; nPairs: number } {
  const diffs: number[] = [];
  for (let i = 0; i < a.length; i++) {
    if (aPreserved[i].length !== bPreserved[i].length) {
      throw computeError(
        `the variants have ${aPreserved[i].length} and ${bPreserved[i].length} preserved ` +
          "tokens, so they cannot be paired",
      );
    }
    for (let j = 0; j < aPreserved[i].length; j++) {
      diffs.push(b[i].nll[bPreserved[i][j]] - a[i].nll[aPreserved[i][j]]);
    }
  }
  const m = mean(diffs);
  let se = NaN;
  if (diffs.length > 1) {
    let ss = 0;
    for (const d of diffs) ss += (d - m) * (d - m);
    se = Math.sqrt(ss / (diffs.length - 1) / diffs.length);
  }
  return { nats: m, se, nPairs: diffs.length };
}

/**
 * Evenly spaced excerpts of the shipped corpus — MIRROR of `default_passages` (Python),
 * the configuration the reference numbers of §8.3a were measured in. The first eighth of
 * the book is skipped: it is the title page and table of contents, a list of titles
 * rather than English.
 */
export function defaultVacancyPassages(
  text: string,
  count = VACANCY_PASSAGE_COUNT,
  words = VACANCY_PASSAGE_WORDS,
): string[] {
  const blocks: string[] = [];
  let current: string[] = [];
  let n = 0;
  const counter = new RegExp(WORD_RE.source, "g");
  for (const line of text.split("\n")) {
    current.push(line);
    n += (line.match(counter) ?? []).length;
    if (n >= words) {
      blocks.push(current.join("\n").replace(/^\n+|\n+$/g, ""));
      current = [];
      n = 0;
    }
  }
  if (blocks.length === 0) {
    throw computeError("the shipped corpus produced no passage of the requested size");
  }
  const start = Math.max(1, Math.round(blocks.length * VACANCY_FRONT_MATTER_FRACTION));
  const step = Math.max(1, Math.floor((blocks.length - start) / Math.max(1, count)));
  return Array.from({ length: count }, (_, i) =>
    blocks[Math.min(start + i * step, blocks.length - 1)],
  );
}

/**
 * Refusal: absolute log-likelihoods, from a quantized model. MEASURED, not cautious.
 */
export const VACANCY_ABSOLUTE_REFUSAL: ArchVacancyRefusal = {
  type: "StaticModeError",
  message:
    "Absolute log-likelihoods are not reportable from a quantized model: q8 shifts " +
    "nllPreserved by −0.19 nats on gpt2 and +0.40 on SmolLM2-135M — the sign is not " +
    "even stable across models. The pooled DIFFERENCES below survive quantization; " +
    "these numbers do not. The full stack reports them at float32.",
};

/** Refusal: any single-passage delta, from a quantized model. */
export const VACANCY_PER_PASSAGE_REFUSAL: ArchVacancyRefusal = {
  type: "StaticModeError",
  message:
    "A per-passage delta is not reportable under q8: the worst measured discrepancy was " +
    "0.65 nats, 115 % of that passage's own float32 delta, caused by q8 compressing the " +
    "16–18 nat line-initial function words this measurement is precisely about. Only the " +
    "pooled figure has a measured bound. Run the full stack for the per-passage table.",
};

/** Refusal: `nonce − swap`, the contrast quantization destroys. */
export const VACANCY_UNKNOWN_FORM_REFUSAL: ArchVacancyRefusal = {
  type: "StaticModeError",
  message:
    "This is the contrast the quantized model destroys, and it is the one the result " +
    "rests on. It is a small number — 0.16–0.27 nats across the curated models at " +
    "float32, 0.06–0.21 in the independent study — and q8's error on it is 14–23 % of " +
    "that, up to 0.28 nats on a single passage, with a sign flip in one passage of six " +
    "per model. Measured on this very configuration: float32 says 0.273 for gpt2 and q8 " +
    "says 0.235, a 14 % error on the quantity the whole result turns on. The two pooled " +
    "numbers shown here do differ by it — that arithmetic is not a measurement, and the " +
    "difference is not reportable at this dtype. Run the full stack (uvicorn " +
    "llm_geometry.api.app:app), which scores at float32, where ONNX and torch agree to " +
    "5.3e-4 nats.",
};

/**
 * What the quantized static build MAY say about the two differences it computed
 * (contract §8.3a, FR-720a). Pure policy, separated from the measurement so it can be
 * asserted directly: pooled `swap − english` and `nonce − english` carry a stated,
 * MEASURED quantization uncertainty; `nonce − swap` carries a refusal and no number.
 *
 * The refusal is not squeamishness about a wide error bar. q8's error on `nonce − swap`
 * is 14–23 % of an effect whose true value is 0.06–0.21 nats, and it flips sign on one
 * passage in six — so a number here would not be imprecise, it would be wrong in a
 * direction the reader could not detect.
 */
export function staticVacancyDifferences(
  swapMinusEnglish: { nats: number; se: number; nPairs: number },
  nonceMinusEnglish: { nats: number; se: number; nPairs: number },
): ArchVacancyDifference[] {
  return [
    {
      id: "wrong_content",
      label: "the cost of wrong content",
      expr: "nll(swap) − nll(english)",
      headline: true,
      quantizationUncertaintyNats: VACANCY_Q8_UNCERTAINTY_NATS,
      ...swapMinusEnglish,
    },
    {
      id: "unknown_form",
      label: "the cost of unknown form",
      expr: "nll(nonce) − nll(swap)",
      headline: true,
      upperBound: true,
      note: VACANCY_UNKNOWN_FORM_NOTE,
      nats: null,
      se: null,
      nPairs: 0,
      refused: VACANCY_UNKNOWN_FORM_REFUSAL,
    },
    {
      id: "total",
      label: "both costs together",
      expr: "nll(nonce) − nll(english)",
      headline: false,
      note:
        "The sum of the two differences above. It conflates wrong content with unknown " +
        "form and is never the headline.",
      quantizationUncertaintyNats: VACANCY_Q8_UNCERTAINTY_NATS,
      ...nonceMinusEnglish,
    },
  ];
}

export class ArchSection {
  private readonly safetensors = new Map<string, SafetensorsFile>();
  private runtimePromise: Promise<ArchRuntime> | null = null;
  private runtimeLoadError: string | null = null;
  private readonly loadRuntime: RuntimeLoader;

  constructor(
    private readonly assets: StaticAssets,
    runtimeLoader?: RuntimeLoader,
  ) {
    // The default loader dynamically imports transformers.js — the heavy chunk
    // only ever downloads when live tokenize/generate is actually used.
    this.loadRuntime = runtimeLoader ?? (() => import("./transformersRuntime").then((m) => m.runtime));
  }

  // --- model catalog ---------------------------------------------------------------

  async model(modelId: string): Promise<StaticIndexModel> {
    const found = await this.findModel(modelId);
    if (!found) {
      const ids = (await this.assets.index()).arch_models.map((m) => m.model_id);
      throw staticModeError(
        `This build ships a curated set of models: ${ids.join(", ")}. ` +
          "Widening that list is tracked in issue #4.",
      );
    }
    return found;
  }

  async findModel(modelId: string): Promise<StaticIndexModel | null> {
    const idx = await this.assets.index();
    return idx.arch_models.find((m) => m.model_id === modelId) ?? null;
  }

  // --- graph + traces (precomputed) ------------------------------------------------

  async getArchGraph(modelId: string): Promise<ArchGraph> {
    const m = await this.model(modelId);
    return this.assets.json<ArchGraph>(`arch/${m.slug}/graph.json`);
  }

  async tracePresets(modelId: string): Promise<TraceIndexEntry[]> {
    const m = await this.model(modelId);
    const idx = await this.assets.json<TracesIndex>(`arch/${m.slug}/traces/index.json`);
    return idx.traces;
  }

  async getArchTrace(params: ArchTraceParams): Promise<ArchTrace> {
    const m = await this.model(params.model_id);
    const traces = await this.tracePresets(params.model_id);
    const sys = params.system_prompt ?? "";
    const match = traces.find(
      (t) => t.prompt === params.prompt && (t.system_prompt ?? "") === sys,
    );
    const contextOk =
      params.max_context === undefined || params.max_context === TRACE_EXPORT_MAX_CONTEXT;
    if (!match || !contextOk) {
      const labels = traces.map((t) => `“${t.label}” (${JSON.stringify(t.prompt)})`);
      throw staticModeError(
        "Per-layer traces need the model's hidden states, which browser ONNX " +
          "exports don't expose — the static demo ships traces precomputed by the " +
          `real backend for these example prompts only: ${labels.join(", ")}. ` +
          "Pick one of them, or run the full stack (see the README) to trace any prompt.",
      );
    }
    return this.assets.json<ArchTrace>(`arch/${m.slug}/traces/${match.file}`);
  }

  // --- weight inspector (live range reads + precomputed overview tiles) ------------

  async getArchWeights(params: ArchWeightsParams): Promise<ArchWeightsData> {
    const m = await this.model(params.model_id);
    const tiles = await this.assets.json<TilesManifest>(`arch/${m.slug}/tiles.json`);
    const tile = tiles.tiles.find((t) => t.param === params.param);
    if (!tile) {
      throw notFoundError(`Model '${m.model_id}' has no parameter '${params.param}'.`);
    }
    const [rows, cols] = [tile.shape[0], tile.shape[1] ?? 1];
    const r0 = params.r0 ?? 0;
    const c0 = params.c0 ?? 0;
    const r1 = params.r1 ?? rows;
    const c1 = params.c1 ?? cols;
    const maxCells = params.max_cells ?? DEFAULT_MAX_CELLS;
    if (maxCells < 1) throw invalidParamError(`max_cells must be >= 1, got ${maxCells}`);
    if (!(0 <= r0 && r0 < r1 && r1 <= rows && 0 <= c0 && c0 < c1 && c1 <= cols)) {
      throw invalidParamError(
        `Window [${r0}:${r1}, ${c0}:${c1}] is out of range for '${params.param}' ` +
          `with shape [${rows}, ${cols}].`,
      );
    }

    const cells = (r1 - r0) * (c1 - c0);
    if (cells <= maxCells && cells <= EXACT_CELLS_HARD_CAP) {
      return this.exactWindow(m, tile, r0, r1, c0, c1);
    }

    // Over-budget window → the precomputed strided-mean overview of the FULL
    // tensor (the backend's own full-window response, uint8-quantized at build
    // time). r0..c1 report what is actually served — the whole tensor — so the
    // response never claims a sub-window resolution it doesn't have.
    const bin = await this.assets.bin(`arch/${m.slug}/${tiles.bin}`);
    const bytes = new Uint8Array(bin, tile.offset, tile.nbytes);
    const [gr, gc] = tile.grid_shape;
    return {
      param: tile.param,
      shape: [rows, cols],
      r0: 0,
      r1: rows,
      c0: 0,
      c1: cols,
      downsampled: true,
      grid_shape: [gr, gc],
      values: dequantizeTile(bytes, gr, gc, tile.vmin, tile.vmax),
      stats: tile.stats,
      method: "strided_mean",
    };
  }

  private async exactWindow(
    m: StaticIndexModel,
    tile: TileEntry,
    r0: number,
    r1: number,
    c0: number,
    c1: number,
  ): Promise<ArchWeightsData> {
    const meta = await this.assets.json<ArchMeta>(`arch/${m.slug}/meta.json`);
    let file = this.safetensors.get(m.model_id);
    if (!file) {
      file = new SafetensorsFile(meta.safetensors_url, (i, init) => this.assets.rawFetch(i, init));
      this.safetensors.set(m.model_id, file);
    }
    const name = await this.resolveTensorName(file, tile.param);
    const win = await file.readWindow(name, r0, r1, c0, c1);
    // Integrity: the safetensors tensor must have the exact shape the backend
    // exported tiles for — otherwise we'd be windowing a different tensor.
    if (win.rows !== tile.shape[0] || win.cols !== (tile.shape[1] ?? 1)) {
      throw computeError(
        `safetensors tensor '${name}' has shape [${win.rows}, ${win.cols}] but the ` +
          `export recorded [${tile.shape[0]}, ${tile.shape[1] ?? 1}] for '${tile.param}'`,
      );
    }
    const nRows = r1 - r0;
    const nCols = c1 - c0;
    const values: number[][] = [];
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let sumSq = 0;
    for (let r = 0; r < nRows; r++) {
      const row = new Array<number>(nCols);
      for (let c = 0; c < nCols; c++) {
        const v = win.values[r * nCols + c];
        row[c] = v;
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sumSq += v * v;
      }
      values.push(row);
    }
    const n = nRows * nCols;
    const mean = sum / n;
    const variance = Math.max(0, sumSq / n - mean * mean);
    return {
      param: tile.param,
      shape: [win.rows, win.cols],
      r0,
      r1,
      c0,
      c1,
      downsampled: false,
      grid_shape: [nRows, nCols],
      values,
      stats: { min, max, mean, std: Math.sqrt(variance) },
      method: "exact",
    };
  }

  /**
   * state_dict param paths don't always match safetensors keys: gpt2's file
   * drops the "transformer." prefix, and tied lm_head weights are stored only
   * as the embedding. Candidates are tried against the REAL header — never
   * guessed blind.
   */
  private async resolveTensorName(file: SafetensorsFile, param: string): Promise<string> {
    const header = await file.header();
    const candidates = [param];
    if (param.startsWith("transformer.")) candidates.push(param.slice("transformer.".length));
    if (param === "lm_head.weight") {
      candidates.push("model.embed_tokens.weight", "transformer.wte.weight", "wte.weight");
    }
    for (const c of candidates) {
      const entry = header.tensors[c];
      if (entry) return c;
    }
    throw notFoundError(
      `Tensor '${param}' (tried: ${candidates.join(", ")}) is not in the safetensors ` +
        `index of ${file.url}.`,
    );
  }

  // --- live runtime (transformers.js) ----------------------------------------------

  runtimeInfo(): RuntimeGenerationInfo {
    if (this.runtimeLoadError) {
      return { ...IDLE_GENERATION_INFO, status: "error", error: this.runtimeLoadError };
    }
    if (!this.loadedRuntime) return { ...IDLE_GENERATION_INFO };
    return this.loadedRuntime.info();
  }

  private loadedRuntime: ArchRuntime | null = null;

  private runtime(): Promise<ArchRuntime> {
    if (!this.runtimePromise) {
      this.runtimePromise = this.loadRuntime()
        .then((r) => {
          this.loadedRuntime = r;
          return r;
        })
        .catch((e: unknown) => {
          this.runtimePromise = null;
          this.runtimeLoadError = e instanceof Error ? e.message : String(e);
          throw computeError(
            `Could not load the in-browser inference runtime: ${this.runtimeLoadError}`,
          );
        });
    }
    return this.runtimePromise;
  }

  /** LIVE tokenization from the pinned original-repo tokenizer files. */
  async tokenizeLive(modelId: string, text: string): Promise<TokenizeResult> {
    const m = await this.model(modelId);
    const rt = await this.runtime();
    return rt.tokenize(m.model_id, m.revision, text);
  }

  /**
   * The pretrained arm of the vacancy instrument, in the browser (contract §8).
   *
   * Everything is computed here for real: the three variants come from the TypeScript
   * vacancy transform (the same one the Lexicon Lab runs, golden-tested against Python),
   * and each is scored by one real forward pass on the community ONNX export.
   *
   * WHAT IT MAY THEN SAY is narrower than what it computed, and that is the point
   * (§8.3a / FR-720a). This build runs a **q8** export, and the measurement of §8.3a
   * bounds q8's error only on POOLED differences over several hundred preserved tokens:
   *
   *   - absolute `nllPreserved` is refused — q8 moves it by −0.19 nats on gpt2 and +0.40
   *     on SmolLM2-135M, so even its SIGN is not stable across models;
   *   - per-passage deltas are refused — worst case 0.65 nats, 115 % of that passage's
   *     own fp32 delta;
   *   - `nonce − swap` is refused — its true value is 0.06–0.21 nats and q8's error on it
   *     is 14–23 % pooled with sign flips, so quantization eats exactly the contrast that
   *     makes the result mean something;
   *   - pooled `swap − english` and `nonce − english` are reported, with the measured
   *     ±0.1 nats quantization uncertainty stated beside the sampling standard error.
   *
   * A dtype with no measured bound is refused outright rather than given a ± copied from
   * a different dtype: a stated error bar that was never measured is a fabrication, and
   * worse than no number.
   */
  async archVacancyScore(body: ArchVacancyScoreBody): Promise<ArchVacancyScoreResult> {
    const m = await this.model(body.model_id);
    const repo = ONNX_REPOS[m.model_id];
    if (!repo) {
      throw staticModeError(
        `No browser (ONNX) export is wired up for ${m.model_id}, so this measurement ` +
          "cannot be run on it here — it covers: " +
          Object.keys(ONNX_REPOS).join(", ") +
          ". Run the full stack (see the README) for other models.",
      );
    }
    if (body.passage !== undefined && body.passages !== undefined) {
      throw invalidParamError(
        "send either `passage` (one) or `passages` (several), not both — they would " +
          "silently disagree about what was scored",
      );
    }
    const requested =
      body.passages ?? (body.passage !== undefined ? [body.passage] : await this.defaultPassages());
    if (requested.length === 0 || requested.some((t) => !t.trim())) {
      throw invalidParamError("every passage must be a non-empty string");
    }
    if (requested.length > VACANCY_MAX_PASSAGES) {
      throw invalidParamError(
        `at most ${VACANCY_MAX_PASSAGES} passages per request, got ${requested.length}; ` +
          "each one costs three real forward passes",
      );
    }
    const p = body.p ?? 1.0;
    if (!(p >= 0 && p <= 1)) throw invalidParamError(`p must lie in [0, 1], got ${p}`);
    const seed = body.seed ?? 0;
    const matchProsody = body.match_prosody ?? true;
    const keep = body.keep ?? [];

    // NFC once, up front: every byte span downstream indexes THIS string. Qwen's
    // tokenizer normalizes internally and gpt2's/SmolLM2's do not, so without this a
    // decomposed character would shift every span after it (§8.2).
    const passages = requested.map((t) => t.normalize("NFC"));
    const rt = await this.runtime();

    const scored: Record<string, RuntimeScoredText[]> = { english: [], swap: [], nonce: [] };
    const preservedIdx: Record<string, number[][]> = { english: [], swap: [], nonce: [] };
    const previews: Record<string, string> = { english: "", swap: "", nonce: "" };
    for (const passage of passages) {
      const texts = vacancyVariantTexts(passage, { p, seed, matchProsody, keep });
      const { words, preserved } = preservedWordIndices(texts);
      if (preserved.size === 0) {
        throw computeError(
          "this passage has no word that survives the transform, so there is no " +
            "scaffolding to score. Lower p, or use a passage with closed-class words.",
        );
      }
      const order = VACANCY_VARIANTS.map((name) => texts[name]);
      const results = await rt.scoreTexts(repo, order);
      VACANCY_VARIANTS.forEach((name, i) => {
        if (!previews[name]) previews[name] = texts[name];
        const variantWords = name === "english" ? words : wordSpans(texts[name], WORD_RE);
        const spans = tokenByteSpans(results[i].pieces, texts[name]);
        scored[name].push(results[i]);
        preservedIdx[name].push(
          preservedTokenIndices(spans, variantWords, preserved).filter((j) => j > 0),
        );
      });
    }

    const info = rt.info();
    const dtype = info.dtype ?? "unknown";
    const device = info.device ?? "unknown";
    if (!VACANCY_MEASURED_DTYPES.includes(dtype)) {
      throw staticModeError(
        `This browser loaded the model at dtype "${dtype}", and no error bound has been ` +
          "measured for it. Quantization moves absolute log-likelihoods by nats and can " +
          "reverse the sign of the effect this panel measures, so stating a number here " +
          "would mean inventing an error bar. Run the full stack (uvicorn " +
          "llm_geometry.api.app:app), which scores at float32.",
      );
    }

    const pooledPreserved = preservedIdx.english.reduce((n, list) => n + list.length, 0);
    if (pooledPreserved < VACANCY_MIN_POOLED_PRESERVED) {
      throw staticModeError(
        `Pooled over ${pooledPreserved} preserved tokens, this is below the ` +
          `${VACANCY_MIN_POOLED_PRESERVED} at which q8's error on the pooled difference ` +
          "was measured (≤ 0.054 nats). Below it the only honest answer is no number: a " +
          "single-passage delta under q8 was wrong by up to 115 % of its own value. Add " +
          "more passages, or run the full stack, which scores at float32 and reports " +
          "every per-passage number.",
      );
    }

    const differences = staticVacancyDifferences(
      pairedDifference(scored.english, preservedIdx.english, scored.swap, preservedIdx.swap),
      pairedDifference(scored.english, preservedIdx.english, scored.nonce, preservedIdx.nonce),
    );

    return {
      model_id: m.model_id,
      revision: m.revision,
      stack: "static",
      dtype,
      device,
      p,
      seed,
      match_prosody: matchProsody,
      keep: [...keep],
      alignment: {
        mechanism: "byte-level pieces → UTF-8 byte spans",
        unit: "utf8_bytes",
        verified: true,
        note:
          "Token→word attribution is verified at run time by reconstructing the passage " +
          "from the token byte spans; a mismatch raises rather than mis-attributing.",
      },
      variants: VACANCY_VARIANTS.map((name) => ({
        id: name,
        pooled: pooledStats(scored[name], preservedIdx[name]),
        preview: previews[name].slice(0, 400),
        refused: VACANCY_ABSOLUTE_REFUSAL,
      })),
      passages_used: passages,
      differences,
      passages: null,
      passagesRefused: VACANCY_PER_PASSAGE_REFUSAL,
      tiny_arm: VACANCY_TINY_ARM,
      confound: VACANCY_CONFOUND,
    };
  }

  /** The six evenly spaced corpus excerpts the measurement of §8.3a used. */
  private async defaultPassages(): Promise<string[]> {
    const corpus = await this.assets.json<{ text?: string }>("lex/corpus.json");
    if (typeof corpus?.text !== "string" || !corpus.text) {
      throw staticModeError(
        "This build did not export the corpus (static-data/lex/corpus.json), so the " +
          "default passage set cannot be cut. Paste a passage, or run the full stack.",
      );
    }
    return defaultVacancyPassages(corpus.text.normalize("NFC"));
  }

  /** LIVE generation on the model's community ONNX export. */
  async archGenerate(body: ArchGenerateBody): Promise<ArchGenerateResult> {
    const m = await this.model(body.model_id);
    const repo = ONNX_REPOS[m.model_id];
    if (!repo) {
      throw staticModeError(
        `No browser (ONNX) export is wired up for ${m.model_id} — live chat in the ` +
          "static demo covers: " +
          Object.keys(ONNX_REPOS).join(", ") +
          ". Run the full stack (see the README) for other models.",
      );
    }
    const rt = await this.runtime();
    return rt.generate(body, repo);
  }
}
