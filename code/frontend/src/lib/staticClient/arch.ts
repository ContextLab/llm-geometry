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
  ArchWeightsData,
  ArchWeightsParams,
  TokenizeResult,
} from "../dataClient";
import type { StaticAssets, StaticIndexModel } from "./assets";
import { computeError, invalidParamError, notFoundError, staticModeError } from "./errors";
import { IDLE_GENERATION_INFO, type ArchRuntime, type RuntimeGenerationInfo, type RuntimeLoader } from "./runtimeTypes";
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
