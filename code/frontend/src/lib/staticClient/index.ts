/**
 * createStaticClient — the GitHub Pages data layer (feature 003, FR-201).
 *
 * Implements the SAME interface as dataClient's createClient(), with the
 * Python backend absent. Per method, data is either:
 *   LIVE       — computed in the browser (TS geoEngine; transformers.js;
 *                safetensors HTTP Range reads at pinned revisions), or
 *   PRECOMPUTED — served verbatim from build-time exports of the real backend
 *                (static-data/: architecture graphs, example traces, weight tiles,
 *                the geo checkpoint), or
 *   REFUSED    — a typed ApiError("StaticModeError", …) naming what IS
 *                available; nothing is ever fabricated (FR-203).
 *
 * NOTE (import cycle): dataClient re-exports `client` from clientProvider,
 * which imports this module — so nothing here may touch dataClient's runtime
 * bindings (ApiError) at module top level. Everything below only uses them
 * inside functions.
 */

import type {
  ArchGenerateBody,
  ArchGenerateResult,
  ArchGraph,
  ArchTrace,
  ArchTraceParams,
  ArchWeightsData,
  ArchWeightsParams,
  Client,
  GeoFinetuneBody,
  GeoFinetuneResult,
  GeoSpec,
  GeoTokenizeResult,
  GeoTrace,
  GeoTrainResult,
  GeoVectorFieldData,
  GeoVectorFieldParams,
  GeoWeightsData,
  GeoWeightsParams,
  GeoWeightsPostBody,
  GeoWeightsPostResult,
  JobSnapshot,
  ModelReference,
  TokenizeResult,
} from "../dataClient";
import { ArchSection, type TraceIndexEntry } from "./arch";
import { StaticAssets, type FetchLike } from "./assets";
import { LocalJobRegistry } from "./jobs";
import type { RuntimeLoader, StaticRuntimeInfo } from "./runtimeTypes";
import { GeoSection } from "./geo";

export { STATIC_MODE_ERROR } from "./errors";
export type { StaticRuntimeInfo } from "./runtimeTypes";

/** Friendly names for the curated static models (labels only, not data). */
const DISPLAY_NAMES: Record<string, string> = {
  "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B Instruct (default)",
  gpt2: "GPT-2 124M (base — completes text)",
  "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2 360M Instruct",
  "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2 135M Instruct (smallest)",
};

export interface StaticClientOptions {
  baseUrl?: string;
  fetchImpl?: FetchLike;
  /** Test seam: replaces the dynamic transformers.js import. */
  runtimeLoader?: RuntimeLoader;
}

export interface StaticExtras {
  /** Marker + type guard hook for views that need static-only affordances. */
  readonly staticMode: true;
  /** The example prompts with precomputed traces for a curated model. */
  staticArchTracePresets(modelId: string): Promise<TraceIndexEntry[]>;
  /** Device/dtype report for the in-browser generation runtime. */
  staticRuntimeInfo(): StaticRuntimeInfo;
}

export type StaticClient = Client & StaticExtras;

export function isStaticClient(c: Client): c is StaticClient {
  return (c as Partial<StaticExtras>).staticMode === true;
}

export function createStaticClient(opts: StaticClientOptions = {}): StaticClient {
  const assets = new StaticAssets(opts);
  const jobs = new LocalJobRegistry();
  const geo = new GeoSection(assets, jobs);
  const arch = new ArchSection(assets, opts.runtimeLoader);

  // --- curated model catalog -------------------------------------------------------

  async function modelReference(modelId: string): Promise<ModelReference> {
    const m = await arch.model(modelId); // StaticModeError for unknown ids
    const graph = await arch.getArchGraph(modelId); // real traced meta (cached)
    return {
      model_id: m.model_id,
      revision: m.revision,
      source: "curated",
      display_name: DISPLAY_NAMES[m.model_id] ?? m.model_id,
      status: "supported",
      capabilities: {
        num_layers: graph.meta.n_layers,
        hidden_size: graph.meta.hidden,
        vocab_size: graph.meta.vocab,
        // Real logits are available live (transformers.js) and in the
        // precomputed traces; hidden states are NOT exposable in-browser.
        exposes_token_probs: true,
        exposes_hidden_states: false,
      },
    };
  }

  async function listModels(): Promise<{ models: ModelReference[] }> {
    const idx = await assets.index();
    const models = await Promise.all(idx.arch_models.map((m) => modelReference(m.model_id)));
    return { models };
  }

  function resolveModel(model_id: string): Promise<ModelReference> {
    return modelReference(model_id);
  }

  // --- jobs (geo training / fine-tuning run locally) -------------------------------

  async function getJob(job_id: string): Promise<JobSnapshot> {
    return jobs.snapshot(job_id);
  }

  async function pollJob(
    job_id: string,
    onProgress?: (progress: number, message: string) => void,
  ): Promise<void> {
    await jobs.wait(job_id, onProgress);
  }

  function subscribeProgress(
    job_id: string,
    handlers: {
      onProgress?: (progress: number, message: string) => void;
      onDone?: (data?: Record<string, unknown>) => void;
      onError?: (type: string, message: string) => void;
    },
  ): () => void {
    return jobs.subscribe(job_id, handlers);
  }

  /** LIVE: the model's real tokenizer files at the pinned revision (transformers.js). */
  function tokenize(model_id: string, text: string): Promise<TokenizeResult> {
    return arch.tokenizeLive(model_id, text);
  }

  // --- assembled client ------------------------------------------------------------

  const staticClient: StaticClient = {
    listModels,
    resolveModel,
    getJob,
    tokenize,
    pollJob,
    subscribeProgress,
    // Geometry Lab: fully live in TypeScript (golden-tested geoEngine)
    getGeoSpec: (): Promise<GeoSpec> => geo.getGeoSpec(),
    geoTrain: (seed?: number): Promise<GeoTrainResult> => geo.geoTrain(seed),
    geoTokenize: (text: string): Promise<GeoTokenizeResult> => geo.geoTokenize(text),
    getGeoTrace: (prompt: string, weightsToken?: string, _signal?: AbortSignal): Promise<GeoTrace> =>
      geo.getGeoTrace(prompt, weightsToken),
    getGeoVectorField: (
      params: GeoVectorFieldParams,
      _signal?: AbortSignal,
    ): Promise<GeoVectorFieldData> => geo.getGeoVectorField(params),
    getGeoWeights: (params: GeoWeightsParams, _signal?: AbortSignal): Promise<GeoWeightsData> =>
      geo.getGeoWeights(params),
    postGeoWeights: (body: GeoWeightsPostBody): Promise<GeoWeightsPostResult> =>
      geo.postGeoWeights(body),
    geoFinetune: (body: GeoFinetuneBody): Promise<GeoFinetuneResult> => geo.geoFinetune(body),
    geoFinetuneFile: (
      file: Blob,
      filename: string,
      options: Omit<GeoFinetuneBody, "text" | "hf_dataset"> = {},
    ): Promise<GeoFinetuneResult> => geo.geoFinetuneFile(file, filename, options),
    // Architecture Explorer: precomputed graph/traces, live weights/tokenize/chat
    getArchGraph: (model_id: string): Promise<ArchGraph> => arch.getArchGraph(model_id),
    getArchWeights: (params: ArchWeightsParams, _signal?: AbortSignal): Promise<ArchWeightsData> =>
      arch.getArchWeights(params),
    getArchTrace: (params: ArchTraceParams, _signal?: AbortSignal): Promise<ArchTrace> =>
      arch.getArchTrace(params),
    archGenerate: (body: ArchGenerateBody): Promise<ArchGenerateResult> => arch.archGenerate(body),
    // Static-only extras (views reach these via isStaticClient())
    staticMode: true,
    staticArchTracePresets: (modelId: string): Promise<TraceIndexEntry[]> =>
      arch.tracePresets(modelId),
    staticRuntimeInfo: (): StaticRuntimeInfo => ({ mode: "static", generation: arch.runtimeInfo() }),
  };

  return staticClient;
}
