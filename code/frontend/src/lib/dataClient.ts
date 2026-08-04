// Cached-data access layer (FR-016). Typed calls to contracts/api.md, plus a
// precompute → progress → ready flow. `fetchImpl`/`EventSourceImpl` are injectable so
// the client logic is unit-testable without a live server; the real client↔backend
// path is exercised by the Playwright e2e against real gpt2.

export interface Capabilities {
  num_layers: number | null;
  hidden_size: number | null;
  vocab_size: number | null;
  exposes_token_probs: boolean;
  exposes_hidden_states: boolean;
}

export interface ModelReference {
  model_id: string;
  revision: string;
  source: string;
  display_name: string;
  status: string;
  capabilities: Capabilities;
  reason?: string;
}

export interface JobSnapshot {
  job_id: string;
  cache_key: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  error: { type: string; message: string } | null;
  version: number;
}

export interface TokenizeResult {
  model_id: string;
  tokens: { token: number; token_str: string }[];
}

// ---------------------------------------------------------------------------
// Feature 002 — Interactive Model Explorer
// (specs/002-interactive-model-explorer/contracts/api.md, FROZEN).
// Geometry Lab (/api/geo/*) + Architecture Explorer (/api/arch/*). Tensors are
// nested JSON lists of finite floats, row-major, 6 significant digits. Trace and
// field getters accept an AbortSignal so views can cancel-and-restart (FR-108).
// ---------------------------------------------------------------------------

export interface GeoModelSpec {
  d_model: number;
  n_layers: number;
  n_heads: number;
  mlp_hidden: number;
  vocab_size: number;
  context_window: number;
  tied_unembedding: boolean;
  corpus: string;
  seed: number;
}

export interface GeoCheckpointStatus {
  status: "ready" | "missing" | "training";
  checkpoint_id: string | null;
  final_loss: number | null;
  coverage_uniformity: number | null;
  field_directional_entropy: number | null;
  job_id: string | null;
}

export interface GeoSpec {
  model: GeoModelSpec;
  special_tokens: { unk: number; eos: number; pad: number };
  checkpoint: GeoCheckpointStatus;
}

// 200 (cache hit) carries checkpoint_id/status; 202 carries job_id for SSE progress.
export interface GeoTrainResult {
  ready: boolean;
  checkpoint_id?: string;
  status?: string;
  job_id?: string;
}

export interface GeoToken {
  id: number;
  text: string;
  unk: boolean;
}

export interface GeoTokenizeResult {
  tokens: GeoToken[];
  n_unk: number;
  truncated: boolean;
}

export interface GeoLogitsTopk {
  ids: number[];
  texts: string[];
  probs: number[];
}

export interface GeoTraceLayer {
  layer: number;
  attention: number[][]; // (T,T) row-stochastic, causal
  q: number[][]; // (T,3)
  k: number[][];
  v: number[][];
  hidden_in: number[][];
  attn_out: number[][];
  mlp_out: number[][];
  hidden_out: number[][];
}

export interface GeoTrace {
  tokens: GeoToken[]; // T ≤ 50
  embeddings: number[][]; // (T,3) input embeddings, unit-norm
  layers: GeoTraceLayer[]; // × 4
  probs: number[]; // (1003,) next-token distribution
  logits_topk: GeoLogitsTopk;
  next_token: { id: number; text: string };
}

export type GeoFieldMode = "next_next" | "force";
export type GeoLayerParam = number | "full"; // 0..3 or "full" (force mode: per-layer only)

/** POST /api/geo/train_scratch (feature 004). */
export interface GeoTrainScratchBody {
  text?: string | null;
  hf_dataset?: string | null;
  hf_split?: string;
  max_samples?: number;
  epochs?: number;
}

export interface GeoTrainScratchResult {
  ready: boolean;
  job_id?: string;
  weights_token?: string;
  vocab_size?: number;
  final_loss?: number;
  /** ln(vocab_size): the cross-entropy a model reaches by learning NOTHING. */
  uniform_baseline?: number;
  /** Whether `final_loss` actually cleared that baseline (see geo/scratch.py). */
  learned?: boolean;
  n_tokens?: number;
  n_distinct?: number;
  epochs?: number;
}

/** GET /api/geo/corpus_stats — is this corpus big enough to train on? */
export interface CorpusStatsResult {
  n_tokens: number;
  n_distinct: number;
  vocab_words_required: number;
}

/** GET/POST /api/geo/model — the portable model file. */
export interface GeoModelBundleShape {
  format: string;
  version: number;
  weights_token: string;
  config: Record<string, number>;
  vocab: string;
  weights: Record<string, { shape: number[]; data: string }>;
}

export interface GeoVectorFieldParams {
  mode: GeoFieldMode;
  layer: GeoLayerParam;
  prompt: string;
  weights_token?: string;
  temperature?: number; // default 0
  top_m?: number; // default 1
  antisymmetrize?: boolean; // default false; force mode only
}

export interface GeoArrow {
  origin_index: number; // index into points
  vec: number[]; // [dx,dy,dz]
  weight: number; // 0..1
}

export interface GeoSequenceForce {
  position: number;
  vec: number[];
  normal_residual: number;
}

export interface GeoVectorFieldData {
  mode: GeoFieldMode;
  layer: GeoLayerParam;
  points: number[][]; // (V,3), V = 1003
  token_ids: number[];
  arrows: GeoArrow[];
  sequence_forces: GeoSequenceForce[] | null; // force mode only
  tangent_exact: boolean; // true iff force mode + antisymmetrize
}

export type GeoMatrixName = "W_Q" | "W_K" | "W_V" | "W_O" | "embedding";
export type GeoPresetName =
  | "identity"
  | "toeplitz_fuzzy"
  | "random"
  | "random_autocorr"
  | "zero"
  | "learned";
export type GeoWeightSource = "learned" | "edited" | `preset:${string}`;

export interface GeoWeightsParams {
  matrix: GeoMatrixName;
  layer?: number; // ignored for matrix=embedding
  weights_token?: string; // omitted ⇒ canonical learned checkpoint
}

export interface GeoWeightsData {
  values: number[][];
  shape: number[]; // e.g. [3,3] | [1003,3] | [3,12]
  source: GeoWeightSource;
}

export interface GeoWeightEdit {
  layer: number;
  matrix: GeoMatrixName;
  preset?: GeoPresetName | null; // exactly one of preset/values
  values?: number[][] | null;
  seed?: number; // default 0
}

export interface GeoWeightsPostBody {
  base: "learned" | string; // "learned" or an existing weights_token
  edits: GeoWeightEdit[];
}

export interface GeoWeightsPostResult {
  weights_token: string; // content hash over the full resulting weight set
  edited: { layer: number; matrix: GeoMatrixName; source: GeoWeightSource }[];
}

export interface GeoFinetuneBody {
  text?: string | null;
  hf_dataset?: string | null;
  hf_split?: string; // default "train"
  max_samples?: number; // default 200
  steps?: number; // ≤ 500, default 100
  lr?: number; // default 1e-2
  base?: "learned" | string;
}

// 200 (content-hash cache hit) carries the token + losses; 202 carries job_id.
export interface GeoFinetuneResult {
  ready: boolean;
  job_id?: string;
  weights_token?: string;
  loss_before?: number;
  loss_after?: number;
  /** How much of the fine-tuning stream the ACTIVE model's vocabulary actually knew.
   * A loss drop is only "on your text" to the extent your text was in the vocabulary,
   * so a client that shows the loss must be able to show this beside it. */
  n_tokens?: number;
  n_unk?: number;
  unk_rate?: number;
}

export type ArchNodeKind =
  | "embedding"
  | "linear"
  | "layernorm"
  | "rmsnorm"
  | "rope"
  | "attention_softmax"
  | "residual_add"
  | "activation"
  | "mlp"
  | "lm_head"
  | "other";

export interface ArchParamInfo {
  name: string; // "weight" | "bias"
  shape: number[];
  param_path: string; // state_dict key
  tied_to: string | null; // tied tensors appear once, aliased
}

export interface ArchNode {
  id: string; // stable dotted path, e.g. "model.layers.0.self_attn.q_proj"
  kind: ArchNodeKind;
  op: "module" | "functional";
  label: string;
  layer: number | null;
  group: string; // "stem" | "layer_<k>" | "head"
  params: ArchParamInfo[];
}

export interface ArchEdge {
  from: string;
  to: string;
  tensor_shape: (number | string)[]; // e.g. ["T", 576]
}

export interface ArchGraphMeta {
  n_layers: number;
  hidden: number;
  heads: number;
  kv_heads: number;
  vocab: number;
  total_params: number;
  traced_seq_len: number;
}

export interface ArchGraph {
  model_id: string;
  schema_version: number;
  meta: ArchGraphMeta;
  nodes: ArchNode[];
  edges: ArchEdge[];
}

export interface ArchWeightsParams {
  model_id: string;
  param: string; // param_path
  r0?: number; // default 0
  r1?: number;
  c0?: number; // default 0
  c1?: number;
  max_cells?: number; // default 4096
}

export interface ArchWeightsData {
  param: string;
  shape: number[]; // [R,C]; 1-D params use C=1
  r0: number;
  r1: number;
  c0: number;
  c1: number;
  downsampled: boolean;
  grid_shape: number[]; // [gr,gc]
  values: number[][]; // gr×gc
  stats: { min: number; max: number; mean: number; std: number };
  method: "exact" | "strided_mean";
}

export interface ArchTraceParams {
  model_id: string;
  prompt: string;
  system_prompt?: string;
  max_context?: number; // default 64 (truncates left)
}

export interface ArchTraceLayer {
  layer: number;
  attention: number[][][]; // [heads][T][T], downsampled to ≤64×64 per head
  attention_downsampled: boolean;
  hidden_norm: number[]; // L2 of residual stream out, × T
  hidden_pca3: number[][]; // (T,3) PCA of hidden states (viz aid)
}

export interface ArchNodeActivation {
  node_id: string;
  out_norm: number;
  out_shape: (number | string)[]; // e.g. ["T", 576]
}

export interface ArchTrace {
  tokens: { id: number; text: string }[]; // T ≤ max_context
  chat_template_used: boolean;
  truncated: boolean; // prompt exceeded max_context and was LEFT-truncated
  layers: ArchTraceLayer[];
  logits_topk: GeoLogitsTopk;
  node_activations: ArchNodeActivation[]; // one entry per traced node
}

export interface ArchGenerateBody {
  model_id: string;
  prompt: string;
  system_prompt?: string | null;
  temperature?: number; // default 0.8
  max_new_tokens?: number; // ≤ 128, default 64
  seed?: number | null;
}

export interface ArchGeneratedToken {
  id: number;
  text: string;
  prob: number;
  topk: { ids: number[]; texts: string[]; probs: number[] };
  // Optional honesty annotation (static build): set when the quantized decode's
  // greedy pick disagrees with the full-precision re-scoring pass's argmax.
  note?: string;
}

export interface ArchGenerateResult {
  text: string; // full decoded reply
  tokens: ArchGeneratedToken[];
  finish_reason: "eos" | "length";
}

// --- Feature 007: the pretrained arm of the vacancy instrument (contract §8) ---

export interface ArchVacancyScoreBody {
  model_id: string;
  /** One passage (the panel's editable excerpt) … */
  passage?: string;
  /** … or several, pooled at the token level. Omit both for the shipped default set. */
  passages?: string[];
  p?: number; // default 1.0 — full vacancy, the measured condition
  seed?: number;
  match_prosody?: boolean;
  keep?: string[];
}

/** A quantity this stack measured but may not report, with the reason and the fix. */
export interface ArchVacancyRefusal {
  type: string; // the typed-error name the full stack would raise
  message: string;
}

/** Contract §8.1, per variant. Absolutes are `null` where the running dtype has no bound. */
export interface ArchVacancyStats {
  nllPreserved: number | null;
  nllAll: number | null;
  bitsPerChar: number | null;
  nTokens: number;
  nPreservedTokens: number;
  nChars: number;
}

export interface ArchVacancyVariant {
  id: "english" | "swap" | "nonce";
  pooled: ArchVacancyStats;
  preview: string;
  /** Present when the absolute NLLs are withheld (quantized stack). */
  refused?: ArchVacancyRefusal;
}

export interface ArchVacancyDifference {
  /** `wrong_content` = swap − english; `unknown_form` = nonce − swap; `total` = their sum. */
  id: "wrong_content" | "unknown_form" | "total";
  label: string;
  expr: string;
  /** `total` is false: it conflates the two and is never a headline (contract §8.3). */
  headline: boolean;
  nats: number | null;
  se: number | null;
  nPairs: number;
  upperBound?: boolean;
  note?: string;
  /** Stated only where it was MEASURED for the dtype that ran; never invented. */
  quantizationUncertaintyNats?: number;
  refused?: ArchVacancyRefusal;
}

export interface ArchVacancyPassage {
  index: number;
  nWords: number;
  nPreservedWords: number;
  variants: Record<string, ArchVacancyStats>;
}

export interface ArchVacancyScoreResult {
  model_id: string;
  revision?: string;
  /** "backend" (torch, fp32) or "static" (transformers.js, quantized ONNX). */
  stack: "backend" | "static";
  dtype: string;
  device?: string;
  p: number;
  seed: number;
  match_prosody: boolean;
  keep: string[];
  alignment: { mechanism: string; unit: string; verified: boolean; note: string };
  variants: ArchVacancyVariant[];
  /** The English passages exactly as scored (NFC-normalized), so the UI can show them. */
  passages_used: string[];
  differences: ArchVacancyDifference[];
  /** Per-passage rows, or `null` where they are refused (quantized stack). */
  passages: ArchVacancyPassage[] | null;
  passagesRefused?: ArchVacancyRefusal;
  tiny_arm: { delta_nats: number; exact: boolean; label: string; note: string };
  confound: string;
}

export class ApiError extends Error {
  type: string;
  constructor(type: string, message: string) {
    super(message);
    this.type = type;
    this.name = "ApiError";
  }
}

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;
type ProgressFn = (progress: number, message: string) => void;

export interface ClientOptions {
  fetchImpl?: FetchLike;
  baseUrl?: string;
  EventSourceImpl?: typeof EventSource;
  pollIntervalMs?: number;
}

export function createClient(opts: ClientOptions = {}) {
  const doFetch: FetchLike = opts.fetchImpl ?? ((i, init) => fetch(i, init));
  const base = opts.baseUrl ?? "";
  const ESImpl =
    opts.EventSourceImpl ?? (typeof EventSource !== "undefined" ? EventSource : undefined);
  const pollIntervalMs = opts.pollIntervalMs ?? 250;

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await doFetch(base + path, init);
    } catch (e) {
      // Cancellation is not an API failure — let callers distinguish it (FR-108).
      if (e instanceof DOMException && e.name === "AbortError") throw e;
      const msg = e instanceof Error ? e.message : String(e);
      throw new ApiError("NetworkError", `Could not reach the server: ${msg}`);
    }
    const text = await res.text();
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        // Non-JSON body (proxy error page, truncated stream): still a typed error,
        // never a raw SyntaxError escaping to the UI (FR-107). The backend's own
        // errors ALWAYS carry the JSON envelope, so a non-JSON 5xx means the request
        // never reached it (dev proxy turns connection-refused into a bare 500) —
        // surface that as NetworkError so views show the "backend unreachable" copy.
        if (!res.ok && [500, 502, 503, 504].includes(res.status)) {
          throw new ApiError("NetworkError", "Could not reach the backend server");
        }
        throw new ApiError(
          res.ok ? "BadResponse" : "HttpError",
          res.ok ? "Server returned invalid JSON" : `HTTP ${res.status}`,
        );
      }
    }
    if (!res.ok) {
      const env = (data as { error?: { type?: string; message?: string } } | null)?.error;
      throw new ApiError(env?.type ?? "HttpError", env?.message ?? `HTTP ${res.status}`);
    }
    return data as T;
  }

  function qs(params: Record<string, unknown>): string {
    const u = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) u.set(k, String(v));
    }
    const s = u.toString();
    return s ? `?${s}` : "";
  }

  const jsonInit = (body: unknown): RequestInit => ({
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  function listModels(): Promise<{ models: ModelReference[] }> {
    return request("/api/models");
  }

  function resolveModel(model_id: string): Promise<ModelReference> {
    return request("/api/models/resolve", jsonInit({ model_id }));
  }

  function getJob(job_id: string): Promise<JobSnapshot> {
    return request(`/api/jobs/${encodeURIComponent(job_id)}`);
  }

  function tokenize(model_id: string, text: string): Promise<TokenizeResult> {
    return request("/api/tokenize" + qs({ model_id, text }));
  }

  async function pollJob(job_id: string, onProgress?: ProgressFn): Promise<void> {
    for (;;) {
      const job = await getJob(job_id);
      onProgress?.(job.progress, job.message);
      if (job.status === "done") return;
      if (job.status === "error") {
        throw new ApiError(job.error?.type ?? "ComputeError", job.error?.message ?? "job failed");
      }
      await new Promise((r) => setTimeout(r, pollIntervalMs));
    }
  }

  // Live progress via SSE when available; polling otherwise. Returns an unsubscribe fn.
  function subscribeProgress(
    job_id: string,
    handlers: {
      onProgress?: ProgressFn;
      // Receives the done event's payload (e.g. {cache_key, checkpoint_id} for geo
      // training, {weights_token, loss_before, loss_after} for fine-tunes).
      onDone?: (data?: Record<string, unknown>) => void;
      onError?: (type: string, message: string) => void;
    },
  ): () => void {
    if (!ESImpl) {
      let cancelled = false;
      pollJob(job_id, (p, m) => !cancelled && handlers.onProgress?.(p, m))
        .then(async () => {
          if (cancelled) return;
          let data: Record<string, unknown> | undefined;
          try {
            const snap = (await getJob(job_id)) as unknown as {
              result?: Record<string, unknown> | null;
            };
            data = snap.result ?? undefined;
          } catch {
            data = undefined; // the job finished; a missing snapshot shouldn't fail done
          }
          if (!cancelled) handlers.onDone?.(data);
        })
        .catch((e: ApiError) => !cancelled && handlers.onError?.(e.type, e.message));
      return () => {
        cancelled = true;
      };
    }
    const es = new ESImpl(base + `/api/jobs/${encodeURIComponent(job_id)}/events`);
    es.addEventListener("progress", (ev: MessageEvent) => {
      const d = JSON.parse(ev.data);
      handlers.onProgress?.(d.progress, d.message);
    });
    es.addEventListener("done", (ev: MessageEvent) => {
      let data: Record<string, unknown> | undefined;
      try {
        data = JSON.parse(ev.data);
      } catch {
        data = undefined;
      }
      handlers.onDone?.(data);
      es.close();
    });
    es.addEventListener("error", (ev: MessageEvent) => {
      try {
        const d = JSON.parse(ev.data);
        handlers.onError?.(d.type, d.message);
      } catch {
        handlers.onError?.("StreamError", "progress stream error");
      }
      es.close();
    });
    return () => es.close();
  }

  // --- Feature 002: Geometry Lab (/api/geo/*, frozen contract) ---

  function getGeoSpec(): Promise<GeoSpec> {
    return request("/api/geo/spec");
  }

  // Idempotent, single-flight; omitted seed defers to the server default (0).
  function geoTrain(seed?: number): Promise<GeoTrainResult> {
    return request("/api/geo/train", jsonInit(seed === undefined ? {} : { seed }));
  }

  function geoTokenize(text: string, weightsToken?: string): Promise<GeoTokenizeResult> {
    return request("/api/geo/tokenize" + qs({ text, weights_token: weightsToken }));
  }

  function getGeoTrace(
    prompt: string,
    weightsToken?: string,
    signal?: AbortSignal,
  ): Promise<GeoTrace> {
    return request("/api/geo/trace" + qs({ prompt, weights_token: weightsToken }), { signal });
  }

  function getGeoVectorField(
    params: GeoVectorFieldParams,
    signal?: AbortSignal,
  ): Promise<GeoVectorFieldData> {
    const { mode, layer, prompt, weights_token, temperature, top_m, antisymmetrize } = params;
    return request(
      "/api/geo/vector_field" +
        qs({ mode, layer, prompt, weights_token, temperature, top_m, antisymmetrize }),
      { signal },
    );
  }

  function getGeoWeights(params: GeoWeightsParams, signal?: AbortSignal): Promise<GeoWeightsData> {
    const { matrix, layer, weights_token } = params;
    return request("/api/geo/weights" + qs({ weights_token, layer, matrix }), { signal });
  }

  function postGeoWeights(body: GeoWeightsPostBody): Promise<GeoWeightsPostResult> {
    return request("/api/geo/weights", jsonInit(body));
  }

  function geoFinetune(body: GeoFinetuneBody): Promise<GeoFinetuneResult> {
    return request("/api/geo/finetune", jsonInit(body));
  }

  // Multipart variant: a .txt/.md `file` field replaces `text` (exactly one source).
  // The browser sets the multipart content-type (with boundary) itself.
  function geoFinetuneFile(
    file: Blob,
    filename: string,
    options: Omit<GeoFinetuneBody, "text" | "hf_dataset"> = {},
  ): Promise<GeoFinetuneResult> {
    const form = new FormData();
    form.append("file", file, filename);
    for (const [k, v] of Object.entries(options)) {
      if (v !== undefined && v !== null) form.append(k, String(v));
    }
    return request("/api/geo/finetune", { method: "POST", body: form });
  }

  // --- Feature 004: from-scratch training + portable models ---

  function geoTrainScratch(body: GeoTrainScratchBody): Promise<GeoTrainScratchResult> {
    return request("/api/geo/train_scratch", jsonInit(body));
  }

  function geoCorpusStats(text: string): Promise<CorpusStatsResult> {
    return request("/api/geo/corpus_stats" + qs({ text }));
  }

  function geoExportModel(weightsToken?: string): Promise<GeoModelBundleShape> {
    return request("/api/geo/model" + qs({ weights_token: weightsToken ?? "learned" }));
  }

  function geoImportModel(
    bundle: unknown,
  ): Promise<{ weights_token: string; vocab_size: number }> {
    return request("/api/geo/model", jsonInit(bundle as Record<string, unknown>));
  }

  // --- Feature 002: Architecture Explorer (/api/arch/*, frozen contract) ---

  function getArchGraph(model_id: string): Promise<ArchGraph> {
    return request("/api/arch/graph" + qs({ model_id }));
  }

  function getArchWeights(
    params: ArchWeightsParams,
    signal?: AbortSignal,
  ): Promise<ArchWeightsData> {
    const { model_id, param, r0, r1, c0, c1, max_cells } = params;
    return request("/api/arch/weights" + qs({ model_id, param, r0, r1, c0, c1, max_cells }), {
      signal,
    });
  }

  function getArchTrace(params: ArchTraceParams, signal?: AbortSignal): Promise<ArchTrace> {
    const { model_id, prompt, system_prompt, max_context } = params;
    return request("/api/arch/trace" + qs({ model_id, prompt, system_prompt, max_context }), {
      signal,
    });
  }

  function archGenerate(body: ArchGenerateBody): Promise<ArchGenerateResult> {
    return request("/api/arch/generate", jsonInit(body));
  }

  function archVacancyScore(
    body: ArchVacancyScoreBody,
    signal?: AbortSignal,
  ): Promise<ArchVacancyScoreResult> {
    return request("/api/arch/vacancy-score", { ...jsonInit(body), signal });
  }

  return {
    listModels,
    resolveModel,
    getJob,
    tokenize,
    pollJob,
    subscribeProgress,
    // Feature 002 (frozen contract): Geometry Lab + Architecture Explorer
    getGeoSpec,
    geoTrain,
    geoTokenize,
    getGeoTrace,
    getGeoVectorField,
    getGeoWeights,
    postGeoWeights,
    geoFinetune,
    geoFinetuneFile,
    geoTrainScratch,
    geoCorpusStats,
    geoExportModel,
    geoImportModel,
    getArchGraph,
    getArchWeights,
    getArchTrace,
    archGenerate,
    archVacancyScore,
  };
}

export type Client = ReturnType<typeof createClient>;

// Default client used by the app: the backend client (same-origin; Vite proxies
// /api in dev), or the static Pages client when built with VITE_DATA_MODE=static.
// The pick lives in clientProvider.ts; the type is identical either way.
export { client, DATA_MODE } from "./clientProvider";

// ---------------------------------------------------------------------------
// Trailing-edge debounce for interactive controls (FR-108 cancel-and-restart):
// the wrapped fn runs once, `ms` after the last invocation, with the last args;
// `cancel()` drops any PENDING (not-yet-fired) call — it does NOT abort a fetch
// that already started. Callers implementing cancel-and-restart (FR-108) must pair
// this with an AbortController: abort the previous request's controller when the
// debounced fn fires, and call `cancel()` + `controller.abort()` on teardown.
// ---------------------------------------------------------------------------

export interface DebouncedFn<A extends unknown[]> {
  (...args: A): void;
  cancel: () => void;
}

export function debounced<A extends unknown[]>(fn: (...args: A) => void, ms = 400): DebouncedFn<A> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const wrapped = ((...args: A) => {
    if (timer !== undefined) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = undefined;
      fn(...args);
    }, ms);
  }) as DebouncedFn<A>;
  wrapped.cancel = () => {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
  };
  return wrapped;
}
