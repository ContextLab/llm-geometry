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

export interface PrecomputeResult {
  cache_key: string;
  job_id: string | null;
  status: string;
  ready: boolean;
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

export interface TopToken {
  token_id: number;
  token_str: string;
  prob: number;
}

export interface Distribution {
  model_id: string;
  revision: string;
  temperature: number;
  top_token: number;
  top_token_str: string;
  top?: TopToken[];
  tail_mass?: number;
  probs?: number[];
}

export interface Reduction2D {
  model_id: string;
  method: string;
  coords: number[][];
  token_ids: number[];
  grid?: { n: number; vertices: number[][]; reference_token_ids: number[] };
}

export interface VectorField {
  grid_n: number;
  layer_from: number;
  layer_to: number;
  num_layers: number;
  temperature: number;
  fanout: number;
  reference_points: number;
  response_step: number;
  seed: number;
  spread_mu: number;
  starts: number[][];
  ends: number[][];
  probs: number[];
  start_token_strs: string[];
  end_token_strs: string[];
  trajectory?: number[][];
  trajectory_probs?: number[];
  trajectory_token_strs?: string[];
}

export interface TokenizeResult {
  model_id: string;
  tokens: { token: number; token_str: string }[];
}

export interface ManifoldAnimation {
  n_frames: number;
  n_vertices: number;
  token_strs: string[];
  trajectory_token_strs: string[];
  faces: number[][]; // static
  token_points: number[][]; // static (radius-2 token positions)
  traj_points: number[][]; // static (response tokens on the sphere)
  vertices: number[][][]; // [frame][vertex][x,y,z] — the morphing mesh
  warp: number[][]; // [frame][vertex]
  token_emis: number[][]; // [frame][token]
  // per-frame surface flow field (top emitters → predicted next token), placed on the sphere
  surface_src: number[][][]; // [frame][k][x,y,z]
  surface_dst: number[][][]; // [frame][k][x,y,z]
  surface_src_strs: string[][]; // [frame][k]
  surface_dst_strs: string[][]; // [frame][k]
  surface_probs: number[][]; // [frame][k]
}

export interface VectorFieldAnimation {
  n_frames: number;
  layer_to: number;
  num_layers: number;
  grid_n: number;
  reference_points: number; // G = grid_n^2 (STATIC vertices)
  arrow_len: number; // in plot (PCA) units
  token_strs: Record<string, string>; // token id (as string) -> decoded token
  trajectory_token_strs: string[];
  grid: number[][]; // [G][x,y] — fixed vertices, identical every frame
  from_tokens: number[][]; // [frame][vertex] nearest reference token id (the token this spot refers to)
  to_tokens: number[][]; // [frame][vertex] its predicted next token id
  dirs: number[][][]; // [frame][vertex][x,y] unit arrow direction
  probs: number[][]; // [frame][vertex]
  trajectory: number[][]; // [token][x,y] in the consistent frame
  trajectory_probs: number[];
}

export interface TokenCloud {
  model_id: string;
  vocab_size: number;
  seed: number;
  spread_mu: number;
  coords: number[][]; // one [x, y] per printable token (the spread layout)
  token_ids: number[];
  token_strs: string[]; // real decoded strings, aligned with coords/token_ids
}

export interface SankeyNode {
  pos: number;
  token: number;
  count: number;
  prob: number; // empirical marginal share at this position
}
export interface SankeyLink {
  pos: number;
  source_token: number;
  target_token: number;
  value: number;
  cond: number; // empirical P(target | source) at this position
}
export interface SankeyHighlight {
  pos: number;
  token: number;
  token_str: string;
  prob: number; // teacher-forced P(token | prompt + response[:pos])
}
export interface SankeyData {
  n_steps: number;
  n_particles: number;
  nodes: SankeyNode[];
  links: SankeyLink[];
  token_strs: Record<string, string>;
  per_position: { pos: number; top: { token: number; prob: number }[] }[];
  token_order: number[]; // fixed token rows (top → bottom), identical at every position
  max_pos: number; // last sequence position reached
}
// The user's response path — a cheap, swarm-independent overlay (so editing it is instant).
export interface SankeyHighlightData {
  highlight: SankeyHighlight[];
  token_strs: Record<string, string>;
}

export interface ManifoldData {
  vertices: number[][];
  faces: number[][];
  warp: number[];
  token_points: number[][];
  token_emis: number[];
  token_strs: string[];
  token_ids: number[];
  top_tokens: { token_str: string; prob: number }[];
  traj_points: number[][]; // response tokens on the radius-2 sphere (the trajectory line)
  trajectory_token_strs?: string[];
  trajectory_emis?: number[];
  // surface flow field: from a likely token (src) to its predicted next token (dst), on the sphere
  surface_src: number[][];
  surface_dst: number[][];
  surface_src_strs: string[];
  surface_dst_strs: string[];
  surface_probs: number[];
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

  function precompute(
    artifact_type: string,
    model_id: string,
    params: Record<string, unknown> = {},
    inputs: Record<string, unknown> = {},
  ): Promise<PrecomputeResult> {
    return request("/api/precompute", jsonInit({ artifact_type, model_id, params, inputs }));
  }

  function getJob(job_id: string): Promise<JobSnapshot> {
    return request(`/api/jobs/${encodeURIComponent(job_id)}`);
  }

  function getDistribution(
    model_id: string,
    prefix_text: string,
    temperature: number,
    top_k?: number,
  ): Promise<Distribution> {
    return request("/api/distribution" + qs({ model_id, prefix_text, temperature, top_k }));
  }

  function getReduction2d(
    model_id: string,
    params: Record<string, unknown> = {},
  ): Promise<Reduction2D> {
    return request("/api/reduction/2d" + qs({ model_id, ...params }));
  }

  function getVectorField(model_id: string, params: Record<string, unknown> = {}): Promise<VectorField> {
    return request("/api/vector_field" + qs({ model_id, ...params }));
  }

  function getVectorFieldAnimation(model_id: string, params: Record<string, unknown> = {}): Promise<VectorFieldAnimation> {
    return request("/api/vector_field_animation" + qs({ model_id, ...params }));
  }

  function getSankey(model_id: string, params: Record<string, unknown> = {}): Promise<SankeyData> {
    return request("/api/sankey" + qs({ model_id, ...params }));
  }

  function getSankeyHighlight(model_id: string, params: Record<string, unknown> = {}): Promise<SankeyHighlightData> {
    return request("/api/sankey_highlight" + qs({ model_id, ...params }));
  }

  function getManifold(model_id: string, params: Record<string, unknown> = {}): Promise<ManifoldData> {
    return request("/api/manifold" + qs({ model_id, ...params }));
  }

  function getManifoldAnimation(model_id: string, params: Record<string, unknown> = {}): Promise<ManifoldAnimation> {
    return request("/api/manifold_animation" + qs({ model_id, ...params }));
  }

  // The full-vocab cloud only depends on (model, seed, spread_mu) and is multi-MB, so it's
  // fetched once and memoized; the vector field re-fetches only its (small) arrows.
  const cloudCache = new Map<string, Promise<TokenCloud>>();
  function getTokenCloud(model_id: string, seed = 0, spread_mu = 0.65): Promise<TokenCloud> {
    const k = `${model_id}|${seed}|${spread_mu}`;
    let p = cloudCache.get(k);
    if (!p) {
      p = request<TokenCloud>("/api/token_cloud" + qs({ model_id, seed, spread_mu })).catch((e) => {
        cloudCache.delete(k); // don't memoize failures
        throw e;
      });
      cloudCache.set(k, p);
    }
    return p;
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

  // Ensure an artifact is cached (compute once if needed), reporting progress.
  async function ensureArtifact(
    artifact_type: string,
    model_id: string,
    params: Record<string, unknown> = {},
    inputs: Record<string, unknown> = {},
    onProgress?: ProgressFn,
  ): Promise<string> {
    const result = await precompute(artifact_type, model_id, params, inputs);
    if (result.ready) {
      onProgress?.(1, "cached");
      return result.cache_key;
    }
    if (!result.job_id) throw new ApiError("NoJob", "precompute returned no job id");
    await pollJob(result.job_id, onProgress);
    return result.cache_key;
  }

  // --- Feature 002: Geometry Lab (/api/geo/*, frozen contract) ---

  function getGeoSpec(): Promise<GeoSpec> {
    return request("/api/geo/spec");
  }

  // Idempotent, single-flight; omitted seed defers to the server default (0).
  function geoTrain(seed?: number): Promise<GeoTrainResult> {
    return request("/api/geo/train", jsonInit(seed === undefined ? {} : { seed }));
  }

  function geoTokenize(text: string): Promise<GeoTokenizeResult> {
    return request("/api/geo/tokenize" + qs({ text }));
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

  return {
    listModels,
    resolveModel,
    precompute,
    getJob,
    getDistribution,
    getReduction2d,
    getVectorField,
    getVectorFieldAnimation,
    getSankey,
    getSankeyHighlight,
    getManifold,
    getManifoldAnimation,
    getTokenCloud,
    tokenize,
    pollJob,
    subscribeProgress,
    ensureArtifact,
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
    getArchGraph,
    getArchWeights,
    getArchTrace,
    archGenerate,
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
