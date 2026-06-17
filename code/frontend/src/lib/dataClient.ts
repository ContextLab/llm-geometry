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
  prob: number; // empirical marginal share (or teacher-forced model prob for highlight nodes)
  highlight: boolean;
}
export interface SankeyLink {
  pos: number;
  source_token: number;
  target_token: number;
  value: number;
  cond: number; // empirical P(target | source) at this position
  highlight: boolean;
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
  highlight: SankeyHighlight[]; // the user's response path (empty if none specified)
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
    const res = await doFetch(base + path, init);
    const text = await res.text();
    const data = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const env = data?.error;
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
    handlers: { onProgress?: ProgressFn; onDone?: () => void; onError?: (type: string, message: string) => void },
  ): () => void {
    if (!ESImpl) {
      let cancelled = false;
      pollJob(job_id, (p, m) => !cancelled && handlers.onProgress?.(p, m))
        .then(() => !cancelled && handlers.onDone?.())
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
    es.addEventListener("done", () => {
      handlers.onDone?.();
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
    getManifold,
    getManifoldAnimation,
    getTokenCloud,
    tokenize,
    pollJob,
    subscribeProgress,
    ensureArtifact,
  };
}

export type Client = ReturnType<typeof createClient>;

// Default client used by the app (same-origin; Vite proxies /api in dev).
export const client = createClient();
