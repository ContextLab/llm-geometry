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
  layer: number;
  num_layers: number;
  temperature: number;
  fanout: number;
  reference_points: number;
  starts: number[][];
  ends: number[][];
  probs: number[];
  start_token_strs: string[];
  end_token_strs: string[];
  trajectory?: number[][];
  trajectory_probs?: number[];
  trajectory_token_strs?: string[];
}

export interface SankeyNode {
  pos: number;
  token: number;
  count: number;
}
export interface SankeyLink {
  pos: number;
  source_token: number;
  target_token: number;
  value: number;
}
export interface SankeyData {
  n_steps: number;
  nodes: SankeyNode[];
  links: SankeyLink[];
  token_strs: Record<string, string>;
  per_position: { pos: number; top: { token: number; prob: number }[] }[];
}

export interface ManifoldData {
  vertices: number[][];
  faces: number[][];
  warp: number[];
  token_points: number[][];
  token_emis: number[];
  top_tokens: { token_str: string; prob: number }[];
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

  function getSankey(model_id: string, params: Record<string, unknown> = {}): Promise<SankeyData> {
    return request("/api/sankey" + qs({ model_id, ...params }));
  }

  function getManifold(model_id: string, params: Record<string, unknown> = {}): Promise<ManifoldData> {
    return request("/api/manifold" + qs({ model_id, ...params }));
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
    getSankey,
    getManifold,
    pollJob,
    subscribeProgress,
    ensureArtifact,
  };
}

export type Client = ReturnType<typeof createClient>;

// Default client used by the app (same-origin; Vite proxies /api in dev).
export const client = createClient();
