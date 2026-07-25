/**
 * Static-asset access for the Pages build: everything under
 * `${BASE_URL}static-data/` was exported by the REAL Python backend at build
 * time (scripts/export_static_assets.py — see notes/agent-reports/003-A.md for
 * the schemas). This module only fetches + caches; it never synthesizes data.
 */

import { ApiError } from "../dataClient";

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

// --- index.json (the export manifest) ---------------------------------------------

export interface StaticIndexModel {
  model_id: string;
  slug: string; // directory name under static-data/arch/
  revision: string; // pinned HF commit for weights + tokenizer
  n_params: number;
}

export interface StaticPresetEntry {
  n: number;
  label: string;
  file: string;
  bytes: number;
}

export interface StaticIndex {
  schema_version: number;
  generated_at: string;
  git_sha: string;
  quick: boolean;
  geo: { checkpoint_id: string };
  arch_models: StaticIndexModel[];
  preset_model: string;
  presets: Record<string, StaticPresetEntry[]>;
  files: Record<string, number>;
}

export interface StaticAssetsOptions {
  /** Site base (defaults to import.meta.env.BASE_URL, "/" in dev). */
  baseUrl?: string;
  fetchImpl?: FetchLike;
}

export class StaticAssets {
  readonly base: string;
  private readonly fetchImpl: FetchLike;
  private readonly jsonCache = new Map<string, Promise<unknown>>();
  private readonly binCache = new Map<string, Promise<ArrayBuffer>>();

  constructor(opts: StaticAssetsOptions = {}) {
    const base = opts.baseUrl ?? import.meta.env.BASE_URL ?? "/";
    this.base = base.endsWith("/") ? base : `${base}/`;
    this.fetchImpl = opts.fetchImpl ?? ((i, init) => fetch(i, init));
  }

  url(rel: string): string {
    return `${this.base}static-data/${rel}`;
  }

  /** Fetch + parse a JSON asset, memoized. Failures are typed and NOT memoized. */
  json<T>(rel: string): Promise<T> {
    let p = this.jsonCache.get(rel);
    if (!p) {
      p = this.load(rel, "json").catch((e) => {
        this.jsonCache.delete(rel);
        throw e;
      });
      this.jsonCache.set(rel, p);
    }
    return p as Promise<T>;
  }

  /** Fetch a binary asset (e.g. tiles.bin), memoized. */
  bin(rel: string): Promise<ArrayBuffer> {
    let p = this.binCache.get(rel);
    if (!p) {
      p = this.load(rel, "bin").catch((e) => {
        this.binCache.delete(rel);
        throw e;
      }) as Promise<ArrayBuffer>;
      this.binCache.set(rel, p);
    }
    return p;
  }

  index(): Promise<StaticIndex> {
    return this.json<StaticIndex>("index.json");
  }

  /** Raw fetch through the injected impl (used by the safetensors reader). */
  rawFetch(input: string, init?: RequestInit): Promise<Response> {
    return this.fetchImpl(input, init);
  }

  private async load(rel: string, kind: "json" | "bin"): Promise<unknown> {
    const url = this.url(rel);
    let res: Response;
    try {
      res = await this.fetchImpl(url);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new ApiError("NetworkError", `Could not load static asset ${rel}: ${msg}`);
    }
    if (!res.ok) {
      if (res.status === 404) {
        throw new ApiError(
          "NotFoundError",
          `Static asset ${rel} is missing from this build (HTTP 404).`,
        );
      }
      throw new ApiError("HttpError", `Static asset ${rel}: HTTP ${res.status}`);
    }
    if (kind === "bin") return res.arrayBuffer();
    const text = await res.text();
    try {
      return JSON.parse(text) as unknown;
    } catch {
      throw new ApiError("BadResponse", `Static asset ${rel} is not valid JSON.`);
    }
  }
}
