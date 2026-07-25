/**
 * Preset-backed serving for the 001 views (vector field / Sankey / manifold).
 *
 * Each `static-data/presets/<view>/<n>.json` was produced by replaying the
 * views' EXACT request param dicts against the real backend (003-A), so a
 * request is served if and only if its full param dict matches a recorded one.
 * Anything else is an honest StaticModeError naming the available presets —
 * never an approximation (FR-203).
 */

import type { StaticAssets, StaticPresetEntry } from "./assets";
import { staticModeError } from "./errors";

export interface PresetRequest {
  endpoint: string; // e.g. "/api/vector_field"
  params: Record<string, unknown>; // the exact query dict the view sends
  response: unknown; // the real backend's response, verbatim
}

export interface PresetFile {
  schema_version: number;
  view: string;
  n: number;
  label: string;
  model_id: string;
  state: Record<string, unknown>;
  requests: PresetRequest[];
}

export type PresetView = "vector" | "sankey" | "manifold";
export const PRESET_VIEWS: readonly PresetView[] = ["vector", "sankey", "manifold"];

/** Which view's preset files can contain a given endpoint (loads stay lazy). */
const ENDPOINT_VIEWS: Record<string, readonly PresetView[]> = {
  "/api/vector_field": ["vector"],
  "/api/vector_field_animation": ["vector"],
  "/api/sankey": ["sankey"],
  "/api/sankey_highlight": ["sankey"],
  "/api/manifold": ["manifold"],
  "/api/manifold_animation": ["manifold"],
  // tokenize calls ride along inside response-bearing presets of every view
  "/api/tokenize": PRESET_VIEWS,
};

/**
 * Compare a live request's params with a recorded preset's params. Both sides
 * are normalized the way the live client's query string does it — String(v),
 * dropping null/undefined — so 1 vs 1.0 and true vs "true" compare equal.
 */
export function paramsMatch(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
): boolean {
  const na = normalize(a);
  const nb = normalize(b);
  const ka = Object.keys(na);
  if (ka.length !== Object.keys(nb).length) return false;
  return ka.every((k) => nb[k] === na[k]);
}

function normalize(p: Record<string, unknown>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(p)) {
    if (v === undefined || v === null) continue;
    out[k] = String(v);
  }
  return out;
}

export class PresetStore {
  constructor(private readonly assets: StaticAssets) {}

  async manifest(view: PresetView): Promise<StaticPresetEntry[]> {
    const idx = await this.assets.index();
    return idx.presets[view] ?? [];
  }

  async presetFiles(view: PresetView): Promise<PresetFile[]> {
    const entries = await this.manifest(view);
    return Promise.all(
      entries.map((e) => this.assets.json<PresetFile>(`presets/${view}/${e.file}`)),
    );
  }

  /** The single full-vocab token-cloud request (seed 0, spread_mu 0.65). */
  tokenCloud(): Promise<PresetRequest> {
    return this.assets.json<PresetRequest>("presets/token_cloud.json");
  }

  /**
   * Serve `params` for `endpoint` from the recorded presets, or throw a
   * StaticModeError that names what IS available.
   */
  async serve(endpoint: string, params: Record<string, unknown>): Promise<unknown> {
    const hit = await this.find(endpoint, params);
    if (hit !== null) return hit;
    throw staticModeError(await this.missMessage(endpoint));
  }

  /** Like serve(), but returns null on a miss (for live-fallback paths). */
  async find(endpoint: string, params: Record<string, unknown>): Promise<unknown | null> {
    if (endpoint === "/api/token_cloud") {
      const tc = await this.tokenCloud();
      return paramsMatch(params, tc.params) ? tc.response : null;
    }
    for (const view of ENDPOINT_VIEWS[endpoint] ?? PRESET_VIEWS) {
      for (const file of await this.presetFiles(view)) {
        for (const req of file.requests) {
          if (req.endpoint === endpoint && paramsMatch(params, req.params)) {
            return req.response;
          }
        }
      }
    }
    return null;
  }

  async missMessage(endpoint: string): Promise<string> {
    const views = ENDPOINT_VIEWS[endpoint] ?? PRESET_VIEWS;
    const labels: string[] = [];
    for (const view of views) {
      for (const e of await this.manifest(view)) labels.push(`“${e.label}”`);
    }
    const available =
      labels.length > 0
        ? ` Available presets: ${labels.join(", ")}.`
        : " No presets cover this endpoint.";
    return (
      `This is the static demo build — ${endpoint} is served from precomputed ` +
      `presets only, and this exact request is not one of them.${available} ` +
      "Run the full stack locally (see the README) for arbitrary inputs."
    );
  }
}
