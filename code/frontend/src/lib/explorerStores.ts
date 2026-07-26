// Per-view stores for the two explorer tabs (specs/002-interactive-model-explorer,
// User Story 3 / §3b): the Architecture and Geometry tabs own their controls outright,
// so the existing global stores' semantics (model/prompt/temperature for the three
// embedding-geometry views) are never overloaded and no state leaks across meanings.
import { writable } from "svelte/store";

import type { GeoFieldMode } from "./dataClient";

export type { GeoFieldMode } from "./dataClient";

// ---------------------------------------------------------------------------
// Architecture Explorer (/api/arch/*)
// ---------------------------------------------------------------------------

/**
 * The traced open-weights model. Qwen2.5-0.5B-Instruct is the default (feature 004):
 * the previous 135M default is near the floor of usable reply quality no matter how
 * the sampling is tuned, and the smaller models stay in the menu for speed.
 */
export const archModelId = writable<string>("Qwen/Qwen2.5-0.5B-Instruct");
export const archPrompt = writable<string>("What is the capital of France?");
export const archSystemPrompt = writable<string>("");
/** POST /api/arch/generate default per the frozen contract. */
export const archTemperature = writable<number>(0.8);
/** ≤ 128 per the contract; default 64. */
export const archMaxNewTokens = writable<number>(64);
/** Currently selected diagram node id (stable dotted path), or null. */
export const archSelectedNode = writable<string | null>(null);

// ---------------------------------------------------------------------------
// Geometry Lab (/api/geo/*)
// ---------------------------------------------------------------------------

/** UI layer selection for the fixed 4-layer GeoTransformer ("full" = final layer readout). */
export type GeoLayerSelection = "full" | 0 | 1 | 2 | 3;

// The tiny model is trained on a public-domain corpus with a ~1000-word vocab, so the
// default prompt sticks to common corpus words (out-of-vocab words surface as <unk>).
export const geoPrompt = writable<string>(
  "alice was beginning to get very tired of sitting by her sister",
);
export const geoFieldMode = writable<GeoFieldMode>("next_next");
export const geoLayer = writable<GeoLayerSelection>("full");
/** Vector-field sampling temperature (contract default 0 = argmax arrows). */
export const geoTemperature = writable<number>(0);
/** Arrows per point (contract default 1). */
export const geoTopM = writable<number>(1);
/**
 * Force mode only: use (W_V − W_Vᵀ)/2 so the per-point field is exactly tangent.
 *
 * ON by default (feature 004): the raw W_V·z field has a radial component, so it draws
 * as arrows shooting off the sphere the points live on — unreadable as a surface field.
 * Unchecking it shows the raw operator, which is a legitimate thing to look at; the
 * "tangent: exact" badge disappears and the residual badge takes over.
 */
export const geoAntisymmetrize = writable<boolean>(true);

// ---------------------------------------------------------------------------
// geoWeightsToken — persisted to sessionStorage (spec acceptance 2.3: a page
// refresh preserves weight edits). init ← read; subscribe → write; null clears.
// Tokens are stateless content hashes (contract), so a stored token stays valid
// across page loads and backend restarts.
// ---------------------------------------------------------------------------

export const GEO_WEIGHTS_TOKEN_KEY = "llm-geometry:geo-weights-token";

function readStoredWeightsToken(): string | null {
  try {
    if (typeof sessionStorage === "undefined") return null;
    return sessionStorage.getItem(GEO_WEIGHTS_TOKEN_KEY);
  } catch {
    return null; // storage unavailable (e.g. privacy mode) — run without persistence
  }
}

/** Active edited-weights token; null = canonical learned checkpoint. */
export const geoWeightsToken = writable<string | null>(readStoredWeightsToken());

geoWeightsToken.subscribe((token) => {
  try {
    if (typeof sessionStorage === "undefined") return;
    if (token === null) sessionStorage.removeItem(GEO_WEIGHTS_TOKEN_KEY);
    else sessionStorage.setItem(GEO_WEIGHTS_TOKEN_KEY, token);
  } catch {
    // storage unavailable — the token still lives in memory for this session
  }
});
