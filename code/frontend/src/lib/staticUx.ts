/**
 * Shared UX helpers for the static (GitHub Pages) build — feature 003, agent D.
 *
 * FR-203 (honesty): in static mode nothing silently no-ops and nothing fakes.
 * These helpers give every view a consistent way to (a) detect the static
 * build, (b) recognize the data layer's typed "not precomputed" refusals, and
 * (c) apply a precomputed preset's saved control state so the view's normal
 * fetch path hits the recorded artifact exactly.
 */

import { get } from "svelte/store";

import { client, DATA_MODE } from "./dataClient";
import { isStaticClient, type StaticClient } from "./staticClient";
import {
  fanout,
  layerFrom,
  layerTo,
  modelId,
  nParticles,
  nSteps,
  prefixText,
  rbfWidth,
  responseText,
  temperature,
} from "./stores";

/** Statically-replaced at build time (VITE_DATA_MODE) — backend builds tree-shake on it. */
export const STATIC_MODE: boolean = DATA_MODE === "static";

/** Where the masthead badge + every affordance note sends people for the full stack. */
export const README_URL = "https://github.com/ContextLab/llm-geometry#quickstart";

/** The static client's extras (presets, trace examples, runtime info), or null live. */
export function staticExtras(): StaticClient | null {
  return isStaticClient(client) ? client : null;
}

/**
 * True for the data layer's typed StaticModeError ("this build can't do that",
 * with a plain-language message naming what IS available). Duck-typed on
 * `.type` so callers don't need the ApiError class.
 */
export function isStaticMiss(e: unknown): e is { type: string; message: string } {
  return (
    typeof e === "object" &&
    e !== null &&
    (e as { type?: unknown }).type === "StaticModeError" &&
    typeof (e as { message?: unknown }).message === "string"
  );
}

/**
 * Apply a preset's saved control state (the `state` field the exporter recorded
 * alongside each replayed request) to the shared stores. Keys mirror
 * scripts/export_static_assets.py's preset_specs; unknown keys are ignored.
 */
export function applyPresetState(state: Record<string, unknown>): void {
  const num = (v: unknown): number => (typeof v === "number" ? v : Number(v));
  for (const [key, v] of Object.entries(state)) {
    switch (key) {
      case "prefix_text":
        prefixText.set(String(v));
        break;
      case "response_text":
        responseText.set(String(v));
        break;
      case "temperature":
        temperature.set(num(v));
        break;
      case "layer_from":
        layerFrom.set(num(v));
        break;
      case "layer_to":
        layerTo.set(num(v));
        break;
      case "fanout":
        fanout.set(num(v));
        break;
      case "n_particles":
        nParticles.set(num(v));
        break;
      case "n_steps":
        nSteps.set(num(v));
        break;
      case "width":
        rbfWidth.set(num(v));
        break;
    }
  }
}

/** Normalized (String-ified, like the query string) comparison of one state field. */
export function stateFieldEqual(a: unknown, b: unknown): boolean {
  return String(a) === String(b);
}

// --- static boot ------------------------------------------------------------------------

let presetModelPromise: Promise<string | null> | null = null;

/** The model id every 001 preset was recorded against (index.json `preset_model`). */
export function staticPresetModel(): Promise<string | null> {
  if (!presetModelPromise) {
    presetModelPromise = fetch(`${import.meta.env.BASE_URL}static-data/index.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((idx: { preset_model?: unknown } | null) =>
        idx && typeof idx.preset_model === "string" ? idx.preset_model : null,
      )
      .catch(() => null);
  }
  return presetModelPromise;
}

/**
 * One-time static boot: point the shared stores at preset 1 of every 001 view
 * (and at the recorded preset model), so the first render of each tab is an
 * exact preset hit. With full exports preset 1 == the built-in defaults and
 * this is a no-op; with --quick exports (different model / particle counts) it
 * is what makes the default state render instantly instead of missing.
 * Failures fall back to the built-in defaults — the views then surface their
 * own designed notes (never a silent blank).
 */
export async function applyStaticDefaults(): Promise<void> {
  const sc = staticExtras();
  if (!sc) return;
  try {
    const model = await staticPresetModel();
    // Only steer the model if the user hasn't already picked one this session.
    if (model && get(modelId) !== model) modelId.set(model);
    const merged: Record<string, unknown> = {};
    for (const view of ["vector", "sankey", "manifold"] as const) {
      const presets = await sc.staticPresets(view);
      if (presets[0]) Object.assign(merged, presets[0].state);
    }
    applyPresetState(merged);
  } catch {
    // static-data unreachable — proceed with defaults; views report honestly.
  }
}
