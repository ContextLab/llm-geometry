/**
 * Shared UX helpers for the static (GitHub Pages) build.
 *
 * FR-203 (honesty): in static mode nothing silently no-ops and nothing fakes.
 * These helpers give every view a consistent way to (a) detect the static build,
 * (b) recognize the data layer's typed "not available in this build" refusals,
 * and (c) reach the static-only extras (trace examples, runtime info).
 *
 * Feature 004 removed the 001 preset machinery that used to live here along with
 * the three views it served.
 */

import { client, DATA_MODE } from "./dataClient";
import { isStaticClient, type StaticClient } from "./staticClient";

/** Statically-replaced at build time (VITE_DATA_MODE) — backend builds tree-shake on it. */
export const STATIC_MODE: boolean = DATA_MODE === "static";

/** Where the masthead badge + every affordance note sends people for the full stack. */
export const README_URL = "https://github.com/ContextLab/llm-geometry#quickstart";

/** The static client's extras (trace examples, runtime info), or null when live. */
export function staticExtras(): StaticClient | null {
  return isStaticClient(client) ? client : null;
}

/**
 * True for the data layer's typed StaticModeError ("this build can't do that",
 * with a plain-language message naming what IS available). Duck-typed on `.type`
 * so callers don't need the ApiError class.
 */
export function isStaticMiss(e: unknown): e is { type: string; message: string } {
  return (
    typeof e === "object" &&
    e !== null &&
    (e as { type?: unknown }).type === "StaticModeError" &&
    typeof (e as { message?: unknown }).message === "string"
  );
}
