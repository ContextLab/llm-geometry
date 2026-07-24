// Shared interaction parameters (FR-015): the controls every visualization reuses.
import { writable } from "svelte/store";

export const modelId = writable<string>("Qwen/Qwen2.5-0.5B-Instruct");
export const prefixText = writable<string>("The capital of France is");
export const temperature = writable<number>(1.0);

// Layer RANGE (the doc's "layer(s)"). Single layer = from === to.
export const layerFrom = writable<number>(0);
export const layerTo = writable<number>(0);
export const numLayers = writable<number>(0);

// Optional response string traced as a trajectory through embedding space (§1/§2), and
// the animation step over its tokens (§1/§3: "how it changes with each subsequent token").
export const responseText = writable<string>("");
export const responseStep = writable<number>(0);
export const responseTokenCount = writable<number>(0);
export const isPlaying = writable<boolean>(false);

// Particle-swarm controls (Sankey): how many particles, and the max sequence length.
export const nParticles = writable<number>(1000);
export const nSteps = writable<number>(10);

// Manifold control: RBF cap width on the unit sphere (smaller = tighter, more localized domes).
export const rbfWidth = writable<number>(0.18);
// Manifold: overlay the surface flow field (from a likely token → its predicted next token).
export const showSurface = writable<boolean>(false);

// Which view is active in the main panel.
export type View = "vector" | "sankey" | "manifold" | "architecture" | "geometry";
export const view = writable<View>("vector");

// Bumped by the "Recompute" button to force the active view to re-fetch (bypassing the cache).
export const refreshNonce = writable<number>(0);

export function clampTemperature(value: number): number {
  if (Number.isNaN(value) || value < 0) return 0;
  return Math.min(value, 2);
}
