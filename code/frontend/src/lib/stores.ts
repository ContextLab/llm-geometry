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

// Which view is active in the main panel.
export type View = "vector" | "sankey" | "manifold";
export const view = writable<View>("vector");

export function clampTemperature(value: number): number {
  if (Number.isNaN(value) || value < 0) return 0;
  return Math.min(value, 2);
}
