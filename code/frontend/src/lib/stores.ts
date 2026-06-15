// Shared interaction parameters (FR-015): the controls every visualization reuses.
import { writable } from "svelte/store";

export const modelId = writable<string>("Qwen/Qwen2.5-0.5B-Instruct");
export const prefixText = writable<string>("The capital of France is");
export const temperature = writable<number>(1.0);
export const layer = writable<number>(0);
// Optional response string traced as a trajectory through embedding space (§1/§2).
export const responseText = writable<string>("");

// Capabilities of the currently-selected model (e.g., layer count for the slider).
export const numLayers = writable<number>(0);

// Which view is active in the main panel.
export type View = "vector" | "sankey" | "manifold" | "preview";
export const view = writable<View>("vector");

export function clampTemperature(value: number): number {
  if (Number.isNaN(value) || value < 0) return 0;
  return Math.min(value, 2);
}

export function clampLayer(value: number, maxLayer: number): number {
  if (Number.isNaN(value) || value < 0) return 0;
  return Math.min(Math.round(value), Math.max(0, maxLayer));
}
