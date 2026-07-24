// Shared cursor-following tooltip used for interactive hover across the views.
import { writable } from "svelte/store";

export interface TooltipState {
  show: boolean;
  x: number;
  y: number;
  text: string;
}

export const tooltip = writable<TooltipState>({ show: false, x: 0, y: 0, text: "" });

export function showTip(event: { clientX: number; clientY: number }, text: string): void {
  tooltip.set({ show: true, x: event.clientX, y: event.clientY, text });
}

export function hideTip(): void {
  tooltip.update((t) => ({ ...t, show: false }));
}
