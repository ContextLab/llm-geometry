// App-level shell state.
//
// Feature 004 removed the three embedding-geometry views, and with them every
// "shared control" store (model / prefix / temperature / layer range / particle
// swarm / RBF width / response animation). The two explorer tabs own their
// controls outright — see lib/explorerStores.ts — so the only shell state left
// is which tab is showing.
import { writable } from "svelte/store";

/** Which view is active in the main panel. */
export type View = "architecture" | "geometry";

export const view = writable<View>("architecture");
