// App-level shell state.
//
// Feature 004 removed the three embedding-geometry views, and with them every
// "shared control" store (model / prefix / temperature / layer range / particle
// swarm / RBF width / response animation). The two explorer tabs own their
// controls outright — see lib/explorerStores.ts — so the only shell state left
// is which tab is showing.
import { writable } from "svelte/store";

/** Which view is active in the main panel. */
export type View = "architecture" | "geometry" | "lexicon" | "info";

const VIEWS: View[] = ["architecture", "geometry", "lexicon", "info"];

const isView = (v: string): v is View => (VIEWS as string[]).includes(v);

/**
 * The active tab, mirrored into `location.hash`.
 *
 * Without this the app has exactly one URL: you cannot send a colleague a link to the
 * Info tab, a reload always lands on Architecture, and Back does nothing. The hash is
 * the whole state — deliberately: everything else (model, prompt, weights) is either
 * expensive or session-scoped, and a URL that promises to restore it would lie.
 *
 * Guarded for non-browser contexts (SSR/unit tests) via `typeof window`.
 */
function initialView(): View {
  if (typeof window === "undefined") return "architecture";
  const h = window.location.hash.replace(/^#/, "");
  return isView(h) ? h : "architecture";
}

export const view = writable<View>(initialView());

if (typeof window !== "undefined") {
  // Store -> URL. replaceState, not a hash assignment: tab switching should not fill
  // the back stack with every click, but the address bar must stay copyable.
  // The first emission is skipped so simply loading the page does not rewrite a clean
  // URL into `#architecture`; the hash appears once the reader actually navigates.
  let first = true;
  view.subscribe((v) => {
    if (first) {
      first = false;
      return;
    }
    const target = `#${v}`;
    if (window.location.hash !== target) {
      window.history.replaceState(null, "", target);
    }
  });

  // URL -> store, so Back/Forward and a pasted link both work.
  window.addEventListener("hashchange", () => {
    const h = window.location.hash.replace(/^#/, "");
    if (isView(h)) view.set(h);
  });
}
