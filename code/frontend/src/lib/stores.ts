// App-level shell state.
//
// Feature 004 removed the three embedding-geometry views, and with them every
// "shared control" store (model / prefix / temperature / layer range / particle
// swarm / RBF width / response animation). The explorer tabs own their controls
// outright — see lib/explorerStores.ts — so the only shell state left is which
// tab is showing.
import { writable } from "svelte/store";

/** Which view is active in the main panel. */
export type View = "architecture" | "geometry" | "lexicon" | "info";

const VIEWS: View[] = ["architecture", "geometry", "lexicon", "info"];

/** Where a cold load with no usable fragment lands. */
const LANDING: View = "architecture";

/**
 * The view a fragment names, or `null`. Case-insensitive: a mail client that title-cases
 * `#Info`, or a reader who types it, meant the Info tab, and answering that with the
 * Architecture tab and no explanation is not a defensible reading of the URL.
 */
function parseHash(hash: string): View | null {
  const h = hash.replace(/^#/, "").trim().toLowerCase();
  return (VIEWS as string[]).includes(h) ? (h as View) : null;
}

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
  if (typeof window === "undefined") return LANDING;
  return parseHash(window.location.hash) ?? LANDING;
}

export const view = writable<View>(initialView());

if (typeof window !== "undefined") {
  // A fragment we did not honour is corrected ONCE, at load, so the address bar never
  // keeps advertising a view the reader is not looking at: `#Info` becomes `#info`, and
  // `#not-a-tab` becomes `#architecture` (which is what it actually rendered). This is a
  // replaceState — correcting the current entry, not adding one — and a hash-free URL is
  // left alone, because `/` legitimately means "the landing tab".
  {
    const raw = window.location.hash.replace(/^#/, "");
    const canonical = initialView();
    if (raw !== "" && raw !== canonical) {
      window.history.replaceState(null, "", `#${canonical}`);
    }
  }

  // True while a history event is being applied to the store, so the subscriber below
  // does not push a duplicate entry for a navigation the browser already performed.
  let applyingFromUrl = false;

  // Store -> URL. pushState, so Back and Forward step through the tabs the reader
  // actually visited — the third thing this store exists to fix, and the one that was
  // silently missing while the docstring claimed it. One entry per NAVIGATION, not per
  // click: re-selecting the tab already showing changes no hash and pushes nothing, so
  // the back stack stays as long as the reader's actual path and no longer.
  //
  // The first emission is skipped so simply loading the page does not rewrite a clean
  // URL into `#architecture`; the hash appears once the reader actually navigates.
  let first = true;
  view.subscribe((v) => {
    if (first) {
      first = false;
      return;
    }
    if (applyingFromUrl) return;
    const target = `#${v}`;
    if (window.location.hash !== target) {
      window.history.pushState(null, "", target);
    }
  });

  // URL -> store, so Back/Forward and a pasted link both work. `popstate` is what a
  // history traversal fires; `hashchange` covers a fragment typed into the address bar
  // of the page already open.
  const syncFromUrl = (): void => {
    const raw = window.location.hash.replace(/^#/, "");
    // An unfamiliar fragment arriving mid-session is left alone: the reader is looking
    // at a tab, and a stray hash must not yank them out of it. Only a cold load falls
    // back to the landing tab (and rewrites the URL to say so).
    if (raw !== "" && parseHash(raw) === null) return;
    const next = parseHash(raw) ?? LANDING;
    applyingFromUrl = true;
    try {
      view.set(next);
    } finally {
      applyingFromUrl = false;
    }
  };
  window.addEventListener("popstate", syncFromUrl);
  window.addEventListener("hashchange", syncFromUrl);
}
