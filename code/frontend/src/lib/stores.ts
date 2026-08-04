// App-level shell state.
//
// Feature 004 removed the three embedding-geometry views, and with them every
// "shared control" store (model / prefix / temperature / layer range / particle
// swarm / RBF width / response animation). The explorer tabs own their controls
// outright — see lib/explorerStores.ts — so the only shell state left is which
// tab is showing.
import { get, writable, type Readable } from "svelte/store";

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

/**
 * Work that a tab switch would destroy, registered by whoever owns it.
 *
 * `App.svelte` renders exactly one tab (`{#if $view === …}`), so switching tabs unmounts
 * the other three. That is the right structure — the Geometry Lab holds a WebGL context
 * and the Architecture Explorer holds a model — but it means a component's `onDestroy` is
 * also the app's "discard everything you were doing" path. The Lexicon Lab's training
 * worker is terminated there, so a run at step 114 of 400 vanished on a tab click: the
 * button returned to idle, nothing was shown, and roughly forty seconds of real training
 * was gone. Every `Explain` deep-dive ends with a button that navigates to the Info tab,
 * and the landing tab offers "New here? Start with Info →", so the app actively invited
 * the click that destroyed the work (red-team D, F2).
 *
 * The registry is a plain list rather than a boolean because the message has to NAME what
 * would be lost. A component registers when its work starts and releases it in every
 * terminal branch AND in `onDestroy` — releasing on destroy is what makes a confirmed
 * navigation leave the registry empty rather than latched.
 */
export interface PendingWork {
  /** Unique per registrant; re-registering the same id replaces the label. */
  id: string;
  /** What the reader loses, in their words: "a training run in the Lexicon Lab". */
  label: string;
}

const pending = writable<readonly PendingWork[]>([]);

/** Read-only view of the registry, for anything that wants to warn about it. */
export const pendingWork: Readable<readonly PendingWork[]> = { subscribe: pending.subscribe };

export function registerWork(id: string, label: string): void {
  pending.update((list) => [...list.filter((w) => w.id !== id), { id, label }]);
}

export function releaseWork(id: string): void {
  pending.update((list) => list.filter((w) => w.id !== id));
}

/**
 * A navigation held back because it would destroy registered work, or `null`.
 *
 * The prompt is in-app rather than `window.confirm`: a native dialog is not stylable, is
 * suppressed in some embedding contexts, and — the deciding reason — cannot say which run
 * is at stake without the caller building the sentence anyway.
 */
const navHold = writable<{ target: View; work: readonly PendingWork[] } | null>(null);
export const pendingNavigation: Readable<{ target: View; work: readonly PendingWork[] } | null> = {
  subscribe: navHold.subscribe,
};

/** Go through with the held navigation. The work's owner unmounts and releases itself. */
export function confirmNavigation(): void {
  const held = get(navHold);
  if (held === null) return;
  navHold.set(null);
  current.set(held.target);
}

/** Abandon the held navigation and stay where the work is. */
export function cancelNavigation(): void {
  navHold.set(null);
}

const current = writable<View>(initialView());

/**
 * The active tab.
 *
 * `set` is guarded rather than plain, so there is exactly ONE way to change the view and
 * it cannot silently discard work. With the guard in the eleven call sites instead, the
 * twelfth would have been the one that lost a training run. When nothing is registered —
 * which is almost always — this is an ordinary `writable`.
 */
export const view = {
  subscribe: current.subscribe,
  set(next: View): void {
    if (next === get(current) || get(pending).length === 0) {
      current.set(next);
      return;
    }
    navHold.set({ target: next, work: get(pending) });
  },
  update(fn: (v: View) => View): void {
    view.set(fn(get(current)));
  },
};

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
  current.subscribe((v) => {
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
    // Back is a tab switch too, and it destroys work exactly as a click does. The
    // traversal has already happened by the time we hear about it, so the address bar is
    // put back on the tab still showing and the same prompt is raised. That costs the
    // Forward stack, which is the cheaper of the two things on offer here.
    if (next !== get(current) && get(pending).length > 0) {
      window.history.pushState(null, "", `#${get(current)}`);
      navHold.set({ target: next, work: get(pending) });
      return;
    }
    applyingFromUrl = true;
    try {
      current.set(next);
    } finally {
      applyingFromUrl = false;
    }
  };
  window.addEventListener("popstate", syncFromUrl);
  window.addEventListener("hashchange", syncFromUrl);
}
