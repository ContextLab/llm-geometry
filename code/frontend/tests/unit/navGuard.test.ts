/**
 * Leaving a tab must not destroy work in it without saying so (red-team D, F2).
 *
 * `App.svelte` renders one tab at a time, so switching tabs unmounts the one you leave
 * and runs its `onDestroy`. The Lexicon Lab terminates its training worker there, so a
 * run at step 114 of 400 disappeared on a tab click: the button returned to idle, no
 * message, no way to resume — and the app invites exactly that click, with "New here?
 * Start with Info →" and an Info-tab button at the end of every `Explain`.
 *
 * The fix is a registry in `lib/stores.ts`: whoever owns destructible work registers it,
 * and any navigation while the registry is non-empty is HELD until the reader answers.
 * These tests drive the real store and the real `App`, in jsdom, with real history
 * entries — including browser Back, which is a tab switch too.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it, vi } from "vitest";
import { get } from "svelte/store";

const SRC = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "src");

/**
 * Load a fresh copy of the store module with the browser sitting at `url`.
 *
 * `vi.resetModules()` gives each test its own module instance, but every instance keeps
 * its `popstate` listener on the shared `window` — so a registration left behind by an
 * earlier test would make an EARLIER module hold this test's Back. `releaseAll` below is
 * the test-level stand-in for the `onDestroy` a real component runs, and every test that
 * registers work calls it.
 */
const loaded: {
  registerWork: (id: string, label: string) => void;
  releaseWork: (id: string) => void;
}[] = [];
const registered = new Set<string>();

async function loadShell(url: string) {
  window.history.replaceState(null, "", url);
  vi.resetModules();
  const mod = await import("../../src/lib/stores");
  loaded.push(mod);
  return mod;
}

/** Register work through the module under test, remembering it for `releaseAll`. */
function register(
  mod: { registerWork: (id: string, label: string) => void },
  id: string,
  label: string,
): void {
  registered.add(id);
  mod.registerWork(id, label);
}

/** Release every id any test registered, in every module instance still listening. */
function releaseAll(): void {
  for (const mod of loaded) for (const id of registered) mod.releaseWork(id);
  registered.clear();
}

afterEach(releaseAll);

/** jsdom runs history traversal as a task, so the assertion has to wait for it. */
async function until(pred: () => boolean, label: string, ms = 2000): Promise<void> {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (pred()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`timed out waiting for ${label} (hash=${window.location.hash})`);
}

describe("a tab switch that would destroy work is held, not performed", () => {
  it("switches tabs normally while nothing is registered", async () => {
    const { view, pendingNavigation } = await loadShell("#lexicon");
    view.set("info");
    expect(get(view)).toBe("info");
    expect(get(pendingNavigation)).toBeNull();
  });

  it("holds the navigation and names the work instead of discarding it", async () => {
    const shell = await loadShell("#lexicon");
    const { view, pendingNavigation } = shell;
    register(shell, "lex-train", "a training run in the Lexicon Lab");

    view.set("info");

    // The run is still on screen and the URL still points at it: nothing was destroyed.
    expect(get(view)).toBe("lexicon");
    expect(window.location.hash).toBe("#lexicon");
    const held = get(pendingNavigation);
    expect(held?.target).toBe("info");
    expect(held?.work.map((w) => w.label)).toEqual(["a training run in the Lexicon Lab"]);
  });

  it("stays put when the reader declines, and the run keeps its registration", async () => {
    const shell = await loadShell("#lexicon");
    const { view, cancelNavigation, pendingNavigation, pendingWork } = shell;
    register(shell, "lex-train", "a training run in the Lexicon Lab");
    view.set("info");
    cancelNavigation();

    expect(get(view)).toBe("lexicon");
    expect(get(pendingNavigation)).toBeNull();
    // Declining is not "the work finished": a second attempt is held again.
    expect(get(pendingWork).length).toBe(1);
    view.set("info");
    expect(get(pendingNavigation)?.target).toBe("info");
  });

  it("goes through when the reader accepts, and updates the URL", async () => {
    const shell = await loadShell("#lexicon");
    const { view, confirmNavigation, pendingNavigation } = shell;
    register(shell, "lex-train", "a training run in the Lexicon Lab");
    view.set("info");
    confirmNavigation();

    expect(get(view)).toBe("info");
    expect(window.location.hash).toBe("#info");
    expect(get(pendingNavigation)).toBeNull();
  });

  it("holds browser Back too, and puts the address bar back on the running tab", async () => {
    // Back is a tab switch. Guarding only the buttons would have left the single most
    // reflexive way out of a tab as the one that still destroys the run in silence.
    const shell = await loadShell("#architecture");
    const { view, pendingNavigation, cancelNavigation } = shell;
    // Two real pushes, so the entry Back leads to is this test's own rather than one
    // another test left in jsdom's shared history.
    view.set("geometry");
    view.set("lexicon");
    expect(window.location.hash).toBe("#lexicon");
    register(shell, "lex-train", "a training run in the Lexicon Lab");

    window.history.back();
    await until(() => get(pendingNavigation) !== null, "Back to be held");

    expect(get(view)).toBe("lexicon");
    expect(get(pendingNavigation)?.target).toBe("geometry");
    await until(() => window.location.hash === "#lexicon", "the address bar to be restored");

    cancelNavigation();
    expect(get(view)).toBe("lexicon");
  });

  it("stops holding once the work releases itself", async () => {
    const shell = await loadShell("#lexicon");
    const { view, releaseWork, pendingNavigation } = shell;
    register(shell, "lex-train", "a training run in the Lexicon Lab");
    releaseWork("lex-train");

    view.set("info");
    expect(get(view)).toBe("info");
    expect(get(pendingNavigation)).toBeNull();
  });

  it("never holds a re-selection of the tab already showing", async () => {
    const shell = await loadShell("#lexicon");
    const { view, pendingNavigation } = shell;
    register(shell, "lex-train", "a training run in the Lexicon Lab");
    view.set("lexicon");
    expect(get(pendingNavigation)).toBeNull();
  });
});

describe("the shell shows the held navigation and both ways out", () => {
  async function mountApp(url: string) {
    const stores = await loadShell(url);
    // Re-imported AFTER vi.resetModules(), like shell.test.ts: mounting a component
    // compiled against a second Svelte instance orphans its `$effect`s.
    const svelte = await import("svelte");
    const App = (await import("../../src/App.svelte")).default;
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = svelte.mount(App, { target });
    svelte.flushSync();
    return {
      stores,
      target,
      flush: svelte.flushSync,
      done: () => (svelte.unmount(app), target.remove()),
    };
  }

  it("names the work and the destination, and does not switch on its own", async () => {
    const { stores, target, flush, done } = await mountApp("#info");
    try {
      register(stores, "lex-train", "a training run in the Lexicon Lab");
      flush();
      expect(target.querySelector('[data-testid="nav-hold"]')).toBeNull();

      const geometry = [...target.querySelectorAll('[data-testid="view-tabs"] button')].find(
        (b) => b.textContent?.trim() === "Geometry",
      ) as HTMLButtonElement;
      geometry.click();
      flush();

      const hold = target.querySelector<HTMLElement>('[data-testid="nav-hold"]');
      expect(hold, "the tab switch was performed silently").not.toBeNull();
      expect(hold!.textContent).toContain("a training run in the Lexicon Lab");
      expect(hold!.textContent).toContain("Geometry");
      expect(get(stores.view)).toBe("info");
    } finally {
      done();
    }
  });

  it("its Stay button keeps the tab and its Discard button leaves it", async () => {
    for (const [testid, expected] of [
      ["nav-hold-stay", "info"],
      ["nav-hold-discard", "geometry"],
    ] as const) {
      const { stores, target, flush, done } = await mountApp("#info");
      try {
        register(stores, "lex-train", "a training run in the Lexicon Lab");
        flush();
        (
          [...target.querySelectorAll('[data-testid="view-tabs"] button')].find(
            (b) => b.textContent?.trim() === "Geometry",
          ) as HTMLButtonElement
        ).click();
        flush();
        target.querySelector<HTMLButtonElement>(`[data-testid="${testid}"]`)!.click();
        flush();

        expect(get(stores.view), testid).toBe(expected);
        expect(target.querySelector('[data-testid="nav-hold"]')).toBeNull();
      } finally {
        done();
      }
    }
  });
});

describe("the panels that own destructible work register it", () => {
  // These panels cannot be driven to `busy` in jsdom — a run needs a real `Worker`, and
  // jsdom has none (`typeof Worker === "undefined"`). What is asserted here is the
  // WIRING: that each panel registers on the same flag it disables its button with, and
  // releases in `onDestroy`. The store-level behaviour above is exercised for real.
  const PANELS = [
    ["viz/lex/TrainPanel.svelte", "lex-train"],
    ["viz/geo/TrainPanel.svelte", "geo-train"],
    ["viz/geo/FinetunePanel.svelte", "geo-finetune"],
  ] as const;

  for (const [file, id] of PANELS) {
    it(`${file} registers while busy and releases on destroy`, () => {
      const src = readFileSync(path.join(SRC, file), "utf8");
      expect(src, `${file} does not import the registry`).toContain(
        'from "../../lib/stores"',
      );
      expect(src, `${file} does not register on \`busy\``).toMatch(
        new RegExp(`if \\(busy\\) registerWork\\(WORK_ID,[\\s\\S]*?else releaseWork\\(WORK_ID\\)`),
      );
      expect(src, `${file} does not name itself`).toContain(`"${id}"`);
      // Released on destroy, or a confirmed navigation would latch the registry and hold
      // every later navigation against work that no longer exists.
      expect(src, `${file} does not release in onDestroy`).toMatch(
        /onDestroy\(\(\) => \{[\s\S]*?releaseWork\(WORK_ID\);[\s\S]*?\}\)/,
      );
    });
  }
});
