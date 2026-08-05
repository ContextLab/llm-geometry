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

describe("leaving the document is held too", () => {
  /**
   * `popstate` fires only for a traversal that stays in THIS document, so the in-app
   * prompt — which is the right answer for a tab click and for an in-document Back —
   * covered neither a reload nor a close nor the Back that steps off a cold deep link.
   * Measured on the running app before this listener existed: cold-load `#lexicon`, train,
   * press Back → `url = about:blank`, run gone, `native dialogs fired during reload: []`.
   *
   * The browser will not wait for a component to render, so this one navigation has to use
   * the native dialog. Here that is `beforeunload` being cancelled, which is exactly the
   * signal a browser acts on.
   */
  function fireBeforeUnload(): Event {
    const e = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(e);
    return e;
  }

  it("asks the browser to confirm a reload or close while work is registered", async () => {
    const shell = await loadShell("#lexicon");
    expect(fireBeforeUnload().defaultPrevented, "an idle app must never interrupt a reload").toBe(
      false,
    );

    register(shell, "lex-train", "a training run in the Lexicon Lab");
    expect(fireBeforeUnload().defaultPrevented, "the run was discarded silently").toBe(true);

    shell.releaseWork("lex-train");
    expect(fireBeforeUnload().defaultPrevented).toBe(false);
  });
});

describe("a panel's registration survives being driven for real", () => {
  /**
   * The one panel whose work CAN be driven end to end here, driven end to end here.
   *
   * `trainInWorker` runs the same job function inline where `Worker` is undefined —
   * "one code path, no mock", `lib/lexEngine/index.ts` — and Node is such a place, so
   * clicking Run in this test really trains the Lexicon Lab's demonstration twice. That
   * makes the whole chain observable: real click → real `demoBusy` → the registry → a
   * real tab switch that is held and names the run.
   *
   * It exists because the source-text checks below cannot see the difference between
   *
   *     $effect(() => { if (demoBusy) registerWork(…); else releaseWork(…); });
   *
   * and the same three lines in a bare block. The block is not reactive: it runs once at
   * init with `demoBusy === false`, releases an id nobody registered, and never fires
   * again — so the two training runs are never registered and a tab click destroys them
   * in silence, which is the regression this registration was added to fix. Both forms
   * contain the text the regex matches; only one of them registers anything.
   */
  const DEMO_TEXT = [
    "The little children ran through the garden and picked the flowers.",
    "The gardener planted seeds and watered them every morning.",
    "Singing birds landed on the branches while the kittens were sleeping.",
    "The farmer wanted apples, pears and berries from the orchard.",
    "Mother baked bread, cakes and puddings for the hungry travellers.",
  ].join("\n");

  async function mountVacancyPanel(url: string) {
    const stores = await loadShell(url);
    // Everything the panel touches has to be re-imported after `vi.resetModules()`, or the
    // component registers into a DIFFERENT copy of the store than the one asserted on.
    const svelte = await import("svelte");
    const VacancyPanel = (await import("../../src/viz/lex/VacancyPanel.svelte")).default;
    const { LexVocab, tokenize } = await import("../../src/lib/lexEngine");
    const { buildVacancyMap, vacancyDomain, vacancyParams, vacateText } = await import(
      "../../src/lib/lexEngine/vacancy"
    );

    const params = vacancyParams({ p: 0.5, seed: 5, mint: "nonce" });
    const tokens = tokenize(DEMO_TEXT);
    const map = buildVacancyMap(vacancyDomain(tokens), vacancyParams({ seed: 5, mint: "nonce" }));
    const words = [...new Set(tokens)].sort();
    const vocab = new LexVocab(words, "frequency", "full");

    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = svelte.mount(VacancyPanel, {
      target,
      props: {
        corpusText: DEMO_TEXT,
        vacatedText: vacateText(DEMO_TEXT, map, params),
        map,
        params,
        baseVocab: vocab,
        vocab,
        condition: "consistent",
        revealAfter: 1,
        mint: "nonce",
        refusal: "",
        onP: () => {},
        onSeed: () => {},
        onCondition: () => {},
        onRevealAfter: () => {},
        onProsody: () => {},
        onMint: () => {},
      },
    });
    svelte.flushSync();
    const run = target.querySelector<HTMLButtonElement>('[data-testid="lex-vacancy-demo-run"]');
    if (run === null) throw new Error("the demonstration's Run button did not render");
    return {
      stores,
      target,
      run,
      flush: svelte.flushSync,
      done: () => (svelte.unmount(app), target.remove()),
    };
  }

  /** The ids the registry is holding, through the same module instance the panel used. */
  const heldIds = (stores: Awaited<ReturnType<typeof loadShell>>): string[] =>
    get(stores.pendingWork).map((w) => w.id);

  it("holds a tab switch while the vacancy demonstration is really training", async () => {
    const { stores, run, flush, done } = await mountVacancyPanel("#lexicon");
    try {
      expect(heldIds(stores), "an idle panel must register nothing").toEqual([]);

      // Clicking really starts the first of the two runs; `runDemo` sets `demoBusy`
      // before its first `await`, so the flag is up when control comes back here.
      run.click();
      flush();
      expect(heldIds(stores), "the demonstration's training runs were never registered").toEqual([
        "lex-vacancy-demo",
      ]);

      // …and that registration is the thing that saves the run: the tab switch this very
      // panel invites (it offers two "Architecture Explorer" buttons) is held, not made.
      stores.view.set("architecture");
      expect(get(stores.view)).toBe("lexicon");
      expect(get(stores.pendingNavigation)?.work.map((w) => w.label)).toEqual([
        "the two training runs in the Lexicon Lab's transform demo",
      ]);
      stores.cancelNavigation();

      // Both runs finish, and the registry empties itself again.
      await until(() => heldIds(stores).length === 0, "the demonstration to release its work");
      flush();
      stores.view.set("architecture");
      expect(get(stores.view), "a finished demonstration must not keep holding").toBe(
        "architecture",
      );
    } finally {
      done();
    }
  });

  it("releases the work when the tab is left, so a later navigation is not held", async () => {
    const { stores, run, flush, done } = await mountVacancyPanel("#lexicon");
    run.click();
    flush();
    expect(heldIds(stores)).toEqual(["lex-vacancy-demo"]);

    // What a CONFIRMED navigation does: the tab unmounts mid-run. If the panel did not
    // release here the registry would latch, and every later navigation would be held
    // against work that no longer exists.
    done();
    flush();
    expect(heldIds(stores), "the registry latched after the panel was destroyed").toEqual([]);
  });
});

describe("the panels that own destructible work register it", () => {
  // The three panels here cannot be driven to `busy` in jsdom — `lex/TrainPanel` needs a
  // real `Worker` and the two geo panels need the backend — so what is asserted for them
  // is the WIRING: that each registers on the same flag it disables its button with, and
  // releases in `onDestroy`. That is a source-text check and it cannot see a semantic
  // change; `viz/lex/VacancyPanel.svelte`, the one panel whose work runs inline in Node,
  // is driven for real in the block above, and the store-level behaviour further up is
  // exercised for real too.
  //
  // `[file, work id, the panel's own busy flag]`. The registry only protects work that
  // OPTS IN, so the list is the whole guarantee: with three entries the Lexicon Lab's
  // vacancy demonstration — TWO real training runs, in a file that itself offers two
  // "Architecture Explorer" buttons — was still destroyed in silence by a tab click.
  const PANELS = [
    ["viz/lex/TrainPanel.svelte", "lex-train", "busy"],
    ["viz/lex/VacancyPanel.svelte", "lex-vacancy-demo", "demoBusy"],
    ["viz/geo/TrainPanel.svelte", "geo-train", "busy"],
    ["viz/geo/FinetunePanel.svelte", "geo-finetune", "busy"],
  ] as const;

  it("covers every panel that holds abortable work of its own", () => {
    // A panel that aborts something in `onDestroy` is a panel with work to lose. If this
    // list grows, the one above has to grow with it or the new panel is the next silent loss.
    const owners = PANELS.map(([file]) => file);
    expect(owners).toContain("viz/lex/VacancyPanel.svelte");
    expect(new Set(PANELS.map(([, id]) => id)).size).toBe(PANELS.length);
  });

  for (const [file, id, flag] of PANELS) {
    it(`${file} registers while ${flag} and releases on destroy`, () => {
      const src = readFileSync(path.join(SRC, file), "utf8");
      expect(src, `${file} does not import the registry`).toContain(
        'from "../../lib/stores"',
      );
      // `$effect(` is part of the pattern, not decoration around it. The same three
      // statements in a bare `{ … }` block run ONCE, at init, with the flag still false —
      // so the panel releases an id nobody registered and never registers again. Both
      // forms contain `if (busy) registerWork(…) else releaseWork(…)`, which is why the
      // wrapper has to be matched too; VacancyPanel's is additionally driven for real
      // above, and this is the closest the other three can be pinned without a Worker.
      expect(src, `${file} does not register REACTIVELY on \`${flag}\``).toMatch(
        new RegExp(
          `\\$effect\\(\\(\\) => \\{\\s*if \\(${flag}\\) registerWork\\(WORK_ID,` +
            `[\\s\\S]*?else releaseWork\\(WORK_ID\\);\\s*\\}\\);`,
        ),
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
