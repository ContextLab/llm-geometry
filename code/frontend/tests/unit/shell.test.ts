// The app shell: the hash router in `lib/stores.ts`, the tab strip's accessibility, and
// the segmented controls' accessibility. jsdom, real history entries, real events — the
// point of these tests is that the SHELL BEHAVIOURS the documentation promises are
// pinned the same way its numbers are (red-team D, F1/F6/F8/F14).
//
// `tests/e2e/docs.spec.ts` covers the same contract in a real browser; this file is what
// makes a regression cheap to find, because it needs neither a backend nor a build.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushSync, mount, unmount, type ComponentProps } from "svelte";
import { get } from "svelte/store";

import SegmentedControl from "../../src/lib/SegmentedControl.svelte";
import ArchModelPicker from "../../src/viz/arch/ArchModelPicker.svelte";
// Statically imported: a dynamic import after `vi.resetModules()` would compile against
// a second Svelte instance and break `mount`.
import InfoTab from "../../src/viz/info/InfoTab.svelte";

const SRC = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "src");

/**
 * The accessible name of an element, from the three sources that can supply one here:
 * `aria-label`, `aria-labelledby`, and an associated or wrapping `<label>`. Not a
 * general implementation of the accname spec — enough of it to answer "would a screen
 * reader say anything at all about this control?", which is the question F9 asked.
 */
function accessibleName(el: Element): string {
  const label = el.getAttribute("aria-label")?.trim();
  if (label) return label;
  const ids = el.getAttribute("aria-labelledby")?.split(/\s+/) ?? [];
  const referenced = ids
    .map((id) => el.ownerDocument.getElementById(id)?.textContent?.trim() ?? "")
    .filter(Boolean)
    .join(" ");
  if (referenced) return referenced;
  const id = el.getAttribute("id");
  const forLabel = id
    ? el.ownerDocument.querySelector(`label[for="${CSS.escape(id)}"]`)?.textContent?.trim()
    : "";
  if (forLabel) return forLabel;
  return el.closest("label")?.textContent?.trim() ?? "";
}

/** Load a fresh copy of the store module with the browser sitting at `url`. */
async function loadShell(url: string) {
  window.history.replaceState(null, "", url);
  vi.resetModules();
  return await import("../../src/lib/stores");
}

/** jsdom runs history traversal as a task, so the assertion has to wait for it. */
async function until(pred: () => boolean, label: string, ms = 2000): Promise<void> {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (pred()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`timed out waiting for ${label} (hash=${window.location.hash})`);
}

describe("hash router", () => {
  it("moves between tabs with Back and Forward", async () => {
    // The docstring in stores.ts promises "Back/Forward and a pasted link both work".
    // With replaceState only the pasted-link half was true: Back skipped every tab the
    // reader had visited and left the site entirely.
    const { view } = await loadShell("#architecture");

    view.set("geometry");
    expect(window.location.hash).toBe("#geometry");
    view.set("lexicon");
    expect(window.location.hash).toBe("#lexicon");

    window.history.back();
    await until(() => get(view) === "geometry", "Back to Geometry");
    expect(window.location.hash).toBe("#geometry");

    window.history.back();
    await until(() => get(view) === "architecture", "Back to Architecture");

    window.history.forward();
    await until(() => get(view) === "geometry", "Forward to Geometry");
  });

  it("does not push an entry for re-selecting the tab already showing", async () => {
    // The design constraint the old comment stated, kept: one entry per NAVIGATION, not
    // one per click. Clicking the active tab is not a navigation.
    const { view } = await loadShell("#architecture");
    view.set("geometry");
    view.set("geometry");
    view.set("geometry");

    window.history.back();
    await until(() => get(view) === "architecture", "a single Back to leave Geometry");
  });

  it("canonicalizes a mis-cased hash at load instead of silently ignoring it", async () => {
    // `#Info` — a mail client capitalised the fragment — used to render Architecture
    // with no explanation, and the address bar kept advertising Info.
    const { view } = await loadShell("#Info");
    expect(get(view)).toBe("info");
    expect(window.location.hash).toBe("#info");
  });

  it("corrects an unknown hash at load rather than leaving a lying URL", async () => {
    // Falling back to the landing tab is deliberate (docs.spec.ts pins it). Keeping
    // `#not-a-tab` in a URL the reader may copy is not: it promises a view nobody gets.
    const { view } = await loadShell("#not-a-tab");
    expect(get(view)).toBe("architecture");
    expect(window.location.hash).toBe("#architecture");
  });

  it("leaves a hash-free URL alone until the reader navigates", async () => {
    const { view } = await loadShell("/");
    expect(get(view)).toBe("architecture");
    expect(window.location.hash).toBe("");
  });

  it("leaves the current tab alone when an unknown hash arrives mid-session", async () => {
    // Pinned by docs.spec.ts: a stray fragment must not yank a reader out of the tab
    // they are reading. Only a COLD load falls back.
    const { view } = await loadShell("#geometry");
    window.location.hash = "#not-a-tab";
    await until(() => window.location.hash === "#not-a-tab", "the hash to change");
    await new Promise((r) => setTimeout(r, 20));
    expect(get(view)).toBe("geometry");
  });

  it("follows a valid hash typed mid-session", async () => {
    const { view } = await loadShell("#geometry");
    window.location.hash = "#info";
    await until(() => get(view) === "info", "the store to follow the hash");
  });
});

describe("the tab strip", () => {
  // Mounted at #info because the Info tab is static prose: no backend, no WebGL. The
  // shell is the same on every tab.
  async function mountApp() {
    await loadShell("#info");
    // Re-imported AFTER vi.resetModules(): the statically imported `mount` belongs to
    // the previous module graph, and mounting a component compiled against a second
    // Svelte instance orphans its `$effect`s.
    const svelte = await import("svelte");
    const App = (await import("../../src/App.svelte")).default;
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = svelte.mount(App, { target });
    svelte.flushSync();
    return {
      target,
      flush: svelte.flushSync,
      done: () => (svelte.unmount(app), target.remove()),
    };
  }

  it("names itself and marks the showing view with aria-current", async () => {
    // Before: `<nav class="tabs">` with four plain buttons whose only "selected" signal
    // was a background gradient. A screen-reader user heard four ordinary buttons and
    // had no way to know which view was showing (red-team F6, issue #7).
    const { target, done } = await mountApp();
    const nav = target.querySelector<HTMLElement>('[data-testid="view-tabs"]')!;
    expect(nav.getAttribute("aria-label")).toBeTruthy();

    const buttons = [...nav.querySelectorAll("button")];
    expect(buttons.map((b) => b.textContent?.trim())).toEqual([
      "Architecture",
      "Geometry",
      "Lexicon",
      "Info",
    ]);
    const current = buttons.filter((b) => b.getAttribute("aria-current") === "page");
    expect(current.map((b) => b.textContent?.trim())).toEqual(["Info"]);
    // Exactly one, and it is not merely a class.
    for (const b of buttons) {
      if (b.getAttribute("aria-current") !== "page") {
        expect(b.getAttribute("aria-current")).toBeNull();
      }
    }
    done();
  });

  it("moves aria-current with the selection", async () => {
    const { target, flush, done } = await mountApp();
    const nav = target.querySelector<HTMLElement>('[data-testid="view-tabs"]')!;
    const geometry = [...nav.querySelectorAll("button")].find(
      (b) => b.textContent?.trim() === "Geometry",
    )!;
    geometry.click();
    flush();
    expect(geometry.getAttribute("aria-current")).toBe("page");
    expect(
      [...nav.querySelectorAll("button")].filter(
        (b) => b.getAttribute("aria-current") === "page",
      ).length,
    ).toBe(1);
    done();
  });
});

describe("SegmentedControl", () => {
  // The Geometry Lab's `field` and `layer` controls were `role="tablist"` wrapped around
  // plain buttons: no `role`, no `aria-selected`, so the only signal of which field or
  // which layer was active was a background colour (red-team F5, issue #7). They are not
  // tabs — they own no panels — they are a one-of-N setting, so they are a radio group.
  const OPTIONS = [
    { value: "full", label: "full" },
    { value: "0", label: "0" },
    { value: "1", label: "1" },
  ];

  function render(props: ComponentProps<typeof SegmentedControl>) {
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(SegmentedControl, { target, props });
    flushSync();
    const group = target.querySelector<HTMLElement>(".seg")!;
    const buttons = () => [...group.querySelectorAll("button")];
    return { target, group, buttons, done: () => (unmount(app), target.remove()) };
  }

  it("exposes the selected option to assistive technology", () => {
    const picks: string[] = [];
    const { group, buttons, done } = render({
      label: "layer",
      options: OPTIONS,
      value: "0",
      onSelect: (v: string) => picks.push(v),
      testid: "geo-layer",
    });

    expect(group.getAttribute("role")).toBe("radiogroup");
    expect(group.getAttribute("aria-label")).toBe("layer");
    expect(group.dataset.testid).toBe("geo-layer");
    expect(buttons().map((b) => b.getAttribute("role"))).toEqual(["radio", "radio", "radio"]);
    expect(buttons().map((b) => b.getAttribute("aria-checked"))).toEqual([
      "false",
      "true",
      "false",
    ]);

    buttons()[2].click();
    expect(picks).toEqual(["1"]);
    done();
  });

  it("is one tab stop, with the selected option carrying it", () => {
    const { buttons, done } = render({
      label: "layer",
      options: OPTIONS,
      value: "1",
      onSelect: () => {},
    });
    expect(buttons().map((b) => b.tabIndex)).toEqual([-1, -1, 0]);
    done();
  });

  it("moves the selection with the arrow keys, Home and End", () => {
    const picks: string[] = [];
    const { group, done } = render({
      label: "field",
      options: OPTIONS,
      value: "0",
      onSelect: (v: string) => picks.push(v),
    });
    const key = (k: string) =>
      group.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true }));

    key("ArrowRight");
    expect(picks).toEqual(["1"]);
    key("ArrowLeft");
    expect(picks).toEqual(["1", "full"]);
    key("End");
    expect(picks).toEqual(["1", "full", "1"]);
    key("Home");
    expect(picks).toEqual(["1", "full", "1", "full"]);
    done();
  });

  it("wraps around and skips disabled options", () => {
    // The Geometry Lab really disables `full` while the force field is showing, so a
    // keyboard user must not be able to land on it.
    const picks: string[] = [];
    const { group, buttons, done } = render({
      label: "layer",
      options: [{ ...OPTIONS[0], disabled: true }, OPTIONS[1], OPTIONS[2]],
      value: "0",
      onSelect: (v: string) => picks.push(v),
    });
    expect(buttons()[0].disabled).toBe(true);
    group.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
    expect(picks).toEqual(["1"]); // wrapped past the disabled "full"
    done();
  });

  it("leaves keys it does not own to the page", () => {
    const { group, done } = render({
      label: "field",
      options: OPTIONS,
      value: "0",
      onSelect: () => {},
    });
    const e = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    group.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(false);
    done();
  });
});

describe("the Info tab's table of contents", () => {
  it("moves focus to the section it jumps to, not just the viewport", () => {
    // Before: `scrollIntoView` alone. Focus stayed on the TOC button, so the next Tab
    // took a keyboard user back to the top of the page and the TOC was mouse-only
    // (red-team D F7, WCAG 2.4.3).
    //
    // jsdom has no layout, so it does not implement scrollIntoView. Supplying it is a
    // gap in the environment, not a stand-in for our code: the assertion below is about
    // focus, and the call is recorded only to prove the scroll still happens too.
    const scrolled: string[] = [];
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      configurable: true,
      value(this: Element) {
        scrolled.push(this.id);
      },
    });

    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(InfoTab, { target });
    flushSync();

    const toc = target.querySelector<HTMLElement>(".toc")!;
    const limits = [...toc.querySelectorAll("button")].find(
      (b) => b.textContent?.trim() === "Known limits",
    )!;
    limits.click();
    flushSync();

    const heading = target.querySelector<HTMLElement>("#limits")!;
    expect(heading.tabIndex).toBe(-1); // focusable, but not a tab stop of its own
    expect(document.activeElement).toBe(heading);
    expect(scrolled).toEqual(["limits"]);

    unmount(app);
    target.remove();
  });
});

describe("the Geometry Lab's primary controls", () => {
  // GeometryLab.svelte cannot be mounted here — it boots WebGL and a real model — so the
  // wiring is asserted against its source and the behaviour against the component it now
  // renders. Reading the source to pin a fact is the pattern docs.spec.ts already uses.
  const src = () => readFileSync(path.join(SRC, "viz", "geo", "GeometryLab.svelte"), "utf8");

  it("are radio groups rather than tablists wrapped around plain buttons", () => {
    expect(src()).not.toMatch(/data-testid="geo-mode"[^>]*role="tablist"/);
    expect(src()).not.toMatch(/data-testid="geo-layer"[^>]*role="tablist"/);
  });

  it("render through SegmentedControl, so the ARIA above applies to them", () => {
    for (const testid of ["geo-mode", "geo-layer"]) {
      const use = new RegExp(`<SegmentedControl[^>]*?testid="${testid}"`, "s");
      expect(src(), `${testid} is not a SegmentedControl`).toMatch(use);
    }
  });
});

describe("the Architecture model picker", () => {
  it("gives its select an accessible name", () => {
    // The only unnamed interactive control on any of the four tabs: the visible "Model"
    // text was a sibling <span>, so a screen reader announced "combo box, Qwen2.5 0.5B
    // Instruct" with no indication of what was being chosen (red-team F9).
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(ArchModelPicker, { target, props: {} });
    flushSync();
    const select = target.querySelector<HTMLSelectElement>(
      '[data-testid="arch-model-select"]',
    )!;
    expect(select).not.toBeNull();
    expect(accessibleName(select)).toBe("Model");
    unmount(app);
    target.remove();
  });
});

beforeEach(() => {
  document.body.innerHTML = "";
});
