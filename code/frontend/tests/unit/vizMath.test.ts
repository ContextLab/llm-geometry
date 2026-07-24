// Unit tests for the fix-A pure helpers (redteam-vector F3, redteam-sankey S1/S7)
// and the ModelSelector invalid-model UX (redteam-vector F5).
import { describe, expect, it } from "vitest";
import { flushSync, mount, unmount } from "svelte";
import { get } from "svelte/store";

import { robustMax, capLinkWidth, plural, LINK_WIDTH_CAP } from "../../src/lib/vizMath";
import { modelError } from "../../src/lib/stores";
import ModelSelector from "../../src/controls/ModelSelector.svelte";

describe("robustMax (vector F3 — high-temperature washout)", () => {
  it("ignores a single outlier so the rest of the field keeps contrast", () => {
    // 1151 near-uniform faint arrows + 1 bright outlier (the temp-2 repro).
    const probs = [...Array(1151).fill(0.001), 0.9];
    const norm = robustMax(probs);
    expect(norm).toBeLessThan(0.01); // ~95th percentile, nowhere near the 0.9 outlier
    // With max-normalisation the typical arrow sat at rel ≈ 0.0011 (invisible);
    // with robust normalisation it clamps to full strength.
    expect(Math.min(1, 0.001 / norm)).toBeGreaterThan(0.9);
  });

  it("equals the max for a uniform field and stays >= the 1e-6 floor", () => {
    expect(robustMax([0.5, 0.5, 0.5])).toBeCloseTo(0.5, 12);
    expect(robustMax([])).toBe(1e-6);
    expect(robustMax([0, 0, 0])).toBe(1e-6);
  });
});

describe("capLinkWidth (sankey S1 — giant-ribbon blob)", () => {
  it("caps the 102px low-diversity ribbon at the absolute cap", () => {
    // temp 0, seqlen 2, particles 1000 → 2 rows → rowH ≈ 256; dominant link = maxVal.
    expect(capLinkWidth(1000, 1000, 256)).toBe(LINK_WIDTH_CAP);
  });

  it("stays proportional in the normal regime and never drops below 1px", () => {
    expect(capLinkWidth(50, 100, 25)).toBeCloseTo(10, 12); // 0.5 * (25 * 0.8)
    expect(capLinkWidth(1, 100000, 25)).toBe(1);
  });
});

describe("plural (sankey S7 — caption pluralisation)", () => {
  it("says '1 transition' and '2 transitions'", () => {
    expect(plural(1, "transition")).toBe("1 transition");
    expect(plural(2, "transition")).toBe("2 transitions");
    expect(plural(1, "token row")).toBe("1 token row");
  });
});

describe("ModelSelector invalid-model UX (vector F5)", () => {
  it("shows a concise first line, keeps the raw detail in the title, and flags modelError", async () => {
    modelError.set("");
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(ModelSelector, { target });
    flushSync();
    // jsdom/Node fetch rejects the relative /api URL → resolve fails like an unreachable
    // backend; the component must surface a short plain-language message, not the raw error.
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    const msg = target.querySelector<HTMLElement>('[data-testid="model-message"]');
    const badge = target.querySelector<HTMLElement>('[data-testid="model-status"]');
    expect(badge?.textContent).toBe("error");
    expect(msg).not.toBeNull();
    expect(msg!.textContent!.length).toBeLessThan(120); // concise, single line
    expect(msg!.textContent).toMatch(/could not|not found/i); // plain language (e2e regex compatible)
    expect(msg!.textContent).not.toMatch(/Traceback|hf auth login/); // no raw dump in the DOM text
    expect(msg!.getAttribute("title")).toBeTruthy(); // raw detail lives behind the tooltip
    // Views read this to dim their previous-model content as stale.
    expect(get(modelError)).not.toBe("");
    unmount(app);
    target.remove();
    modelError.set("");
  });
});
