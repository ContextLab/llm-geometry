import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

const SRC = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "src");

// The Info tab and the in-tab explainers, against the REAL backend.
//
// The point of this file is not "does the text render" — it is that the documentation
// states NUMBERS, and numbers rot. Each drift check reads a fact from the running
// system (the live /api/geo/spec, the real slider bounds, the real controls) and
// asserts the prose still agrees with it. If someone changes the fine-tune ceiling or
// the vocabulary size, this fails instead of the site quietly lying to a reader.

async function openInfo(page: Page) {
  await page.goto("/");
  await page.getByTestId("tab-info").click();
  await expect(page.getByTestId("info-view")).toBeVisible();
}

test.describe("Info tab", () => {
  test("renders every documented section", async ({ page }) => {
    await openInfo(page);
    for (const heading of [
      "Which tab do I want?",
      "Notation",
      "The Architecture Explorer",
      "The Geometry Lab",
      "The Lexicon Lab",
      "The vacancy transform",
      "What's real, and where it runs",
      "Known limits",
      "Source & references",
    ]) {
      await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    }
  });

  test("is reachable by URL, so it can be shared", async ({ page }) => {
    // The point of the whole feature: the author can send a colleague a link that lands
    // on the explanation rather than on a visualization they cannot read.
    await page.goto("/#info");
    await expect(page.getByTestId("info-view")).toBeVisible();
    await expect(page.getByTestId("arch-model-picker")).toHaveCount(0);

    await page.goto("/#geometry");
    await expect(page.getByTestId("geo-view")).toBeVisible();

    // An unknown fragment must not blank the app. Two distinct paths, because a hash
    // change within the same document does NOT re-run initialization:
    //   (a) navigating to one mid-session leaves the current tab alone,
    await page.goto("/#not-a-tab");
    await expect(page.getByTestId("geo-view")).toBeVisible();
    //   (b) loading one cold falls back to the landing tab.
    await page.reload();
    await expect(page.getByTestId("arch-model-picker")).toBeVisible();

    // Switching tabs updates the URL, so what you copy is what you are looking at.
    await page.getByTestId("tab-info").click();
    await expect(page).toHaveURL(/#info$/);
  });

  test("Back and Forward move between tabs, as the shell documents", async ({ page }) => {
    // The behaviour claim, pinned the way this file pins the numeric ones. `stores.ts`
    // says "Back/Forward and a pasted link both work"; it used `replaceState`, so Back
    // skipped every tab the reader had visited and left the site (red-team D F1). The
    // unit suite covers the store in isolation — this is the claim in a real browser,
    // with real history entries.
    await page.goto("/#architecture");
    await page.getByTestId("tab-geometry").click();
    await expect(page).toHaveURL(/#geometry$/);
    await page.getByTestId("tab-info").click();
    await expect(page).toHaveURL(/#info$/);

    await page.goBack();
    await expect(page).toHaveURL(/#geometry$/);
    await expect(page.getByTestId("geo-view")).toBeVisible();

    await page.goForward();
    await expect(page).toHaveURL(/#info$/);
    await expect(page.getByTestId("info-view")).toBeVisible();
  });

  test("a hash the app did not honour is corrected instead of left in the address bar", async ({
    page,
  }) => {
    // Falling back to the landing tab is deliberate; keeping a URL that promises a view
    // the recipient will not get is not (red-team D F8). Both directions of the fix:
    //   (a) an unknown fragment is rewritten to the tab actually rendered,
    await page.goto("/#not-a-tab");
    await expect(page.getByTestId("arch-model-picker")).toBeVisible();
    await expect(page).toHaveURL(/#architecture$/);
    //   (b) and a mis-cased one resolves rather than silently falling back.
    await page.goto("/#Info");
    await expect(page.getByTestId("info-view")).toBeVisible();
    await expect(page).toHaveURL(/#info$/);
  });

  test("the tab strip and the Geometry Lab's controls expose their state to assistive tech", async ({
    page,
  }) => {
    // Issue #7: the active tab was a background gradient and nothing else, and the two
    // controls the Geometry Lab is about were `role="tablist"` around plain buttons.
    // Asserted as ARIA, in a real browser, rather than as a screenshot.
    await page.goto("/#architecture");
    const strip = page.getByTestId("view-tabs");
    await expect(strip).toHaveAttribute("aria-label", /.+/);
    await expect(strip.locator("button[aria-current=page]")).toHaveText("Architecture");
    await page.getByTestId("tab-geometry").click();
    await expect(strip.locator("button[aria-current=page]")).toHaveText("Geometry");

    await expect
      .poll(async () => page.getByTestId("geo-view").getAttribute("data-ready"), {
        timeout: 220_000,
      })
      .toBe("1");

    for (const testid of ["geo-mode", "geo-layer"]) {
      const group = page.getByTestId(testid);
      await expect(group).toHaveAttribute("role", "radiogroup");
      await expect(group).toHaveAttribute("aria-label", /.+/);
      // Exactly one checked option, and every child says whether it is the one.
      await expect(group.locator('[role=radio][aria-checked=true]')).toHaveCount(1);
      const n = await group.locator("button").count();
      await expect(group.locator("[role=radio]")).toHaveCount(n);
    }

    // …and the arrow keys really move it, which is what makes the group one tab stop.
    const mode = page.getByTestId("geo-mode");
    const before = await mode.locator("[role=radio][aria-checked=true]").innerText();
    await mode.locator("[role=radio][aria-checked=true]").focus();
    await page.keyboard.press("ArrowRight");
    await expect(mode.locator("[role=radio][aria-checked=true]")).not.toHaveText(before);
  });

  test("the table of contents moves the reading position, not just the viewport", async ({
    page,
  }) => {
    // WCAG 2.4.3: scrolling without moving focus left a keyboard user's next Tab back at
    // the top of the page, 12,000 px from what they had just asked to read (F7).
    await openInfo(page);
    await page.locator(".toc button", { hasText: "Known limits" }).click();
    await expect(page.locator("#limits")).toBeFocused();
  });

  test("a first-time visitor is pointed at it from the landing tab", async ({ page }) => {
    await page.goto("/");
    const pointer = page.getByTestId("info-pointer");
    await expect(pointer).toBeVisible();
    await pointer.click();
    await expect(page.getByTestId("info-view")).toBeVisible();

    // ...and it retires itself once they have actually read it.
    await page.getByTestId("tab-architecture").click();
    await expect(page.getByTestId("info-pointer")).toHaveCount(0);
    await page.reload();
    await expect(page.getByTestId("info-pointer")).toHaveCount(0);
  });

  test("the cards switch to the tab they describe", async ({ page }) => {
    await openInfo(page);
    await page.getByRole("button", { name: /Open the Geometry Lab/ }).click();
    await expect(page.getByTestId("geo-view")).toBeVisible();

    await page.getByTestId("tab-info").click();
    await page.getByRole("button", { name: /Open the Architecture Explorer/ }).click();
    await expect(page.getByTestId("arch-model-picker")).toBeVisible();
  });

  test("the documented model shape matches what the API actually serves", async ({
    page,
    request,
  }) => {
    const spec = await (await request.get("/api/geo/spec")).json();
    expect(spec.model.d_model).toBe(3);
    expect(spec.model.n_layers).toBe(4);
    expect(spec.model.n_heads).toBe(1);
    expect(spec.model.vocab_size).toBe(1003);

    await openInfo(page);
    const info = page.getByTestId("info-view");
    // The prose states each of these outright; a change to the architecture must not
    // leave the description behind.
    await expect(info).toContainText("d_model = 3");
    await expect(info).toContainText("4 layers, 1 head");
    await expect(info).toContainText(String(spec.model.vocab_size));
  });

  test("documented training limits match the real controls", async ({ page }) => {
    await openInfo(page);
    const info = page.getByTestId("info-view");
    await expect(info).toContainText("up to 500 steps, default 100");
    await expect(info).toContainText("epochs slider runs 1–30 and starts at 12");

    // …and those are the bounds the Geometry Lab's sliders really have.
    await page.getByTestId("tab-geometry").click();
    await expect
      .poll(async () => page.getByTestId("geo-view").getAttribute("data-ready"), {
        timeout: 220_000,
      })
      .toBe("1");

    const epochs = page.getByTestId("geo-train-epochs");
    await expect(epochs).toHaveAttribute("min", "1");
    await expect(epochs).toHaveAttribute("max", "30");
    await expect(epochs).toHaveValue("12");

    const steps = page.locator(".steps input[type=range]");
    await expect(steps).toHaveAttribute("max", "500");
    await expect(steps).toHaveValue("100");
  });

  test("documented decoding constraints match the backend config", async ({ page, request }) => {
    // Sampling is filtered to top-k 50 ∩ top-p 0.9 with a 1.1 repetition penalty. There
    // is no endpoint that reports those, so assert the prose and let the backend's own
    // contract test hold the values — what must not happen is the two disagreeing after
    // a one-sided edit, which the string below makes visible in review.
    await openInfo(page);
    await expect(page.getByTestId("info-view")).toContainText("top-k 50 ∩ top-p 0.9");
    await expect(page.getByTestId("info-view")).toContainText("repetition penalty of 1.1");

    const health = await (await request.get("/api/health")).json();
    expect(health.status).toBe("ok");
  });

  test("every external reference is a well-formed, safely-targeted https link", async ({
    page,
  }) => {
    await openInfo(page);
    const links = page.getByTestId("info-view").locator("a[href^=http]");
    const n = await links.count();
    expect(n).toBeGreaterThan(3);
    for (let i = 0; i < n; i++) {
      const a = links.nth(i);
      expect(await a.getAttribute("href")).toMatch(/^https:\/\//);
      // Opening a new tab without rel=noopener hands the opener to the target page.
      expect(await a.getAttribute("rel")).toContain("noopener");
      await expect(a).toHaveAttribute("target", "_blank");
    }
  });
});

test.describe("in-tab explainers", () => {
  test("the Architecture tab explains the diagram and its controls", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("tab-architecture").click();

    const diagram = page.getByTestId("arch-explain-diagram");
    await expect(diagram).toBeVisible();
    // Collapsed by default: the visualization stays above the fold.
    await expect(diagram).not.toHaveAttribute("open", "");
    await expect(diagram).toContainText("How to read the diagram");

    await diagram.locator("summary").click();
    await expect(diagram).toContainText("tied_to");
    await expect(diagram).toContainText("1.5B parameters");

    const controls = page.getByTestId("arch-explain-controls");
    await controls.locator("summary").click();
    await expect(controls).toContainText("top-k 50 ∩ top-p 0.9");
    // The explainer must route a reader to the reference tab, and it must work.
    await controls.getByRole("button", { name: "Info tab" }).click();
    await expect(page.getByTestId("info-view")).toBeVisible();
  });

  test("the Geometry tab explains the model, the fields, and the controls", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("tab-geometry").click();
    await expect
      .poll(async () => page.getByTestId("geo-view").getAttribute("data-ready"), {
        timeout: 220_000,
      })
      .toBe("1");

    const model = page.getByTestId("geo-explain-model");
    await model.locator("summary").click();
    await expect(model).toContainText("logit lens");
    await expect(model).toContainText("no layer norm");

    const fields = page.getByTestId("geo-explain-fields");
    await fields.locator("summary").click();
    // The tangency subtlety is the one thing a reader must not get wrong: the text has
    // to say that antisymmetrizing does NOT make the aggregate forces tangent, and that
    // the projection's cost is reported rather than hidden.
    await expect(fields).toContainText("does not make the amber arrows tangent");
    await expect(fields).toContainText("largest radial component that projection removed");

    const controls = page.getByTestId("geo-explain-controls");
    await controls.locator("summary").click();
    await expect(controls).toContainText("W_Q and W_K");
    await controls.getByRole("button", { name: "Info tab" }).click();
    await expect(page.getByTestId("info-view")).toBeVisible();
  });
});

test.describe("Lexicon Lab documentation", () => {
  // Feature 006's numbers, pinned the same way: read the fact from the running system,
  // then assert the prose still agrees. This tab has already produced two stale-number
  // defects during development — a parameter count that moved when the defaults changed,
  // and a hand-copied effective-rank increment — so nothing here is left unpinned.

  test("documented budget sizes match the real Dolch lists", async ({ page, request }) => {
    const spec = await (await request.get("/api/lex/spec")).json();
    const sizes = spec.budgets.map((b: { size: number }) => b.size);

    // Measured from the shipped word lists, never quoted. The largest is 314, not the
    // widely-cited 315: `Santa Claus` contains a space and no word tokenizer can match it.
    expect(sizes).toEqual([40, 92, 133, 220, 314]);

    await openInfo(page);
    const info = page.getByTestId("info-view");
    await expect(info).toContainText("40 / 92 / 133 / 220 / 314");
    await expect(info).toContainText("314");
  });

  test("documented coverage figures match a real backend measurement", async ({
    page,
    request,
  }) => {
    // The prose claims the descriptive budget beats the prescribed one at matched |V|:
    // "70.7% against 60.8% at |V| = 314". Recompute both against the shipped corpus.
    const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

    const dolch = await (
      await request.post("/api/lex/coverage", {
        data: { source: "dolch", budget: "full" },
      })
    ).json();
    const freq = await (
      await request.post("/api/lex/coverage", {
        data: { source: "frequency", budget: "full" },
      })
    ).json();

    expect(dolch.size).toBe(freq.size); // the comparison must be at matched |V|
    expect(freq.coverage.token_coverage).toBeGreaterThan(dolch.coverage.token_coverage);

    await openInfo(page);
    const info = page.getByTestId("info-view");
    await expect(info).toContainText(pct(freq.coverage.token_coverage));
    await expect(info).toContainText(pct(dolch.coverage.token_coverage));
    // …and the <unk> rate, which is the measurable form of what a budget cannot say.
    await expect(info).toContainText(pct(dolch.coverage.unk_rate));
    await expect(info).toContainText(
      `${dolch.coverage.whole_lines_in_budget} of ${dolch.coverage.total_lines.toLocaleString("en-US")}`,
    );
  });

  test("the Lexicon Lab is described as browser-only, because it is", async ({ page }) => {
    // It trains in a worker in BOTH modes and never calls the backend. The what's-real
    // table used to say "Both tabs run against real PyTorch", which was false for it.
    await openInfo(page);
    const info = page.getByTestId("info-view");
    await expect(info).toContainText(/never calls the backend|entirely in the browser/i);
    await expect(info).not.toContainText("Both tabs run against real PyTorch");
  });

  test("the corpus is identified exactly, and the tab verifies what it loaded", async ({
    page,
    request,
  }) => {
    const spec = await (await request.get("/api/lex/spec")).json();
    expect(spec.corpus.gutenberg_id).toBe(10607);
    expect(spec.corpus.year).toBe(1916);

    await openInfo(page);
    const info = page.getByTestId("info-view");
    await expect(info).toContainText("The Real Mother Goose");
    await expect(info).toContainText("1916");
    await expect(info).toContainText("10607");
  });
});

test.describe("vacancy transform documentation", () => {
  // Feature 007's numbers, pinned by feature 005's rule: read the fact from the running
  // system — the real transform, the real panel, the real constant in the source — and then
  // assert the sentence still agrees with it. Nothing in the Info tab's vacancy section is a
  // number someone typed and nobody checks again.
  //
  // The source document's own prosody figures are ITS numbers on a corpus we do not have,
  // and are transcribed nowhere; every figure asserted here is measured on Mother Goose.

  const en = (x: number): string => x.toLocaleString("en-US");

  test("the documented counts are what the transform really produces", async ({
    page,
    request,
  }) => {
    // p = 1, seed 0 on the shipped corpus — the configuration the prose quotes.
    const res = await request.post("/api/lex/vacancy", { data: { p: 1, seed: 0 } });
    expect(res.ok(), await res.text()).toBe(true);
    const body = await res.json();
    const s = body.vacancy_stats;

    // Full vacancy means these identities hold; the sentence about 8,125 tokens is only
    // true while they do, so they are asserted rather than assumed.
    expect(s.stemsVacated).toBe(s.stemsTotal);
    expect(s.corpusTypesVacated).toBe(s.corpusTypesEligible);
    expect(body.bijective).toBe(true);

    await openInfo(page);
    const info = page.getByTestId("info-view");
    // Domain scope governs the map; corpus scope is what a reader can see in the text. The
    // prose states both, labelled — an unprefixed "types" is forbidden (contract §10).
    for (const value of [
      s.domainTypesTotal, // 2,233 = corpus types ∪ the FULL Dolch list
      s.corpusTypesTotal, // 2,211 of them are the corpus's own
      s.domainTypesEligible, // 1,940 eligible — also the total size of the swap pools
      s.stemsTotal, // 1,676 distinct stems (the size of the map only under `nonce`)
      s.tokensVacated, // 8,125 rewritten word occurrences at p = 1
      s.tokensTotal, // out of 16,000
    ]) {
      await expect(info, `the prose no longer states ${en(value)}`).toContainText(en(value));
    }

    // The honesty number beside every prosody statistic: what fraction of this corpus's
    // tokens the unverified hand table actually covers.
    await expect(info).toContainText(`${(s.stressFromTableBefore * 100).toFixed(1)}%`);
  });

  test("the documented swap collisions are the ones the engine measures", async ({ page }) => {
    // Contract §5.2a as a number rather than as an adjective. `swap` draws its replacements
    // FROM the domain, so at an intermediate p a vacated type lands on one that has not
    // moved. The prose states the measured triple; this reads it off the running panel, so a
    // change to the map, the pool or the tie rule fails here instead of quietly making the
    // documentation wrong.
    await page.goto("/#lexicon");
    await expect(page.getByTestId("lex-vacancy")).toBeVisible({ timeout: 30_000 });
    await page
      .getByTestId("lex-vacancy-mint")
      .getByRole("radio", { name: "swap", exact: true })
      .click();

    const slider = page.getByTestId("lex-vacancy-p");
    const lost = page.getByTestId("lex-vacancy-lost-slots");
    const measured: string[] = [];
    for (const p of ["0.25", "0.5", "0.75"]) {
      await slider.fill(p);
      await slider.dispatchEvent("input");
      await expect(page.getByTestId("lex-vacancy")).toContainText(`p = ${Number(p).toFixed(2)}`);
      // The refusal is SHOWN, not worked around: no clamped p, no silent fall back to nonce.
      await expect(page.getByTestId("lex-vacancy-refusal")).toBeVisible();
      await expect(page.getByTestId("lex-vacancy-refusal-message")).toContainText("§5.2a");
      measured.push((await lost.innerText()).trim());
    }

    // …and 0 at full vacancy, where swap IS a bijection of the domain and the invariance
    // theorem holds for it exactly as it does for nonce.
    await slider.fill("1");
    await slider.dispatchEvent("input");
    await expect(lost).toHaveText("0");
    await expect(page.getByTestId("lex-vacancy-refusal")).toHaveCount(0);
    await expect(page.getByTestId("lex-vacancy-invariance-verdict")).toHaveText(/identical/i);

    await openInfo(page);
    await expect(page.getByTestId("info-view")).toContainText(measured.join(" / "));
  });

  test("the documented stress-table size is the table the engine actually has", async ({
    page,
  }) => {
    await page.goto("/#lexicon");
    await expect(page.getByTestId("lex-vacancy")).toBeVisible({ timeout: 30_000 });
    const honesty = await page.getByTestId("lex-vacancy-prosody-honesty").innerText();
    const entries = honesty.match(/(\d+) hand-set entries/)?.[1];
    expect(entries, `no entry count in: ${honesty}`).toBeDefined();

    await openInfo(page);
    await expect(page.getByTestId("info-view")).toContainText(`hand table of ${entries} entries`);
  });

  test("the documented static-mode limits are the constants the static client enforces", async ({
    page,
  }) => {
    // These two live in the static client rather than behind an endpoint, so the fact is
    // read from the source that enforces it. A stated ± that was never measured is a
    // fabricated error bar — which is exactly why the number must not be retyped.
    const src = readFileSync(path.join(SRC, "lib", "staticClient", "arch.ts"), "utf8");
    const uncertainty = src.match(/VACANCY_Q8_UNCERTAINTY_NATS = ([\d.]+)/)?.[1];
    const floor = src.match(/VACANCY_MIN_POOLED_PRESERVED = (\d+)/)?.[1];
    expect(uncertainty, "VACANCY_Q8_UNCERTAINTY_NATS is gone or renamed").toBeDefined();
    expect(floor, "VACANCY_MIN_POOLED_PRESERVED is gone or renamed").toBeDefined();

    await openInfo(page);
    const info = page.getByTestId("info-view");
    await expect(info).toContainText(`±${uncertainty} nats`);
    await expect(info).toContainText(`${floor} preserved tokens`);
  });

  test("the caveats that make the numbers readable are all present", async ({ page }) => {
    // Each of these is a claim the instrument would be dishonest without, so each is
    // asserted rather than left to survive an edit by luck.
    await openInfo(page);
    const info = page.getByTestId("info-view");
    // The decomposition, and that its second term is a BOUND rather than a measurement.
    await expect(info).toContainText("the cost of wrong content");
    await expect(info).toContainText("the cost of unknown form");
    await expect(info).toContainText(/upper bound/i);
    // What the static build refuses, by name.
    await expect(info).toContainText("nonce − swap");
    await expect(info).toContainText(/Per-passage rows: refused/i);
    await expect(info).toContainText(/dtype without a measured bound: refused/i);
    // The coverage gap that belongs in the documentation rather than in a commit message.
    // `\s+` rather than a space: a regex matcher sees the raw textContent, newlines and
    // source indentation included, so a rewrapped paragraph must not fail this.
    await expect(info).toContainText(/CI only ever\s+exercises the WASM rung/i);
    // The exact zero, framed as the finding rather than as a missing curve.
    await expect(info).toContainText(/exact zero is the result/i);
  });
});
