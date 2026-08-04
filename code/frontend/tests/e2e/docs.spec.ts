import { expect, test, type Page } from "@playwright/test";

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
