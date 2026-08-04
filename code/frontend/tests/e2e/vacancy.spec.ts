import { expect, test, type Page } from "@playwright/test";

/**
 * The vacancy panel of the Lexicon Lab (feature 007, `ui.md` §1), driven as a visitor
 * drives it.
 *
 * These tests exist for the claims this panel would be worthless without, and every one
 * of them is checked against the LIVE computation rather than against a string the panel
 * could print unconditionally:
 *
 *   * the transform really acts on the corpus, and moving `p` really changes it (FR-710);
 *   * nesting and stability are VISIBLE — a cell that turns minted never reverts, and the
 *     minted string is the same string in every later column (FR-711);
 *   * the instant invariance check reports identical id streams under `consistent` AND a
 *     real difference under `inconsistent` (FR-714 — a hard-coded tick would pass the
 *     first assertion and fail the second, which is the point of testing both);
 *   * the on-demand demonstration trains twice and reports max |Δloss| = exactly 0;
 *   * the tab still trains on the vacated corpus, at any budget (FR-713/FR-716).
 *
 * Every test that exercises a run watches for `pageerror` and console errors: a duplicate
 * -key crash once shipped to the live site because nothing here was watching.
 */

const LEX = "#lexicon";

/** Fail on any client-side exception or console error raised during the test. */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  return errors;
}

/** Move the `p` slider and wait for the synchronous re-derivation to land. */
async function setP(page: Page, value: string): Promise<void> {
  const slider = page.getByTestId("lex-vacancy-p");
  await slider.fill(value);
  await slider.dispatchEvent("input");
  await expect(page.getByTestId("lex-vacancy")).toContainText(`p = ${Number(value).toFixed(2)}`);
}

test.beforeEach(async ({ page }) => {
  await page.goto(`/${LEX}`);
  await expect(page.getByTestId("lex-view")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("lex-corpus-error")).toHaveCount(0);
  await expect(page.getByTestId("lex-vacancy")).toBeVisible({ timeout: 30_000 });
});

test("the panel renders with its controls, corpus view, ribbon and statistics", async ({
  page,
}) => {
  const errors = watchErrors(page);
  for (const id of [
    "lex-vacancy-p",
    "lex-vacancy-seed",
    "lex-vacancy-condition",
    "lex-vacancy-prosody",
    "lex-vacancy-mint",
    "lex-vacancy-corpus",
    "lex-vacancy-legend",
    "lex-vacancy-ribbon",
    "lex-vacancy-stats",
    "lex-vacancy-invariance",
    "lex-vacancy-framing",
  ]) {
    await expect(page.getByTestId(id), `missing ${id}`).toBeVisible();
  }

  // The corpus view is the real corpus, not a placeholder.
  const corpus = await page.getByTestId("lex-vacancy-corpus").innerText();
  expect(corpus.trim().length).toBeGreaterThan(200);

  // §1.4: corpus scope, never domain scope. `domainTypes*` must never reach a reader.
  const stats = page.getByTestId("lex-vacancy-stats");
  await expect(stats).toContainText("types vacated");
  await expect(stats).not.toContainText(/domainTypes/i);
  await expect(page.getByTestId("lex-vacancy-bijective")).toContainText("injective");

  // §1.4 / SC-708: no prosody number without the three-way split and the caveat beside it.
  await expect(page.getByTestId("lex-vacancy-prosody-stats")).toContainText("mean anapest");
  const split = page.getByTestId("lex-vacancy-stress-split");
  await expect(split).toContainText("hand table");
  await expect(split).toContainText("minted");
  await expect(split).toContainText("spelling rule");
  await expect(page.getByTestId("lex-vacancy-prosody-honesty")).toContainText(
    /seeded by rule and never checked/i,
  );

  // §1.6: the null is framed as the finding, with the pretrained arm named.
  await expect(page.getByTestId("lex-vacancy-framing")).toContainText(/exact zero/i);
  await expect(page.getByTestId("lex-vacancy-framing")).toContainText(/Architecture Explorer/);

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("moving p rewrites the corpus, and the statistics move with it", async ({ page }) => {
  const errors = watchErrors(page);
  const corpus = page.getByTestId("lex-vacancy-corpus");
  const tokens = page.getByTestId("lex-vacancy-tokens");
  const types = page.getByTestId("lex-vacancy-types");

  // At p = 0 nothing is vacated — that is the definition, not a claim about rendering.
  await expect(tokens).toContainText(/^0\b/);
  const at0 = await corpus.innerText();

  await setP(page, "0.5");
  const at50 = await corpus.innerText();
  expect(at50).not.toBe(at0);
  const tokens50 = await tokens.innerText();
  const types50 = await types.innerText();
  expect(Number(tokens50.split("/")[0].replace(/[^\d]/g, ""))).toBeGreaterThan(0);

  await setP(page, "1");
  const at100 = await corpus.innerText();
  expect(at100).not.toBe(at50);
  // Monotone: more of the corpus is vacated at p = 1 than at p = 0.5 (nesting, in counts).
  const num = (s: string) => Number(s.split("/")[0].replace(/[^\d]/g, ""));
  expect(num(await tokens.innerText())).toBeGreaterThan(num(tokens50));
  expect(num(await types.innerText())).toBeGreaterThan(num(types50));

  // At full vacancy every eligible type has moved: the two halves of the ratio agree.
  const full = await types.innerText();
  const [vacated, eligible] = full.split("/").map((s) => Number(s.replace(/[^\d]/g, "")));
  expect(vacated).toBe(eligible);

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("the ribbon shows nesting and stability, cell by cell (FR-711)", async ({ page }) => {
  const errors = watchErrors(page);
  const rows = page.getByTestId("lex-vacancy-ribbon").locator("tbody tr");
  await expect(rows.first()).toBeVisible();
  const count = await rows.count();
  expect(count).toBeGreaterThanOrEqual(4);

  let sawFlip = false;
  for (let r = 0; r < count; r++) {
    const cells = rows.nth(r).locator("td.cell");
    expect(await cells.count()).toBe(5);
    const classes: string[] = [];
    const forms: string[] = [];
    for (let c = 0; c < 5; c++) {
      classes.push((await cells.nth(c).getAttribute("class")) ?? "");
      forms.push((await cells.nth(c).innerText()).trim());
    }
    const minted = classes.map((c) => c.includes("minted"));

    // NESTING — once minted, never reverts as p grows.
    let flipped = false;
    for (let c = 0; c < 5; c++) {
      if (minted[c]) flipped = true;
      else expect(flipped, `row ${r} reverted at column ${c}: ${forms.join(" | ")}`).toBe(false);
    }
    // STABILITY — every minted cell in a row carries the SAME string.
    const mintedForms = forms.filter((_, c) => minted[c]);
    if (mintedForms.length > 0) {
      expect(new Set(mintedForms).size, `row ${r} minted forms: ${mintedForms.join(" | ")}`).toBe(1);
      // ...and it is not the English stem it replaced.
      expect(mintedForms[0]).not.toBe(forms[0]);
    }
    if (mintedForms.length > 0 && mintedForms.length < 5) sawFlip = true;
  }
  // The rows span the u range, so at least one flips somewhere in the middle rather than
  // every row being all-English or all-minted.
  expect(sawFlip).toBe(true);

  // p = 0 is never vacated and p = 1 always is, so the first and last columns are the
  // extremes of the demonstration.
  await expect(rows.first().locator("td.cell").first()).toHaveClass(/open/);
  await expect(rows.first().locator("td.cell").last()).toHaveClass(/minted/);

  await expect(page.getByTestId("lex-vacancy-ribbon-caption")).toContainText(/Nesting/);
  await expect(page.getByTestId("lex-vacancy-ribbon-caption")).toContainText(/Stability/);

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("the instant invariance check is LIVE: identical under consistent, broken under inconsistent", async ({
  page,
}) => {
  const errors = watchErrors(page);
  const verdict = page.getByTestId("lex-vacancy-invariance-verdict");
  const box = page.getByTestId("lex-vacancy-invariance");

  await setP(page, "0.6");
  // The theorem: the mapped condition encodes to the identical id stream.
  await expect(verdict).toHaveText(/identical/i);
  await expect(box).toContainText(/ids compared/);
  const compared = Number(
    (await box.innerText()).match(/([\d,]+) ids compared/)?.[1].replace(/,/g, "") ?? "0",
  );
  expect(compared).toBeGreaterThan(1000);

  // ...and the control conditions really break it. A hard-coded tick passes above and
  // fails here, which is exactly why both halves are asserted in one test.
  await page
    .getByTestId("lex-vacancy-condition")
    .getByRole("radio", { name: "inconsistent", exact: true })
    .click();
  await expect(verdict).toHaveText(/differ/i);
  const differing = Number(
    (await box.innerText()).match(/([\d,]+) of [\d,]+ positions/)?.[1].replace(/,/g, "") ?? "0",
  );
  expect(differing).toBeGreaterThan(0);
  // Coverage collapses: a fresh type per occurrence cannot be in any fixed budget.
  const unk = (await box.innerText()).match(/rate ([\d.]+)% → ([\d.]+)%/);
  expect(unk, `no <unk> rates in: ${await box.innerText()}`).not.toBeNull();
  expect(Number(unk![2])).toBeGreaterThan(Number(unk![1]));

  // Partial reveal splits every vacated type in two, and breaks it too.
  await page
    .getByTestId("lex-vacancy-condition")
    .getByRole("radio", { name: "partial reveal", exact: true })
    .click();
  await expect(verdict).toHaveText(/differ/i);

  // Back to the mapped condition and it is identical again — the check tracks the state.
  await page
    .getByTestId("lex-vacancy-condition")
    .getByRole("radio", { name: "consistent", exact: true })
    .click();
  await expect(verdict).toHaveText(/identical/i);

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("changing the seed re-mints the corpus but keeps the theorem (SC-702/SC-703)", async ({
  page,
}) => {
  const errors = watchErrors(page);
  await setP(page, "0.7");
  const before = await page.getByTestId("lex-vacancy-corpus").innerText();

  await page.getByTestId("lex-vacancy-seed").fill("7");
  await page.getByTestId("lex-vacancy-seed").dispatchEvent("input");
  await expect
    .poll(async () => page.getByTestId("lex-vacancy-corpus").innerText(), { timeout: 30_000 })
    .not.toBe(before);

  await expect(page.getByTestId("lex-vacancy-invariance-verdict")).toHaveText(/identical/i);
  await expect(page.getByTestId("lex-vacancy-bijective")).toContainText("injective");

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("the on-demand demonstration trains twice and reports max |Δloss| = 0 (FR-714)", async ({
  page,
}) => {
  test.setTimeout(300_000);
  const errors = watchErrors(page);

  await setP(page, "0.75");
  await page.getByTestId("lex-vacancy-demo-run").click();

  const delta = page.getByTestId("lex-vacancy-demo-delta");
  await expect(delta).toBeVisible({ timeout: 240_000 });
  // Reported as an exact 0 — not "≈0", not rounded away. §1.6 is a hard requirement.
  await expect(delta).toContainText("max |Δloss| = 0");
  await expect(delta).not.toContainText(/should have been exactly 0/);
  await expect(page.getByTestId("lex-vacancy-demo-result")).toBeVisible();

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("the tab trains on the VACATED corpus, at a non-default budget (FR-713/FR-716)", async ({
  page,
}) => {
  test.setTimeout(300_000);
  const errors = watchErrors(page);

  // Compose with the tab's own controls: a frequency budget, a small model, a real run.
  await page.getByTestId("lex-budget-source").getByRole("radio", { name: /corpus top-N/i }).click();
  await setP(page, "0.8");

  // The budget's coverage is measured against the VACATED text, so it is live before the
  // first gradient step — that is the whole argument of the budget panel, preserved here.
  await expect(page.getByTestId("lex-coverage-tokens")).toHaveText(/^\d+(\.\d+)?%$/);
  await expect(page.getByTestId("lex-train-active-corpus")).toContainText(/vacated p=0\.80/);

  await page.getByTestId("lex-dmodel").getByRole("radio", { name: "16" }).click();
  // A multiple of the default sampleEvery (50): the periodic sampler and the final sample
  // then land on the same step, which is the case that once threw `each_key_duplicate`.
  await page.getByTestId("lex-steps").fill("100");
  await page.getByTestId("lex-train-run").click();

  const done = page.getByTestId("lex-train-done");
  await expect(done).toBeVisible({ timeout: 240_000 });
  const nums = [...(await done.innerText()).matchAll(/(\d+\.\d+)/g)].map((m) => Number(m[1]));
  expect(nums.length).toBeGreaterThanOrEqual(2);
  expect(Math.min(...nums)).toBeLessThan(Math.max(...nums));
  await expect(page.getByTestId("lex-active-model")).toContainText(/vacated p=0\.80/);

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("the swap control runs, and refuses the p it cannot support (FR-719a, §5.2a)", async ({
  page,
}) => {
  const errors = watchErrors(page);
  const corpus = page.getByTestId("lex-vacancy-corpus");
  const lost = page.getByTestId("lex-vacancy-lost-slots");
  const mint = page.getByTestId("lex-vacancy-mint");

  // The control is live, not decoration: switching it rewrites the corpus with real English
  // words drawn by frequency rank instead of invented ones.
  await setP(page, "1");
  const withNonce = await corpus.innerText();
  await expect(lost).toHaveText("0");
  await mint.getByRole("radio", { name: "swap", exact: true }).click();
  await expect(mint.getByRole("radio", { name: "swap", exact: true })).toHaveAttribute(
    "aria-checked",
    "true",
  );
  const withSwap = await corpus.innerText();
  expect(withSwap).not.toBe(withNonce);

  // At FULL vacancy swap is a bijection of the domain, so the theorem holds for it exactly
  // as it does for nonce — that is the check that the control is implemented correctly.
  await expect(lost).toHaveText("0");
  await expect(page.getByTestId("lex-vacancy-invariance-verdict")).toHaveText(/identical/i);
  await expect(page.getByTestId("lex-vacancy-refusal")).toHaveCount(0);

  // In between it CANNOT be injective (contract §5.2a). The panel must show the engine's
  // refusal and the measured cost — never a clamped p, never a silent nonce map.
  await setP(page, "0.5");
  const refusal = page.getByTestId("lex-vacancy-refusal");
  await expect(refusal).toBeVisible();
  await expect(page.getByTestId("lex-vacancy-refusal-message")).toContainText("§5.2a");
  await expect(page.getByTestId("lex-vacancy-refusal-message")).toContainText("swap");
  expect(Number((await lost.innerText()).replace(/[^\d]/g, ""))).toBeGreaterThan(0);
  // The slider really is still where the reader put it, and nothing was computed in place
  // of the refused vocabulary.
  await expect(page.getByTestId("lex-vacancy")).toContainText("p = 0.50");
  await expect(mint.getByRole("radio", { name: "swap", exact: true })).toHaveAttribute(
    "aria-checked",
    "true",
  );
  await expect(page.getByTestId("lex-vacancy-invariance")).toHaveCount(0);
  await expect(page.getByTestId("lex-vacancy-demo-run")).toBeDisabled();

  // The way out is offered explicitly and works.
  await page.getByTestId("lex-vacancy-refusal-p1").click();
  await expect(refusal).toHaveCount(0);
  await expect(page.getByTestId("lex-vacancy")).toContainText("p = 1.00");
  await expect(page.getByTestId("lex-vacancy-invariance-verdict")).toHaveText(/identical/i);

  // The inconsistent control is refused under swap for a countable reason: it needs a fresh
  // type per occurrence and the corpus has no supply of real words at that rate.
  await page
    .getByTestId("lex-vacancy-condition")
    .getByRole("radio", { name: "inconsistent", exact: true })
    .click();
  await expect(refusal).toBeVisible();
  await expect(page.getByTestId("lex-vacancy-refusal-message")).toContainText(/consistent/);

  expect(errors, `client-side errors: ${errors.join(" | ")}`).toEqual([]);
});

test("no horizontal page overflow at 390px with the panel present", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`/${LEX}`);
  await expect(page.getByTestId("lex-vacancy")).toBeVisible({ timeout: 30_000 });
  await setP(page, "0.65");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test("the corpus view opens on the verse, not on the book's table of contents", async ({
  page,
}) => {
  const errors = watchErrors(page);

  // The shipped corpus carries 618 token-producing lines of front matter -- a title page, a
  // list of rhymes, and an index of first lines -- before LITTLE BO-PEEP starts the verse at
  // line 619. Landing there makes the transform look like it rewrites an index, which is the
  // least interesting thing it does. No general rule separates the two honestly: the index of
  // first lines has verse-length lines, so a line-length heuristic stops inside the front
  // matter. The panel therefore pins the boundary, and this is the assertion that it is right.
  const windowLabel = page.getByTestId("lex-vacancy-window");
  await expect(windowLabel).toContainText("601");

  // At p = 0 the corpus is untransformed, so the real words must be on screen.
  const corpus = page.getByTestId("lex-vacancy-corpus");
  await expect(corpus).toContainText("Bo-Peep");

  // Paging back must still work, and must reach the front matter it deliberately skipped.
  await page.getByTestId("lex-vacancy-prev").click();
  await expect(windowLabel).toContainText("561");

  expect(errors, `console errors: ${errors.join(" | ")}`).toEqual([]);
});
