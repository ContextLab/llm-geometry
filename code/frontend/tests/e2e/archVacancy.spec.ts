import { expect, test } from "@playwright/test";

/**
 * The pretrained arm of the vacancy instrument, driven against the REAL backend
 * (feature 007, contract §8; SC-707/707a/707b).
 *
 * Two runs: the pooled default set (which resolves the small effect and lets the
 * measured ordering be asserted), then a single short passage, which must NOT be
 * presented as if it resolved anything.
 */

/** A short passage: ~60 preserved tokens, far too few for a 0.1-nat effect. */
const PASSAGE = [
  "Hey diddle diddle, the cat and the fiddle,",
  "The cow jumped over the moon;",
  "The little dog laughed to see such sport,",
  "And the dish ran away with the spoon.",
  "Little Jack Horner sat in a corner,",
  "Eating a Christmas pie;",
  "He put in his thumb, and pulled out a plum,",
  "And said, What a good boy am I!",
].join("\n");

test("scores a passage and reports the decomposition, never the conflated total", async ({
  page,
}) => {
  test.setTimeout(300_000);
  await page.goto("/");
  await page.getByTestId("tab-architecture").click();
  const panel = page.getByTestId("arch-vacancy");
  await panel.scrollIntoViewIfNeeded();
  await expect(panel).toBeVisible();

  // The default: the six shipped corpus excerpts, pooled — the configuration the
  // reference numbers were measured in. Pooling matters here rather than being a
  // nicety: the "unknown form" effect is a tenth of a nat, and one short passage
  // cannot resolve its SIGN, let alone its size.
  await page.getByTestId("arch-vac-run").click();

  // Real download + three forward passes on a cold cache.
  await expect(page.getByTestId("arch-vac-table")).toBeVisible({ timeout: 280_000 });
  await expect(page.getByTestId("arch-vac-error")).toHaveCount(0);

  // The three variants, each with its own real statistics.
  for (const v of ["english", "swap", "nonce"]) {
    await expect(page.getByTestId(`arch-vac-row-${v}`)).toBeVisible();
  }

  // The two LABELLED differences, in words, with real numbers.
  const wrong = page.getByTestId("arch-vac-wrong_content");
  const form = page.getByTestId("arch-vac-unknown_form");
  await expect(wrong).toContainText(/\d\.\d{3}/);
  await expect(form).toContainText(/\d\.\d{3}/);
  await expect(panel).toContainText("the cost of wrong content");
  await expect(panel).toContainText("the cost of unknown form");

  // nll(nonce) − nll(english) appears ONLY as the small, explicitly-labelled sum.
  const total = page.getByTestId("arch-vac-total");
  await expect(total).toContainText("nll(nonce) − nll(english)");
  await expect(total).toContainText("conflates");

  // The tiny arm's exact 0 is beside it — that juxtaposition IS the 2×2 (FR-719).
  await expect(page.getByTestId("arch-vac-tiny-arm")).toContainText("0");
  await expect(page.getByTestId("arch-vac-tiny-arm")).toContainText("exactly");

  // The honesty block: the residual, the confound, the alignment mechanism.
  const honesty = page.getByTestId("arch-vac-honesty");
  await expect(honesty).toContainText("UPPER BOUND");
  await expect(honesty).toContainText("higher entropy");
  await expect(honesty).toContainText("UTF-8 byte spans");

  // The measured ordering: wrong content costs more than unknown form (SC-707b).
  const value = async (testid: string): Promise<number> => {
    const text = (await page.getByTestId(testid).innerText()).match(/-?\d+\.\d{3}/);
    if (!text) throw new Error(`no number rendered in ${testid}`);
    return Number(text[0]);
  };
  const wrongNats = await value("arch-vac-wrong_content");
  const formNats = await value("arch-vac-unknown_form");
  expect(wrongNats).toBeGreaterThan(0);
  expect(formNats).toBeGreaterThan(0);
  expect(wrongNats).toBeGreaterThan(formNats);

  await panel.screenshot({
    path: "tests/e2e/__screenshots__/arch-vacancy-score.png",
  });
});

test("a passage too short to resolve the small effect says so instead of concluding", async ({
  page,
}) => {
  test.setTimeout(300_000);
  await page.goto("/");
  await page.getByTestId("tab-architecture").click();
  const panel = page.getByTestId("arch-vacancy");
  await panel.scrollIntoViewIfNeeded();

  await page.getByTestId("arch-vac-defaults").uncheck();
  await page.getByTestId("arch-vac-passage").fill(PASSAGE);
  await page.getByTestId("arch-vac-run").click();
  await expect(page.getByTestId("arch-vac-table")).toBeVisible({ timeout: 280_000 });
  await expect(page.getByTestId("arch-vac-error")).toHaveCount(0);

  // ~60 preserved tokens: the standard error swamps a tenth of a nat, and the verdict
  // must say that rather than reading a conclusion off the sign.
  const pairs = Number(
    (await page.getByTestId("arch-vac-unknown_form-err").innerText()).match(
      /([\d,]+) paired tokens/,
    )?.[1]?.replace(/,/g, "") ?? "0",
  );
  expect(pairs).toBeGreaterThan(0);
  expect(pairs).toBeLessThan(300);
  await expect(page.getByTestId("arch-vac-verdict")).toContainText("does not resolve");
});
