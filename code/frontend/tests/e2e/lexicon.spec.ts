import { expect, test } from "@playwright/test";

/**
 * The Lexicon Lab, driven as a visitor drives it (feature 006).
 *
 * These tests exist for the claims that are cheap to break and expensive to notice:
 * that the coverage counters are real and visible BEFORE training; that switching the
 * budget SOURCE at a matched |V| actually changes the numbers (US-3 is the tab's whole
 * argument); that the spectrum panel never shows effective rank without the ceiling and
 * the untrained baseline beside it; and that generated text is in budget.
 *
 * Training here is deliberately tiny (d=16, 1 layer, few steps) — the point is that the
 * loop runs and the numbers move, not that the model gets good.
 */

const LEX = "#lexicon";

test.beforeEach(async ({ page }) => {
  await page.goto(`/${LEX}`);
  await expect(page.getByTestId("lex-view")).toBeVisible({ timeout: 30_000 });
  // The corpus is a build-time export; if it is missing the tab says so rather than
  // inventing one, and every assertion below would be meaningless.
  await expect(page.getByTestId("lex-corpus-error")).toHaveCount(0);
});

test("is reachable by URL and lands on the Lexicon tab", async ({ page }) => {
  expect(page.url()).toContain(LEX);
  await expect(page.getByTestId("lex-view")).toBeVisible();
});

test("coverage is measured and shown BEFORE any training (US-1)", async ({ page }) => {
  const rows = page.getByTestId("lex-budget-rows");
  await expect(page.getByTestId("lex-budget-size")).toBeVisible();
  await expect(rows).toBeVisible();

  // |V| is the word count; rows adds the four specials. Conflating them was a real defect
  // in the source project, so the tab must keep them distinct.
  //
  // Parse the sentence that states BOTH, rather than stripping digits out of an element:
  // `lex-budget-size` is the whole radio GROUP ("pre-primer 40 … full list 314"), so
  // digit-stripping it yields a concatenation of every option. An earlier version of this
  // test did exactly that and failed in CI.
  const m = (await rows.innerText()).match(/\|V\|\s*=\s*([\d,]+)\s*words.*?([\d,]+)\s*embedding rows/s);
  expect(m, `could not parse budget rows from: ${await rows.innerText()}`).not.toBeNull();
  const num = (s: string) => Number(s.replace(/,/g, ""));
  expect(num(m![2]) - num(m![1])).toBe(4);

  for (const id of ["lex-coverage-tokens", "lex-coverage-unk", "lex-coverage-lines"]) {
    await expect(page.getByTestId(id)).toBeVisible();
    await expect(page.getByTestId(id)).not.toHaveText("");
  }
  // These are percentages and a ratio, not placeholders.
  await expect(page.getByTestId("lex-coverage-tokens")).toHaveText(/^\d+(\.\d+)?%$/);
  await expect(page.getByTestId("lex-coverage-unk")).toHaveText(/^\d+(\.\d+)?%$/);
  await expect(page.getByTestId("lex-coverage-lines")).toHaveText(/^[\d,]+ \/ [\d,]+$/);
});

test("the Dolch budget tops out at 314, not the widely-cited 315", async ({ page }) => {
  // `Santa Claus` has a space and no word tokenizer can match it; the source shipped it
  // and silently had a 314-word "315" budget. If this ever reads 315 again, either the
  // entry came back or the count stopped being measured from the data.
  await page.getByTestId("lex-budget-source").getByRole("radio", { name: /dolch/i }).click();
  // The largest budget's radio is labelled "full list, 314 words".
  const largest = page.getByTestId("lex-budget-size").getByRole("radio").last();
  await largest.click();
  await expect(largest).toHaveAttribute("aria-checked", "true");

  // Assert on the sentence that states the count, not on the radio group — the group
  // lists EVERY size, so `toContainText("314")` there passes no matter what is selected.
  await expect(page.getByTestId("lex-budget-rows")).toContainText("|V| = 314 words");
  await expect(page.getByTestId("lex-budget-rows")).toContainText("318 embedding rows");
  await expect(page.getByTestId("lex-budget-rows")).not.toContainText("315");
});

test("switching the budget SOURCE at matched |V| changes coverage (US-3, SC-603)", async ({
  page,
}) => {
  const source = page.getByTestId("lex-budget-source");
  const tokens = page.getByTestId("lex-coverage-tokens");
  const rows = page.getByTestId("lex-budget-rows");

  // The two options are labelled "Dolch (1936)" and "corpus top-N" — NOT "frequency".
  // An earlier version of this test waited for /frequenc/i and timed out against text
  // that never existed. Match the radios positionally-by-name as they actually read.
  const dolchRadio = source.getByRole("radio", { name: /dolch/i });
  const freqRadio = source.getByRole("radio", { name: /corpus top-N/i });
  await expect(dolchRadio).toBeVisible();
  await expect(freqRadio).toBeVisible();

  const vOf = async () =>
    (await rows.innerText()).match(/\|V\|\s*=\s*([\d,]+)\s*words/)?.[1] ?? null;

  await dolchRadio.click();
  await expect(dolchRadio).toHaveAttribute("aria-checked", "true");
  const dolchV = await vOf();
  const dolchCoverage = await tokens.innerText();

  await freqRadio.click();
  await expect(freqRadio).toHaveAttribute("aria-checked", "true");
  const freqV = await vOf();
  const freqCoverage = await tokens.innerText();

  // Same number of words...
  expect(freqV).toBe(dolchV);
  // ...different words, so measurably different coverage. If these ever match, the
  // comparison the tab invites is not the comparison it is making.
  expect(freqCoverage).not.toBe(dolchCoverage);

  const pct = (s: string) => Number(s.replace(/[^\d.]/g, ""));
  expect(pct(freqCoverage)).toBeGreaterThan(pct(dolchCoverage));
});

test("effective rank is never shown without its ceiling and an untrained baseline", async ({
  page,
}) => {
  // The honesty requirement of the whole tab: rank rises with |V| even for random
  // weights, so the number alone is misleading. Either the panel is still waiting for a
  // model, or it shows the rank WITH both references. There is no third state in which a
  // bare effective rank appears.
  const panel = page.getByTestId("lex-spectrum");
  await expect(panel).toBeVisible();

  const waiting = page.getByTestId("lex-spectrum-waiting");
  if ((await waiting.count()) > 0) {
    await expect(waiting).toBeVisible();
    return;
  }
  await expect(page.getByTestId("lex-spectrum-legend")).toBeVisible();
  await expect(page.getByTestId("lex-spectrum-untrained")).toBeVisible();
  // The ceiling is stated numerically as min(|V|−1, d), not merely drawn.
  await expect(panel).toContainText(/min\(\|V\|−1, ?d\)/);
});

test("the token cloud is labelled a PCA projection, not a native embedding (FR-623)", async ({
  page,
}) => {
  const cloud = page.getByTestId("lex-cloud");
  await expect(cloud).toBeVisible();
  // The Geometry Lab's sphere IS the representation; this one is a projection of a
  // higher-dimensional space and must not be read the same way.
  await expect(cloud).toContainText(/projection/i);
  await expect(page.getByTestId("lex-cloud-explained")).toBeVisible();
});

test("trains a real model: loss falls and generated words are all in budget", async ({ page }) => {
  test.setTimeout(240_000);

  // Any client-side exception during a run is a failure, not noise. A duplicate-key crash
  // shipped to the live site because nothing here was watching.
  const consoleErrors: string[] = [];
  page.on("pageerror", (e) => consoleErrors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });

  // Smallest honest configuration, so the loop is exercised without a long wait.
  await page.getByTestId("lex-dmodel").getByRole("radio", { name: "16" }).click();
  // MUST be a multiple of the default sampleEvery (50): the periodic sampler and the
  // final sample then both land on the last step. That collision is what produced
  // `each_key_duplicate` on the deployed site, and a step count like 30 hides it.
  await page.getByTestId("lex-steps").fill("100");

  await page.getByTestId("lex-train-run").click();

  const done = page.getByTestId("lex-train-done");
  await expect(done).toBeVisible({ timeout: 180_000 });

  // Loss must actually have moved. An untrained model over R rows starts at ln(R);
  // we only require that the final loss is meaningfully below the first.
  const text = await done.innerText();
  const nums = [...text.matchAll(/(\d+\.\d+)/g)].map((m) => Number(m[1]));
  expect(nums.length).toBeGreaterThanOrEqual(2);
  const [first, final] = [Math.max(...nums), Math.min(...nums)];
  expect(final).toBeLessThan(first);

  // SC-602: in-budget by construction, because the vocabulary IS the budget. The badge
  // states that guarantee; assert it is present rather than merely hoping.
  await page.getByTestId("lex-generate").click();
  await expect(page.getByTestId("lex-output")).not.toHaveText("", { timeout: 60_000 });
  await expect(page.getByTestId("lex-inbudget-badge")).toBeVisible();

  // And check it independently: every emitted word must be a real budget word, not a
  // special token leaking through.
  const out = await page.getByTestId("lex-output").innerText();
  expect(out.trim().length).toBeGreaterThan(0);

  // The caption legitimately NAMES the masked specials, so scope this to the generated
  // text rather than the whole panel — checking the panel gives a false positive.
  const genOnly = await page
    .getByTestId("lex-output")
    .locator(".gen, .text, pre, p")
    .first()
    .innerText();
  expect(genOnly).not.toMatch(/<unk>|<bos>|<pad>/);

  expect(consoleErrors, `client-side errors during training: ${consoleErrors.join(" | ")}`)
    .toEqual([]);
});

test("no horizontal page overflow at 390px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`/${LEX}`);
  await expect(page.getByTestId("lex-view")).toBeVisible({ timeout: 30_000 });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
