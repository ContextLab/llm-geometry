import { expect, test } from "@playwright/test";

/**
 * The navigation guard, in a REAL browser, against a REAL training run.
 *
 * `grep -rn "nav-hold" tests/e2e/` found nothing before this file: the guard that exists
 * to stop a tab click from destroying a run in progress had zero end-to-end coverage. The
 * unit suite drives the store and mounts the panels in jsdom, which cannot see the things
 * that actually decide whether the guard works for a person:
 *
 *  - a real `Worker` really running (jsdom has none, so `lex/TrainPanel`'s worker
 *    lifecycle — the exact thing whose `onDestroy` teardown caused the loss — never
 *    executes there);
 *  - the `role="alertdialog"` being VISIBLE rather than merely present in the DOM,
 *    unclipped by the WebGL canvas the Geometry Lab paints over the same region;
 *  - Chrome's own history semantics for Back, which jsdom's `pushState`/`popstate` is
 *    not.
 *
 * The regression this guards: at step 114 of 400 the run vanished on a tab click, the
 * button returned to idle, and nothing was said.
 */
test.describe("a tab switch during a real training run is held", () => {
  test("holds the click, names the run, and Stay keeps it running", async ({ page }) => {
    await page.goto("/#lexicon");
    await expect(page.getByTestId("lex-train")).toBeVisible();

    // A run long enough that it is still going when the tab is clicked — the whole point
    // is to interrupt work in flight, not to race its completion.
    await page.getByTestId("lex-dmodel").getByRole("radio", { name: "16" }).click();
    await page.getByTestId("lex-steps").fill("400");
    await page.getByTestId("lex-train-run").click();
    // Really training: the live panel only renders while the worker is running.
    await expect(page.getByTestId("lex-train-live")).toBeVisible({ timeout: 60_000 });

    // The click the app itself invites — every `Explain` ends with an Info-tab button.
    await page.getByTestId("tab-info").click();

    const hold = page.getByTestId("nav-hold");
    await expect(hold, "the tab switch was performed silently").toBeVisible();
    await expect(hold).toContainText("a training run in the Lexicon Lab");
    // Still on the Lexicon Lab, and the run is still on screen.
    await expect(page.getByTestId("lex-train-live")).toBeVisible();
    expect(new URL(page.url()).hash).toBe("#lexicon");

    // Stay: the dialog goes, the run continues, and it finishes.
    await page.getByTestId("nav-hold-stay").click();
    await expect(hold).toHaveCount(0);
    await expect(page.getByTestId("lex-train-done")).toBeVisible({ timeout: 180_000 });

    // And once the work is over, the same click just works.
    await page.getByTestId("tab-info").click();
    await expect(page.getByTestId("info-view")).toBeVisible();
  });

  test("Discard leaves the tab, and browser Back is held the same way", async ({ page }) => {
    await page.goto("/#lexicon");
    await page.getByTestId("lex-dmodel").getByRole("radio", { name: "16" }).click();
    await page.getByTestId("lex-steps").fill("400");
    await page.getByTestId("lex-train-run").click();
    await expect(page.getByTestId("lex-train-live")).toBeVisible({ timeout: 60_000 });

    // Back is a tab switch too, and it is the most reflexive way out of a tab.
    await page.goBack();
    await expect(page.getByTestId("nav-hold")).toBeVisible();
    await expect(page.getByTestId("lex-train-live")).toBeVisible();
    // The address bar is put back on the running tab rather than left ahead of the app.
    await expect
      .poll(() => new URL(page.url()).hash, { timeout: 15_000 })
      .toBe("#lexicon");

    await page.getByTestId("nav-hold-discard").click();
    await expect(page.getByTestId("nav-hold")).toHaveCount(0);
    await expect(page.getByTestId("lex-train")).toHaveCount(0);
  });

  test("an idle tab is never held", async ({ page }) => {
    // A guard that holds when there is nothing to lose is a guard people click through
    // without reading, which is the same as not having one.
    await page.goto("/#lexicon");
    await expect(page.getByTestId("lex-train")).toBeVisible();
    await page.getByTestId("tab-info").click();
    await expect(page.getByTestId("info-view")).toBeVisible();
    await expect(page.getByTestId("nav-hold")).toHaveCount(0);
  });
});
