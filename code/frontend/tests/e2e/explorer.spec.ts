import { expect, test, type Page } from "@playwright/test";

// Feature 002 (issue #1): the two explorer tabs, driven against the REAL backend —
// SmolLM2-135M-Instruct for the Architecture Explorer and the trained-from-scratch
// GeoTransformer for the Geometry Lab. First runs may download/train; timeouts are
// generous for cold caches (CI) and fast on warm ones.

async function openArchitecture(page: Page) {
  await page.goto("/");
  await page.getByTestId("tab-architecture").click();
  // Graph build (download + trace) can take a while cold.
  await expect(page.locator('[data-testid^="diagram-node-"]').first()).toBeVisible({
    timeout: 220_000,
  });
}

async function openGeometry(page: Page) {
  await page.goto("/");
  await page.getByTestId("tab-geometry").click();
  // Cold cache trains the tiny model (~30 s); warm is instant.
  await expect
    .poll(async () => page.getByTestId("geo-view").getAttribute("data-ready"), {
      timeout: 220_000,
    })
    .toBe("1");
}

test.describe("Architecture Explorer", () => {
  test("loads the traced diagram with real model metadata", async ({ page }) => {
    await openArchitecture(page);
    await expect(page.getByTestId("arch-meta")).toContainText("layers");
    await expect(page.getByTestId("arch-meta")).toContainText("params");
    // Functional (parameterless) ops are first-class nodes (FR-101): at least one
    // softmax node exists in the visible layer group.
    await expect(
      page.locator('[data-testid*="attention_softmax"], [data-testid*="rope"]').first(),
    ).toBeAttached();
  });

  test("traces a typed prompt into the token strip and top-10", async ({ page }) => {
    await openArchitecture(page);
    await page.getByTestId("arch-prompt").fill("The capital of France is");
    await expect(page.getByTestId("arch-trace-strip")).toContainText("capital", {
      timeout: 120_000,
    });
    // (the label is lowercase in the DOM; CSS uppercases it)
    await expect(page.getByTestId("arch-breakdown")).toContainText("next-token top-10");
  });

  test("clicking a diagram node opens the inspector with real weights", async ({ page }) => {
    await openArchitecture(page);
    await page.locator('[data-testid="diagram-node-model.embed_tokens"]').click();
    await expect(page.getByTestId("arch-inspector")).toBeVisible();
    await expect(page.getByTestId("arch-inspector")).toContainText("token embedding");
    await expect(
      page.getByTestId("arch-inspector").getByTestId("matrix-heatmap"),
    ).toBeVisible({ timeout: 60_000 });
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("arch-inspector")).not.toBeVisible();
  });

  test("generates a real reply with per-token probabilities", async ({ page }) => {
    await openArchitecture(page);
    await page.getByTestId("arch-prompt").fill("Say hi in three words.");
    await page.getByTestId("arch-generate").click();
    await expect(page.getByTestId("arch-reply")).not.toBeEmpty({ timeout: 180_000 });
    // Reply tokens are probability-underlined spans with an accessible tooltip label.
    const tok = page.getByTestId("arch-reply").locator(".tok").first();
    await expect(tok).toBeAttached();
    await expect(tok).toHaveAttribute("aria-label", /%|prob/i);
  });
});

test.describe("Geometry Lab", () => {
  test("reaches ready and renders the sphere with gate metrics", async ({ page }) => {
    await openGeometry(page);
    await expect(page.getByTestId("geo-canvas")).toBeVisible();
    await expect(page.getByTestId("geo-view")).toContainText("coverage");
    await expect(page.getByTestId("geo-view")).toContainText("field entropy");
  });

  test("tokenization strip marks out-of-vocabulary words", async ({ page }) => {
    await openGeometry(page);
    await page.getByTestId("geo-prompt").fill("alice met a zorblatt");
    await expect(page.getByTestId("geo-tokenize-strip")).toContainText("zorblatt", {
      timeout: 60_000,
    });
    await expect(page.getByTestId("geo-view")).toContainText("unknown");
  });

  test("force mode shows the residual badge; antisymmetrize makes it exact", async ({ page }) => {
    await openGeometry(page);
    await page.getByTestId("geo-mode").getByText("force").click();
    await expect(page.getByTestId("geo-view")).toContainText("max normal residual", {
      timeout: 120_000,
    });
    await page.getByTestId("geo-view").getByText("antisymmetrize").click();
    await expect(page.getByTestId("geo-view")).toContainText("tangent: exact", {
      timeout: 120_000,
    });
  });

  test("applying the identity preset mints an edited-weights token", async ({ page }) => {
    await openGeometry(page);
    await page.getByTestId("geo-preset").selectOption("identity");
    await page.getByTestId("geo-apply").click();
    await expect(page.getByTestId("geo-weight-panel")).toContainText("edited", {
      timeout: 120_000,
    });
    await page.getByTestId("geo-reset").click();
    await expect(page.getByTestId("geo-weight-panel")).toContainText("learned", {
      timeout: 60_000,
    });
  });
});
