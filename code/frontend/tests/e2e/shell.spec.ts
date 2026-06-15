import { expect, test, type Page } from "@playwright/test";

// Uses the fast (but real) tiny-gpt2 model for the interactive assertions so the e2e
// stays reliable; gpt2 is the app default and is exercised on first load.
const TINY = "sshleifer/tiny-gpt2";

async function selectTinyModel(page: Page) {
  await page.getByTestId("model-custom").fill(TINY);
  await page.getByTestId("model-custom").press("Enter");
  await expect(page.getByTestId("model-status")).toHaveText("ok");
}

async function ready(page: Page, testid: string) {
  await expect
    .poll(async () => page.getByTestId(testid).getAttribute("data-ready"), { timeout: 200_000 })
    .toBe("1");
}

test("shell renders with controls and the view switcher", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "llm-geometry" })).toBeVisible();
  await expect(page.getByTestId("controls")).toBeVisible();
  await expect(page.getByTestId("view-tabs")).toBeVisible();
  for (const id of ["vector", "sankey", "manifold", "preview"]) {
    await expect(page.getByTestId(`tab-${id}`)).toBeVisible();
  }
});

test("vector field renders real arrows", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-vector").click();
  await ready(page, "viz-vector");
  await expect(page.locator('[data-testid="vector-svg"] line').first()).toBeVisible();
  await page.screenshot({ path: "tests/e2e/__screenshots__/viz-vector.png", fullPage: true });
});

test("sankey renders the particle swarm", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-sankey").click();
  await ready(page, "viz-sankey");
  // node rects (real area) or the "not enough transitions" note — both are valid renders.
  // (Flat link <path>s can report a ~zero bounding box, so assert on the rects.)
  await expect(page.locator('[data-testid="sankey-svg"] rect, [data-testid="sankey-svg"] text').first()).toBeVisible();
  await page.screenshot({ path: "tests/e2e/__screenshots__/viz-sankey.png", fullPage: true });
});

test("manifold renders a 3D canvas", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-manifold").click();
  await ready(page, "viz-manifold");
  await expect(page.locator('[data-testid="manifold-canvas"] canvas')).toBeVisible();
  await page.screenshot({ path: "tests/e2e/__screenshots__/viz-manifold.png", fullPage: true });
});

test("an unsupported model shows a clear error and no fabricated data", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("model-custom").fill("definitely-not-a-real-model-xyz-123");
  await page.getByTestId("model-custom").press("Enter");
  await expect(page.getByTestId("model-status")).toHaveText("error");
  await expect(page.getByTestId("model-message")).toContainText(
    /could not|not exist|Unsupported|configuration|gated/i,
  );
});
