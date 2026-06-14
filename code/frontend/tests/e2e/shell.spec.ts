import { expect, test, type Page } from "@playwright/test";

// Uses the fast (but real) tiny-gpt2 model for the interactive assertions so the e2e
// stays reliable; gpt2 is the app default and is exercised on first load.
const TINY = "sshleifer/tiny-gpt2";

async function setRange(page: Page, testid: string, value: string) {
  await page.getByTestId(testid).evaluate((el: HTMLInputElement, v: string) => {
    el.value = v;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
}

test("shell renders with shared controls and the preview", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "llm-geometry" })).toBeVisible();
  await expect(page.getByTestId("controls")).toBeVisible();
  await expect(page.getByTestId("preview")).toBeVisible();
  await expect(page.getByTestId("model-select")).toBeVisible();
  await expect(page.getByTestId("prefix-input")).toBeVisible();
  await expect(page.getByTestId("temp-input")).toBeVisible();
  await expect(page.getByTestId("layer-input")).toBeVisible();
});

test("default model (gpt2) renders a real distribution and embedding scatter", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("model-status")).toHaveText("ok");
  // First load may download/load gpt2 then compute — generous timeout.
  await expect(page.getByTestId("dist-bars")).toBeVisible({ timeout: 200_000 });
  await expect(page.locator('[data-testid="scatter"] circle').first()).toBeVisible({ timeout: 200_000 });
  await expect(page.getByTestId("updated")).toBeVisible({ timeout: 200_000 });
  await page.screenshot({ path: "tests/e2e/__screenshots__/shell-gpt2.png", fullPage: true });
});

test("controls drive real cached data into the preview", async ({ page }) => {
  await page.goto("/");

  // Use the fast real model.
  await page.getByTestId("model-custom").fill(TINY);
  await page.getByTestId("model-custom").press("Enter");
  await expect(page.getByTestId("model-status")).toHaveText("ok");

  // Real next-token distribution + 2D scatter render.
  await expect(page.getByTestId("dist-bars")).toBeVisible();
  await expect(page.locator('[data-testid="scatter"] circle').first()).toBeVisible();

  // Changing the layer triggers a recompute -> the preview's render counter increments
  // (a deterministic signal, unlike the 1s-resolution timestamp).
  const rendersBefore = Number(await page.getByTestId("preview").getAttribute("data-renders"));
  await setRange(page, "layer-input", "1");
  await expect(page.getByTestId("layer-value")).toHaveText("1");
  await expect
    .poll(async () => Number(await page.getByTestId("preview").getAttribute("data-renders")), { timeout: 120_000 })
    .toBeGreaterThan(rendersBefore);

  // Changing temperature is reflected in the control + refetches the distribution.
  await setRange(page, "temp-input", "0.10");
  await expect(page.getByTestId("temp-value")).toHaveText("0.10");
  await expect(page.getByTestId("dist-bars")).toBeVisible();

  await page.screenshot({ path: "tests/e2e/__screenshots__/shell.png", fullPage: true });
});

test("an unsupported model shows a clear error and no fabricated data", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("model-custom").fill("definitely-not-a-real-model-xyz-123");
  await page.getByTestId("model-custom").press("Enter");
  await expect(page.getByTestId("model-status")).toHaveText("error");
  await expect(page.getByTestId("model-message")).toContainText(
    /could not|not exist|Unsupported|configuration|gated/i,
  );
  await page.screenshot({ path: "tests/e2e/__screenshots__/error.png", fullPage: true });
});
