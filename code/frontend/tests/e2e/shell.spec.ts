import { expect, test } from "@playwright/test";

// App shell e2e against the REAL backend (feature 004: two explorer tabs, no shared
// control sidebar — each explorer owns its own controls; feature 005 added the Info
// reference tab). The per-tab behavior lives in explorer.spec.ts and docs.spec.ts;
// this file covers only the shell contract.

test("shell renders the masthead and the three tabs", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "llm-geometry" })).toBeVisible();

  const tabs = page.getByTestId("view-tabs").locator("button");
  await expect(tabs).toHaveCount(3);
  await expect(tabs.nth(0)).toHaveText("Architecture");
  await expect(tabs.nth(1)).toHaveText("Geometry");
  await expect(tabs.nth(2)).toHaveText("Info");

  // The removed views must not be reachable by any leftover affordance.
  for (const gone of ["tab-vector", "tab-sankey", "tab-manifold", "controls", "recompute"]) {
    await expect(page.getByTestId(gone)).toHaveCount(0);
  }
});

test("the tab switcher moves between the two explorers", async ({ page }) => {
  await page.goto("/");
  // Architecture is the landing tab.
  await expect(page.getByTestId("arch-model-picker")).toBeVisible();
  await expect(page.getByTestId("geo-view")).toHaveCount(0);

  await page.getByTestId("tab-geometry").click();
  await expect(page.getByTestId("geo-view")).toBeVisible();
  await expect(page.getByTestId("arch-model-picker")).toHaveCount(0);

  await page.getByTestId("tab-architecture").click();
  await expect(page.getByTestId("arch-model-picker")).toBeVisible();

  await page.getByTestId("tab-info").click();
  await expect(page.getByTestId("info-view")).toBeVisible();
  await expect(page.getByTestId("arch-model-picker")).toHaveCount(0);
  await expect(page.getByTestId("geo-view")).toHaveCount(0);
});

test("no request is made to a removed endpoint", async ({ page }) => {
  const removed: string[] = [];
  page.on("request", (r) => {
    const u = new URL(r.url());
    if (
      /^\/api\/(vector_field|sankey|manifold|token_cloud|distribution|reduction|embeddings|precompute)/.test(
        u.pathname,
      )
    ) {
      removed.push(u.pathname);
    }
  });
  await page.goto("/");
  await page.getByTestId("tab-geometry").click();
  await expect(page.getByTestId("geo-view")).toBeVisible();
  await page.getByTestId("tab-architecture").click();
  await expect(page.getByTestId("arch-model-picker")).toBeVisible();
  expect(removed).toEqual([]);
});
