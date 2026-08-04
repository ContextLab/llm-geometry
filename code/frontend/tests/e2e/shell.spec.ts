import { expect, test } from "@playwright/test";

// App shell e2e against the REAL backend (feature 004: two explorer tabs, no shared
// control sidebar — each explorer owns its own controls; feature 005 added the Info
// reference tab; feature 006 added the Lexicon Lab). The per-tab behavior lives in
// explorer.spec.ts, docs.spec.ts and lexicon.spec.ts; this file covers only the shell
// contract.

test("shell renders the masthead and every tab, in order", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "llm-geometry" })).toBeVisible();

  // Named rather than counted: a bare count tells you the number changed but not what
  // to do about it, and this assertion is the one that notices a tab being added or
  // renamed. It caught feature 006 adding Lexicon.
  const expected = ["Architecture", "Geometry", "Lexicon", "Info"];
  const tabs = page.getByTestId("view-tabs").locator("button");
  await expect(tabs).toHaveText(expected);

  // The removed views must not be reachable by any leftover affordance.
  for (const gone of ["tab-vector", "tab-sankey", "tab-manifold", "controls", "recompute"]) {
    await expect(page.getByTestId(gone)).toHaveCount(0);
  }
});

test("the tab switcher shows exactly one explorer at a time", async ({ page }) => {
  await page.goto("/");

  // Every tab, with the view it owns. Driven from a table so a new tab cannot be added
  // without either appearing here or failing the ordering assertion above.
  const views: Record<string, string> = {
    "tab-architecture": "arch-model-picker",
    "tab-geometry": "geo-view",
    "tab-lexicon": "lex-view",
    "tab-info": "info-view",
  };

  // Architecture is the landing tab.
  await expect(page.getByTestId("arch-model-picker")).toBeVisible();

  for (const [tab, view] of Object.entries(views)) {
    await page.getByTestId(tab).click();
    await expect(page.getByTestId(view)).toBeVisible();
    // …and none of the others is mounted alongside it.
    for (const other of Object.values(views).filter((v) => v !== view)) {
      await expect(page.getByTestId(other)).toHaveCount(0);
    }
  }
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
