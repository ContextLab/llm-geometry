import { expect, test, type Page } from "@playwright/test";

// Uses the fast (but real) tiny-gpt2 model for the interactive assertions so the e2e
// stays reliable; gpt2 is the app default and is exercised on first load.
const TINY = "sshleifer/tiny-gpt2";

async function selectTinyModel(page: Page) {
  await page.getByTestId("model-custom").fill(TINY);
  await page.getByTestId("model-custom").press("Enter");
  await expect(page.getByTestId("model-status")).toHaveText("ok");
}

async function setRange(page: Page, testid: string, value: string) {
  await page.getByTestId(testid).evaluate((el: HTMLInputElement, v: string) => {
    el.value = v;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
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
  for (const id of ["vector", "sankey", "manifold"]) {
    await expect(page.getByTestId(`tab-${id}`)).toBeVisible();
  }
});

test("vector field renders real arrows", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-vector").click();
  await ready(page, "viz-vector");
  // The drift flow field is a regular grid of many uniform arrows + grid-origin dots.
  // (Assert the field rendered rather than one arrow's bbox: a horizontal arrow has a
  // ~zero-height bounding box and would read as "hidden" — same flat-geometry caveat as
  // the Sankey links.)
  await expect.poll(async () => page.locator('[data-testid="vector-svg"] g.arrows line').count(),
    { timeout: 60_000 }).toBeGreaterThan(50);
  await expect.poll(async () => page.locator('[data-testid="vector-svg"] g.origins circle').count())
    .toBeGreaterThan(20);
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

test("manifold morphs through key frames when a response is set", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("response-input").fill("the cat sat");
  await page.getByTestId("tab-manifold").click();
  await ready(page, "viz-manifold");
  // animation mode: a 3D canvas + a caption announcing the morphing key frames
  await expect(page.locator('[data-testid="manifold-canvas"] canvas')).toBeVisible();
  await expect(page.getByTestId("viz-manifold")).toContainText(/key frames/, { timeout: 60_000 });
  // animation exports (GIF/MP4) become available once there are frames to play
  await expect(page.locator('[data-testid="export-bar"] button', { hasText: "GIF" }).first()).toBeVisible();
  await page.screenshot({ path: "tests/e2e/__screenshots__/viz-manifold-anim.png", fullPage: true });
});

test("the readout-layer slider re-renders the vector field", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-vector").click();
  await ready(page, "viz-vector");
  await setRange(page, "layer-to", "1"); // tiny-gpt2 has 2 layers -> read out at layer 1
  await expect(page.getByTestId("layer-value")).toHaveText("1");
  await expect
    .poll(async () => page.getByTestId("viz-vector").getAttribute("data-ready"), { timeout: 200_000 })
    .toBe("1");
  await expect(page.getByTestId("viz-vector")).toContainText(/layers/);
});

test("response animator plays through the tokens", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("response-input").fill("Paris is nice");
  await expect(page.getByTestId("play-button")).toBeVisible({ timeout: 60_000 });
  const before = await page.getByTestId("step-label").textContent();
  await page.getByTestId("play-button").click();
  await expect
    .poll(async () => page.getByTestId("step-label").textContent(), { timeout: 60_000 })
    .not.toBe(before);
});

test("interactive hover shows a tooltip on the vector field", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-vector").click();
  await ready(page, "viz-vector");
  await page.locator('[data-testid="vector-svg"] line').first().hover({ force: true });
  await expect(page.getByTestId("hover-tooltip")).toBeVisible({ timeout: 10_000 });
});

test("prompt presets dropdown updates the prefix", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("prompt-presets").selectOption({ index: 2 });
  const txt = await page.getByTestId("prefix-input").inputValue();
  expect(txt.length).toBeGreaterThan(0);
});

test("controls are view-specific (layers→vector, swarm→sankey, width→manifold)", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("tab-vector").click();
  await expect(page.getByTestId("layer-to")).toBeVisible(); // readout layer on the vector field
  await expect(page.getByTestId("response-input")).toBeVisible(); // response animates the vector field
  await expect(page.getByTestId("particles-input")).toHaveCount(0); // no swarm controls
  await expect(page.getByTestId("rbfwidth-input")).toHaveCount(0); // no manifold controls
  await page.getByTestId("tab-sankey").click();
  await expect(page.getByTestId("particles-input")).toBeVisible(); // swarm controls on the sankey
  await expect(page.getByTestId("seqlen-input")).toBeVisible();
  await expect(page.getByTestId("response-input")).toBeVisible(); // response highlights a path on the swarm
  await expect(page.getByTestId("play-button")).toHaveCount(0); // ...but no stepper (the Sankey has its own Play)
  await expect(page.getByTestId("layer-to")).toHaveCount(0); // no layer control
  await page.getByTestId("tab-manifold").click();
  await expect(page.getByTestId("rbfwidth-input")).toBeVisible(); // RBF width + surface-field toggle on the manifold
  await expect(page.getByTestId("surface-toggle")).toBeVisible();
  await expect(page.getByTestId("particles-input")).toHaveCount(0);
});

test("export bar saves a vector SVG of the figure", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-vector").click();
  await ready(page, "viz-vector");
  const downloadPromise = page.waitForEvent("download");
  await page.locator('[data-testid="export-bar"] button', { hasText: "SVG" }).first().click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.svg$/);
});

test("the Recompute button force-refreshes the active view", async ({ page }) => {
  await page.goto("/");
  await selectTinyModel(page);
  await page.getByTestId("tab-vector").click();
  await ready(page, "viz-vector");
  let forced = false;
  page.on("request", (r) => {
    if (r.url().includes("/api/vector_field") && r.url().includes("force=true")) forced = true;
  });
  await page.getByTestId("recompute").click();
  await expect.poll(() => forced, { timeout: 60_000 }).toBe(true); // bypasses the cache
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
