// Baseline reproduction for feature 004 issue #2 ("architecture animations don't
// update correctly; head and layer can't be viewed at the same time").
// Drives the REAL app on :5173 against the REAL backend and reports observed state.
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const OUT = process.argv[2] ?? "/tmp/004-baseline";
mkdirSync(OUT, { recursive: true });

const b = await chromium.launch();
const page = await b.newPage({ viewport: { width: 1400, height: 1000 } });
page.on("console", (m) => m.type() === "error" && console.log("  [console.error]", m.text()));
page.on("pageerror", (e) => console.log("  [pageerror]", e.message));

await page.goto("http://localhost:5173/");
console.log("== waiting for the graph + first trace (real model download on cold cache) ==");
await page.getByTestId("arch-breakdown").waitFor({ timeout: 300_000 });
await page.getByTestId("arch-trace-strip").waitFor({ timeout: 300_000 });
await page.screenshot({ path: `${OUT}/01-arch-loaded.png`, fullPage: true });

const layerSlider = page.locator('input[aria-label="detail layer"]');
const headSelect = page.locator('select[aria-label="attention head"]');
const playhead = page.locator(".playhead");

console.log("layer slider present:", await layerSlider.count());
console.log("head select present:", await headSelect.count());
console.log("head options:", await headSelect.locator("option").count());

// --- the suspected defect: does playback overwrite a user-chosen layer? -------------
await layerSlider.evaluate((el) => {
  el.value = "7";
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
const pinned = await layerSlider.inputValue();
console.log(`\n== user pinned layer ${pinned}; starting playback ==`);

await page.getByTestId("arch-play").click();
const samples = [];
for (let i = 0; i < 8; i++) {
  await page.waitForTimeout(400);
  samples.push({
    layer: await layerSlider.inputValue(),
    head: await headSelect.inputValue(),
    playhead: (await playhead.innerText()).replace(/\s+/g, " ").slice(0, 90),
  });
}
for (const s of samples) console.log(`  layer=${s.layer} head=${s.head} | ${s.playhead}`);

const drifted = samples.some((s) => s.layer !== pinned);
console.log(`\nRESULT: user-pinned layer ${drifted ? "WAS OVERWRITTEN by the playhead" : "held"}`);
const advanced = new Set(samples.map((s) => s.playhead)).size > 1;
console.log(`RESULT: playhead ${advanced ? "advanced" : "DID NOT ADVANCE"} during playback`);

await page.screenshot({ path: `${OUT}/02-arch-playing.png`, fullPage: true });

// --- can a whole layer's heads be seen at once? ------------------------------------
const heatmaps = await page.locator("canvas").count();
console.log(`RESULT: ${heatmaps} canvas element(s) in the detail area — one attention`);
console.log("        head is shown at a time via the <select> (no all-heads view)");

// --- geometry tab baseline ----------------------------------------------------------
await page.getByTestId("tab-geometry").click();
await page.getByTestId("geo-view").waitFor({ timeout: 300_000 });
await page.waitForFunction(
  () => document.querySelector('[data-testid="geo-view"]')?.getAttribute("data-ready") === "1",
  null,
  { timeout: 300_000 },
);
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/03-geo-next-next.png`, fullPage: true });

// force mode = the tangency/occlusion defect
await page.getByTestId("geo-mode").locator("button", { hasText: "force" }).click();
await page.waitForTimeout(2500);
await page.screenshot({ path: `${OUT}/04-geo-force.png`, fullPage: true });
const residual = await page.locator(".badge").allInnerTexts();
console.log("\nGEO force-mode badges:", residual);

const antisym = await page.locator('input[type="checkbox"]').first().isChecked();
console.log(`RESULT: antisymmetrize default = ${antisym}`);

// is the sphere auto-rotating with no way to stop it?
const rotateControls = await page.getByText(/rotat/i).count();
console.log(`RESULT: ${rotateControls} rotation control(s) in the Geometry tab`);

await b.close();
console.log(`\nscreenshots -> ${OUT}`);
