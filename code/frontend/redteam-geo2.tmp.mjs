// Red-team harness part 2. TEMPORARY.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
const OUT = "/tmp/redteam-geo";
fs.mkdirSync(OUT, { recursive: true });
const STAGE = process.argv[2] || "1b";
const log = (...a) => console.log(...a);
function hashFile(p) { const b = fs.readFileSync(p); let h = 0; for (let i = 0; i < b.length; i++) h = (h * 31 + b[i]) >>> 0; return h.toString(16) + ":" + b.length; }
async function canvasShot(page, name) { const p = path.join(OUT, `${name}.png`); await page.locator('[data-testid="geo-canvas"]').screenshot({ path: p }); return p; }

async function openGeo(browser, opts = {}) {
  const ctx = await browser.newContext({ viewport: opts.viewport || { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const errs = [];
  page.on("console", (m) => { if (m.type() === "error") errs.push(`[error] ${m.text()}`); });
  page.on("pageerror", (e) => errs.push(`[pageerror] ${e.message}`));
  await page.goto("http://localhost:5173/", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Geometry", exact: false }).first().click();
  await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 180000 });
  await page.waitForTimeout(2500);
  return { ctx, page, errs };
}

// count pixels of the token-dot colour (#6ea8fe) in the live GL canvas
async function dotPixels(page) {
  return page.evaluate(() => {
    const gl = document.querySelector('[data-testid="geo-canvas"] canvas');
    const c = document.createElement("canvas");
    c.width = gl.width; c.height = gl.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(gl, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let dot = 0, arrow = 0, lit = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2], a = d[i + 3];
      if (a < 10) continue;
      if (r + g + b > 120) lit++;
      // dot colour 110,168,254 : blue dominant, green mid, red low
      if (b > 170 && g > 110 && g < 210 && r < 150 && b - r > 80) dot++;
      // arrow colours run 42,58,110 -> 183,148,246 (purple: r ~ b)
      if (b > 120 && r > 120 && Math.abs(r - b) < 90 && g < r) arrow++;
    }
    return { w: c.width, h: c.height, dot, arrow, lit };
  });
}

const browser = await chromium.launch();

if (STAGE === "1b") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== 1b: hover / dots / drag-settle ==");
  const box = await page.locator('[data-testid="geo-canvas"] canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;

  log("DOT PIXELS (next_next default):", JSON.stringify(await dotPixels(page)));

  // hover probe with the CORRECT selector
  const hits = [];
  const misses = [];
  for (let dx = -220; dx <= 220 && hits.length < 12; dx += 8) {
    for (let dy = -220; dy <= 220 && hits.length < 12; dy += 8) {
      await page.mouse.move(cx + dx, cy + dy);
      await page.waitForTimeout(16);
      const t = await page.evaluate(() => {
        const el = document.querySelector('[data-testid="hover-tooltip"]');
        return el ? el.textContent.trim() : null;
      });
      if (t) hits.push(`(${dx},${dy}) ${t}`); else misses.push(1);
    }
  }
  log("HOVER hits:", hits.length, JSON.stringify(hits, null, 1));
  log("HOVER misses:", misses.length);
  await page.screenshot({ path: path.join(OUT, "50-hover.png") });

  // verify a hovered label against the API vocabulary
  const check = await page.evaluate(async () => {
    const r = await fetch("/api/geo/tokenize?text=" + encodeURIComponent("alice rabbit queen"));
    return await r.json();
  });
  log("tokenize sanity:", JSON.stringify(check).slice(0, 300));

  // drag then LONG settle
  await page.mouse.move(cx, cy); await page.mouse.down();
  await page.mouse.move(cx + 90, cy + 20, { steps: 10 }); await page.mouse.up();
  await page.waitForTimeout(4000);
  const s1 = await canvasShot(page, "51-settle-a");
  await page.waitForTimeout(2500);
  const s2 = await canvasShot(page, "52-settle-b");
  log("AFTER-DRAG settled identical?", hashFile(s1) === hashFile(s2), hashFile(s1), hashFile(s2));
  log("spin aria:", await page.locator('[data-testid="geo-autorotate"]').getAttribute("aria-pressed"));

  log("CONSOLE:", JSON.stringify(errs));
  await ctx.close();
}

if (STAGE === "occl") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== occl: force mode occlusion from many angles ==");
  await page.locator('[data-testid="geo-mode"] button', { hasText: "force" }).click();
  await page.waitForTimeout(3500);
  await page.locator('[data-testid="geo-prompt"]').fill("alice rabbit queen said the little door");
  await page.waitForTimeout(3500);
  log("badges:", (await page.locator(".controls-row").innerText()).replace(/\n/g, " | "));
  const box = await page.locator('[data-testid="geo-canvas"] canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  // OrbitControls: full canvas height drag == pi rotation vertically; horizontally 2*pi over clientHeight
  const pxPerRad = box.height / (2 * Math.PI);
  for (let k = 0; k < 8; k++) {
    await canvasShot(page, `60-orbit-${String(k).padStart(2, "0")}`);
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    const dpx = Math.round((Math.PI / 4) * pxPerRad);
    for (let s = 1; s <= 10; s++) await page.mouse.move(cx + (dpx * s) / 10, cy, { steps: 1 });
    await page.mouse.up();
    await page.waitForTimeout(1400);
  }
  log("CONSOLE:", JSON.stringify(errs));
  await ctx.close();
}

await browser.close();
log("DONE", STAGE);
