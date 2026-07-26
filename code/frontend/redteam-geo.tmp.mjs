// Red-team harness for the Geometry Lab tab. TEMPORARY — delete when done.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const OUT = "/tmp/redteam-geo";
fs.mkdirSync(OUT, { recursive: true });
const STAGE = process.argv[2] || "all";

const log = (...a) => console.log(...a);

async function shot(page, name) {
  const p = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: p });
  return p;
}
async function canvasShot(page, name) {
  const el = page.locator('[data-testid="geo-canvas"]');
  const p = path.join(OUT, `${name}.png`);
  await el.screenshot({ path: p });
  return p;
}
function hashFile(p) {
  const b = fs.readFileSync(p);
  let h = 0;
  for (let i = 0; i < b.length; i++) h = (h * 31 + b[i]) >>> 0;
  return h.toString(16) + ":" + b.length;
}

async function openGeo(browser, opts = {}) {
  const ctx = await browser.newContext({ viewport: opts.viewport || { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const errs = [];
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") errs.push(`[${m.type()}] ${m.text()}`);
  });
  page.on("pageerror", (e) => errs.push(`[pageerror] ${e.message}`));
  page.on("requestfailed", (r) => {
    const f = r.failure()?.errorText || "";
    if (!/ABORTED/i.test(f)) errs.push(`[requestfailed] ${r.url()} ${f}`);
  });
  await page.goto("http://localhost:5173/", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Geometry", exact: false }).first().click();
  await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 180000 });
  await page.waitForTimeout(2500);
  return { ctx, page, errs };
}

const browser = await chromium.launch();

// =====================================================================================
if (STAGE === "1" || STAGE === "all") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== STAGE 1: load, header, autorotate, hover, zoom ==");

  log("HEADER:", (await page.locator('[data-testid="geo-view"] header').innerText()).replace(/\n/g, " | "));
  log("CONTROLS:", (await page.locator(".controls-row").innerText()).replace(/\n/g, " | "));
  log("CAPTION:", (await page.locator(".caption").first().innerText()).replace(/\n/g, " | "));
  await shot(page, "01-full-load");

  // --- autorotate off on load?
  const a = await canvasShot(page, "02-rot-a");
  await page.waitForTimeout(1600);
  const b = await canvasShot(page, "03-rot-b");
  log("AUTOROTATE-OFF check: identical?", hashFile(a) === hashFile(b), hashFile(a), hashFile(b));
  log("spin button text:", await page.locator('[data-testid="geo-autorotate"]').innerText(),
      "aria-pressed:", await page.locator('[data-testid="geo-autorotate"]').getAttribute("aria-pressed"));

  // --- spin toggle starts it
  await page.locator('[data-testid="geo-autorotate"]').click();
  await page.waitForTimeout(300);
  const c = await canvasShot(page, "04-spin-a");
  await page.waitForTimeout(1600);
  const d = await canvasShot(page, "05-spin-b");
  log("SPIN-ON check: changed?", hashFile(c) !== hashFile(d),
      "btn:", await page.locator('[data-testid="geo-autorotate"]').innerText(),
      "aria:", await page.locator('[data-testid="geo-autorotate"]').getAttribute("aria-pressed"));

  // --- drag stops spin
  const box = await page.locator('[data-testid="geo-canvas"] canvas').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 60, box.y + box.height / 2 + 10, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(1200);
  const e1 = await canvasShot(page, "06-afterdrag-a");
  await page.waitForTimeout(1600);
  const e2 = await canvasShot(page, "07-afterdrag-b");
  log("DRAG-STOPS-SPIN: still identical?", hashFile(e1) === hashFile(e2),
      "aria:", await page.locator('[data-testid="geo-autorotate"]').getAttribute("aria-pressed"));

  // --- hover a token dot: find one by probing a grid of points near sphere
  let hoverHits = [];
  for (let dx = -140; dx <= 140 && hoverHits.length < 6; dx += 14) {
    for (let dy = -140; dy <= 140 && hoverHits.length < 6; dy += 14) {
      await page.mouse.move(box.x + box.width / 2 + dx, box.y + box.height / 2 + dy);
      await page.waitForTimeout(22);
      const tip = await page.locator(".tooltip, [data-testid='tooltip'], #tooltip").first();
      const n = await page.locator("body").evaluate(() => {
        const el = document.querySelector(".tooltip, #tooltip, [data-testid=tooltip]");
        return el && el.offsetParent !== null ? el.textContent : null;
      });
      if (n) hoverHits.push(`(${dx},${dy}) -> ${n}`);
    }
  }
  log("HOVER hits:", JSON.stringify(hoverHits, null, 1));
  await shot(page, "08-hover");

  // --- scroll zoom
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  const zBefore = await canvasShot(page, "09-zoom-before");
  await page.mouse.wheel(0, -600);
  await page.waitForTimeout(900);
  const zAfter = await canvasShot(page, "10-zoom-after");
  log("SCROLL-ZOOM changed?", hashFile(zBefore) !== hashFile(zAfter));

  log("CONSOLE:", JSON.stringify(errs, null, 1));
  await ctx.close();
}

// =====================================================================================
if (STAGE === "2" || STAGE === "all") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== STAGE 2: force mode, badges, occlusion, tangency ==");
  await page.locator('[data-testid="geo-mode"] button', { hasText: "force" }).click();
  await page.waitForTimeout(3000);
  log("CONTROLS(force):", (await page.locator(".controls-row").innerText()).replace(/\n/g, " | "));
  log("CAPTION(force):", (await page.locator(".caption").first().innerText()).replace(/\n/g, " | "));
  const tb = page.locator('[data-testid="geo-tangent-badge"]');
  const rb = page.locator('[data-testid="geo-residual-badge"]');
  log("tangent badge visible?", await tb.count(), await tb.count() ? await tb.innerText() : "");
  log("residual badge:", await rb.count() ? await rb.innerText() : "(absent)");
  log("residual title:", await rb.count() ? await rb.getAttribute("title") : "");
  // layer full disabled?
  const full = page.locator('[data-testid="geo-layer"] button', { hasText: "full" });
  log("full disabled in force?", await full.isDisabled(), "title:", await full.getAttribute("title"));
  await shot(page, "20-force-default");

  // orbit to several angles & screenshot for occlusion inspection
  const box = await page.locator('[data-testid="geo-canvas"] canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const drags = [[0, "front"], [180, "back"], [90, "side"], [270, "side2"]];
  let acc = 0;
  for (const [deg, name] of drags) {
    const delta = deg - acc; acc = deg;
    if (delta !== 0) {
      await page.mouse.move(cx, cy);
      await page.mouse.down();
      // OrbitControls: horizontal px -> rotation. canvas width maps ~2pi over height*? use px per deg
      const px = delta * (box.height / 180) * 1.0;
      let moved = 0;
      const step = 40;
      while (Math.abs(moved) < Math.abs(px)) {
        const s = Math.sign(px) * Math.min(step, Math.abs(px) - Math.abs(moved));
        moved += s;
        await page.mouse.move(cx + moved, cy, { steps: 2 });
      }
      await page.mouse.up();
      await page.waitForTimeout(700);
    }
    await canvasShot(page, `21-force-orbit-${name}`);
  }

  // antisymmetrize OFF
  await page.locator('.ctl.check input[type=checkbox]').uncheck();
  await page.waitForTimeout(2500);
  log("CONTROLS(antisym off):", (await page.locator(".controls-row").innerText()).replace(/\n/g, " | "));
  log("tangent badge count (antisym off):", await tb.count());
  log("residual badge (antisym off):", await rb.count() ? await rb.innerText() : "(absent)");
  await canvasShot(page, "22-force-antisym-off");
  await shot(page, "23-force-antisym-off-full");

  // Compare API residual with badge
  const apiMax = await page.evaluate(async () => {
    const r = await fetch("/api/geo/vector_field?mode=force&layer=0&prompt=" + encodeURIComponent(document.querySelector('[data-testid=geo-prompt]').value) + "&antisymmetrize=false");
    const d = await r.json();
    return Math.max(...d.sequence_forces.map((f) => f.normal_residual));
  });
  log("API max normal_residual:", apiMax, " badge:", await rb.count() ? await rb.innerText() : "(absent)");

  // measure actual drawn force-arrow tangency via three.js scene? not accessible. Instead
  // recompute from API: are returned sequence_forces perpendicular to their anchors?
  const tang = await page.evaluate(async () => {
    const prompt = document.querySelector('[data-testid=geo-prompt]').value;
    const [f, t] = await Promise.all([
      fetch("/api/geo/vector_field?mode=force&layer=0&prompt=" + encodeURIComponent(prompt) + "&antisymmetrize=false").then((r) => r.json()),
      fetch("/api/geo/trace?prompt=" + encodeURIComponent(prompt)).then((r) => r.json()),
    ]);
    const out = [];
    for (const sf of f.sequence_forces) {
      const z = t.embeddings[sf.position];
      const dot = z[0] * sf.vec[0] + z[1] * sf.vec[1] + z[2] * sf.vec[2];
      const m = Math.hypot(...z) * Math.hypot(...sf.vec);
      out.push({ pos: sf.position, cos: m > 1e-12 ? dot / m : 0, zn: Math.hypot(...z), fmag: Math.hypot(...sf.vec) });
    }
    return out;
  });
  log("SEQ FORCE tangency vs trace anchors:", JSON.stringify(tang));

  log("CONSOLE:", JSON.stringify(errs, null, 1));
  await ctx.close();
}

// =====================================================================================
if (STAGE === "3" || STAGE === "all") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== STAGE 3: layers / sliders / no-op detection ==");
  const shots = {};
  const snap = async (n) => { shots[n] = hashFile(await canvasShot(page, `30-${n}`)); };
  await page.waitForTimeout(1200);
  await snap("nn-full-T0");
  for (const l of ["0", "1", "2", "3"]) {
    await page.locator('[data-testid="geo-layer"] button', { hasText: new RegExp(`^${l}$`) }).click();
    await page.waitForTimeout(2200);
    await snap(`nn-layer${l}`);
  }
  // temperature
  await page.locator('[data-testid="geo-layer"] button', { hasText: /^full$/ }).click();
  await page.waitForTimeout(2200);
  const tempSlider = page.locator('.ctl.slider input[type=range]').first();
  await tempSlider.fill("1");
  await page.waitForTimeout(2400);
  await snap("nn-T1");
  log("controls after T=1:", (await page.locator(".controls-row").innerText()).replace(/\n/g, " | "));
  log("caption after T=1:", (await page.locator(".caption").first().innerText()).replace(/\n/g, " | "));
  const topM = page.locator('.ctl.slider.narrow input[type=range]');
  await topM.fill("5");
  await page.waitForTimeout(2400);
  await snap("nn-T1-m5");
  log("caption T1 m5:", (await page.locator(".caption").first().innerText()).replace(/\n/g, " | "));
  await shot(page, "31-nn-T1-m5-full");
  // T=0 with topM=5 (claim: only argmax arrow emitted)
  await tempSlider.fill("0");
  await page.waitForTimeout(2400);
  await snap("nn-T0-m5");
  log("caption T0 m5:", (await page.locator(".caption").first().innerText()).replace(/\n/g, " | "));

  log("CANVAS HASHES:", JSON.stringify(shots, null, 1));
  const vals = Object.entries(shots);
  const dupes = [];
  for (let i = 0; i < vals.length; i++) for (let j = i + 1; j < vals.length; j++)
    if (vals[i][1] === vals[j][1]) dupes.push(`${vals[i][0]} == ${vals[j][0]}`);
  log("IDENTICAL RENDERS (possible no-ops):", JSON.stringify(dupes));

  log("CONSOLE:", JSON.stringify(errs, null, 1));
  await ctx.close();
}

// =====================================================================================
if (STAGE === "4" || STAGE === "all") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== STAGE 4: edge cases (empty prompt, oov, long prompt, 390px) ==");
  const prompt = page.locator('[data-testid="geo-prompt"]');
  log("placeholder:", await prompt.getAttribute("placeholder"));
  log("default prompt:", await prompt.inputValue());

  await prompt.fill("");
  await page.waitForTimeout(2500);
  log("EMPTY prompt: strip?", await page.locator('[data-testid="geo-tokenize-strip"]').count(),
      "errors:", await page.locator('[data-testid="geo-error"]').count(),
      (await page.locator('[data-testid="geo-error"]').count()) ? await page.locator('[data-testid="geo-error"]').first().innerText() : "");
  await shot(page, "40-empty-prompt");

  await prompt.fill("zzqqx blorptastic hyperzoid");
  await page.waitForTimeout(2500);
  log("OOV strip:", await page.locator('[data-testid="geo-tokenize-strip"]').innerText());
  await shot(page, "41-oov");
  log("attention panel:", (await page.locator('[data-testid="geo-attention"]').innerText()).slice(0, 400).replace(/\n/g, " | "));

  const longp = Array.from({ length: 70 }, (_, i) => ["the", "queen", "said", "alice", "rabbit", "little", "very"][i % 7]).join(" ");
  await prompt.fill(longp);
  await page.waitForTimeout(3000);
  const strip = await page.locator('[data-testid="geo-tokenize-strip"]').innerText();
  log("LONG prompt strip (tail):", strip.slice(-200).replace(/\n/g, " | "));
  log("truncation chip present?", strip.includes("truncated"));
  await shot(page, "42-long-prompt");

  // rapid control changes
  await prompt.fill("the queen said");
  await page.waitForTimeout(600);
  for (let i = 0; i < 8; i++) {
    await page.locator('[data-testid="geo-mode"] button', { hasText: i % 2 ? "next-next" : "force" }).click();
    await page.waitForTimeout(90);
  }
  await page.waitForTimeout(4000);
  log("after rapid toggles: mode row:", (await page.locator(".controls-row").innerText()).replace(/\n/g, " | "));
  log("errors after rapid:", await page.locator('[data-testid="geo-error"]').count());
  await shot(page, "43-rapid");

  // tab switching mid-compute
  await prompt.fill("alice was beginning to get very tired");
  await page.waitForTimeout(150);
  await page.getByRole("button", { name: "Architecture", exact: false }).first().click();
  await page.waitForTimeout(900);
  await page.getByRole("button", { name: "Geometry", exact: false }).first().click();
  await page.waitForTimeout(5000);
  log("after tab switch ready?", await page.locator('[data-testid="geo-view"][data-ready="1"]').count());
  log("errors:", await page.locator('[data-testid="geo-error"]').count());
  await shot(page, "44-tabswitch-back");

  // 390 px
  await page.setViewportSize({ width: 390, height: 900 });
  await page.waitForTimeout(2500);
  const overflow = await page.evaluate(() => ({
    docW: document.documentElement.scrollWidth,
    clientW: document.documentElement.clientWidth,
  }));
  log("390px overflow:", JSON.stringify(overflow));
  await page.screenshot({ path: path.join(OUT, "45-390-top.png") });
  await page.evaluate(() => window.scrollTo(0, 900));
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, "46-390-mid.png") });
  await page.evaluate(() => window.scrollTo(0, 2000));
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, "47-390-bot.png") });

  log("CONSOLE:", JSON.stringify(errs, null, 1));
  await ctx.close();
}

await browser.close();
log("DONE stage", STAGE);
