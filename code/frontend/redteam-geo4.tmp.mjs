// Red-team harness part 3: weight lab / fine-tune / train+save/load. TEMPORARY.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
const OUT = "/tmp/redteam-geo";
fs.mkdirSync(OUT, { recursive: true });
const STAGE = process.argv[2];
const log = (...a) => console.log(...a);
function hashFile(p) { const b = fs.readFileSync(p); let h = 0; for (let i = 0; i < b.length; i++) h = (h * 31 + b[i]) >>> 0; return h.toString(16) + ":" + b.length; }
async function canvasShot(page, name) { const p = path.join(OUT, `${name}.png`); await page.locator('[data-testid="geo-canvas"]').screenshot({ path: p }); return p; }

async function openGeo(browser, opts = {}) {
  const ctx = await browser.newContext({
    viewport: opts.viewport || { width: 1440, height: 1100 },
    acceptDownloads: true,
  });
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
const headerText = (page) => page.locator('[data-testid="geo-view"] header').innerText();

const browser = await chromium.launch();

// ------------------------------------------------------------------ weight lab
if (STAGE === "wl") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== WEIGHT LAB ==");
  log("initial badge:", (await page.locator('[data-testid="geo-weight-panel"] .head').innerText()).replace(/\n/g, " | "));
  log("meta:", (await page.locator(".matrix-meta").innerText()).replace(/\n/g, " | "));
  const base = hashFile(await canvasShot(page, "70-wl-base"));
  const seen = { base };

  const presets = ["identity", "toeplitz_fuzzy", "random", "random_autocorr", "zero", "learned"];
  for (const m of ["W_V", "W_Q", "W_K", "W_O", "embedding"]) {
    await page.locator('[data-testid="geo-matrix"]').selectOption(m);
    await page.waitForTimeout(900);
    log(`--- matrix ${m}: meta =`, (await page.locator(".matrix-meta").innerText()).replace(/\n/g, " | "));
    for (const p of presets) {
      await page.locator('[data-testid="geo-preset"]').selectOption(p);
      await page.locator('[data-testid="geo-apply"]').click();
      await page.waitForTimeout(2600);
      const err = await page.locator('[data-testid="geo-weight-panel"] .error').count();
      const errTxt = err ? await page.locator('[data-testid="geo-weight-panel"] .error').innerText() : "";
      const badge = (await page.locator('[data-testid="geo-weight-panel"] .head').innerText()).replace(/\n/g, " | ");
      const h = hashFile(await canvasShot(page, `71-wl-${m}-${p}`));
      const hdr = (await headerText(page)).replace(/\n/g, " | ");
      log(`  ${m}/${p}: err="${errTxt}" badge="${badge}" changed=${h !== seen.base} hash=${h}`);
      log(`     header: ${hdr.slice(hdr.indexOf("active model") >= 0 ? hdr.indexOf("active model") : hdr.indexOf("shipped"))}`);
      seen[`${m}-${p}`] = h;
    }
  }
  // cell edit
  await page.locator('[data-testid="geo-matrix"]').selectOption("W_V");
  await page.locator('[data-testid="geo-preset"]').selectOption("learned");
  await page.locator('[data-testid="geo-apply"]').click();
  await page.waitForTimeout(2500);
  const beforeEdit = hashFile(await canvasShot(page, "72-wl-before-edit"));
  log("after preset=learned, badge:", (await page.locator('[data-testid="geo-weight-panel"] .head').innerText()).replace(/\n/g, " | "));
  // click a heatmap cell
  const hm = page.locator('[data-testid="geo-weight-panel"] canvas').first();
  const hb = await hm.boundingBox();
  log("heatmap box:", JSON.stringify(hb));
  page.once("dialog", async (d) => { log("DIALOG:", d.type(), JSON.stringify(d.message()), "default:", JSON.stringify(d.defaultValue())); await d.accept("2.5"); });
  await hm.click({ position: { x: hb.width / 6, y: hb.height / 6 } });
  await page.waitForTimeout(3000);
  const afterEdit = hashFile(await canvasShot(page, "73-wl-after-edit"));
  log("cell edit changed sphere?", beforeEdit !== afterEdit);
  log("badge after edit:", (await page.locator('[data-testid="geo-weight-panel"] .head').innerText()).replace(/\n/g, " | "));
  log("header after edit:", (await headerText(page)).replace(/\n/g, " | "));
  log("weight-lab error:", await page.locator('[data-testid="geo-weight-panel"] .error').count() ? await page.locator('[data-testid="geo-weight-panel"] .error').innerText() : "(none)");
  await page.screenshot({ path: path.join(OUT, "74-wl-after-edit-full.png") });
  // reset
  const reset = page.locator('[data-testid="geo-reset"]');
  log("reset button present?", await reset.count());
  if (await reset.count()) { await reset.click(); await page.waitForTimeout(2800); }
  const afterReset = hashFile(await canvasShot(page, "75-wl-after-reset"));
  log("reset back to base?", afterReset === base, afterReset, base);
  log("badge after reset:", (await page.locator('[data-testid="geo-weight-panel"] .head').innerText()).replace(/\n/g, " | "));
  log("header after reset:", (await headerText(page)).replace(/\n/g, " | "));
  log("CONSOLE:", JSON.stringify(errs));
  await ctx.close();
}

// ------------------------------------------------------------------ fine-tune
if (STAGE === "ft") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== FINE-TUNE ==");
  const ft = page.locator('[data-testid="geo-finetune"]');
  log("panel:", (await ft.innerText()).replace(/\n/g, " | "));
  await ft.locator("textarea").fill(
    "The queen said the little door was very small. Alice found the key on the glass table. " +
    "She said it was curious and the rabbit hurried away down the hall. ".repeat(6),
  );
  const before = hashFile(await canvasShot(page, "80-ft-before"));
  await ft.getByRole("button", { name: /Fine-tune/ }).click();
  await page.waitForTimeout(1500);
  log("progress visible:", await ft.locator(".progress, [data-testid=progress]").count());
  await page.waitForSelector('[data-testid="geo-finetune-loss"]', { timeout: 240000 });
  log("LOSS:", await page.locator('[data-testid="geo-finetune-loss"]').innerText());
  await page.waitForTimeout(3500);
  const after = hashFile(await canvasShot(page, "81-ft-after"));
  log("sphere changed?", before !== after);
  log("header:", (await headerText(page)).replace(/\n/g, " | "));
  log("weight-lab badge:", (await page.locator('[data-testid="geo-weight-panel"] .head').innerText()).replace(/\n/g, " | "));
  await page.screenshot({ path: path.join(OUT, "82-ft-after-full.png") });

  // HF dataset tab: is it enabled?
  await ft.getByRole("tab", { name: "HF dataset" }).click();
  await page.waitForTimeout(300);
  const hfTab = ft.getByRole("tab", { name: "HF dataset" });
  log("HF tab disabled?", await hfTab.isDisabled(), "selected:", await hfTab.getAttribute("aria-selected"));
  // nonsense dataset first (fast fail)
  await ft.locator('input[type=text]').fill("definitely/not-a-real-dataset-xyz123");
  await ft.getByRole("button", { name: /Fine-tune/ }).click();
  await page.waitForTimeout(20000);
  log("nonsense-dataset error:", await ft.locator('[data-testid="geo-error"]').count() ? await ft.locator('[data-testid="geo-error"]').innerText() : "(none yet)");
  await page.screenshot({ path: path.join(OUT, "83-ft-bad-dataset.png") });
  // real dataset
  await ft.locator('input[type=text]').fill("roneneldan/TinyStories");
  await ft.locator('input[type=range]').fill("50");
  await ft.getByRole("button", { name: /Fine-tune/ }).click();
  try {
    await page.waitForSelector('[data-testid="geo-finetune-loss"]', { timeout: 300000 });
    log("HF LOSS:", await page.locator('[data-testid="geo-finetune-loss"]').innerText());
  } catch {
    log("HF fine-tune did NOT produce a loss in 300s. error:",
      await ft.locator('[data-testid="geo-error"]').count() ? await ft.locator('[data-testid="geo-error"]').innerText() : "(none)");
  }
  log("header after HF:", (await headerText(page)).replace(/\n/g, " | "));
  await page.screenshot({ path: path.join(OUT, "84-ft-hf.png") });
  log("CONSOLE:", JSON.stringify(errs));
  await ctx.close();
}

// ------------------------------------------------------------------ train from scratch
if (STAGE === "train") {
  const { ctx, page, errs } = await openGeo(browser);
  log("== TRAIN FROM SCRATCH ==");
  const tp = page.locator('[data-testid="geo-train"]');
  log("panel:", (await tp.innerText()).replace(/\n/g, " | "));
  // short text gate
  await page.locator('[data-testid="geo-train-text"]').fill("the cat sat on the mat and the dog ran away fast");
  await page.waitForTimeout(900);
  log("SHORT stats:", await page.locator('[data-testid="geo-train-stats"]').innerText());
  log("SHORT button disabled?", await page.locator('[data-testid="geo-train-run"]').isDisabled());
  await page.screenshot({ path: path.join(OUT, "90-train-short.png") });

  // big corpus
  const corpus = fs.readFileSync("/Users/jmanning/llm-geometry/code/backend/src/llm_geometry/geo/data/alice-in-wonderland.txt", "utf8");
  log("corpus chars:", corpus.length);
  await page.locator('[data-testid="geo-train-text"]').fill(corpus);
  await page.waitForTimeout(1500);
  log("BIG stats:", await page.locator('[data-testid="geo-train-stats"]').innerText());
  log("BIG button disabled?", await page.locator('[data-testid="geo-train-run"]').isDisabled());
  await page.locator('[data-testid="geo-train-epochs"]').fill("3");
  const before = hashFile(await canvasShot(page, "91-train-before"));
  const beforeHdr = (await headerText(page)).replace(/\n/g, " | ");
  log("header BEFORE training:", beforeHdr);
  await page.locator('[data-testid="geo-train-run"]').click();
  await page.waitForTimeout(2500);
  log("progress element count:", await tp.locator("progress, .bar, [data-testid=progress]").count());
  log("progress text:", (await tp.innerText()).split("\n").filter((l) => /epoch|%|training|submitting/i.test(l)).join(" | "));
  await page.screenshot({ path: path.join(OUT, "92-train-running.png") });
  try {
    await page.waitForSelector('[data-testid="geo-train-result"]', { timeout: 900000 });
  } catch (e) {
    log("TRAIN TIMED OUT. error:", await tp.locator('[data-testid="geo-train-error"]').count() ? await tp.locator('[data-testid="geo-train-error"]').innerText() : "(none)");
    log("CONSOLE:", JSON.stringify(errs)); await ctx.close(); await browser.close(); process.exit(0);
  }
  log("RESULT:", await page.locator('[data-testid="geo-train-result"]').innerText());
  await page.waitForTimeout(5000);
  const after = hashFile(await canvasShot(page, "93-train-after"));
  log("sphere changed?", before !== after);
  const afterHdr = (await headerText(page)).replace(/\n/g, " | ");
  log("header AFTER training:", afterHdr);
  log("HEADER still claims shipped checkpoint?", afterHdr.includes("shipped checkpoint"), "| claims alice corpus?", afterHdr.includes("gutenberg-11-alice"), "| claims final loss 4.89?", afterHdr.includes("final loss 4.89"));
  log("weight-lab badge:", (await page.locator('[data-testid="geo-weight-panel"] .head').innerText()).replace(/\n/g, " | "));
  await page.screenshot({ path: path.join(OUT, "94-train-after-full.png") });

  // token labels under the new vocab
  const box = await page.locator('[data-testid="geo-canvas"] canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const hits = [];
  for (let dx = -200; dx <= 200 && hits.length < 8; dx += 9)
    for (let dy = -200; dy <= 200 && hits.length < 8; dy += 9) {
      await page.mouse.move(cx + dx, cy + dy); await page.waitForTimeout(14);
      const t = await page.evaluate(() => { const e = document.querySelector('[data-testid="hover-tooltip"]'); return e ? e.textContent.trim() : null; });
      if (t) hits.push(t);
    }
  log("HOVER labels under NEW vocab:", JSON.stringify(hits));
  // prompt / tokenize under new vocab
  await page.locator('[data-testid="geo-prompt"]').fill("alice rabbit queen");
  await page.waitForTimeout(2500);
  log("tokenize strip (new vocab):", await page.locator('[data-testid="geo-tokenize-strip"]').count() ? (await page.locator('[data-testid="geo-tokenize-strip"]').innerText()).replace(/\n/g, " ") : "(none)");
  log("attention/topk:", (await page.locator('[data-testid="geo-attention"]').innerText()).replace(/\n/g, " | ").slice(0, 350));

  // SAVE
  const [dl] = await Promise.all([
    page.waitForEvent("download", { timeout: 60000 }),
    page.locator('[data-testid="geo-save-model"]').click(),
  ]);
  const saved = path.join(OUT, "model.llmgeo.json");
  await dl.saveAs(saved);
  log("DOWNLOAD name:", dl.suggestedFilename(), "bytes:", fs.statSync(saved).size);
  await page.waitForTimeout(700);
  log("io note:", await page.locator('[data-testid="geo-io-note"]').count() ? await page.locator('[data-testid="geo-io-note"]').innerText() : "(none)");
  const bundle = JSON.parse(fs.readFileSync(saved, "utf8"));
  log("bundle keys:", Object.keys(bundle).join(","));
  log("bundle weights_token:", bundle.weights_token, "vocab len:", bundle.vocab?.length ?? bundle.vocab_size);
  log("CONSOLE:", JSON.stringify(errs));
  await ctx.close();
}

// ------------------------------------------------------------------ load / corrupt
if (STAGE === "load") {
  const saved = path.join(OUT, "model.llmgeo.json");
  const bundle = JSON.parse(fs.readFileSync(saved, "utf8"));
  // corrupt: flip one weight
  const bad = JSON.parse(JSON.stringify(bundle));
  const findArr = (o, p = "") => {
    for (const k of Object.keys(o)) {
      const v = o[k];
      if (Array.isArray(v) && Array.isArray(v[0]) && typeof v[0][0] === "number") return `${p}${k}`;
      if (Array.isArray(v) && typeof v[0] === "number") return `${p}${k}`;
      if (v && typeof v === "object" && !Array.isArray(v)) { const r = findArr(v, `${p}${k}.`); if (r) return r; }
    }
    return null;
  };
  log("first numeric array path:", findArr(bad));
  // mutate embedding[0][0] if present
  const paths = ["embedding", "weights.embedding", "model.embedding", "tensors.embedding"];
  let mutated = null;
  const walk = (o, cb, p = "") => { for (const k of Object.keys(o)) { const v = o[k]; if (Array.isArray(v) && Array.isArray(v[0]) && typeof v[0][0] === "number") cb(`${p}${k}`, v); else if (v && typeof v === "object" && !Array.isArray(v)) walk(v, cb, `${p}${k}.`); } };
  walk(bad, (p, v) => { if (!mutated) { v[0][0] = v[0][0] + 1.2345; mutated = p; } });
  log("mutated tensor:", mutated);
  const badPath = path.join(OUT, "model-corrupt-weight.llmgeo.json");
  fs.writeFileSync(badPath, JSON.stringify(bad));
  // corrupt hash
  const badHash = JSON.parse(JSON.stringify(bundle));
  const hkey = Object.keys(badHash).find((k) => /hash|token|checksum|digest/i.test(k));
  log("hash-ish keys:", Object.keys(badHash).filter((k) => /hash|token|checksum|digest/i.test(k)).join(","));
  if (hkey) badHash[hkey] = "0".repeat(String(badHash[hkey]).length);
  const badHashPath = path.join(OUT, "model-corrupt-hash.llmgeo.json");
  fs.writeFileSync(badHashPath, JSON.stringify(badHash));
  const notJson = path.join(OUT, "not-json.llmgeo.json");
  fs.writeFileSync(notJson, "this is definitely not json {{{");

  const { ctx, page, errs } = await openGeo(browser);
  log("== LOAD (fresh page) ==");
  log("header before load:", (await headerText(page)).replace(/\n/g, " | "));
  const beforeHash = hashFile(await canvasShot(page, "A0-load-before"));

  const tryLoad = async (p, tag) => {
    await page.locator('[data-testid="geo-load-model-input"]').setInputFiles(p);
    await page.waitForTimeout(6000);
    const e = await page.locator('[data-testid="geo-io-error"]').count() ? await page.locator('[data-testid="geo-io-error"]').innerText() : "";
    const n = await page.locator('[data-testid="geo-io-note"]').count() ? await page.locator('[data-testid="geo-io-note"]').innerText() : "";
    const hdr = (await headerText(page)).replace(/\n/g, " | ");
    const h = hashFile(await canvasShot(page, `A1-load-${tag}`));
    log(`LOAD ${tag}: error="${e}" note="${n}" spherechanged=${h !== beforeHash}`);
    log(`   header: ${hdr}`);
    await page.screenshot({ path: path.join(OUT, `A2-load-${tag}-full.png`) });
    return h;
  };
  await tryLoad(notJson, "notjson");
  await tryLoad(badHashPath, "corrupt-hash");
  await tryLoad(badPath, "corrupt-weight");
  const good = await tryLoad(saved, "good");
  log("good load restored a different sphere than shipped?", good !== beforeHash);
  // hover labels after restore
  const box = await page.locator('[data-testid="geo-canvas"] canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const hits = [];
  for (let dx = -180; dx <= 180 && hits.length < 6; dx += 11)
    for (let dy = -180; dy <= 180 && hits.length < 6; dy += 11) {
      await page.mouse.move(cx + dx, cy + dy); await page.waitForTimeout(14);
      const t = await page.evaluate(() => { const e = document.querySelector('[data-testid="hover-tooltip"]'); return e ? e.textContent.trim() : null; });
      if (t) hits.push(t);
    }
  log("HOVER after load:", JSON.stringify(hits));
  log("CONSOLE:", JSON.stringify(errs));
  await ctx.close();
}

await browser.close();
log("DONE", STAGE);
