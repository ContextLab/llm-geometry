// Red-team probe of the STATIC GitHub-Pages build. Temporary; deleted when done.
import { chromium } from "@playwright/test";
import fs from "node:fs";

const BASE = "http://localhost:4173/llm-geometry/";
const OUT = "/tmp/redteam-static";
fs.mkdirSync(OUT, { recursive: true });

const PHASE = process.argv[2] ?? "recon";

export function mkRecorder(page, log) {
  page.on("request", (r) => log.requests.push(`${r.method()} ${r.url()}`));
  page.on("requestfailed", (r) =>
    log.failed.push(`${r.method()} ${r.url()} :: ${r.failure()?.errorText}`),
  );
  page.on("response", (r) => {
    if (r.status() >= 400) log.badStatus.push(`${r.status()} ${r.url()}`);
  });
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning")
      log.console.push(`[${m.type()}] ${m.text()}`.slice(0, 400));
  });
  page.on("pageerror", (e) => log.pageErrors.push(String(e).slice(0, 400)));
}

function newLog() {
  return { requests: [], failed: [], badStatus: [], console: [], pageErrors: [] };
}

function hosts(log) {
  const h = {};
  for (const r of log.requests) {
    try {
      const u = new URL(r.split(" ")[1]);
      h[u.host] = (h[u.host] ?? 0) + 1;
    } catch {}
  }
  return h;
}

async function dumpText(page, name) {
  const txt = await page.evaluate(() => document.body.innerText);
  fs.writeFileSync(`${OUT}/${name}.txt`, txt);
  return txt;
}

async function shot(page, name, full = true) {
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: full });
}

// ---------------------------------------------------------------- recon
async function recon() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await ctx.newPage();
  const log = newLog();
  mkRecorder(page, log);

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(3000);
  await shot(page, "01-arch-initial");
  const archText = await dumpText(page, "01-arch-initial");

  // enumerate every interactive control on the arch tab
  const archControls = await page.evaluate(() => {
    const out = [];
    document
      .querySelectorAll("button, input, select, textarea, a, [role=button], summary")
      .forEach((el, i) => {
        const r = el.getBoundingClientRect();
        out.push({
          i,
          tag: el.tagName.toLowerCase(),
          type: el.getAttribute("type"),
          testid: el.getAttribute("data-testid"),
          text: (el.innerText || el.value || el.placeholder || "").slice(0, 70).replace(/\n/g, " | "),
          href: el.getAttribute("href"),
          disabled: el.disabled ?? null,
          visible: r.width > 0 && r.height > 0,
          title: el.getAttribute("title"),
        });
      });
    return out;
  });
  fs.writeFileSync(`${OUT}/arch-controls.json`, JSON.stringify(archControls, null, 2));

  // switch to geometry
  await page.getByTestId("tab-geometry").click();
  await page.waitForTimeout(6000);
  await shot(page, "02-geo-initial");
  const geoText = await dumpText(page, "02-geo-initial");

  const geoControls = await page.evaluate(() => {
    const out = [];
    document
      .querySelectorAll("button, input, select, textarea, a, [role=button], summary")
      .forEach((el, i) => {
        const r = el.getBoundingClientRect();
        out.push({
          i,
          tag: el.tagName.toLowerCase(),
          type: el.getAttribute("type"),
          testid: el.getAttribute("data-testid"),
          text: (el.innerText || el.value || el.placeholder || "").slice(0, 70).replace(/\n/g, " | "),
          href: el.getAttribute("href"),
          disabled: el.disabled ?? null,
          visible: r.width > 0 && r.height > 0,
          title: el.getAttribute("title"),
        });
      });
    return out;
  });
  fs.writeFileSync(`${OUT}/geo-controls.json`, JSON.stringify(geoControls, null, 2));

  // all links anywhere
  const links = await page.evaluate(() =>
    [...document.querySelectorAll("a[href]")].map((a) => a.href),
  );

  fs.writeFileSync(
    `${OUT}/recon-log.json`,
    JSON.stringify({ ...log, hosts: hosts(log), links: [...new Set(links)] }, null, 2),
  );
  console.log("HOSTS:", JSON.stringify(hosts(log), null, 2));
  console.log("API-ish requests:", log.requests.filter((r) => /:8000|\/api\//.test(r)));
  console.log("FAILED:", log.failed);
  console.log("BAD STATUS:", log.badStatus);
  console.log("CONSOLE:", log.console.slice(0, 30));
  console.log("PAGE ERRORS:", log.pageErrors);
  console.log("\n=== ARCH TEXT ===\n", archText);
  console.log("\n=== GEO TEXT ===\n", geoText);
  await browser.close();
}

// ---------------------------------------------------------------- arch
async function arch() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const log = newLog();
  mkRecorder(page, log);
  const notes = [];
  const N = (s) => { notes.push(s); console.log("  * " + s); };

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);

  // --- 1. free-form prompt: designed note or blank/error? -----------------------
  console.log("\n## free-form prompt");
  const ta = page.locator("textarea").first();
  await ta.fill("Tell me about octopuses in the deep sea");
  await page.waitForTimeout(2500);
  await shot(page, "10-arch-freeform");
  const note = page.getByTestId("arch-static-note");
  N(`static-note visible=${await note.count()} text=${await note.count() ? JSON.stringify((await note.innerText()).slice(0, 400)) : "-"}`);
  const errs = page.getByTestId("arch-error");
  N(`arch-error count=${await errs.count()} ${await errs.count() ? JSON.stringify(await errs.first().innerText()) : ""}`);
  N(`trace strip still present? ${await page.getByTestId("arch-trace-strip").count()}`);

  // --- 2. example prompt dropdown ------------------------------------------------
  console.log("\n## example prompt dropdown");
  const promptSel = page.locator("select").first();
  const opts = await promptSel.locator("option").allTextContents();
  N(`prompt dropdown options: ${JSON.stringify(opts)}`);
  const before = await page.getByTestId("arch-trace-strip").innerText().catch(() => "(none)");
  for (const o of opts.slice(1)) {
    await promptSel.selectOption({ label: o });
    await page.waitForTimeout(2500);
    const strip = await page.getByTestId("arch-trace-strip").innerText().catch(() => "(none)");
    const nerr = await page.getByTestId("arch-error").count();
    N(`preset "${o}": strip=${JSON.stringify(strip.replace(/\n/g, " ").slice(0, 120))} errors=${nerr}`);
  }
  await shot(page, "11-arch-preset2");

  // --- 3. play the trace ----------------------------------------------------------
  console.log("\n## trace playback");
  const diagBefore = await page.getByTestId("pipeline-diagram").innerHTML();
  await page.getByTestId("arch-play").click();
  await page.waitForTimeout(1800);
  await shot(page, "12-arch-playing");
  const diagMid = await page.getByTestId("pipeline-diagram").innerHTML();
  N(`diagram HTML changed during playback: ${diagBefore !== diagMid}`);
  await page.getByTestId("arch-play").click();
  await page.waitForTimeout(300);

  // scrub slider
  const scrub = page.locator('input[type=range]').first();
  const box = await scrub.boundingBox();
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height / 2);
  await page.mouse.down(); await page.mouse.move(box.x + box.width * 0.7, box.y + box.height / 2); await page.mouse.up();
  await page.waitForTimeout(800);
  const diagScrub = await page.getByTestId("pipeline-diagram").innerHTML();
  N(`diagram changed on scrub: ${diagScrub !== diagMid}`);
  await shot(page, "13-arch-scrubbed");

  // speed selector
  const speed = page.locator("select").last();
  N(`speed options: ${JSON.stringify(await speed.locator("option").allTextContents())}`);

  // --- 4. layer slider + head grid ------------------------------------------------
  console.log("\n## layer slider + heads");
  const heads = page.getByTestId("arch-head-grid").locator("> *");
  N(`attention head tiles rendered: ${await heads.count()}`);
  const ranges = page.locator("input[type=range]");
  N(`range inputs on page: ${await ranges.count()}`);
  // layer slider is the 2nd range
  const layerSlider = ranges.nth(1);
  const attBefore = await page.getByTestId("arch-head-tile-0").innerHTML().catch(() => "");
  await layerSlider.evaluate((el) => { el.value = "7"; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); });
  await page.waitForTimeout(1200);
  const attAfter = await page.getByTestId("arch-head-tile-0").innerHTML().catch(() => "");
  N(`layer slider -> head-tile-0 changed: ${attBefore !== attAfter}`);
  await shot(page, "14-arch-layer7");
  const bodyTxt = await page.evaluate(() => document.body.innerText);
  N(`layer label now: ${JSON.stringify((bodyTxt.match(/layer \d+ \/ \d+/) || [])[0] || "?")}`);

  // --- 5. inspector: click a diagram node -> real weights from HF -----------------
  console.log("\n## weight inspector (HF range reads)");
  const beforeReqs = log.requests.length;
  const node = page.locator('[data-testid^="diagram-node-"]').first();
  await node.click({ force: true });
  await page.waitForTimeout(6000);
  await shot(page, "15-arch-inspector");
  const insp = page.getByTestId("arch-inspector");
  N(`inspector visible=${await insp.count()}`);
  if (await insp.count()) {
    const t = await insp.innerText();
    fs.writeFileSync(`${OUT}/15-inspector.txt`, t);
    N(`inspector text: ${JSON.stringify(t.replace(/\n/g, " | ").slice(0, 600))}`);
  }
  const newReqs = log.requests.slice(beforeReqs);
  N(`requests during inspect: ${JSON.stringify(newReqs.slice(0, 12))}`);

  // click several nodes to find one with weights
  for (const id of ["wte", "lm_head", "attn", "mlp"]) {
    const n2 = page.locator(`[data-testid^="diagram-node-"]`).filter({ hasText: /projection|embedding|head/i }).first();
    if (await n2.count()) break;
  }

  fs.writeFileSync(`${OUT}/arch-log.json`, JSON.stringify({ ...log, hosts: hosts(log), notes }, null, 2));
  console.log("\nHOSTS:", JSON.stringify(hosts(log)));
  console.log("API-ish:", log.requests.filter((r) => /:8000|\/api\//.test(r)));
  console.log("FAILED:", log.failed);
  console.log("BAD STATUS:", log.badStatus);
  console.log("CONSOLE:", log.console.slice(0, 20));
  console.log("PAGE ERRORS:", log.pageErrors);
  await browser.close();
}

// ------------------------------------------------------- arch2 (inspector, sliders, chat)
async function arch2() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const log = newLog();
  mkRecorder(page, log);
  const notes = [];
  const N = (s) => { notes.push(s); console.log("  * " + s); };

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);

  // enumerate every range + select with identity
  const ctrls = await page.evaluate(() =>
    [...document.querySelectorAll("input[type=range], select")].map((el, i) => ({
      i, tag: el.tagName, testid: el.dataset.testid,
      min: el.min, max: el.max, value: el.value,
      label: el.closest("label")?.innerText?.slice(0, 40) ??
             el.previousElementSibling?.innerText?.slice(0, 40) ?? "",
      opts: el.tagName === "SELECT" ? [...el.options].map((o) => o.text) : undefined,
    })),
  );
  fs.writeFileSync(`${OUT}/arch-ranges.json`, JSON.stringify(ctrls, null, 2));
  console.log("CONTROLS:", JSON.stringify(ctrls, null, 1));

  // --- layer slider: find the one whose max is 11 -------------------------------
  const layerIdx = ctrls.findIndex((c) => c.tag === "INPUT" && c.max === "11");
  N(`layer slider index=${layerIdx}`);
  const headBefore = await page.getByTestId("arch-head-tile-3").innerHTML().catch(() => "");
  const labelBefore = await page.evaluate(() => (document.body.innerText.match(/layer \d+ \/ \d+/) || [])[0]);
  if (layerIdx >= 0) {
    await page.locator("input[type=range]").nth(
      ctrls.slice(0, layerIdx).filter((c) => c.tag === "INPUT").length,
    ).evaluate((el) => { el.value = "9"; el.dispatchEvent(new Event("input", { bubbles: true })); });
    await page.waitForTimeout(1200);
  }
  const headAfter = await page.getByTestId("arch-head-tile-3").innerHTML().catch(() => "");
  const labelAfter = await page.evaluate(() => (document.body.innerText.match(/layer \d+ \/ \d+/) || [])[0]);
  N(`layer slider: label ${labelBefore} -> ${labelAfter}; head tile changed=${headBefore !== headAfter}`);
  await shot(page, "20-arch-layer9");

  // --- residual/attention panel text ---------------------------------------------
  const panelTxt = await page.getByTestId("arch-breakdown").innerText();
  fs.writeFileSync(`${OUT}/20-breakdown.txt`, panelTxt);

  // --- inspector: click nodes (scroll into view first) ----------------------------
  console.log("\n## inspector");
  const nodeIds = await page.evaluate(() =>
    [...document.querySelectorAll('[data-testid^="diagram-node-"]')].map((e) => e.dataset.testid),
  );
  N(`diagram nodes: ${nodeIds.length} -> ${JSON.stringify(nodeIds.slice(0, 20))}`);

  for (const tid of [nodeIds[0], nodeIds.find((t) => /wte|embed/.test(t)), nodeIds.find((t) => /c_attn|attn|qkv/i.test(t)), nodeIds.find((t) => /lm_head/i.test(t))].filter(Boolean)) {
    const mark = log.requests.length;
    await page.evaluate((t) => {
      const el = document.querySelector(`[data-testid="${t}"]`);
      el?.scrollIntoView({ block: "center" });
      el?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }, tid);
    await page.waitForTimeout(7000);
    const insp = page.getByTestId("arch-inspector");
    const txt = (await insp.count()) ? await insp.innerText() : "(no inspector)";
    const reqs = log.requests.slice(mark).filter((r) => !/localhost:4173\/llm-geometry\/(assets|static-data)/.test(r));
    N(`node ${tid}: inspector=${JSON.stringify(txt.replace(/\n/g, " | ").slice(0, 420))}`);
    N(`   remote requests: ${JSON.stringify(reqs.slice(0, 8))}`);
    fs.writeFileSync(`${OUT}/21-inspector-${tid.replace(/[^\w]/g, "_")}.txt`, txt);
    await shot(page, `21-inspector-${tid.replace(/[^\w]/g, "_")}`, false);
  }

  fs.writeFileSync(`${OUT}/arch2-log.json`, JSON.stringify({ ...log, hosts: hosts(log), notes }, null, 2));
  console.log("\nHOSTS:", JSON.stringify(hosts(log)));
  console.log("API-ish:", log.requests.filter((r) => /:8000|\/api\//.test(r)));
  console.log("FAILED:", log.failed);
  console.log("BAD:", log.badStatus);
  console.log("CONSOLE:", log.console.slice(0, 20));
  console.log("PAGEERR:", log.pageErrors);
  await browser.close();
}

// ------------------------------------------------------- arch3 (pixel diffs, HF range, chat)
async function arch3() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const log = newLog();
  mkRecorder(page, log);
  const notes = [];
  const N = (s) => { notes.push(s); console.log("  * " + s); };

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);

  const layerSlider = page.locator("input[type=range]").last();
  const grid = page.getByTestId("arch-head-grid");
  const setLayer = async (v) => {
    await layerSlider.evaluate((el, val) => {
      el.value = String(val);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, v);
    await page.waitForTimeout(1000);
  };

  // pixel-compare head grid across layers
  await setLayer(0);
  const g0 = await grid.screenshot({ path: `${OUT}/30-heads-L0.png` });
  await setLayer(9);
  const g9 = await grid.screenshot({ path: `${OUT}/30-heads-L9.png` });
  await setLayer(5);
  const g5 = await grid.screenshot({ path: `${OUT}/30-heads-L5.png` });
  N(`head grid pixels: L0 vs L9 differ=${!g0.equals(g9)}; L0 vs L5 differ=${!g0.equals(g5)}`);

  // residual bar chart per layer
  const resid = page.getByTestId("arch-breakdown");
  await setLayer(0);
  const r0 = await resid.screenshot();
  await setLayer(11);
  const r11 = await resid.screenshot({ path: `${OUT}/31-breakdown-L11.png` });
  N(`breakdown pixels L0 vs L11 differ=${!r0.equals(r11)}`);

  // click individual head tile -> big heatmap changes?
  const big0 = await page.locator('[data-testid="matrix-heatmap"]').nth(12).screenshot().catch(() => null);
  await page.getByTestId("arch-head-tile-7").click();
  await page.waitForTimeout(800);
  const big7 = await page.locator('[data-testid="matrix-heatmap"]').nth(12).screenshot().catch(() => null);
  N(`head tile click -> enlarged heatmap changed=${big0 && big7 ? !big0.equals(big7) : "n/a"}`);
  await shot(page, "32-head7-selected");

  // --- inspector: zoom into an exact window (should hit huggingface.co) ---------
  console.log("\n## HF range read");
  await page.evaluate(() => {
    const el = document.querySelector('[data-testid="diagram-node-transformer.h.0.attn.c_attn"]');
    el?.scrollIntoView({ block: "center" });
    el?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await page.waitForTimeout(4000);
  const mark = log.requests.length;
  const hm = page.getByTestId("arch-inspector").locator('[data-testid="matrix-heatmap"]').first();
  const hb = await hm.boundingBox();
  N(`inspector heatmap box=${JSON.stringify(hb)}`);
  await page.mouse.click(hb.x + hb.width * 0.4, hb.y + hb.height * 0.4);
  await page.waitForTimeout(10000);
  await shot(page, "33-inspector-zoomed", false);
  const inspTxt = await page.getByTestId("arch-inspector").innerText();
  fs.writeFileSync(`${OUT}/33-inspector-zoomed.txt`, inspTxt);
  N(`after zoom click: ${JSON.stringify(inspTxt.replace(/\n/g, " | ").slice(0, 500))}`);
  const remote = log.requests.slice(mark).filter((r) => !r.includes("localhost:4173"));
  N(`remote requests after zoom: ${JSON.stringify(remote)}`);

  // hover a cell for a value
  await page.mouse.move(hb.x + hb.width * 0.5, hb.y + hb.height * 0.5);
  await page.waitForTimeout(900);
  const tip = page.getByTestId("hover-tooltip");
  N(`hover tooltip: ${(await tip.count()) ? JSON.stringify(await tip.innerText()) : "(none)"}`);
  await shot(page, "34-inspector-hover", false);

  // Esc closes
  await page.keyboard.press("Escape");
  await page.waitForTimeout(500);
  N(`Esc closed inspector: ${(await page.getByTestId("arch-inspector").count()) === 0}`);

  fs.writeFileSync(`${OUT}/arch3-log.json`, JSON.stringify({ ...log, hosts: hosts(log), notes }, null, 2));
  console.log("HOSTS:", JSON.stringify(hosts(log)));
  console.log("FAILED:", log.failed, "BAD:", log.badStatus);
  console.log("CONSOLE:", log.console.slice(0, 15), "PAGEERR:", log.pageErrors);
  await browser.close();
}

// ------------------------------------------------------- chat (live transformers.js)
async function chat() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const log = newLog();
  mkRecorder(page, log);
  const notes = [];
  const N = (s) => { notes.push(s); console.log("  * " + s); };

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(6000);
  await shot(page, "39-chat-preload");
  N(`ranges present at t=6s: ${await page.locator("input[type=range]").count()}`);
  N(`body head: ${JSON.stringify((await page.evaluate(() => document.body.innerText)).slice(0, 300))}`);
  N(`html: ${JSON.stringify((await page.content()).slice(0, 600))}`);
  N(`EARLY console: ${JSON.stringify(log.console.slice(0, 8))}`);
  N(`EARLY pageErrors: ${JSON.stringify(log.pageErrors)}`);
  N(`EARLY failed: ${JSON.stringify(log.failed)}`);
  N(`EARLY bad: ${JSON.stringify(log.badStatus)}`);
  N(`EARLY reqs: ${JSON.stringify(log.requests)}`);

  const badge = page.getByTestId("static-runtime-badge");
  N(`runtime badge before: ${(await badge.count()) ? JSON.stringify(await badge.innerText()) + " title=" + JSON.stringify(await badge.getAttribute("title")) : "(none)"}`);

  // deterministic-ish: temperature to 0
  await page.locator("input[type=range]").nth(0).evaluate((el) => {
    el.value = "0"; el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.locator("input[type=range]").nth(1).evaluate((el) => {
    el.value = "32"; el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForTimeout(300);
  N(`temp/maxtok labels: ${JSON.stringify((await page.evaluate(() => document.body.innerText)).match(/temperature [\d.]+|max tokens \d+/g))}`);

  const t0 = Date.now();
  await page.getByTestId("arch-generate").click();
  await shot(page, "40-chat-busy", false);
  try {
    await page.getByTestId("arch-reply").waitFor({ timeout: 600000 });
  } catch (e) {
    N(`NO REPLY within 600s: ${e.message.slice(0, 120)}`);
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(0);
  await page.waitForTimeout(1500);
  await shot(page, "41-chat-reply");
  const reply = (await page.getByTestId("arch-reply").count()) ? await page.getByTestId("arch-reply").innerText() : "(none)";
  N(`generation took ${secs}s`);
  N(`REPLY: ${JSON.stringify(reply.replace(/\n/g, " ").slice(0, 800))}`);
  const err = page.getByTestId("arch-error");
  N(`errors: ${(await err.count()) ? JSON.stringify(await err.first().innerText()) : "none"}`);
  N(`runtime badge after: ${(await badge.count()) ? JSON.stringify(await badge.innerText()) + " title=" + JSON.stringify(await badge.getAttribute("title")) : "(none)"}`);

  // hover a generated token for probabilities
  const tok = page.getByTestId("arch-reply").locator("span").nth(3);
  if (await tok.count()) {
    await tok.hover();
    await page.waitForTimeout(900);
    const tip = page.getByTestId("hover-tooltip");
    N(`token hover tooltip: ${(await tip.count()) ? JSON.stringify((await tip.innerText()).replace(/\n/g, " | ")) : "(none)"}`);
    await shot(page, "42-chat-tokhover", false);
  }

  const remoteHosts = hosts(log);
  fs.writeFileSync(`${OUT}/chat-log.json`, JSON.stringify({ hosts: remoteHosts, notes, failed: log.failed, bad: log.badStatus, console: log.console, pageErrors: log.pageErrors, remote: log.requests.filter((r) => !r.includes("localhost:4173")).slice(0, 60) }, null, 2));
  console.log("HOSTS:", JSON.stringify(remoteHosts));
  console.log("API-ish:", log.requests.filter((r) => /:8000|\/api\//.test(r)));
  console.log("FAILED:", log.failed.slice(0, 10));
  console.log("BAD:", log.badStatus.slice(0, 10));
  console.log("CONSOLE:", log.console.slice(0, 15));
  console.log("PAGEERR:", log.pageErrors);
  await browser.close();
}

// ------------------------------------------------------- geo1 (view controls)
async function geo1() {
  const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const log = newLog();
  mkRecorder(page, log);
  const notes = [];
  const N = (s) => { notes.push(s); console.log("  * " + s); };

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.getByTestId("tab-geometry").click();
  await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 60000 });
  await page.waitForTimeout(4000);

  const canvas = page.getByTestId("geo-canvas");
  const snap = async (tag) => { await canvas.screenshot({ path: `${OUT}/${tag}.png` }); return fs.readFileSync(`${OUT}/${tag}.png`); };

  // --- spin toggle default state -------------------------------------------------
  const spin = page.getByTestId("geo-autorotate");
  N(`spin button: text=${JSON.stringify(await spin.innerText())} aria-pressed=${await spin.getAttribute("aria-pressed")} class=${await spin.getAttribute("class")} title=${JSON.stringify(await spin.getAttribute("title"))}`);
  const s1 = await snap("50-geo-t0");
  await page.waitForTimeout(2500);
  const s2 = await snap("50-geo-t1");
  N(`sphere STILL when spin off (frames identical): ${s1.equals(s2)}`);
  await spin.click();
  await page.waitForTimeout(2000);
  const s3 = await snap("51-geo-spinning");
  N(`spin ON changes the frame: ${!s2.equals(s3)}`);
  await spin.click();
  await page.waitForTimeout(200);

  // --- field mode: next-next vs force -------------------------------------------
  const modeBtns = page.getByTestId("geo-mode").locator("button");
  N(`field mode buttons: ${JSON.stringify(await modeBtns.allTextContents())}`);
  const fBefore = await snap("52-geo-nextnext");
  await modeBtns.nth(1).click();
  await page.waitForTimeout(2500);
  const fAfter = await snap("53-geo-force");
  N(`force mode changes the render: ${!fBefore.equals(fAfter)}`);
  const badges = await page.evaluate(() =>
    [...document.querySelectorAll('[data-testid^="geo-"][data-testid$="-badge"]')].map((e) => ({
      id: e.dataset.testid, text: e.innerText, title: e.title,
    })),
  );
  N(`force-mode badges: ${JSON.stringify(badges)}`);
  const caption = await page.evaluate(() => {
    const el = [...document.querySelectorAll("p,span")].find((e) => /each arrow|aggregate|force/i.test(e.innerText) && e.innerText.length < 300 && e.children.length === 0);
    return el?.innerText ?? null;
  });
  N(`field caption in force mode: ${JSON.stringify(caption)}`);

  // occlusion check: rotate 180 deg by dragging, screenshot the BACK
  const cb = await canvas.boundingBox();
  await page.mouse.move(cb.x + cb.width / 2, cb.y + cb.height / 2);
  await page.mouse.down();
  await page.mouse.move(cb.x + cb.width / 2 + 380, cb.y + cb.height / 2, { steps: 25 });
  await page.mouse.up();
  await page.waitForTimeout(2000);
  await snap("54-geo-force-back");
  await modeBtns.nth(0).click();
  await page.waitForTimeout(1500);
  await snap("55-geo-nextnext-back");

  // --- layer segmented ------------------------------------------------------------
  const layerBtns = page.getByTestId("geo-layer").locator("button");
  N(`layer buttons: ${JSON.stringify(await layerBtns.allTextContents())}`);
  const lPrev = await snap("56-geo-layerfull");
  await layerBtns.nth(2).click(); // layer 1
  await page.waitForTimeout(2500);
  const lNow = await snap("57-geo-layer1");
  N(`layer switch changes render: ${!lPrev.equals(lNow)}`);
  await layerBtns.nth(0).click();
  await page.waitForTimeout(1500);

  // --- sliders --------------------------------------------------------------------
  const ranges = page.locator("input[type=range]");
  const rmeta = await page.evaluate(() =>
    [...document.querySelectorAll("input[type=range]")].map((el) => ({
      min: el.min, max: el.max, value: el.value, testid: el.dataset.testid,
      label: el.closest("label")?.innerText?.replace(/\n/g, " ").slice(0, 40),
    })),
  );
  N(`geo ranges: ${JSON.stringify(rmeta)}`);
  const setRange = async (i, v) => {
    await ranges.nth(i).evaluate((el, val) => { el.value = String(val); el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); }, v);
    await page.waitForTimeout(2500);
  };
  const tBefore = await snap("58-geo-temp0");
  await setRange(0, 1.2); // temperature
  const tAfter = await snap("59-geo-temp12");
  N(`temperature slider changes render: ${!tBefore.equals(tAfter)}; label=${JSON.stringify((await page.evaluate(() => document.body.innerText)).match(/TEMPERATURE [\d.]+/i))}`);
  await setRange(1, 4); // arrows/point
  const aAfter = await snap("60-geo-arrows4");
  N(`arrows/point changes render: ${!tAfter.equals(aAfter)}; label=${JSON.stringify((await page.evaluate(() => document.body.innerText)).match(/ARROWS\/POINT \d+/i))}`);
  await setRange(0, 0); await setRange(1, 1);

  // --- prompt ---------------------------------------------------------------------
  const pBefore = await snap("61-geo-prompt-before");
  await page.getByTestId("geo-prompt").fill("the rabbit hole went straight on like a tunnel");
  await page.waitForTimeout(3000);
  const pAfter = await snap("62-geo-prompt-after");
  N(`prompt change moves the field: ${!pBefore.equals(pAfter)}`);
  N(`token strip: ${JSON.stringify((await page.getByTestId("geo-tokenize-strip").innerText()).replace(/\n/g, " "))}`);
  const attnTxt = await page.getByTestId("geo-attention").innerText();
  N(`attention panel: ${JSON.stringify(attnTxt.replace(/\n/g, " | ").slice(0, 260))}`);

  // unknown word -> <unk>?
  await page.getByTestId("geo-prompt").fill("quantum chromodynamics zzzyx");
  await page.waitForTimeout(2500);
  N(`unknown-word token strip: ${JSON.stringify((await page.getByTestId("geo-tokenize-strip").innerText()).replace(/\n/g, " "))}`);
  N(`geo errors: ${await page.getByTestId("geo-error").count()}`);
  await shot(page, "63-geo-unknown");
  await page.getByTestId("geo-prompt").fill("alice was beginning to get very tired of sitting by her sister");
  await page.waitForTimeout(2500);

  // --- attention layer tabs -------------------------------------------------------
  const attnTabs = page.locator('[data-testid="geo-attention"]').locator("xpath=..").locator("button");
  const attnHeat = page.getByTestId("geo-attention").locator('[data-testid="matrix-heatmap"]').first();
  const a0 = await attnHeat.screenshot();
  const tabTexts = await attnTabs.allTextContents();
  N(`attention tabs: ${JSON.stringify(tabTexts)}`);
  if (tabTexts.length > 1) {
    await attnTabs.nth(2).click();
    await page.waitForTimeout(1500);
    const a2 = await attnHeat.screenshot({ path: `${OUT}/64-geo-attn-layer2.png` });
    N(`attention layer tab changes heatmap: ${!a0.equals(a2)}`);
  }

  // --- hover a dot -----------------------------------------------------------------
  await page.mouse.move(cb.x + cb.width / 2, cb.y + cb.height / 2);
  await page.waitForTimeout(400);
  for (const [dx, dy] of [[0, 0], [30, -20], [-40, 25], [60, 40], [-70, -35]]) {
    await page.mouse.move(cb.x + cb.width / 2 + dx, cb.y + cb.height / 2 + dy);
    await page.waitForTimeout(500);
    const tip = page.getByTestId("hover-tooltip");
    if (await tip.count()) { N(`sphere hover tooltip: ${JSON.stringify(await tip.innerText())}`); break; }
  }
  await shot(page, "65-geo-hover", false);

  fs.writeFileSync(`${OUT}/geo1-log.json`, JSON.stringify({ ...log, hosts: hosts(log), notes }, null, 2));
  console.log("HOSTS:", JSON.stringify(hosts(log)));
  console.log("API-ish:", log.requests.filter((r) => /:8000|\/api\//.test(r)));
  console.log("FAILED:", log.failed, "BAD:", log.badStatus);
  console.log("CONSOLE:", log.console.slice(0, 15));
  console.log("PAGEERR:", log.pageErrors);
  await browser.close();
}

// ------------------------------------------------------- geo2 (weight lab, hover, export)
async function geo2() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await ctx.newPage();
  const log = newLog();
  mkRecorder(page, log);
  const notes = [];
  const N = (s) => { notes.push(s); console.log("  * " + s); };

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.getByTestId("tab-geometry").click();
  await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 60000 });
  await page.waitForTimeout(4000);
  const canvas = page.getByTestId("geo-canvas");
  const cb = await canvas.boundingBox();

  // --- hover scan for a token dot ------------------------------------------------
  let found = null;
  outer:
  for (let gx = 0.25; gx <= 0.75; gx += 0.05) {
    for (let gy = 0.2; gy <= 0.8; gy += 0.08) {
      await page.mouse.move(cb.x + cb.width * gx, cb.y + cb.height * gy);
      await page.waitForTimeout(160);
      const tip = page.getByTestId("hover-tooltip");
      if (await tip.count()) { found = `${(await tip.innerText()).replace(/\n/g, " | ")} @ (${gx.toFixed(2)},${gy.toFixed(2)})`; break outer; }
    }
  }
  N(`sphere hover tooltip: ${found ? JSON.stringify(found) : "NONE FOUND across 143 probe points"}`);
  if (found) await shot(page, "70-geo-hover", false);

  // --- weight lab -----------------------------------------------------------------
  const shot3 = async (t) => { await canvas.screenshot({ path: `${OUT}/${t}.png` }); return fs.readFileSync(`${OUT}/${t}.png`); };
  const selMeta = await page.evaluate(() =>
    [...document.querySelectorAll('[data-testid="geo-weight-panel"] select')].map((s) => ({
      testid: s.dataset.testid, value: s.value, opts: [...s.options].map((o) => o.text),
    })),
  );
  N(`weight lab selects: ${JSON.stringify(selMeta)}`);
  N(`badge before: ${JSON.stringify(await page.locator('[data-testid="geo-weight-panel"] .badge').first().innerText())}`);

  const before = await shot3("71-geo-before-edit");
  await page.getByTestId("geo-preset").selectOption("random");
  await page.waitForTimeout(500);
  const seedSel = page.getByTestId("geo-seed");
  N(`seed selector shown for 'random': ${await seedSel.count()} opts=${(await seedSel.count()) ? JSON.stringify(await seedSel.locator("option").allTextContents()) : "-"} title=${(await seedSel.count()) ? JSON.stringify(await seedSel.getAttribute("title")) : "-"}`);
  await page.getByTestId("geo-apply").click();
  await page.waitForTimeout(4000);
  const after = await shot3("72-geo-after-random");
  N(`Apply random preset changes the sphere: ${!before.equals(after)}`);
  N(`weight panel head after apply: ${JSON.stringify((await page.getByTestId("geo-weight-panel").innerText()).replace(/\n/g, " | ").slice(0, 260))}`);
  N(`geo errors: ${(await page.getByTestId("geo-error").count()) ? JSON.stringify(await page.getByTestId("geo-error").first().innerText()) : "none"}`);
  await shot(page, "72-geo-after-random-full");

  // active-model chip
  N(`active model chip: ${JSON.stringify((await page.getByTestId("geo-active-model").innerText()).replace(/\n/g, " | "))}`);

  // --- unsupported seed? try each option -----------------------------------------
  if (await seedSel.count()) {
    const seeds = await seedSel.locator("option").allTextContents();
    for (const s of seeds.slice(0, 3)) {
      await seedSel.selectOption({ label: s });
      await page.getByTestId("geo-apply").click();
      await page.waitForTimeout(2500);
      N(`seed ${s}: err=${(await page.getByTestId("geo-error").count()) ? JSON.stringify(await page.getByTestId("geo-error").first().innerText()) : "none"}`);
    }
  }

  // --- cell edit -------------------------------------------------------------------
  await page.getByTestId("geo-preset").selectOption("learned");
  await page.getByTestId("geo-apply").click();
  await page.waitForTimeout(2500);
  const preCell = await shot3("73-geo-precell");
  const hm = page.getByTestId("geo-weight-panel").locator('[data-testid="matrix-heatmap"]').first();
  const hb = await hm.boundingBox();
  await page.mouse.click(hb.x + hb.width * 0.2, hb.y + hb.height * 0.2);
  await page.waitForTimeout(600);
  const editor = page.getByTestId("heatmap-cell-editor");
  N(`cell editor appears: ${await editor.count()}`);
  if (await editor.count()) {
    await shot(page, "74-geo-cell-editor", false);
    const inp = editor.locator("input").first();
    await inp.fill("2.5");
    await inp.press("Enter");
    await page.waitForTimeout(4000);
    const postCell = await shot3("75-geo-postcell");
    N(`cell edit changes the sphere: ${!preCell.equals(postCell)}`);
    N(`weight panel after cell edit: ${JSON.stringify((await page.getByTestId("geo-weight-panel").innerText()).replace(/\n/g, " | ").slice(0, 200))}`);
    N(`geo errors after cell edit: ${(await page.getByTestId("geo-error").count()) ? JSON.stringify(await page.getByTestId("geo-error").first().innerText()) : "none"}`);
  }

  // --- reset to learned ------------------------------------------------------------
  const reset = page.getByTestId("geo-reset");
  N(`reset button present: ${await reset.count()}`);
  if (await reset.count()) {
    await reset.click();
    await page.waitForTimeout(3000);
    N(`after reset badge: ${JSON.stringify((await page.getByTestId("geo-weight-panel").innerText()).replace(/\n/g, " | ").slice(0, 140))}`);
  }

  // --- embedding matrix (read-only ribbon) ------------------------------------------
  await page.getByTestId("geo-matrix").selectOption("embedding");
  await page.waitForTimeout(2500);
  N(`embedding view: ${JSON.stringify((await page.getByTestId("geo-weight-panel").innerText()).replace(/\n/g, " | ").slice(0, 240))}`);
  await shot(page, "76-geo-embedding");

  // --- Export PNG --------------------------------------------------------------------
  const dl = page.waitForEvent("download", { timeout: 30000 }).catch(() => null);
  const pngBtn = page.locator("button", { hasText: /^PNG$/ }).first();
  N(`export PNG button: ${await pngBtn.count()}`);
  if (await pngBtn.count()) {
    await pngBtn.click();
    const d = await dl;
    if (d) {
      const p = `${OUT}/77-export.png`;
      await d.saveAs(p);
      N(`export downloaded: ${d.suggestedFilename()} size=${fs.statSync(p).size}`);
    } else N("export produced NO download within 30s");
  }

  fs.writeFileSync(`${OUT}/geo2-log.json`, JSON.stringify({ ...log, hosts: hosts(log), notes }, null, 2));
  console.log("HOSTS:", JSON.stringify(hosts(log)));
  console.log("API-ish:", log.requests.filter((r) => /:8000|\/api\//.test(r)));
  console.log("FAILED:", log.failed, "BAD:", log.badStatus);
  console.log("CONSOLE:", log.console.slice(0, 12));
  console.log("PAGEERR:", log.pageErrors);
  await browser.close();
}

// ------------------------------------------------------- cell (focused cell-edit probe)
async function cell() {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 1100 } })).newPage();
  const log = newLog(); mkRecorder(page, log);
  const N = (s) => console.log("  * " + s);

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.getByTestId("tab-geometry").click();
  await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 60000 });
  await page.waitForTimeout(4000);

  const canvas = page.getByTestId("geo-canvas");
  const shot3 = async (t) => { await canvas.screenshot({ path: `${OUT}/${t}.png` }); return fs.readFileSync(`${OUT}/${t}.png`); };
  const hm = page.getByTestId("geo-weight-panel").locator('[data-testid="matrix-heatmap"]').first();
  N(`heatmap editable class: ${await hm.getAttribute("class")}`);
  const hb = await hm.boundingBox();
  N(`heatmap box: ${JSON.stringify(hb)}`);
  const pre = await shot3("80-cell-pre");
  await hm.click({ position: { x: hb.width / 6, y: hb.height / 6 } });
  await page.waitForTimeout(700);
  const ed = page.getByTestId("heatmap-cell-editor");
  N(`editor after click: ${await ed.count()}`);
  await shot(page, "81-cell-editor", false);
  if (await ed.count()) {
    N(`editor value: ${JSON.stringify(await ed.inputValue())}`);
    await ed.fill("3.75");
    await ed.press("Enter");
    await page.waitForTimeout(4500);
    const post = await shot3("82-cell-post");
    N(`sphere changed after cell edit: ${!pre.equals(post)}`);
    N(`panel: ${JSON.stringify((await page.getByTestId("geo-weight-panel").innerText()).replace(/\n/g, " | ").slice(0, 200))}`);
    N(`active model: ${JSON.stringify((await page.getByTestId("geo-active-model").innerText()).replace(/\n/g, " | "))}`);
    N(`errors: ${(await page.getByTestId("geo-error").count()) ? await page.getByTestId("geo-error").first().innerText() : "none"}`);
    await shot(page, "83-cell-after-full");

    // reload persistence
    await page.reload({ waitUntil: "networkidle" });
    await page.getByTestId("tab-geometry").click();
    await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 60000 });
    await page.waitForTimeout(4000);
    N(`AFTER RELOAD panel: ${JSON.stringify((await page.getByTestId("geo-weight-panel").innerText()).replace(/\n/g, " | ").slice(0, 200))}`);
    N(`AFTER RELOAD active model chip: ${JSON.stringify((await page.getByTestId("geo-active-model").innerText()).replace(/\n/g, " | "))}`);
    const post2 = await shot3("84-cell-after-reload");
    N(`sphere identical to edited state after reload: ${post.equals(post2)}`);
    await shot(page, "85-after-reload-full");
  }
  console.log("CONSOLE:", log.console.slice(0, 10), "PAGEERR:", log.pageErrors, "BAD:", log.badStatus);
  await browser.close();
}

// ------------------------------------------------------- attn (does the edit reach the trace?)
async function attn() {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 1100 } })).newPage();
  const log = newLog(); mkRecorder(page, log);
  const N = (s) => console.log("  * " + s);

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.getByTestId("tab-geometry").click();
  await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 60000 });
  await page.waitForTimeout(4000);

  const attnPanel = page.getByTestId("geo-attention");
  const heat = attnPanel.locator('[data-testid="matrix-heatmap"]').first();
  const read = async () => (await attnPanel.innerText()).replace(/\n/g, " ");
  const base = await read();
  const baseHeat = await heat.screenshot({ path: `${OUT}/90-attn-base.png` });
  N(`BASE dist: ${JSON.stringify(base.slice(0, 260))}`);

  const apply = async (matrix, preset, layer) => {
    await page.getByTestId("geo-matrix").selectOption(matrix);
    await page.waitForTimeout(400);
    if (layer !== undefined) {
      const ls = page.getByTestId("geo-weight-panel").locator("select").nth(1);
      if (await ls.count()) await ls.selectOption(String(layer));
    }
    await page.getByTestId("geo-preset").selectOption(preset);
    await page.waitForTimeout(300);
    await page.getByTestId("geo-apply").click();
    await page.waitForTimeout(5000);
  };

  for (const [m, p, l] of [["W_V", "zero", 0], ["W_Q", "zero", 0], ["W_O", "zero", 0], ["W_V", "random", 1]]) {
    await apply(m, p, l);
    const now = await read();
    const nowHeat = await heat.screenshot({ path: `${OUT}/91-attn-${m}-${p}-L${l}.png` });
    N(`${m}=${p} L${l}: dist CHANGED=${now !== base} | heat CHANGED=${!baseHeat.equals(nowHeat)}`);
    N(`   -> ${JSON.stringify(now.slice(0, 200))}`);
    N(`   panel badge: ${JSON.stringify((await page.getByTestId("geo-active-model").innerText()).replace(/\n/g, " | "))}`);
    N(`   errors: ${(await page.getByTestId("geo-error").count()) ? await page.getByTestId("geo-error").first().innerText() : "none"}`);
  }
  await shot(page, "92-attn-after-zeros");
  console.log("CONSOLE:", log.console.filter((c) => !/WebGL|GPU stall/.test(c)).slice(0, 10));
  console.log("PAGEERR:", log.pageErrors, "BAD:", log.badStatus);
  await browser.close();
}

// ------------------------------------------------------- ft (fine-tune: paste, file, HF dataset)
async function ft() {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 1100 } })).newPage();
  const log = newLog(); mkRecorder(page, log);
  const N = (s) => console.log("  * " + s);

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.getByTestId("tab-geometry").click();
  await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]', { timeout: 60000 });
  await page.waitForTimeout(4000);
  const canvas = page.getByTestId("geo-canvas");
  const snap = async (t) => { await canvas.screenshot({ path: `${OUT}/${t}.png` }); return fs.readFileSync(`${OUT}/${t}.png`); };
  const panel = page.getByTestId("geo-finetune");

  // --- 1. paste text -------------------------------------------------------------
  const before = await snap("100-ft-before");
  const tabs = panel.locator('[role=tab], .tabs button');
  N(`fine-tune tabs: ${JSON.stringify(await tabs.allTextContents())}`);
  const txt = panel.locator("textarea").first();
  await txt.fill("the white rabbit ran across the garden and down the hole. the queen shouted at the cards. alice grew very small and then very tall again. the cat smiled and vanished slowly.".repeat(6));
  await page.locator('[data-testid="geo-finetune"] input[type=range]').evaluate((el) => { el.value = "60"; el.dispatchEvent(new Event("input", { bubbles: true })); });
  await page.waitForTimeout(300);
  const t0 = Date.now();
  await panel.locator("button", { hasText: /Fine-tune/ }).first().click();
  await page.waitForTimeout(400);
  await shot(page, "101-ft-running", false);
  await page.getByTestId("geo-finetune-loss").waitFor({ timeout: 180000 }).catch((e) => N(`no loss chip: ${e.message.slice(0, 90)}`));
  await page.waitForTimeout(2500);
  N(`paste fine-tune took ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  N(`loss chip: ${(await page.getByTestId("geo-finetune-loss").count()) ? JSON.stringify((await page.getByTestId("geo-finetune-loss").innerText()).replace(/\n/g, " ")) : "(none)"}`);
  N(`panel: ${JSON.stringify((await panel.innerText()).replace(/\n/g, " | ").slice(0, 300))}`);
  N(`errors: ${(await page.getByTestId("geo-error").count()) ? await page.getByTestId("geo-error").first().innerText() : "none"}`);
  N(`active model: ${JSON.stringify((await page.getByTestId("geo-active-model").innerText()).replace(/\n/g, " | "))}`);
  const after = await snap("102-ft-after");
  N(`fine-tune moved the sphere: ${!before.equals(after)}`);
  await shot(page, "103-ft-done");

  // --- 2. HF dataset ---------------------------------------------------------------
  console.log("\n## HF DATASET fine-tune (roneneldan/TinyStories)");
  await page.getByTestId("geo-finetune-hf-tab").click();
  await page.waitForTimeout(400);
  await shot(page, "104-ft-hftab", false);
  N(`hf tab panel: ${JSON.stringify((await panel.innerText()).replace(/\n/g, " | ").slice(0, 320))}`);
  const hfInput = panel.locator('input[type=text]').first();
  N(`hf input placeholder: ${JSON.stringify(await hfInput.getAttribute("placeholder"))}`);
  await hfInput.fill("roneneldan/TinyStories");
  const mark = log.requests.length;
  const t1 = Date.now();
  await panel.locator("button", { hasText: /Fine-tune/ }).first().click();
  await page.waitForTimeout(1500);
  await shot(page, "105-ft-hf-running", false);
  const progressTxts = [];
  for (let i = 0; i < 60; i++) {
    const pr = page.getByTestId("progress");
    if (await pr.count()) progressTxts.push((await pr.innerText()).replace(/\n/g, " "));
    if (await page.getByTestId("geo-finetune-loss").count()) break;
    if (await page.getByTestId("geo-error").count()) break;
    await page.waitForTimeout(3000);
  }
  N(`HF fine-tune took ${((Date.now() - t1) / 1000).toFixed(1)}s`);
  N(`progress messages seen: ${JSON.stringify([...new Set(progressTxts)].slice(0, 8))}`);
  N(`loss chip: ${(await page.getByTestId("geo-finetune-loss").count()) ? JSON.stringify((await page.getByTestId("geo-finetune-loss").innerText()).replace(/\n/g, " ")) : "(none)"}`);
  N(`errors: ${(await page.getByTestId("geo-error").count()) ? JSON.stringify(await page.getByTestId("geo-error").first().innerText()) : "none"}`);
  N(`active model: ${JSON.stringify((await page.getByTestId("geo-active-model").innerText()).replace(/\n/g, " | "))}`);
  const hfReqs = log.requests.slice(mark).filter((r) => !r.includes("localhost:4173"));
  N(`dataset requests (${hfReqs.length}): ${JSON.stringify(hfReqs.slice(0, 5))}`);
  await shot(page, "106-ft-hf-done");

  // --- 3. bogus dataset id --------------------------------------------------------
  await hfInput.fill("definitely/not-a-real-dataset-xyzzy");
  await panel.locator("button", { hasText: /Fine-tune/ }).first().click();
  await page.waitForTimeout(15000);
  N(`bogus dataset error: ${(await page.getByTestId("geo-error").count()) ? JSON.stringify(await page.getByTestId("geo-error").first().innerText()) : "NO ERROR SHOWN"}`);
  await shot(page, "107-ft-bogus", false);

  console.log("HOSTS:", JSON.stringify(hosts(log)));
  console.log("API-ish:", log.requests.filter((r) => /:8000|localhost.*\/api\//.test(r)));
  console.log("FAILED:", log.failed.slice(0, 5), "BAD:", log.badStatus.slice(0, 5));
  console.log("CONSOLE:", log.console.filter((c) => !/WebGL|GPU stall/.test(c)).slice(0, 10));
  console.log("PAGEERR:", log.pageErrors);
  await browser.close();
}

const phases = { recon, arch, arch2, arch3, chat, geo1, geo2, cell, attn, ft };
await phases[PHASE]();
