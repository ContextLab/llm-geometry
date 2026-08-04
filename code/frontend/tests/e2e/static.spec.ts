import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// Static-site e2e (feature 003, FR-206): runs against the BUILT static bundle served
// by `npm run preview:static` on :4173 under the GitHub Pages base path, with NO
// Python backend. Everything asserted here is either computed live in the browser
// (geo engine, transformers.js, safetensors HTTP range reads) or precomputed by the
// real backend at build time and served verbatim.
//
// DATA-DRIVEN: expectations come from the export manifests in public/static-data
// (the same files the build serves), so the suite passes against either a full
// export or `scripts/export_static_assets.py --quick` (gpt2 only).

const BASE = "/llm-geometry/";
const DATA = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../public/static-data",
);

interface StaticIndex {
  arch_models: { model_id: string; slug: string; n_params: number }[];
}

function readJson<T>(rel: string): T {
  try {
    return JSON.parse(readFileSync(path.join(DATA, rel), "utf8")) as T;
  } catch (e) {
    throw new Error(
      `Missing/unreadable static export asset '${rel}' under ${DATA} — run ` +
        `\`python scripts/export_static_assets.py --quick\` (repo root) first. (${String(e)})`,
    );
  }
}

const index = readJson<StaticIndex>("index.json");

async function ready(page: Page, testid: string, timeout = 60_000): Promise<void> {
  await expect
    .poll(async () => page.getByTestId(testid).getAttribute("data-ready"), { timeout })
    .toBe("1");
}

// ---------------------------------------------------------------------------------
// [a] shell + masthead badge
// ---------------------------------------------------------------------------------

test("app loads with the static-demo masthead badge", async ({ page }) => {
  await page.goto(BASE);
  const badge = page.getByTestId("static-badge");
  await expect(badge).toBeVisible();
  await expect(badge).toContainText("static demo — computations run in your browser");
  // links out to the README's full-stack instructions
  await expect(badge).toHaveAttribute("href", /github\.com\/ContextLab\/llm-geometry/);
  await expect(page.getByTestId("view-tabs")).toBeVisible();
});

test("the Info tab tells a static visitor which capabilities they actually have", async ({
  page,
}) => {
  await page.goto(BASE);
  await page.getByTestId("tab-info").click();
  const info = page.getByTestId("info-view");
  await expect(info).toBeVisible();
  // The honesty claim this whole build rests on: the reader is told they are on the
  // static site, and told which capability is precomputed rather than live.
  await expect(info).toContainText("You are currently on the static build");
  await expect(info).toContainText("do not expose hidden states");
  // The math must survive the static bundle intact (no backend involved in rendering it).
  await expect(info).toContainText("logit lens");
  await expect(info).toContainText("On Transformer Dynamics");
});

// ---------------------------------------------------------------------------------
// [f] deep links under the Pages base path
// ---------------------------------------------------------------------------------

test("deep links under the base path serve the app", async ({ page }) => {
  // `vite preview` implements the fallback via its SPA single-page handler (index.html
  // for unknown paths under the base), not via 404.html — the deployed Pages site gets
  // the equivalent behavior from pages.yml's `cp dist/index.html dist/404.html`. So this
  // asserts the fallback CONTRACT (deep link → app shell); the 404.html copy itself is
  // exercised by the deployed-site smoke check, which preview cannot emulate.
  const res = await page.goto(`${BASE}some/deep/link`);
  expect(res?.ok()).toBeTruthy();
  await expect(page.getByTestId("view-tabs")).toBeVisible();
  await expect(page.getByTestId("static-badge")).toBeVisible();
});

// ---------------------------------------------------------------------------------
// [b] Geometry Lab — fully live in the browser (US-1)
// ---------------------------------------------------------------------------------

test("geometry lab runs fully live in-browser (engine, edits, worker fine-tune)", async ({
  page,
}) => {
  await page.goto(BASE);
  await page.getByTestId("tab-geometry").click();
  // The TS engine initializes from the shipped checkpoint — no training gate.
  await ready(page, "geo-view", 30_000);
  await expect(page.getByTestId("geo-canvas")).toBeVisible(); // sphere + field render
  // The shipped-checkpoint chip, not just the word somewhere in the tab: the header
  // prose also mentions a corpus, so a tab-wide match would no longer prove the chip.
  await expect(page.getByTestId("geo-active-model")).toContainText("corpus");

  // identity preset mints an edited-weights token (live postGeoWeights)
  await page.getByTestId("geo-preset").selectOption("identity");
  await page.getByTestId("geo-apply").click();
  // The badge names WHAT the active model is (004): calling a scratch-trained or
  // imported model "edited" was wrong, so the provenance is carried explicitly.
  await expect(page.getByTestId("geo-weight-panel")).toContainText("hand-edited weights", {
    timeout: 60_000,
  });
  await expect(page.getByTestId("geo-active-model")).toContainText("hand-edited weights");
  await page.getByTestId("geo-reset").click();
  await expect(page.getByTestId("geo-weight-panel")).toContainText("shipped checkpoint", {
    timeout: 60_000,
  });

  // pasted-text fine-tune: REAL SGD in the Web Worker → a loss trajectory
  await page
    .locator('[data-testid="geo-finetune"] textarea')
    .fill("alice went down the rabbit hole and found a very tiny door in the wall");
  await page.locator('[data-testid="geo-finetune"] .actions button').click();
  const loss = page.getByTestId("geo-finetune-loss");
  await expect(loss).toBeVisible({ timeout: 180_000 });
  await expect(loss).toContainText(/loss .+ → .+ on your text/);

  // Every source now works in the static build (feature 004): the Hub's dataset viewer
  // is CORS-enabled, so this tab is live rather than the disabled stub it used to be.
  await expect(page.getByTestId("geo-finetune-hf-tab")).toBeEnabled();
  await expect(page.getByTestId("geo-finetune-static-note")).toHaveCount(0);

  // Training a NEW model (feature 004) is offered here too, gated on a corpus big
  // enough to fill the vocabulary — the gate must be visible before the long run.
  const train = page.getByTestId("geo-train");
  await expect(train).toBeVisible();
  await page.getByTestId("geo-train-text").fill("alice met the rabbit");
  await expect(page.getByTestId("geo-train-stats")).toContainText(/short/);
  await expect(page.getByTestId("geo-train-run")).toBeDisabled();

  // Saving the ACTIVE model produces a real, self-describing file.
  const download = page.waitForEvent("download", { timeout: 60_000 });
  await page.getByTestId("geo-save-model").click();
  const saved = await download;
  expect(saved.suggestedFilename()).toMatch(/\.llmgeo\.json$/);
  await page.screenshot({
    path: "tests/e2e/__screenshots__/static-geometry.png",
    fullPage: true,
  });
});

/**
 * A model trained here has a vocabulary of its OWN — its token id 17 is not the shipped
 * model's token id 17 — and the static build keeps the model across a reload by
 * persisting the minted weight set to sessionStorage. It used to persist the WEIGHTS
 * ONLY, so after a reload the engine fell back to the shipped tokenizer and `Save model`
 * wrote a `.llmgeo.json` pairing these weights with Alice in Wonderland's word list,
 * hashing THAT list into `vocab_sha256`. The file was internally consistent, so no
 * reader on either side could reject it: save → reload → save silently changed which
 * words the model file described. The full stack was never affected — the Python
 * backend stores the vocabulary beside the weights (`save_weight_set(..., vocab_json=)`).
 *
 * One epoch is enough: this is about what the file SAYS, not about the loss.
 */
test("a model trained here keeps its own vocabulary across a reload (save → reload → save)", async ({
  page,
}) => {
  test.setTimeout(600_000);
  await page.goto(`${BASE}#geometry`);
  await ready(page, "geo-view", 60_000);

  const corpus = readFileSync(
    path.join(
      path.dirname(fileURLToPath(import.meta.url)),
      "../../../backend/src/llm_geometry/lex/data/real-mother-goose.txt",
    ),
    "utf8",
  );
  await page.getByTestId("geo-train-src-paste").click();
  await page.getByTestId("geo-train-text").fill(corpus);
  await expect(page.getByTestId("geo-train-stats")).toContainText(/enough to fill/);
  const epochs = page.getByTestId("geo-train-epochs");
  await epochs.fill("1");
  await epochs.dispatchEvent("input");
  await page.getByTestId("geo-train-run").click();
  await expect(page.getByTestId("geo-train-result")).toBeVisible({ timeout: 480_000 });

  const saveBundle = async (): Promise<{ weights_token: string; vocab: string }> => {
    const dl = page.waitForEvent("download", { timeout: 120_000 });
    await page.getByTestId("geo-save-model").click();
    await expect(page.getByTestId("geo-io-error")).toHaveCount(0);
    const stream = await (await dl).createReadStream();
    const chunks: Buffer[] = [];
    for await (const c of stream) chunks.push(c as Buffer);
    return JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
      weights_token: string;
      vocab: string;
    };
  };

  const before = await saveBundle();
  const wordsBefore = (JSON.parse(before.vocab) as { words: string[] }).words;
  // It really is a vocabulary of its own, not the shipped one.
  expect(wordsBefore).not.toContain("alice");

  await page.reload();
  await ready(page, "geo-view", 60_000);
  const after = await saveBundle();
  expect(after.weights_token).toBe(before.weights_token);
  expect(
    (JSON.parse(after.vocab) as { words: string[] }).words,
    "the saved model file's vocabulary changed across a reload",
  ).toEqual(wordsBefore);
});

// ---------------------------------------------------------------------------------
// [c] Architecture Explorer — precomputed graph/traces, LIVE weight windows (US-2)
// ---------------------------------------------------------------------------------

// The model the tab lands on: the backend default when this export carries it,
// otherwise the first exported model (--quick ships only gpt2).
const archModel =
  index.arch_models.find((m) => m.model_id === "HuggingFaceTB/SmolLM2-135M-Instruct") ??
  index.arch_models[0];

test("architecture: precomputed graph + example traces + real HF weight windows", async ({
  page,
}) => {
  interface TraceEntry {
    n: number;
    label: string;
    prompt: string;
  }
  const traces = readJson<{ traces: TraceEntry[] }>(
    `arch/${archModel.slug}/traces/index.json`,
  ).traces;
  interface GraphNode {
    id: string;
    kind: string;
    params: unknown[];
  }
  const graph = readJson<{ nodes: GraphNode[] }>(`arch/${archModel.slug}/graph.json`);
  const embedNode =
    graph.nodes.find((n) => n.kind === "embedding" && n.params.length > 0) ??
    graph.nodes.find((n) => n.params.length > 0);
  expect(embedNode).toBeTruthy();

  await page.goto(BASE);
  await page.getByTestId("tab-architecture").click();
  // graph renders from the precomputed JSON (no backend, no model download)
  await expect(page.locator('[data-testid^="diagram-node-"]').first()).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.getByTestId("arch-meta")).toContainText("params");

  // FR-412: the free-HF-id input is gone in BOTH builds, replaced by an honest note
  // about the curated list (it promised something the static build cannot deliver).
  await expect(page.getByTestId("arch-model-custom")).toHaveCount(0);
  const curated = page.getByTestId("arch-model-curated-note");
  await expect(curated).toBeVisible();
  await expect(curated).toContainText(/curated/i);
  await expect(curated.locator("a")).toHaveAttribute("href", /issues\/4/);

  // runtime badge sits by the chat controls (idle until the first generate)
  await expect(page.getByTestId("static-runtime-badge")).toBeVisible();
  await expect(page.getByTestId("static-runtime-badge")).toContainText(/in-browser/);

  // example-trace dropdown lists the export's labeled prompts; the tab lands on one
  const dd = page.getByTestId("arch-trace-presets");
  await expect(dd).toBeVisible();
  for (const t of traces) {
    await expect(dd.locator("option", { hasText: t.label })).toHaveCount(1);
  }
  await expect(page.getByTestId("arch-trace-strip")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("arch-topk-label")).toBeVisible();

  // selecting another example loads ITS precomputed trace (strip + attention heatmap)
  const t2 = traces[Math.min(1, traces.length - 1)];
  await dd.selectOption(String(t2.n));
  await expect(page.getByTestId("arch-prompt")).toHaveValue(t2.prompt);
  await expect(page.getByTestId("arch-topk-label")).toBeVisible({
    timeout: 60_000,
  });
  await expect(
    page.getByTestId("arch-breakdown").getByTestId("matrix-heatmap").first(),
  ).toBeVisible({ timeout: 60_000 });

  // a free-form prompt has no precomputed per-layer trace → the designed affordance
  await page.getByTestId("arch-prompt").fill("a prompt nobody precomputed zzz");
  const note = page.getByTestId("arch-static-note");
  await expect(note).toBeVisible({ timeout: 30_000 });
  await expect(note).toContainText(/example prompts/i);
  await dd.selectOption(String(traces[0].n));
  await expect(note).toHaveCount(0, { timeout: 60_000 });

  // weight inspector: overview tile is labeled honestly; zoom fetches REAL exact
  // values via safetensors HTTP Range reads from huggingface.co
  let hfRequest = false;
  page.on("request", (r) => {
    if (r.url().includes("huggingface.co")) hfRequest = true;
  });
  await page.locator(`[data-testid="diagram-node-${embedNode!.id}"]`).click();
  const inspector = page.getByTestId("arch-inspector");
  await expect(inspector).toBeVisible();
  await expect(inspector.getByTestId("matrix-heatmap")).toBeVisible({ timeout: 60_000 });
  // `wte.weight` really is strided-mean downsampled AND uint8; the caption states both,
  // because a 1-D parameter's tile is uint8 without being downsampled at all.
  await expect(inspector).toContainText("overview (whole tensor, strided mean, 8-bit)");
  await inspector.getByRole("button", { name: /zoom into the weights/ }).click();
  await expect(inspector).toContainText("exact values", { timeout: 120_000 });
  await expect.poll(() => hfRequest, { timeout: 10_000 }).toBe(true);
  await page.screenshot({
    path: "tests/e2e/__screenshots__/static-architecture.png",
    fullPage: true,
  });
});

// ---------------------------------------------------------------------------------
// [d] one REAL in-browser generation (transformers.js; ONNX download is real)
// ---------------------------------------------------------------------------------

test("real in-browser generation on the smallest model", async ({ page }) => {
  test.setTimeout(300_000);
  const gen = index.arch_models.find((m) => m.model_id === "gpt2") ?? index.arch_models[0];

  await page.goto(BASE);
  await page.getByTestId("tab-architecture").click();
  await expect(page.locator('[data-testid^="diagram-node-"]').first()).toBeVisible({
    timeout: 60_000,
  });
  if (gen.model_id !== archModel.model_id) {
    await page.getByTestId("arch-model-select").selectOption(gen.model_id);
    await expect(page.getByTestId("arch-model-status")).toHaveText("ok", {
      timeout: 60_000,
    });
  }
  await page.getByTestId("arch-prompt").fill("The capital of France is");
  await page.getByTestId("arch-generate").click();
  // Real ONNX download (100+ MB on a cold cache) + real sampling.
  await expect(page.getByTestId("arch-reply")).not.toBeEmpty({ timeout: 280_000 });
  const tok = page.getByTestId("arch-reply").locator(".tok").first();
  await expect(tok).toBeAttached();
  await expect(tok).toHaveAttribute("aria-label", /%/);
  // The runtime ladder settled on a device/dtype pair that PASSED the load-time
  // non-degeneracy check (transformersRuntime.selfCheck). Both rungs read the same
  // model_quantized.onnx; only the execution provider differs.
  await expect(page.getByTestId("static-runtime-badge")).toContainText(
    /(webgpu|wasm) · q8/,
  );
  await page.screenshot({
    path: "tests/e2e/__screenshots__/static-generation.png",
    fullPage: true,
  });
});

// ---------------------------------------------------------------------------------
// [g] The vacancy instrument's pretrained arm — computed live, reported NARROWLY
// (feature 007, contract §8.3a / FR-720a / SC-707b). The browser runs a quantized
// export, so it may state only what was MEASURED for that dtype: pooled
// `swap − english` and `nonce − english` with the measured ±0.1 nats, and nothing
// else. This drives the built site and asserts it does one or the other — never a
// bare number.

test("the vacancy panel reports only what q8 has a measured bound for", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto(BASE);
  await page.getByTestId("tab-architecture").click();
  const panel = page.getByTestId("arch-vacancy");
  await panel.scrollIntoViewIfNeeded();
  await page.getByTestId("arch-vac-run").click();

  // Real ONNX download + 18 real forward passes (6 pooled excerpts × 3 variants).
  await expect(page.getByTestId("arch-vac-table")).toBeVisible({ timeout: 560_000 });
  await expect(page.getByTestId("arch-vac-error")).toHaveCount(0);

  // Refused: nonce − swap, by name, with the reason and the command that would fix it.
  const refusal = page.getByTestId("arch-vac-refused-unknown_form");
  await expect(refusal).toBeVisible();
  await expect(refusal).toContainText("sign flip");
  await expect(refusal).toContainText("uvicorn");
  await expect(page.getByTestId("arch-vac-unknown_form")).toHaveCount(0);

  // Refused: the absolute NLLs and every per-passage row.
  await expect(page.getByTestId("arch-vac-refused-absolute")).toBeVisible();
  await expect(page.getByTestId("arch-vac-refused-passages")).toBeVisible();
  const english = page.getByTestId("arch-vac-row-english");
  await expect(english).toContainText("—");

  // Reported: the two pooled differences, each with the MEASURED quantization ±.
  const wrong = page.getByTestId("arch-vac-wrong_content");
  await expect(wrong).toContainText(/\d\.\d{3}/);
  await expect(page.getByTestId("arch-vac-wrong_content-err")).toContainText("0.2 (quantization, measured)");
  // …INCLUDING the secondary row. It rendered `0.879 ± 0.074` on the live site and
  // dropped the ±0.2 the same response carried, stating an interval that excludes the
  // float32 value 0.9892 (red team F4 / FR-720a).
  await expect(page.getByTestId("arch-vac-total-err")).toContainText("(sampling)");
  await expect(page.getByTestId("arch-vac-total-err")).toContainText(
    "0.2 (quantization, measured)",
  );

  // The residual caveat is stated even where the number is refused: a reader must not
  // have to earn the caveat by being shown a value.
  await expect(page.getByTestId("arch-vac-honesty")).toContainText("UPPER BOUND");
  await expect(page.getByTestId("arch-vac-honesty")).toContainText("higher entropy");

  await panel.screenshot({
    path: "tests/e2e/__screenshots__/static-arch-vacancy.png",
  });
});
