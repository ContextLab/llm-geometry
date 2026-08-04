import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

/**
 * The WebGPU regression suite — the coverage gap that let a broken dtype ship.
 *
 * Plain headless Chromium exposes NO WebGPU adapter (`requestAdapter()` resolves to
 * `null`), so every other e2e project silently exercises the WASM rung only. That is
 * why `webgpu/q4f16` — which BUILDS a session and then returns logits identical at
 * every position — reached the deployed site: nothing threw, and CI never ran the code
 * path. This project launches Chromium with `--enable-unsafe-webgpu`, which does
 * surface the machine's real adapter, and asserts the property the defect violated:
 * the model's next-token distribution must DEPEND ON POSITION.
 *
 * Where there is no adapter, the test SKIPS with an explicit reason — it must never
 * pass vacuously by falling through to WASM.
 *
 * READ THIS BEFORE TRUSTING A GREEN CI RUN. GitHub-hosted runners have no GPU, and the
 * software adapter Chromium offers there (SwiftShader) advertises no `shader-f16`, so
 * this test SKIPS in CI and the WebGPU path is verified only on a developer machine
 * with a real GPU. That is a named, deliberate gap, not coverage. What CI does always
 * verify: tests/unit/logitsSanity.test.ts (the invariant and the dtype ladder) and
 * tests/e2e/static.spec.ts (a real session built and passed through the same
 * load-time gate, on the WASM rung).
 */

const BASE = "/llm-geometry/";
const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "../../public/static-data");

interface StaticIndex {
  arch_models: { model_id: string; slug: string }[];
}
const index = JSON.parse(readFileSync(path.join(DATA, "index.json"), "utf8")) as StaticIndex;

// The smallest curated model this export ships: both are known-degenerate under
// q4f16 on WebGPU and known-correct under q8, so either discriminates.
const model =
  index.arch_models.find((m) => m.model_id === "HuggingFaceTB/SmolLM2-135M-Instruct") ??
  index.arch_models.find((m) => m.model_id === "gpt2") ??
  index.arch_models[0];

interface AdapterProbe {
  hasNavigatorGpu: boolean;
  adapter: null | { vendor?: string; architecture?: string; shaderF16: boolean };
}

async function probeAdapter(page: Page): Promise<AdapterProbe> {
  return page.evaluate(async () => {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (!gpu) return { hasNavigatorGpu: false, adapter: null };
    const a = (await gpu.requestAdapter()) as null | {
      features: { has(n: string): boolean };
      info?: { vendor?: string; architecture?: string };
    };
    if (!a) return { hasNavigatorGpu: true, adapter: null };
    return {
      hasNavigatorGpu: true,
      adapter: {
        vendor: a.info?.vendor,
        architecture: a.info?.architecture,
        shaderF16: a.features.has("shader-f16"),
      },
    };
  });
}

test("in-browser generation on a REAL WebGPU adapter is not degenerate", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto(BASE);

  const probe = await probeAdapter(page);
  // A loud skip, never a vacuous pass: say exactly what was missing.
  const why = !probe.hasNavigatorGpu
    ? "navigator.gpu is undefined in this browser build"
    : probe.adapter === null
      ? "navigator.gpu.requestAdapter() resolved to null — this environment has no WebGPU adapter " +
        "(the usual case for Linux CI runners and for headless Chromium without --enable-unsafe-webgpu)"
      : !probe.adapter.shaderF16
        ? `the only adapter is ${probe.adapter.vendor}/${probe.adapter.architecture}, which lacks ` +
          "shader-f16 — the app treats that as a software adapter and takes the WASM rung, so this " +
          "test cannot observe the WebGPU path"
        : "";
  if (why) {
    console.warn(
      `\n!!! SKIPPING THE WEBGPU REGRESSION TEST — THE WEBGPU PATH IS UNVERIFIED IN THIS RUN.\n` +
        `!!! Reason: ${why}.\n` +
        `!!! This is expected on GitHub-hosted runners, which have no GPU. A green CI run\n` +
        `!!! therefore does NOT mean the WebGPU path was checked: run it on a machine with a\n` +
        `!!! real GPU — npx playwright test --project webgpu\n`,
    );
    test.skip(
      true,
      `WEBGPU PATH UNVERIFIED (expected on GPU-less CI runners): ${why}. ` +
        "Run `npx playwright test --project webgpu` on a machine with a GPU.",
    );
  }
  console.log(
    `[webgpu] real adapter: ${probe.adapter?.vendor}/${probe.adapter?.architecture} (shader-f16)`,
  );

  await page.getByTestId("tab-architecture").click();
  await expect(page.locator('[data-testid^="diagram-node-"]').first()).toBeVisible({
    timeout: 60_000,
  });
  if (model.model_id !== (await page.getByTestId("arch-model-select").inputValue())) {
    await page.getByTestId("arch-model-select").selectOption(model.model_id);
    await expect(page.getByTestId("arch-model-status")).toHaveText("ok", { timeout: 60_000 });
  }
  await page.getByTestId("arch-prompt").fill("The capital of France is Paris. The capital of Germany is");
  await page.getByTestId("arch-generate").click();

  // Real ONNX download (100+ MB cold) + real sampling on the GPU.
  await expect(page.getByTestId("arch-reply")).not.toBeEmpty({ timeout: 480_000 });

  // (1) The WebGPU path really was the one exercised — otherwise everything below
  //     would be a WASM result wearing a WebGPU test's name.
  const badge = page.getByTestId("static-runtime-badge");
  await expect(badge).toContainText("webgpu");
  // (2) …on a dtype the runtime verified at load time. q4f16 is gone; if it ever
  //     comes back, this pins the failure to the dtype rather than to the symptom.
  await expect(badge).toContainText("webgpu · q8");

  // (3) THE DEFECT ITSELF, as a user sees it. Under webgpu/q4f16 every row of the
  //     [1,T,V] logits was bit-identical, so the tab reported the SAME top-5 at every
  //     generated position. A model whose distribution does not move with the context
  //     has told the user nothing. Measured on this page, 64 generated tokens:
  //         webgpu/q4f16 (pre-fix)  →  2 distinct top-5 lists out of 64
  //         webgpu/q8    (post-fix) → 64 distinct top-5 lists out of 64
  //     so "most positions differ" separates them by the whole range. Requiring ALL to
  //     differ would be a fair statement too, but it is a claim about the model rather
  //     than about the runtime, and a legitimate repeat would then fail the suite.
  const labels = await page.getByTestId("arch-reply").locator(".tok").evaluateAll((els) =>
    els.map((e) => e.getAttribute("aria-label") ?? ""),
  );
  expect(labels.length, "generation produced too few tokens to compare positions").toBeGreaterThan(8);
  const topk = labels
    .map((l) => l.slice(l.indexOf("top-5:")))
    .filter((s) => s.startsWith("top-5:"));
  expect(topk.length).toBe(labels.length);
  expect(
    new Set(topk).size,
    `only ${new Set(topk).size} of ${topk.length} generated positions reported a DIFFERENT ` +
      "top-5 — the logits barely depend on the context, which is exactly the webgpu/q4f16 " +
      "failure this test exists to catch",
  ).toBeGreaterThan(topk.length / 2);

  await page.screenshot({ path: "tests/e2e/__screenshots__/webgpu-generation.png", fullPage: true });
});
