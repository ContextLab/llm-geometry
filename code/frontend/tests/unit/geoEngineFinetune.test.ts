/**
 * Fine-tune golden tests for the TS geoEngine.
 *
 * The backend shuffles batches with torch.randperm (not portable), so — as
 * documented in finetune.ts — the static engine uses its own deterministic PRNG
 * and these tests assert loss-TRAJECTORY properties rather than bit-equality:
 *   - loss_before matches the backend tightly (it is RNG-free: pure eval),
 *   - loss_after < loss_before (real learning),
 *   - loss_after lands within 15% of the backend's for the same text/steps/lr,
 *   - the minted token is the content hash of the resulting weights
 *     (self-consistent + deterministic + cache-hit on identical requests).
 */

import { beforeAll, describe, expect, it } from "vitest";

import { GeoEngine, weightsToken } from "../../src/lib/geoEngine";
import { evalLoss, makeWindows, runFinetune } from "../../src/lib/geoEngine/finetune";
import { CONTEXT_WINDOW, EOS_ID, GeoModel, PAD_ID } from "../../src/lib/geoEngine/model";
import { loadAsset, loadMergedGolden, type GoldenFile } from "./geoGoldenAssets";

let engine: GeoEngine;
let golden: GoldenFile;

beforeAll(() => {
  golden = loadMergedGolden();
  engine = GeoEngine.fromAssets(loadAsset("checkpoint.json"), loadAsset("vocab.json"));
});

describe("makeWindows (geo/train.py semantics)", () => {
  it("pads short streams to a single window", () => {
    const windows = makeWindows([5, 6, 7], CONTEXT_WINDOW, 25);
    expect(windows.length).toBe(1);
    expect(windows[0].length).toBe(CONTEXT_WINDOW + 1);
    expect(Array.from(windows[0].subarray(0, 4))).toEqual([5, 6, 7, PAD_ID]);
    expect(windows[0][CONTEXT_WINDOW]).toBe(PAD_ID);
  });

  it("strides long streams like the backend (starts 0..len-span step stride)", () => {
    const stream = Array.from({ length: 130 }, (_, i) => (i % 900) + 3);
    const windows = makeWindows(stream, CONTEXT_WINDOW, 25);
    // span=51: starts at 0,25,50,75 (100 > 130-51=79 stops) -> 4 windows
    expect(windows.length).toBe(4);
    expect(Array.from(windows[1].subarray(0, 3))).toEqual(stream.slice(25, 28));
    expect(windows[3][50]).toBe(stream[75 + 50]);
  });
});

describe("geoEngine finetune", () => {
  it("matches the backend's loss trajectory properties on every golden run", () => {
    expect(golden.finetune.length).toBeGreaterThan(0);
    for (const goldenCase of golden.finetune) {
      const { text, steps, lr } = goldenCase.body as { text: string; steps: number; lr: number };
      const goldenResult = goldenCase.result;
      const label = `finetune(steps=${steps}, lr=${lr})`;

      const progress: [number, string][] = [];
      const t0 = performance.now();
      const result = engine.finetune({
        text,
        steps,
        lr,
        onProgress: (fraction, message) => progress.push([fraction, message]),
      });
      console.log(
        `[geoEngine perf] finetune ${steps} steps (incl. before/after evals): ${(performance.now() - t0).toFixed(0)}ms`,
      );

      expect(result.ready, label).toBe(true);
      expect(result.weights_token, label).toMatch(/^[0-9a-f]{32}$/);

      // loss_before is RNG-free (pure evaluation of the base weights on the same
      // windows) — it must match the backend tightly.
      const lb = result.loss_before!;
      expect(
        Math.abs(lb - goldenResult.loss_before) / goldenResult.loss_before,
        `${label}: loss_before`,
      ).toBeLessThan(1e-4);

      // Real learning happened, and the final loss lands near the backend's
      // (RNG-divergent batch order => 15% band, per the port contract).
      expect(result.loss_after!, label).toBeLessThan(result.loss_before!);
      expect(
        Math.abs(result.loss_after! - goldenResult.loss_after) / goldenResult.loss_after,
        `${label}: loss_after vs backend ${goldenResult.loss_after}`,
      ).toBeLessThan(0.15);

      // Progress streamed every 10 steps with backend-shaped messages.
      expect(progress.length, label).toBe(Math.floor(steps / 10));
      expect(progress[0][1]).toMatch(/^step 10\/\d+ · loss \d/);

      // The minted token resolves: tracing with it works and its distribution
      // is a real probability distribution.
      const trace = engine.trace("alice was beginning", result.weights_token);
      const sum = trace.probs.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1), label).toBeLessThan(1e-6);

      // Determinism + content-hash caching: an identical request returns the
      // identical token (served from the engine's content-derived cache).
      const again = engine.finetune({ text, steps, lr });
      expect(again.weights_token, label).toBe(result.weights_token);
      expect(again.loss_after, label).toBe(result.loss_after);
    }
  });

  it("runFinetune mints self-consistent content-hash tokens", () => {
    const goldenCase = golden.finetune[0];
    const tokenIds = engine.tokenizer.encodeStream(goldenCase.body.text as string);
    const result = runFinetune({
      baseWeights: engine.canonical,
      tokenIds,
      steps: 5,
      lr: 1e-2,
      seed: 0,
    });
    // The token the engine would mint is exactly the content hash of the weights.
    expect(weightsToken(result.weights)).toMatch(/^[0-9a-f]{32}$/);
    // Same inputs -> bit-identical weights (deterministic PRNG).
    const rerun = runFinetune({
      baseWeights: engine.canonical,
      tokenIds,
      steps: 5,
      lr: 1e-2,
      seed: 0,
    });
    expect(weightsToken(rerun.weights)).toBe(weightsToken(result.weights));
    // And the base was never mutated.
    expect(weightsToken(engine.canonical)).toBe(engine.canonicalToken);
  });

  it("keeps the S^2 embedding invariant through training", () => {
    const goldenCase = golden.finetune[0];
    const tokenIds = engine.tokenizer.encodeStream(goldenCase.body.text as string);
    const { weights } = runFinetune({
      baseWeights: engine.canonical,
      tokenIds,
      steps: 8,
      lr: 5e-2,
      seed: 3,
    });
    const E = weights["embedding"];
    for (let r = 0; r < 1003; r++) {
      const n = Math.hypot(E[r * 3], E[r * 3 + 1], E[r * 3 + 2]);
      expect(Math.abs(n - 1)).toBeLessThan(1e-6);
    }
  });

  it("gradient check: full-batch SGD steps descend the full-batch loss", () => {
    // Independent correctness check of the hand-derived backward pass. The text
    // tokenizes to < window+1 tokens, so there is exactly ONE window: every SGD
    // step optimizes the same full-batch objective that evalLoss measures, and a
    // small-lr step MUST reduce it — any sign/shape error in the gradients would
    // break monotone descent immediately.
    const tokenIds = engine.tokenizer.encodeStream(
      "alice was beginning to get very tired of sitting by her sister on the bank",
    );
    const windows = makeWindows([...tokenIds, EOS_ID], CONTEXT_WINDOW, 25);
    expect(windows.length).toBe(1);
    const before = evalLoss(new GeoModel(engine.canonical), windows);
    const one = runFinetune({ baseWeights: engine.canonical, tokenIds, steps: 1, lr: 1e-2, seed: 0 });
    const five = runFinetune({ baseWeights: engine.canonical, tokenIds, steps: 5, lr: 1e-2, seed: 0 });
    expect(one.lossBefore).toBeCloseTo(before, 10);
    expect(one.lossAfter).toBeLessThan(one.lossBefore);
    expect(five.lossAfter).toBeLessThan(one.lossAfter);
  });

  it("rejects hf_dataset sources with an honest static-mode error", () => {
    try {
      engine.finetune({ hf_dataset: "roneneldan/TinyStories" });
      throw new Error("expected InvalidParamError");
    } catch (err) {
      expect((err as { type?: string }).type).toBe("InvalidParamError");
      expect((err as Error).message).toMatch(/static build/);
    }
  });
});
