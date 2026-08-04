// The load-time non-degeneracy invariant (src/lib/staticClient/logitsSanity.ts).
//
// These exercise the predicate itself over the exact shapes the real failure produced,
// measured in a real browser on a real GPU (Chrome 150 / Apple Metal-3, ORT-web
// 1.26.0-dev, transformers.js 4.2.0):
//   onnx-community/gpt2-ONNX            webgpu/q4f16 → row separation 0 (identical rows)
//   onnx-community/SmolLM2-135M-…-ONNX  webgpu/q4f16 → row separation 0, every logit 0
//   onnx-community/SmolLM2-360M-…-ONNX  webgpu/q4f16 → row separation 0, every logit 0
// versus the healthy sessions on the same machine:
//   gpt2 webgpu/q8 91.99 · SmolLM2-135M webgpu/q8 36.97 · Qwen2.5-0.5B webgpu/q8 20.27
// The end-to-end version of this, against the real model in a WebGPU browser, is
// tests/e2e/webgpu.spec.ts.
import { describe, expect, it } from "vitest";

import {
  MIN_ROW_SEPARATION,
  assertNonDegenerateLogits,
  rowSeparation,
} from "../../src/lib/staticClient/logitsSanity";
import { FP16_ACTIVATION_DTYPES, RUNTIME_LADDER } from "../../src/lib/staticClient/runtimeTypes";

const VOCAB = 64;
const SEQ = 6;

/** A [1, SEQ, VOCAB] buffer whose rows differ, like any working causal LM. */
function healthy(): Float32Array {
  const out = new Float32Array(SEQ * VOCAB);
  for (let t = 0; t < SEQ; t++) {
    for (let v = 0; v < VOCAB; v++) out[t * VOCAB + v] = Math.sin(v * 0.37) * 8 + t * 1.5;
  }
  return out;
}

/** One row, repeated — the gpt2 webgpu/q4f16 failure. */
function repeatedRows(): Float32Array {
  const out = new Float32Array(SEQ * VOCAB);
  for (let t = 0; t < SEQ; t++) {
    for (let v = 0; v < VOCAB; v++) out[t * VOCAB + v] = Math.cos(v * 0.11) * 6;
  }
  return out;
}

describe("rowSeparation", () => {
  it("measures the L∞ gap between the first and last next-token distribution", () => {
    // Constructed so the true answer is known: rows differ by exactly 1.5*(SEQ-1).
    expect(rowSeparation(healthy(), SEQ, VOCAB)).toBeCloseTo(1.5 * (SEQ - 1), 5);
  });

  it("is exactly 0 for repeated rows and for an all-zero tensor", () => {
    expect(rowSeparation(repeatedRows(), SEQ, VOCAB)).toBe(0);
    expect(rowSeparation(new Float32Array(SEQ * VOCAB), SEQ, VOCAB)).toBe(0);
  });

  it("reports NaN when any compared entry is not finite", () => {
    const nan = healthy();
    nan[(SEQ - 1) * VOCAB + 3] = Number.NaN;
    expect(rowSeparation(nan, SEQ, VOCAB)).toBeNaN();
    const inf = healthy();
    inf[7] = Number.POSITIVE_INFINITY;
    expect(rowSeparation(inf, SEQ, VOCAB)).toBeNaN();
  });

  it("refuses a single-position pass, where the invariant is not defined", () => {
    expect(() => rowSeparation(new Float32Array(VOCAB), 1, VOCAB)).toThrow(/at least 2 positions/);
  });
});

// The dtype-preference half of the fix. tests/e2e/webgpu.spec.ts checks it end to end
// on a real GPU, which SKIPS wherever there is no adapter (all GitHub-hosted runners);
// this runs everywhere, so CI always verifies at least that no fp16-activation dtype
// has crept back into the ladder.
describe("the runtime's dtype ladder", () => {
  it("asks only for dtypes verified correct in a real browser", () => {
    expect(RUNTIME_LADDER.length).toBeGreaterThan(0);
    for (const rung of RUNTIME_LADDER) {
      expect(["q8"]).toContain(rung.dtype);
      expect(["webgpu", "wasm"]).toContain(rung.device);
    }
  });

  it("never requests a dtype with fp16 ACTIVATIONS, the path that returns garbage", () => {
    const asked = new Set<string>(RUNTIME_LADDER.map((r) => r.dtype));
    for (const bad of FP16_ACTIVATION_DTYPES) expect(asked.has(bad)).toBe(false);
  });

  it("tries the GPU first and keeps a non-GPU rung to fall back to", () => {
    expect(RUNTIME_LADDER[0].device).toBe("webgpu");
    expect(RUNTIME_LADDER.some((r) => r.device === "wasm")).toBe(true);
    // Both rungs read the same model_quantized.onnx, so a rejection costs no
    // second download — the property that makes the fallback cheap.
    expect(new Set(RUNTIME_LADDER.map((r) => r.dtype)).size).toBe(1);
  });
});

describe("assertNonDegenerateLogits", () => {
  it("accepts a session whose output depends on its input, returning the separation", () => {
    expect(assertNonDegenerateLogits(healthy(), SEQ, VOCAB, "wasm/q8")).toBeCloseTo(7.5, 5);
  });

  it("rejects identical rows, naming the configuration under test", () => {
    expect(() => assertNonDegenerateLogits(repeatedRows(), SEQ, VOCAB, "webgpu/q4f16")).toThrow(
      /webgpu\/q4f16 produced degenerate logits/,
    );
  });

  it("rejects an all-zero tensor as the same single failure, not a special case", () => {
    expect(() =>
      assertNonDegenerateLogits(new Float32Array(SEQ * VOCAB), SEQ, VOCAB, "webgpu/q4f16"),
    ).toThrow(/does not depend on its input/);
  });

  it("rejects NaN output", () => {
    const nan = new Float32Array(SEQ * VOCAB).fill(Number.NaN);
    expect(() => assertNonDegenerateLogits(nan, SEQ, VOCAB, "webgpu/fp16")).toThrow(/NaN/);
  });

  it("accepts separations just above the threshold and rejects just below", () => {
    const near = new Float32Array(SEQ * VOCAB);
    near[(SEQ - 1) * VOCAB] = MIN_ROW_SEPARATION * 2;
    expect(assertNonDegenerateLogits(near, SEQ, VOCAB, "wasm/q8")).toBeCloseTo(
      MIN_ROW_SEPARATION * 2,
      9,
    );
    const under = new Float32Array(SEQ * VOCAB);
    under[(SEQ - 1) * VOCAB] = MIN_ROW_SEPARATION / 2;
    expect(() => assertNonDegenerateLogits(under, SEQ, VOCAB, "wasm/q8")).toThrow(/degenerate/);
  });
});
