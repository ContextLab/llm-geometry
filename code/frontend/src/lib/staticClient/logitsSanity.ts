/**
 * The non-degeneracy invariant every in-browser inference session must satisfy
 * before the app is allowed to report numbers from it.
 *
 * WHY THIS EXISTS. The app used to ask for `q4f16` first. On any machine whose
 * browser exposes a WebGPU adapter with `shader-f16`, that session BUILDS
 * SUCCESSFULLY on this onnxruntime-web build and then returns logits that carry no
 * information about the input at all — measured in a real browser (Chrome 150,
 * Apple Metal-3):
 *
 *   onnx-community/gpt2-ONNX               webgpu/q4f16 → every row of [1,T,V]
 *                                          bit-identical; greedy decode ",,,,,,,,,,"
 *   onnx-community/SmolLM2-135M-…-ONNX     webgpu/q4f16 → every logit exactly 0;
 *                                          every NLL = ln(49152) = 10.80267
 *
 * Nothing throws, so an exception-only fallback ladder is structurally incapable of
 * catching it. The invariant below is what "the session works" actually means, and it
 * is asserted once, at load, in the same spirit as the Geometry Lab's training gates
 * (final loss, coverage uniformity, field directional entropy): a property the system
 * must satisfy, checked explicitly, failing loudly.
 *
 * THE INVARIANT, stated once. A causal LM's output must DEPEND ON ITS INPUT: run a
 * prompt of T ≥ 2 distinct tokens through one forward pass and the next-token
 * distribution at the last position must differ from the one at the first position.
 * A model whose rows are identical — all-zero logits being one such case, not a
 * separate rule — has told you nothing, whatever its perplexity would have been.
 *
 * The threshold is deliberately far below any real model's separation and far above
 * float noise: healthy sessions measure 20–92 nats of separation on the probe prompt
 * (gpt2 webgpu/q8: 91.99; SmolLM2-135M webgpu/q8: 36.97; Qwen2.5-0.5B webgpu/q8:
 * 20.27), degenerate ones measure exactly 0.
 */

/** Minimum L∞ separation, in logit units, between the first and last position. */
export const MIN_ROW_SEPARATION = 1e-3;

/**
 * L∞ distance between the first and last row of a flat [1, T, V] logits buffer.
 * `NaN` if any compared entry is not finite — an all-NaN session is degenerate too,
 * and NaN fails the `>` comparison in {@link assertNonDegenerateLogits} by itself.
 */
export function rowSeparation(logits: ArrayLike<number>, seqLen: number, vocab: number): number {
  if (seqLen < 2) {
    throw new Error(`rowSeparation needs at least 2 positions, got ${seqLen}`);
  }
  const last = (seqLen - 1) * vocab;
  let sep = 0;
  for (let i = 0; i < vocab; i++) {
    const a = logits[i];
    const b = logits[last + i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) return NaN;
    const d = Math.abs(a - b);
    if (d > sep) sep = d;
  }
  return sep;
}

/**
 * Throw unless the session's output depends on its input. `label` names the
 * device/dtype under test so the message says which configuration is broken.
 */
export function assertNonDegenerateLogits(
  logits: ArrayLike<number>,
  seqLen: number,
  vocab: number,
  label: string,
): number {
  const sep = rowSeparation(logits, seqLen, vocab);
  if (!(sep > MIN_ROW_SEPARATION)) {
    throw new Error(
      `${label} produced degenerate logits: the next-token distribution at the last ` +
        `position differs from the first by ${Number.isNaN(sep) ? "NaN" : sep.toExponential(3)} ` +
        `(needs > ${MIN_ROW_SEPARATION}), i.e. the model's output does not depend on its input. ` +
        `This session cannot be trusted and was rejected.`,
    );
  }
  return sep;
}
