/**
 * Shared types for the in-browser transformers.js runtime, split out so
 * arch.ts / index.ts can type against the runtime WITHOUT importing the heavy
 * module (which is only ever loaded via dynamic import on first use).
 */

import type { ArchGenerateBody, ArchGenerateResult, TokenizeResult } from "../dataClient";

/** Devices the ladder in transformersRuntime may select. */
export type RuntimeDevice = "webgpu" | "wasm";
/**
 * Quantizations the ladder may select. `q4f16` is deliberately NOT here: it builds a
 * session and returns input-independent logits on WebGPU — see logitsSanity.ts.
 */
export type RuntimeDtype = "q8";

/**
 * Every dtype the fp16-ACTIVATION defect covers. Measured in a real browser on a real
 * Apple Metal-3 adapter (Chrome 150 / Chromium 148, transformers.js 4.2.0,
 * onnxruntime-web 1.26.0-dev): `q4f16` and `fp16` build a session and then return
 * logits identical at every position for gpt2 (greedy ",,,,,,,,,,") and identically
 * ZERO for SmolLM2-135M and SmolLM2-360M (empty generation; every NLL = ln V). `q4`
 * (4-bit weights, fp32 activations), `q8` and `fp32` are correct on the same adapter,
 * so the defect is the fp16 activation path, not 4-bit weights.
 */
export const FP16_ACTIVATION_DTYPES = ["q4f16", "fp16"] as const;

/**
 * Load order for the in-browser runtime. Both rungs read the SAME
 * `model_quantized.onnx`, so a rejected WebGPU rung costs no second download — and q8
 * is the smallest quantization verified correct here: in every curated ONNX repo
 * `model_q4.onnx` is LARGER than `model_quantized.onnx` (gpt2 498 vs 280 MB;
 * SmolLM2-135M 181 vs 136; SmolLM2-360M 386 vs 363; Qwen2.5-0.5B 786 vs 512), so the
 * smaller `q4f16` was the only download saving on offer, and it does not work.
 */
export const RUNTIME_LADDER: readonly { device: RuntimeDevice; dtype: RuntimeDtype }[] = [
  { device: "webgpu", dtype: "q8" },
  { device: "wasm", dtype: "q8" },
];

export interface RuntimeGenerationInfo {
  status: "idle" | "loading" | "ready" | "error";
  device: RuntimeDevice | null;
  dtype: RuntimeDtype | null;
  model_id: string | null; // the HF model whose ONNX export is loaded
  onnx_repo: string | null;
  error: string | null;
  /**
   * `device/dtype` rungs that were built but REJECTED — by an exception or by the
   * load-time non-degeneracy check — before the reported one was accepted. Non-empty
   * means the user is on a fallback path, and the badge says so: the unforgivable
   * part of the q4f16 defect was that a wrong configuration was invisible.
   */
  rejected: string[];
}

export interface StaticRuntimeInfo {
  mode: "static";
  generation: RuntimeGenerationInfo;
}

/** One text scored by ONE real teacher-forced forward pass (contract §8.1). */
export interface RuntimeScoredText {
  /** Byte-level pieces, in order — the input to the UTF-8 span algorithm (§8.2). */
  pieces: string[];
  /**
   * Per-token negative log-likelihood in nats: `nll[i]` is the cost of predicting token
   * `i` given tokens `< i`. Position 0 has no prediction and is `NaN`, never 0 — a zero
   * there would read as a perfectly predicted first token.
   */
  nll: number[];
  /** Characters of the scored text, for `bitsPerChar`. */
  nChars: number;
}

/** The surface the lazily-imported transformersRuntime module implements. */
export interface ArchRuntime {
  info(): RuntimeGenerationInfo;
  /** Live tokenization from the pinned original-repo tokenizer files. */
  tokenize(modelId: string, revision: string, text: string): Promise<TokenizeResult>;
  /** Live generation (webgpu q8 → wasm q8 fallback) with real per-token probs. */
  /** The ONNX mirror is resolved at `main` — see transformersRuntime's header. */
  generate(body: ArchGenerateBody, onnxRepo: string): Promise<ArchGenerateResult>;
  /**
   * Per-token NLL for each text, one real forward pass each, plus the byte-level pieces
   * the caller needs to attribute those tokens to words. No special tokens and no chat
   * template: the passage is scored exactly as written, so variants of it differ by the
   * transform and by nothing else.
   */
  scoreTexts(onnxRepo: string, texts: readonly string[]): Promise<RuntimeScoredText[]>;
}

export type RuntimeLoader = () => Promise<ArchRuntime>;

export const IDLE_GENERATION_INFO: RuntimeGenerationInfo = {
  status: "idle",
  device: null,
  dtype: null,
  model_id: null,
  onnx_repo: null,
  error: null,
  rejected: [],
};
