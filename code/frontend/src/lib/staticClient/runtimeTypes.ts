/**
 * Shared types for the in-browser transformers.js runtime, split out so
 * arch.ts / index.ts can type against the runtime WITHOUT importing the heavy
 * module (which is only ever loaded via dynamic import on first use).
 */

import type { ArchGenerateBody, ArchGenerateResult, TokenizeResult } from "../dataClient";

export interface RuntimeGenerationInfo {
  status: "idle" | "loading" | "ready" | "error";
  device: "webgpu" | "wasm" | null;
  dtype: "q4f16" | "q8" | null;
  model_id: string | null; // the HF model whose ONNX export is loaded
  onnx_repo: string | null;
  error: string | null;
}

export interface StaticRuntimeInfo {
  mode: "static";
  generation: RuntimeGenerationInfo;
}

/** The surface the lazily-imported transformersRuntime module implements. */
export interface ArchRuntime {
  info(): RuntimeGenerationInfo;
  /** Live tokenization from the pinned original-repo tokenizer files. */
  tokenize(modelId: string, revision: string, text: string): Promise<TokenizeResult>;
  /** Live generation (webgpu q4f16 → wasm q8 fallback) with real per-token probs. */
  /** The ONNX mirror is resolved at `main` — see transformersRuntime's header. */
  generate(body: ArchGenerateBody, onnxRepo: string): Promise<ArchGenerateResult>;
}

export type RuntimeLoader = () => Promise<ArchRuntime>;

export const IDLE_GENERATION_INFO: RuntimeGenerationInfo = {
  status: "idle",
  device: null,
  dtype: null,
  model_id: null,
  onnx_repo: null,
  error: null,
};
