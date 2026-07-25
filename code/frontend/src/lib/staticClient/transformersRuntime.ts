/**
 * The ONLY module that touches @huggingface/transformers (v4). It is loaded
 * exclusively via dynamic import (see arch.ts), so the library + ONNX runtime
 * land in a lazy chunk that only downloads when the user actually asks for
 * live tokenization or generation.
 *
 * Honesty contract (FR-203):
 * - Tokenization uses the ORIGINAL model repo's tokenizer files at the pinned
 *   revision from meta.json — real BPE, no vendored copies.
 * - Generation runs the model's community ONNX export (webgpu/q4f16, falling
 *   back to wasm/q8) and reports per-token probabilities computed from REAL
 *   logits via one teacher-forced forward pass over prompt+reply — the same
 *   quantities the backend reports (chosen-token prob under the temperature
 *   softmax; top-5 alternatives under the plain softmax), just from the
 *   quantized weights. `seed` is NOT honored (transformers.js has no seeded
 *   sampler); callers receive real samples, never fake determinism.
 */

import {
  AutoTokenizer,
  Tensor,
  pipeline,
  type PreTrainedTokenizer,
  type TextGenerationPipeline,
} from "@huggingface/transformers";

import type { ArchGenerateBody, ArchGenerateResult, ArchGeneratedToken, TokenizeResult } from "../dataClient";
import { computeError, invalidParamError } from "./errors";
import { IDLE_GENERATION_INFO, type ArchRuntime, type RuntimeGenerationInfo } from "./runtimeTypes";

const MAX_NEW_TOKENS_LIMIT = 128; // ARCH_MAX_NEW_TOKENS (backend config)
const TOPK = 5;

let generationInfo: RuntimeGenerationInfo = { ...IDLE_GENERATION_INFO };

const tokenizerCache = new Map<string, Promise<PreTrainedTokenizer>>();
let pipelineCache: { key: string; promise: Promise<TextGenerationPipeline> } | null = null;

function getTokenizer(modelId: string, revision: string): Promise<PreTrainedTokenizer> {
  const key = `${modelId}@${revision}`;
  let p = tokenizerCache.get(key);
  if (!p) {
    p = AutoTokenizer.from_pretrained(modelId, { revision });
    tokenizerCache.set(key, p);
    p.catch(() => tokenizerCache.delete(key));
  }
  return p;
}

async function webgpuUsable(): Promise<boolean> {
  try {
    const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (!gpu) return false;
    const adapter = (await gpu.requestAdapter()) as {
      features?: { has(name: string): boolean };
    } | null;
    // q4f16 needs f16 shaders; without them WASM/q8 is the honest fallback.
    return adapter != null && adapter.features?.has("shader-f16") === true;
  } catch {
    return false;
  }
}

async function getPipeline(onnxRepo: string): Promise<TextGenerationPipeline> {
  if (pipelineCache?.key === onnxRepo) return pipelineCache.promise;
  const promise = (async () => {
    generationInfo = {
      status: "loading",
      device: null,
      dtype: null,
      model_id: null,
      onnx_repo: onnxRepo,
      error: null,
    };
    const tryLoad = async (device: "webgpu" | "wasm", dtype: "q4f16" | "q8") => {
      generationInfo = { ...generationInfo, status: "loading", device, dtype };
      const p = (await pipeline("text-generation", onnxRepo, {
        dtype,
        device,
      })) as TextGenerationPipeline;
      generationInfo = { ...generationInfo, status: "ready", device, dtype };
      return p;
    };
    if (await webgpuUsable()) {
      try {
        return await tryLoad("webgpu", "q4f16");
      } catch (e) {
        // Degradation ladder: WebGPU failed → retry on WASM before giving up.
        console.warn(`[staticClient] webgpu/q4f16 load failed for ${onnxRepo}; falling back to wasm/q8:`, e);
      }
    }
    try {
      return await tryLoad("wasm", "q8");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      generationInfo = { ...generationInfo, status: "error", error: msg };
      throw computeError(`Could not load the in-browser model ${onnxRepo}: ${msg}`);
    }
  })();
  pipelineCache = { key: onnxRepo, promise };
  promise.catch(() => {
    if (pipelineCache?.promise === promise) pipelineCache = null;
  });
  return promise;
}

function tensorIds(t: Tensor): number[] {
  return Array.from(t.data as ArrayLike<number | bigint>, (v) => Number(v));
}

function idsTensor(ids: number[]): Tensor {
  return new Tensor("int64", BigInt64Array.from(ids.map((i) => BigInt(i))), [1, ids.length]);
}

function eosIds(model: unknown, tokenizer: PreTrainedTokenizer): Set<number> {
  const out = new Set<number>();
  const add = (v: unknown): void => {
    if (typeof v === "number") out.add(v);
    else if (Array.isArray(v)) for (const i of v) typeof i === "number" && out.add(i);
  };
  const m = model as { generation_config?: { eos_token_id?: unknown }; config?: { eos_token_id?: unknown } };
  add(m.generation_config?.eos_token_id);
  add(m.config?.eos_token_id);
  add((tokenizer as unknown as { eos_token_id?: unknown }).eos_token_id);
  return out;
}

/** Backend tracing.encode_prompt, mirrored: chat template when available. */
async function encodePrompt(
  tokenizer: PreTrainedTokenizer,
  prompt: string,
  systemPrompt: string | null | undefined,
): Promise<number[]> {
  const hasTemplate = Boolean((tokenizer as unknown as { chat_template?: unknown }).chat_template);
  if (hasTemplate) {
    const messages: { role: string; content: string }[] = [];
    if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
    messages.push({ role: "user", content: prompt });
    const rendered = tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      tokenize: false,
    }) as string;
    return tokenizer.encode(rendered, { add_special_tokens: false });
  }
  const text = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;
  return tokenizer.encode(text);
}

function softmaxProb(logits: Float32Array, index: number, temperature: number): number {
  // Numerically stable softmax(logits/T)[index] without materializing probs.
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let denom = 0;
  for (let i = 0; i < logits.length; i++) denom += Math.exp((logits[i] - max) / temperature);
  return Math.exp((logits[index] - max) / temperature) / denom;
}

function topkByLogits(logits: Float32Array, k: number): number[] {
  const idx = Array.from(logits.keys());
  idx.sort((a, b) => (logits[b] !== logits[a] ? logits[b] - logits[a] : a - b));
  return idx.slice(0, k);
}

async function generateImpl(
  body: ArchGenerateBody,
  onnxRepo: string,
  _revision: string,
): Promise<ArchGenerateResult> {
  const prompt = (body.prompt ?? "").trim();
  if (!prompt) throw invalidParamError("prompt must be a non-empty string");
  const temperature = body.temperature ?? 0.8;
  if (!(temperature >= 0)) throw invalidParamError(`temperature must be >= 0, got ${body.temperature}`);
  const maxNew = Math.trunc(body.max_new_tokens ?? 64);
  if (!(maxNew >= 1 && maxNew <= MAX_NEW_TOKENS_LIMIT)) {
    throw invalidParamError(`max_new_tokens must be in 1..${MAX_NEW_TOKENS_LIMIT}, got ${maxNew}`);
  }

  const generator = await getPipeline(onnxRepo);
  const tokenizer = generator.tokenizer;
  const model = generator.model;
  const promptIds = await encodePrompt(tokenizer, body.prompt, body.system_prompt);
  const inputIds = idsTensor(promptIds);
  const attention = new Tensor(
    "int64",
    BigInt64Array.from({ length: promptIds.length }, () => 1n),
    [1, promptIds.length],
  );

  // Real autoregressive decode. temperature==0 → greedy; otherwise sample the
  // full temperature softmax (top_k/top_p disabled to mirror the backend).
  const generated = (await (
    model as unknown as {
      generate(o: Record<string, unknown>): Promise<Tensor>;
    }
  ).generate({
    input_ids: inputIds,
    attention_mask: attention,
    max_new_tokens: maxNew,
    do_sample: temperature > 0,
    ...(temperature > 0 ? { temperature, top_k: 0, top_p: 1.0 } : {}),
  })) as Tensor;
  const fullIds = tensorIds(generated);
  const newIds = fullIds.slice(promptIds.length);
  if (newIds.length === 0) throw computeError("generation produced no tokens");

  // One teacher-forced pass over prompt+reply for per-position logits — the
  // model's REAL distribution at every generated position.
  const fullTensor = idsTensor(fullIds);
  const fullMask = new Tensor(
    "int64",
    BigInt64Array.from({ length: fullIds.length }, () => 1n),
    [1, fullIds.length],
  );
  const out = (await (model as unknown as (o: Record<string, unknown>) => Promise<{ logits: Tensor }>)({
    input_ids: fullTensor,
    attention_mask: fullMask,
  })) as { logits: Tensor };
  const [, seqLen, vocab] = out.logits.dims as number[];
  const logitsData = out.logits.data as Float32Array;
  if (seqLen !== fullIds.length) throw computeError(`teacher-forced pass returned ${seqLen} positions for ${fullIds.length} tokens`);

  const eos = eosIds(model, tokenizer);
  const tokens: ArchGeneratedToken[] = [];
  let finishReason: "eos" | "length" = "length";
  for (let i = 0; i < newIds.length; i++) {
    const pos = promptIds.length + i - 1; // logits at pos predict token pos+1
    const logits = logitsData.subarray(pos * vocab, (pos + 1) * vocab);
    const id = newIds[i];
    const topIds = topkByLogits(logits, Math.min(TOPK, vocab));
    tokens.push({
      id,
      text: tokenizer.decode([id]),
      // Chosen-token prob under the SAMPLING distribution (1.0 when greedy),
      // exactly like the backend's generate loop.
      prob: temperature === 0 ? (id === topIds[0] ? 1.0 : softmaxProb(logits, id, 1)) : softmaxProb(logits, id, temperature),
      topk: {
        ids: topIds,
        texts: topIds.map((t) => tokenizer.decode([t])),
        // Alternatives always report the model's plain softmax (backend rule).
        probs: topIds.map((t) => softmaxProb(logits, t, 1)),
      },
    });
    if (eos.has(id)) {
      finishReason = "eos";
      break;
    }
  }

  return {
    text: tokenizer.decode(tokens.map((t) => t.id), { skip_special_tokens: true }),
    tokens,
    finish_reason: finishReason,
  };
}

export const runtime: ArchRuntime = {
  info: () => ({ ...generationInfo }),

  async tokenize(modelId: string, revision: string, text: string): Promise<TokenizeResult> {
    const tokenizer = await getTokenizer(modelId, revision);
    const ids = tokenizer.encode(text);
    return {
      model_id: modelId,
      tokens: ids.map((id) => ({ token: id, token_str: tokenizer.decode([id]) })),
    };
  },

  generate: generateImpl,
};
