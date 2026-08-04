/**
 * The ONLY module that touches @huggingface/transformers (v4). It is loaded
 * exclusively via dynamic import (see arch.ts), so the library + ONNX runtime
 * land in a lazy chunk that only downloads when the user actually asks for
 * live tokenization or generation.
 *
 * Honesty contract (FR-203):
 * - Tokenization uses the ORIGINAL model repo's tokenizer files at the pinned
 *   revision from meta.json — real BPE, no vendored copies.
 * - Generation runs the model's community ONNX export (webgpu/q8, falling
 *   back to wasm/q8) and reports per-token probabilities computed from REAL
 *   logits via one teacher-forced forward pass over prompt+reply — the same
 *   quantities the backend reports (chosen-token prob under the temperature
 *   softmax; top-5 alternatives under the plain softmax), just from the
 *   quantized weights. `seed` is NOT honored (transformers.js has no seeded
 *   sampler); callers receive real samples, never fake determinism.
 * - The ONNX repo resolves at `main`, and this is the one thing here that is NOT
 *   pinned. The pinned revision we hold belongs to the ORIGINAL model repo (it is
 *   what the tokenizer and the safetensors range reads use); the community ONNX
 *   mirror is a DIFFERENT repository with its own commit history, so that SHA simply
 *   404s against it — measured, not assumed. Recording each mirror's own SHA is
 *   tracked in issue #5.
 */

import {
  AutoTokenizer,
  Tensor,
  env,
  pipeline,
  type PreTrainedTokenizer,
  type TextGenerationPipeline,
} from "@huggingface/transformers";

import type { ArchGenerateBody, ArchGenerateResult, ArchGeneratedToken, TokenizeResult } from "../dataClient";
import { computeError, invalidParamError } from "./errors";
import { assertNonDegenerateLogits } from "./logitsSanity";
import {
  IDLE_GENERATION_INFO,
  RUNTIME_LADDER,
  type ArchRuntime,
  type RuntimeGenerationInfo,
  type RuntimeScoredText,
} from "./runtimeTypes";

const MAX_NEW_TOKENS_LIMIT = 128; // ARCH_MAX_NEW_TOKENS (backend config)
const TOPK = 5;

// Decoding constraints — MIRROR of llm_geometry/config.py (ARCH_TOP_P / ARCH_TOP_K /
// ARCH_REPETITION_PENALTY). Keep the two in sync: the whole point is that the static
// build and the full stack sample identically.
const TOP_P = 0.9;
const TOP_K = 50;
const REPETITION_PENALTY = 1.1;

// Serve the ONNX runtime's WASM from OUR origin. transformers.js defaults this to a
// cdn.jsdelivr.net URL for a pinned PRERELEASE build, which made the deployed site's
// headline feature depend on a third party staying up and keeping that version
// published. vite.config.ts copies the byte-identical files out of the installed
// onnxruntime-web into <base>ort/.
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`;
}

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
    // `shader-f16` is NOT needed by the q8 path — it is kept as the marker of a
    // HARDWARE adapter. Measured across Chromium configurations on this stack: the
    // real Apple Metal-3 adapter advertises it, the software (SwiftShader) adapter
    // does not, and plain headless Chromium has no adapter at all. Software WebGPU
    // is slower than the WASM backend, so it is not worth preferring.
    return adapter != null && adapter.features?.has("shader-f16") === true;
  } catch {
    return false;
  }
}

// One forward pass over a handful of tokens — the whole load-time check.
const SELF_CHECK_PROMPT = "The capital of France is Paris. The capital of Germany is";

/**
 * The load-time correctness gate (see logitsSanity.ts for why a thrown-exception
 * ladder is not enough). Throws if the freshly built session's next-token
 * distribution does not depend on its input.
 */
async function selfCheck(p: TextGenerationPipeline, label: string): Promise<number> {
  const ids = p.tokenizer.encode(SELF_CHECK_PROMPT, { add_special_tokens: false });
  if (ids.length < 2) {
    throw new Error(`${label}: tokenizer returned ${ids.length} tokens for the self-check prompt`);
  }
  const out = (await (p.model as unknown as (o: Record<string, unknown>) => Promise<{ logits: Tensor }>)({
    input_ids: idsTensor(ids),
    attention_mask: new Tensor("int64", BigInt64Array.from(ids, () => 1n), [1, ids.length]),
  })) as { logits: Tensor };
  const [, seqLen, vocab] = out.logits.dims as number[];
  try {
    return assertNonDegenerateLogits(out.logits.data as Float32Array, seqLen, vocab, label);
  } finally {
    (out.logits as unknown as { dispose?: () => void }).dispose?.();
  }
}

async function getPipeline(onnxRepo: string): Promise<TextGenerationPipeline> {
  if (pipelineCache?.key === onnxRepo) return pipelineCache.promise;
  const promise = (async () => {
    generationInfo = {
      ...IDLE_GENERATION_INFO,
      status: "loading",
      onnx_repo: onnxRepo,
    };
    const skipWebgpu = !(await webgpuUsable());
    const rejected: string[] = [];
    let lastError = "";
    for (const { device, dtype } of RUNTIME_LADDER) {
      const label = `${device}/${dtype}`;
      if (device === "webgpu" && skipWebgpu) continue;
      generationInfo = { ...generationInfo, status: "loading", device, dtype, rejected: [...rejected] };
      try {
        const p = (await pipeline("text-generation", onnxRepo, { dtype, device })) as TextGenerationPipeline;
        // "The session constructed" is not evidence that the model works.
        await selfCheck(p, `${onnxRepo} on ${label}`);
        generationInfo = { ...generationInfo, status: "ready", device, dtype, rejected: [...rejected] };
        return p;
      } catch (e) {
        lastError = e instanceof Error ? e.message : String(e);
        rejected.push(label);
        console.warn(`[staticClient] rejected ${label} for ${onnxRepo}: ${lastError}`);
      }
    }
    const msg = lastError || "no usable device/dtype for this browser";
    generationInfo = { ...generationInfo, status: "error", error: msg, rejected: [...rejected] };
    throw computeError(`Could not load the in-browser model ${onnxRepo}: ${msg}`);
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
  // temperature softmax restricted to the top-k ∩ top-p nucleus, with a repetition
  // penalty. These MUST match llm_geometry.config's ARCH_TOP_P/ARCH_TOP_K/
  // ARCH_REPETITION_PENALTY so the static build and the full stack decode the same
  // way. (Sampling the full vocabulary — the previous top_k:0, top_p:1.0 — draws from
  // the long tail every step, which is what made small models produce word salad.)
  // Only the draw is filtered; every probability reported below still comes from the
  // unfiltered distribution, exactly as the backend does it.
  const generated = (await (
    model as unknown as {
      generate(o: Record<string, unknown>): Promise<Tensor>;
    }
  ).generate({
    input_ids: inputIds,
    attention_mask: attention,
    max_new_tokens: maxNew,
    do_sample: temperature > 0,
    repetition_penalty: REPETITION_PENALTY,
    ...(temperature > 0 ? { temperature, top_k: TOP_K, top_p: TOP_P } : {}),
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
    // The reply comes from the quantized incremental decode; the displayed
    // distributions come from this full teacher-forced re-scoring pass. Near-ties
    // can rank differently between the two — annotate instead of contradicting
    // (red-team static finding #1): greedy sampling prob is 1.0 BY DEFINITION of
    // the decode that chose it, and a mismatch gets an explicit note.
    const rescoreMismatch = temperature === 0 && id !== topIds[0];
    tokens.push({
      id,
      text: tokenizer.decode([id]),
      // Chosen-token prob under the SAMPLING distribution (1.0 when greedy),
      // exactly like the backend's generate loop.
      prob: temperature === 0 ? 1.0 : softmaxProb(logits, id, temperature),
      topk: {
        ids: topIds,
        texts: topIds.map((t) => tokenizer.decode([t])),
        // Alternatives always report the model's plain softmax (backend rule).
        probs: topIds.map((t) => softmaxProb(logits, t, 1)),
      },
      ...(rescoreMismatch
        ? {
            note:
              "greedy pick under the quantized in-browser decode; the full-precision " +
              "re-scoring pass ranks a near-tied alternative first",
          }
        : {}),
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

/**
 * Byte-level pieces for a text — the strings the UTF-8 span algorithm decodes (§8.2).
 *
 * `_tokenizer` is the underlying `tokenizers` object; `encode(...).tokens` is the only
 * place transformers.js 4.x surfaces the raw pieces (there is no offsets API at all).
 * It is a private-ish field, so it is checked here and the failure is loud: the
 * alternative — decoding tokens one at a time — provably corrupts multi-byte characters.
 */
function byteLevelPieces(tokenizer: PreTrainedTokenizer, text: string): string[] {
  const inner = (tokenizer as unknown as { _tokenizer?: { encode(t: string): { tokens?: string[] } } })._tokenizer;
  const tokens = inner?.encode(text)?.tokens;
  if (!Array.isArray(tokens)) {
    throw computeError(
      "this build of @huggingface/transformers does not expose byte-level token pieces " +
        "(tokenizer._tokenizer.encode(text).tokens), so tokens cannot be attributed to " +
        "words. Refusing to guess — per-token decoding corrupts multi-byte characters.",
    );
  }
  return tokens;
}

async function scoreTextsImpl(
  onnxRepo: string,
  texts: readonly string[],
): Promise<RuntimeScoredText[]> {
  if (texts.length === 0) throw invalidParamError("scoreTexts needs at least one text");
  const generator = await getPipeline(onnxRepo);
  const tokenizer = generator.tokenizer;
  const model = generator.model;
  const out: RuntimeScoredText[] = [];
  for (const text of texts) {
    const ids = tokenizer.encode(text, { add_special_tokens: false });
    if (ids.length < 2) {
      throw invalidParamError(
        `a passage must tokenize to at least 2 tokens to be scored, got ${ids.length}`,
      );
    }
    const pieces = byteLevelPieces(tokenizer, text);
    if (pieces.length !== ids.length) {
      throw computeError(
        `the tokenizer returned ${ids.length} ids but ${pieces.length} byte-level pieces`,
      );
    }
    const result = (await (
      model as unknown as (o: Record<string, unknown>) => Promise<{ logits: Tensor }>
    )({
      input_ids: idsTensor(ids),
      attention_mask: new Tensor("int64", BigInt64Array.from(ids, () => 1n), [1, ids.length]),
    })) as { logits: Tensor };
    const [, seqLen, vocab] = result.logits.dims as number[];
    if (seqLen !== ids.length) {
      throw computeError(`the forward pass returned ${seqLen} positions for ${ids.length} tokens`);
    }
    const data = result.logits.data as Float32Array;
    const nll = new Array<number>(ids.length).fill(NaN);
    for (let t = 0; t + 1 < ids.length; t++) {
      const row = data.subarray(t * vocab, (t + 1) * vocab);
      // log softmax at the target, computed stably — there is no log_softmax helper on
      // a transformers.js tensor, and materializing probabilities for a 50k vocabulary
      // at every position would be needlessly expensive.
      let max = -Infinity;
      for (let i = 0; i < row.length; i++) if (row[i] > max) max = row[i];
      let sum = 0;
      for (let i = 0; i < row.length; i++) sum += Math.exp(row[i] - max);
      nll[t + 1] = -(row[ids[t + 1]] - max - Math.log(sum));
    }
    (result.logits as unknown as { dispose?: () => void }).dispose?.();
    out.push({ pieces, nll, nChars: text.length });
  }
  return out;
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

  scoreTexts: scoreTextsImpl,
};
