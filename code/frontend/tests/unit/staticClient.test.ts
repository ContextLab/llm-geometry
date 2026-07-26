/**
 * Static client unit tests against the REAL exported assets in
 * public/static-data/ (no mock payloads): preset matching, honest
 * StaticModeError refusals, precomputed arch serving incl. uint8 tile
 * dequantization, geo delegation to the golden-tested engine, and the local
 * job emulation for fine-tunes. transformers.js is NEVER downloaded here —
 * the live runtime is exercised through an injected fake (e2e covers the
 * real one).
 */

import { describe, expect, it } from "vitest";

import type {
  ArchGenerateResult,
  GeoSpec,
  TokenizeResult,
} from "../../src/lib/dataClient";
import { ApiError } from "../../src/lib/dataClient";
import { createStaticClient, isStaticClient } from "../../src/lib/staticClient";
import { dequantizeTile } from "../../src/lib/staticClient/arch";
import type { ArchRuntime } from "../../src/lib/staticClient/runtimeTypes";
import { fsStaticFetch, readStaticJson } from "./staticTestUtils";

interface TilesJson {
  bin: string;
  tiles: {
    param: string;
    shape: number[];
    grid_shape: number[];
    stats: { min: number; max: number; mean: number; std: number };
    offset: number;
    nbytes: number;
    vmin: number;
    vmax: number;
  }[];
}

function makeClient(runtime?: ArchRuntime) {
  return createStaticClient({
    fetchImpl: fsStaticFetch(),
    runtimeLoader: runtime ? async () => runtime : undefined,
  });
}

describe("model catalog", () => {
  it("lists exactly the exported models with real graph capabilities", async () => {
    const c = makeClient();
    const idx = await readStaticJson<{ arch_models: { model_id: string; revision: string }[] }>(
      "index.json",
    );
    const { models } = await c.listModels();
    expect(models.map((m) => m.model_id).sort()).toEqual(
      idx.arch_models.map((m) => m.model_id).sort(),
    );
    for (const m of models) {
      const source = idx.arch_models.find((a) => a.model_id === m.model_id)!;
      expect(m.revision).toBe(source.revision);
      expect(m.status).toBe("supported");
      expect(m.capabilities.num_layers).toBeGreaterThan(0);
      expect(m.capabilities.vocab_size).toBeGreaterThan(1000);
      expect(m.capabilities.exposes_hidden_states).toBe(false);
    }
  });

  it("resolves curated models and refuses everything else", async () => {
    const c = makeClient();
    const ref = await c.resolveModel("gpt2");
    expect(ref.model_id).toBe("gpt2");
    expect(ref.capabilities.num_layers).toBe(12);
    await expect(c.resolveModel("meta-llama/Llama-3.1-8B")).rejects.toMatchObject({
      type: "StaticModeError",
    });
  });
});

describe("arch precomputed serving", () => {
  it("serves the real graph JSON", async () => {
    const c = makeClient();
    const g = await c.getArchGraph("gpt2");
    expect(g.model_id).toBe("gpt2");
    expect(g.nodes.length).toBeGreaterThan(50);
    expect(g.edges.length).toBeGreaterThan(50);
    expect(g.meta.n_layers).toBe(12);
  });

  it("serves precomputed traces for example prompts only", async () => {
    const c = makeClient();
    const presets = await c.staticArchTracePresets("gpt2");
    expect(presets.length).toBeGreaterThanOrEqual(2);
    const first = presets[0];
    const trace = await c.getArchTrace({
      model_id: "gpt2",
      prompt: first.prompt,
      system_prompt: first.system_prompt,
    });
    expect(trace.tokens.length).toBeGreaterThan(0);
    expect(trace.layers.length).toBe(12);
    expect(trace.node_activations.length).toBeGreaterThan(0);
    const err = await c
      .getArchTrace({ model_id: "gpt2", prompt: "an arbitrary un-exported prompt" })
      .then(
        () => null,
        (e: unknown) => e as ApiError,
      );
    expect(err!.type).toBe("StaticModeError");
    expect(err!.message).toContain(first.label);
  });

  it("serves over-budget weight windows from the uint8 overview tiles", async () => {
    const c = makeClient();
    const tiles = await readStaticJson<TilesJson>("arch/gpt2/tiles.json");
    const tile = tiles.tiles.find((t) => t.param === "transformer.wte.weight")!;
    const w = await c.getArchWeights({ model_id: "gpt2", param: "transformer.wte.weight" });
    expect(w.method).toBe("strided_mean");
    expect(w.downsampled).toBe(true);
    expect(w.grid_shape).toEqual(tile.grid_shape);
    expect([w.r0, w.r1, w.c0, w.c1]).toEqual([0, tile.shape[0], 0, tile.shape[1]]);
    expect(w.stats).toEqual(tile.stats);
    expect(w.values.length).toBe(tile.grid_shape[0]);
    for (const row of w.values) {
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(tile.vmin - 1e-9);
        expect(v).toBeLessThanOrEqual(tile.vmax + 1e-9);
      }
    }
  });

  it("dequantizes tiles exactly per the manifest encoding", () => {
    const grid = dequantizeTile(new Uint8Array([0, 128, 255, 51]), 2, 2, -1, 3);
    expect(grid[0][0]).toBeCloseTo(-1, 12);
    expect(grid[0][1]).toBeCloseTo(-1 + (128 / 255) * 4, 12);
    expect(grid[1][0]).toBeCloseTo(3, 12);
    expect(grid[1][1]).toBeCloseTo(-1 + (51 / 255) * 4, 12);
  });

  it("rejects unknown params and bad windows with the backend's error types", async () => {
    const c = makeClient();
    await expect(
      c.getArchWeights({ model_id: "gpt2", param: "no.such.param" }),
    ).rejects.toMatchObject({ type: "NotFoundError" });
    await expect(
      c.getArchWeights({ model_id: "gpt2", param: "transformer.wte.weight", r0: 5, r1: 5 }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
    await expect(
      c.getArchWeights({ model_id: "nope/nope", param: "transformer.wte.weight" }),
    ).rejects.toMatchObject({ type: "StaticModeError" });
  });
});

describe("live runtime seam (no model downloads)", () => {
  const fakeRuntime = (calls: unknown[][]): ArchRuntime => ({
    info: () => ({
      status: "ready",
      device: "wasm",
      dtype: "q8",
      model_id: "gpt2",
      onnx_repo: "onnx-community/gpt2-ONNX",
      error: null,
    }),
    tokenize: async (modelId, revision, text): Promise<TokenizeResult> => {
      calls.push(["tokenize", modelId, revision, text]);
      return { model_id: modelId, tokens: [{ token: 1, token_str: "x" }] };
    },
    generate: async (body, onnxRepo): Promise<ArchGenerateResult> => {
      calls.push(["generate", body.model_id, onnxRepo]);
      return {
        text: "hi",
        tokens: [{ id: 1, text: "hi", prob: 0.5, topk: { ids: [1], texts: ["hi"], probs: [0.5] } }],
        finish_reason: "length",
      };
    },
  });

  it("falls back to live tokenization with the pinned revision on preset miss", async () => {
    const calls: unknown[][] = [];
    const c = makeClient(fakeRuntime(calls));
    const idx = await readStaticJson<{ arch_models: { model_id: string; revision: string }[] }>(
      "index.json",
    );
    const gpt2 = idx.arch_models.find((m) => m.model_id === "gpt2")!;
    const out = await c.tokenize("gpt2", "some text no preset recorded");
    expect(out.tokens.length).toBe(1);
    expect(calls).toEqual([["tokenize", "gpt2", gpt2.revision, "some text no preset recorded"]]);
  });

  it("routes archGenerate through the runtime with the mapped ONNX repo", async () => {
    const calls: unknown[][] = [];
    const c = makeClient(fakeRuntime(calls));
    const r = await c.archGenerate({ model_id: "gpt2", prompt: "hello" });
    expect(r.finish_reason).toBe("length");
    // No revision: the ONNX mirror is a different repo from the pinned model, so the
    // model's SHA cannot pin it (it 404s). See transformersRuntime's header + issue #5.
    expect(calls[0]).toEqual(["generate", "gpt2", "onnx-community/gpt2-ONNX"]);
    expect(c.staticRuntimeInfo().generation.status).toBe("ready");
    await expect(c.archGenerate({ model_id: "nope/nope", prompt: "hello" })).rejects.toMatchObject({
      type: "StaticModeError",
    });
  });

  it("reports an idle runtime before any live call", () => {
    const c = makeClient();
    const info = c.staticRuntimeInfo();
    expect(info.mode).toBe("static");
    expect(info.generation.status).toBe("idle");
    expect(info.generation.device).toBeNull();
  });
});

describe("geo delegation + job emulation", () => {
  it("getGeoSpec matches the exported spec.json (engine is the source)", async () => {
    const c = makeClient();
    const exported = await readStaticJson<GeoSpec>("geo/spec.json");
    const spec = await c.getGeoSpec();
    expect(spec.model).toEqual(exported.model);
    expect(spec.special_tokens).toEqual(exported.special_tokens);
    expect(spec.checkpoint.status).toBe("ready");
    expect(spec.checkpoint.checkpoint_id).toBe(exported.checkpoint.checkpoint_id);
  });

  it("geoTrain resolves instantly with the precomputed checkpoint", async () => {
    const c = makeClient();
    const t = await c.geoTrain();
    expect(t.ready).toBe(true);
    expect(t.checkpoint_id).toMatch(/^[0-9a-f]{32}$/);
    await expect(c.geoTrain(7)).rejects.toMatchObject({ type: "StaticModeError" });
  });

  it("delegates tokenize/trace/field/weights to the engine with contract shapes", async () => {
    const c = makeClient();
    const tok = await c.geoTokenize("Alice was beginning to get very tired");
    expect(tok.tokens.length).toBeGreaterThan(3);
    expect(tok.truncated).toBe(false);
    const trace = await c.getGeoTrace("Alice was beginning");
    expect(trace.layers.length).toBe(4);
    expect(trace.probs.length).toBe(1003);
    expect(trace.logits_topk.ids.length).toBe(10);
    const field = await c.getGeoVectorField({ mode: "force", layer: 0, prompt: "Alice was" });
    expect(field.points.length).toBe(1003);
    expect(field.tangent_exact).toBe(false);
    const w = await c.getGeoWeights({ matrix: "W_Q", layer: 0 });
    expect(w.shape).toEqual([3, 3]);
    expect(w.source).toBe("learned");
    // engine errors surface as the contract's typed ApiError envelope
    await expect(c.getGeoTrace("")).rejects.toMatchObject({ type: "InvalidParamError" });
    await expect(
      c.getGeoWeights({ matrix: "W_Q", layer: 0, weights_token: "deadbeef" }),
    ).rejects.toMatchObject({ type: "NotFoundError" });
  });

  it("runs fine-tunes through the local job registry with the backend's done payload", async () => {
    const c = makeClient();
    const body = { text: "Alice was beginning to get very tired of sitting by her sister", steps: 5 };
    const first = await c.geoFinetune(body);
    expect(first.ready).toBe(false);
    expect(first.job_id).toBeTruthy();
    const progress: number[] = [];
    const done = await new Promise<Record<string, unknown> | undefined>((resolve, reject) => {
      c.subscribeProgress(first.job_id!, {
        onProgress: (p) => progress.push(p),
        onDone: resolve,
        onError: (type, message) => reject(new Error(`${type}: ${message}`)),
      });
    });
    expect(done).toBeDefined();
    expect(done!.weights_token).toMatch(/^[0-9a-f]{32}$/);
    expect(done!.loss_before).toBeTypeOf("number");
    expect(done!.loss_after).toBeTypeOf("number");
    expect(progress[progress.length - 1]).toBe(1);
    // the snapshot surface works too
    const snap = await c.getJob(first.job_id!);
    expect(snap.status).toBe("done");
    // identical request → 200-style cache hit carrying the same token
    const again = await c.geoFinetune(body);
    expect(again.ready).toBe(true);
    expect(again.weights_token).toBe(done!.weights_token);
    // the minted token is immediately usable
    const w = await c.getGeoWeights({ matrix: "W_Q", layer: 0, weights_token: again.weights_token });
    expect(w.source).toBe("edited");
  });

  it("supports pollJob and file-based fine-tunes", async () => {
    const c = makeClient();
    const blob = new Blob(["Alice took up the fan and gloves"], { type: "text/plain" });
    const r = await c.geoFinetuneFile(blob, "alice.txt", { steps: 4 });
    expect(r.ready).toBe(false);
    await c.pollJob(r.job_id!);
    const snap = await c.getJob(r.job_id!);
    expect(snap.status).toBe("done");
    await expect(c.geoFinetuneFile(blob, "alice.pdf", {})).rejects.toMatchObject({
      type: "InvalidParamError",
    });
    // hf_dataset is NO LONGER refused here (feature 004): the Hub's dataset viewer is
    // CORS-enabled, so the static build reads real rows. The network path is covered by
    // tests/unit/hfDatasets.test.ts against the real service.
    await expect(c.pollJob("static-job-does-not-exist")).rejects.toMatchObject({
      type: "NotFoundError",
    });
  });
});
