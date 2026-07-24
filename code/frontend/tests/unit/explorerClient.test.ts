// Feature 002 client methods (specs/002-interactive-model-explorer/contracts/api.md,
// FROZEN): URL construction, body shapes, response parsing, abort plumbing, and the
// error envelope — via the repo's fetchImpl-injection pattern. The real client↔backend
// path is exercised by the Playwright e2e against the live API.
import { describe, expect, it, vi } from "vitest";

import { createClient, debounced } from "../../src/lib/dataClient";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("explorer client — Geometry Lab (/api/geo/*)", () => {
  it("getGeoVectorField serializes every contract param and parses the field", async () => {
    const calls: string[] = [];
    const c = createClient({
      baseUrl: "http://t",
      fetchImpl: async (url) => {
        calls.push(url);
        return jsonResponse({
          mode: "force",
          layer: 2,
          points: [[0.1, 0.2, 0.3]],
          token_ids: [17],
          arrows: [{ origin_index: 0, vec: [0.01, -0.02, 0.005], weight: 0.9 }],
          sequence_forces: [{ position: 0, vec: [0.1, 0, 0], normal_residual: 0.02 }],
          tangent_exact: true,
        });
      },
    });
    const f = await c.getGeoVectorField({
      mode: "force",
      layer: 2,
      prompt: "alice was beginning",
      weights_token: "abc123",
      temperature: 0.7,
      top_m: 3,
      antisymmetrize: true,
    });
    expect(calls[0]).toContain("/api/geo/vector_field?");
    expect(calls[0]).toContain("mode=force");
    expect(calls[0]).toContain("layer=2");
    expect(calls[0]).toContain("prompt=alice+was+beginning");
    expect(calls[0]).toContain("weights_token=abc123");
    expect(calls[0]).toContain("temperature=0.7");
    expect(calls[0]).toContain("top_m=3");
    expect(calls[0]).toContain("antisymmetrize=true");
    expect(f.tangent_exact).toBe(true);
    expect(f.arrows[0].origin_index).toBe(0);
    expect(f.sequence_forces?.[0].normal_residual).toBe(0.02);
  });

  it("getGeoVectorField omits optional params left unset", async () => {
    const calls: string[] = [];
    const c = createClient({
      fetchImpl: async (url) => {
        calls.push(url);
        return jsonResponse({
          mode: "next_next",
          layer: "full",
          points: [],
          token_ids: [],
          arrows: [],
          sequence_forces: null,
          tangent_exact: false,
        });
      },
    });
    await c.getGeoVectorField({ mode: "next_next", layer: "full", prompt: "alice" });
    expect(calls[0]).toContain("mode=next_next");
    expect(calls[0]).toContain("layer=full");
    expect(calls[0]).not.toContain("weights_token");
    expect(calls[0]).not.toContain("temperature");
    expect(calls[0]).not.toContain("top_m");
    expect(calls[0]).not.toContain("antisymmetrize");
  });

  it("postGeoWeights sends the contract body and parses the minted token", async () => {
    let captured: { url: string; init?: RequestInit } | null = null;
    const c = createClient({
      fetchImpl: async (url, init) => {
        captured = { url, init };
        return jsonResponse({
          weights_token: "cafe42",
          edited: [{ layer: 1, matrix: "W_V", source: "preset:identity" }],
        });
      },
    });
    const r = await c.postGeoWeights({
      base: "learned",
      edits: [{ layer: 1, matrix: "W_V", preset: "identity", values: null, seed: 0 }],
    });
    expect(captured!.url).toContain("/api/geo/weights");
    expect(captured!.init?.method).toBe("POST");
    expect((captured!.init?.headers as Record<string, string>)["content-type"]).toBe(
      "application/json",
    );
    const body = JSON.parse(captured!.init?.body as string);
    expect(body).toEqual({
      base: "learned",
      edits: [{ layer: 1, matrix: "W_V", preset: "identity", values: null, seed: 0 }],
    });
    expect(r.weights_token).toBe("cafe42");
    expect(r.edited[0].source).toBe("preset:identity");
  });

  it("getGeoTrace passes weights_token and the AbortSignal through to fetch", async () => {
    let capturedInit: RequestInit | undefined;
    const calls: string[] = [];
    const c = createClient({
      fetchImpl: async (url, init) => {
        calls.push(url);
        capturedInit = init;
        return jsonResponse({
          tokens: [{ id: 17, text: "alice", unk: false }],
          embeddings: [[0, 0, 1]],
          layers: [],
          probs: [],
          logits_topk: { ids: [], texts: [], probs: [] },
          next_token: { id: 3, text: "was" },
        });
      },
    });
    const ac = new AbortController();
    const t = await c.getGeoTrace("alice", "abc123", ac.signal);
    expect(calls[0]).toContain("/api/geo/trace?");
    expect(calls[0]).toContain("prompt=alice");
    expect(calls[0]).toContain("weights_token=abc123");
    expect(capturedInit?.signal).toBe(ac.signal);
    expect(t.next_token.text).toBe("was");
  });

  it("an aborted signal rejects the in-flight trace (cancel-and-restart, FR-108)", async () => {
    const c = createClient({
      fetchImpl: (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          );
        }),
    });
    const ac = new AbortController();
    const pending = c.getGeoTrace("alice", undefined, ac.signal);
    const assertion = expect(pending).rejects.toMatchObject({ name: "AbortError" });
    ac.abort();
    await assertion;
  });

  it("surfaces the 002 error envelope as an ApiError (force + layer=full ⇒ 400)", async () => {
    const c = createClient({
      fetchImpl: async () =>
        jsonResponse(
          { error: { type: "InvalidWeightEditError", message: "bad shape" } },
          422,
        ),
    });
    await expect(
      c.postGeoWeights({ base: "learned", edits: [{ layer: 0, matrix: "W_Q", values: [[1]] }] }),
    ).rejects.toMatchObject({ type: "InvalidWeightEditError", message: "bad shape" });
  });
});

describe("explorer client — Architecture Explorer (/api/arch/*)", () => {
  it("getArchWeights serializes the window params", async () => {
    const calls: string[] = [];
    const c = createClient({
      fetchImpl: async (url) => {
        calls.push(url);
        return jsonResponse({
          param: "model.layers.0.self_attn.q_proj.weight",
          shape: [576, 576],
          r0: 0,
          r1: 64,
          c0: 0,
          c1: 64,
          downsampled: false,
          grid_shape: [64, 64],
          values: [[0.1]],
          stats: { min: -1, max: 1, mean: 0, std: 0.2 },
          method: "exact",
        });
      },
    });
    const w = await c.getArchWeights({
      model_id: "HuggingFaceTB/SmolLM2-135M-Instruct",
      param: "model.layers.0.self_attn.q_proj.weight",
      r0: 0,
      r1: 64,
      c0: 0,
      c1: 64,
      max_cells: 4096,
    });
    expect(calls[0]).toContain("/api/arch/weights?");
    expect(calls[0]).toContain("model_id=HuggingFaceTB%2FSmolLM2-135M-Instruct");
    expect(calls[0]).toContain("param=model.layers.0.self_attn.q_proj.weight");
    expect(calls[0]).toContain("r0=0");
    expect(calls[0]).toContain("r1=64");
    expect(calls[0]).toContain("c0=0");
    expect(calls[0]).toContain("c1=64");
    expect(calls[0]).toContain("max_cells=4096");
    expect(w.method).toBe("exact");
    expect(w.downsampled).toBe(false);
  });

  it("archGenerate posts the contract body and parses generated tokens", async () => {
    let captured: RequestInit | undefined;
    const c = createClient({
      fetchImpl: async (_url, init) => {
        captured = init;
        return jsonResponse({
          text: "Paris.",
          tokens: [
            {
              id: 42,
              text: "Paris",
              prob: 0.91,
              topk: { ids: [42, 7, 8, 9, 10], texts: ["Paris", "a", "b", "c", "d"], probs: [0.91, 0.02, 0.02, 0.01, 0.01] },
            },
          ],
          finish_reason: "eos",
        });
      },
    });
    const r = await c.archGenerate({
      model_id: "HuggingFaceTB/SmolLM2-135M-Instruct",
      prompt: "What is the capital of France?",
      system_prompt: null,
      temperature: 0.8,
      max_new_tokens: 64,
      seed: 0,
    });
    expect(captured?.method).toBe("POST");
    const body = JSON.parse(captured?.body as string);
    expect(body.model_id).toBe("HuggingFaceTB/SmolLM2-135M-Instruct");
    expect(body.temperature).toBe(0.8);
    expect(body.max_new_tokens).toBe(64);
    expect(body.seed).toBe(0);
    expect(r.finish_reason).toBe("eos");
    expect(r.tokens[0].topk.ids).toHaveLength(5);
  });

  it("getArchTrace serializes params and forwards the AbortSignal", async () => {
    let capturedInit: RequestInit | undefined;
    const calls: string[] = [];
    const c = createClient({
      fetchImpl: async (url, init) => {
        calls.push(url);
        capturedInit = init;
        return jsonResponse({
          tokens: [{ id: 1, text: "Hi" }],
          chat_template_used: true,
          layers: [],
          logits_topk: { ids: [], texts: [], probs: [] },
          node_activations: [],
        });
      },
    });
    const ac = new AbortController();
    await c.getArchTrace(
      { model_id: "m", prompt: "Hi", system_prompt: "be brief", max_context: 64 },
      ac.signal,
    );
    expect(calls[0]).toContain("/api/arch/trace?");
    expect(calls[0]).toContain("model_id=m");
    expect(calls[0]).toContain("prompt=Hi");
    expect(calls[0]).toContain("system_prompt=be+brief");
    expect(calls[0]).toContain("max_context=64");
    expect(capturedInit?.signal).toBe(ac.signal);
  });

  it("surfaces ModelTooLargeError from the envelope", async () => {
    const c = createClient({
      fetchImpl: async () =>
        jsonResponse(
          { error: { type: "ModelTooLargeError", message: "over the parameter ceiling" } },
          422,
        ),
    });
    await expect(c.getArchGraph("meta-llama/Llama-3.1-405B")).rejects.toMatchObject({
      type: "ModelTooLargeError",
      message: "over the parameter ceiling",
    });
  });
});

describe("debounced helper", () => {
  it("fires once, trailing-edge, with the last args", () => {
    vi.useFakeTimers();
    try {
      const seen: number[] = [];
      const d = debounced((n: number) => seen.push(n), 400);
      d(1);
      vi.advanceTimersByTime(200);
      d(2);
      vi.advanceTimersByTime(399);
      expect(seen).toEqual([]);
      vi.advanceTimersByTime(1);
      expect(seen).toEqual([2]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("cancel() drops the pending call", () => {
    vi.useFakeTimers();
    try {
      const fn = vi.fn();
      const d = debounced(fn, 400);
      d();
      d.cancel();
      vi.advanceTimersByTime(1000);
      expect(fn).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});
