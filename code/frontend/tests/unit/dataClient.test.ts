import { describe, expect, it } from "vitest";

import { createClient } from "../../src/lib/dataClient";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("dataClient", () => {
  it("builds the distribution query and parses the response", async () => {
    const calls: string[] = [];
    const c = createClient({
      baseUrl: "http://t",
      fetchImpl: async (url) => {
        calls.push(url);
        return jsonResponse({
          model_id: "gpt2", revision: "r", temperature: 0.7, top_token: 5,
          top_token_str: "x", top: [{ token_id: 5, token_str: "x", prob: 0.5 }], tail_mass: 0.5,
        });
      },
    });
    const d = await c.getDistribution("gpt2", "Hello", 0.7, 10);
    expect(calls[0]).toContain("/api/distribution");
    expect(calls[0]).toContain("model_id=gpt2");
    expect(calls[0]).toContain("temperature=0.7");
    expect(calls[0]).toContain("top_k=10");
    expect(d.top_token).toBe(5);
  });

  it("surfaces the error envelope as an ApiError", async () => {
    const c = createClient({
      fetchImpl: async () =>
        jsonResponse({ error: { type: "UnsupportedModelError", message: "nope" } }, 422),
    });
    await expect(c.resolveModel("bad")).rejects.toMatchObject({
      type: "UnsupportedModelError",
      message: "nope",
    });
  });

  it("ensureArtifact returns immediately on a cache hit (no polling)", async () => {
    let calls = 0;
    const c = createClient({
      fetchImpl: async () => {
        calls++;
        return jsonResponse({ cache_key: "k1", job_id: null, status: "complete", ready: true });
      },
    });
    const key = await c.ensureArtifact("embeddings", "gpt2", {});
    expect(key).toBe("k1");
    expect(calls).toBe(1);
  });

  it("ensureArtifact polls the job until done and reports progress", async () => {
    const seq = [
      jsonResponse({ cache_key: "k2", job_id: "j1", status: "running", ready: false }),
      jsonResponse({ job_id: "j1", cache_key: "k2", status: "running", progress: 0.5, message: "half", error: null, version: 1 }),
      jsonResponse({ job_id: "j1", cache_key: "k2", status: "done", progress: 1, message: "done", error: null, version: 2 }),
    ];
    let i = 0;
    const progresses: number[] = [];
    const c = createClient({ pollIntervalMs: 1, fetchImpl: async () => seq[Math.min(i++, seq.length - 1)] });
    const key = await c.ensureArtifact("reduction_2d", "gpt2", {}, {}, (p) => progresses.push(p));
    expect(key).toBe("k2");
    expect(progresses).toContain(0.5);
    expect(progresses[progresses.length - 1]).toBe(1);
  });

  it("pollJob rejects when the job errors", async () => {
    const c = createClient({
      pollIntervalMs: 1,
      fetchImpl: async () =>
        jsonResponse({ job_id: "j", cache_key: "k", status: "error", progress: 0, message: "", error: { type: "ComputeError", message: "boom" }, version: 1 }),
    });
    await expect(c.pollJob("j")).rejects.toMatchObject({ type: "ComputeError", message: "boom" });
  });
});
