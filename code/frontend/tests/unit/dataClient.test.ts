import { describe, expect, it } from "vitest";

import { createClient } from "../../src/lib/dataClient";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("dataClient", () => {
  it("builds the tokenize query and parses the response", async () => {
    const calls: string[] = [];
    const c = createClient({
      baseUrl: "http://t",
      fetchImpl: async (url) => {
        calls.push(url);
        return jsonResponse({
          model_id: "gpt2",
          tokens: [{ token: 15496, token_str: "Hello" }],
        });
      },
    });
    const r = await c.tokenize("gpt2", "Hello");
    expect(calls[0]).toContain("/api/tokenize");
    expect(calls[0]).toContain("model_id=gpt2");
    expect(calls[0]).toContain("text=Hello");
    expect(r.tokens[0].token_str).toBe("Hello");
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

  it("pollJob reports progress until the job is done", async () => {
    const seq = [
      jsonResponse({ job_id: "j1", cache_key: "k2", status: "running", progress: 0.5, message: "half", error: null, version: 1 }),
      jsonResponse({ job_id: "j1", cache_key: "k2", status: "done", progress: 1, message: "done", error: null, version: 2 }),
    ];
    let i = 0;
    const progresses: number[] = [];
    const c = createClient({ pollIntervalMs: 1, fetchImpl: async () => seq[Math.min(i++, seq.length - 1)] });
    await c.pollJob("j1", (p) => progresses.push(p));
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
