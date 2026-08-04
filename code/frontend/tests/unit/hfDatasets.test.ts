/**
 * HuggingFace datasets in the browser (feature 004, FR-411) — against the REAL
 * datasets-server, no mocks. This is the service the static build reads corpora
 * from, so if its shape or CORS policy changes, this test is how we find out.
 */

import { describe, expect, it } from "vitest";

import { fetchDatasetText, listSplits } from "../../src/lib/staticClient/hfDatasets";

const DATASET = "roneneldan/TinyStories";

describe("HuggingFace dataset viewer (real service)", () => {
  it("lists real config/split pairs", async () => {
    const splits = await listSplits(DATASET);
    expect(splits.length).toBeGreaterThan(0);
    expect(splits.every((s) => s.dataset === DATASET && s.config && s.split)).toBe(true);
    expect(splits.some((s) => s.split === "train")).toBe(true);
  }, 60_000);

  it("reads real rows and joins them like the backend does", async () => {
    const seen: string[] = [];
    const result = await fetchDatasetText(DATASET, {
      maxSamples: 20,
      onProgress: (_f, m) => seen.push(m),
    });
    expect(result.dataset).toBe(DATASET);
    expect(result.split).toBeTruthy();
    expect(result.column).toBe("text"); // TinyStories' text column
    expect(result.rows).toBeGreaterThan(0);
    expect(result.rows).toBeLessThanOrEqual(20);
    expect(result.text.length).toBeGreaterThan(200);
    expect(result.text).toContain("\n\n"); // records joined, backend convention
    expect(seen.length).toBeGreaterThan(0); // progress really reported
  }, 60_000);

  it("surfaces a useful error for a dataset that does not exist", async () => {
    await expect(
      listSplits("this-org-does-not-exist-zzz/neither-does-this"),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
  }, 60_000);

  it("rejects an empty dataset id before touching the network", async () => {
    await expect(listSplits("   ")).rejects.toMatchObject({ type: "InvalidParamError" });
  });
});

describe("transient upstream failures", () => {
  // NOT a mock of the service: these drive the real client with a controlled transport to
  // prove its RETRY POLICY. CI saw the live viewer return HTTP 502 while everything else
  // passed, and the same run saw the Hub 429 the static export. A reader who pastes a
  // dataset id during a blip should get their rows, not an error they cannot act on.
  const okBody = {
    splits: [{ dataset: "d", config: "default", split: "train" }],
  };
  const respond = (status: number, body: unknown = {}): Response =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    });

  it("retries a 502 and succeeds when the service recovers", async () => {
    let calls = 0;
    const flaky: typeof fetch = async () => {
      calls += 1;
      return calls < 3 ? respond(502, { error: "Bad Gateway" }) : respond(200, okBody);
    };
    const splits = await listSplits("roneneldan/TinyStories", flaky);
    expect(calls).toBe(3);
    expect(splits.length).toBeGreaterThan(0);
  });

  it("retries a 429 the same way", async () => {
    let calls = 0;
    const limited: typeof fetch = async () => {
      calls += 1;
      return calls < 2 ? respond(429, { error: "Too Many Requests" }) : respond(200, okBody);
    };
    await listSplits("roneneldan/TinyStories", limited);
    expect(calls).toBe(2);
  });

  it("does NOT retry a 404 — an unknown dataset is a real answer", async () => {
    let calls = 0;
    const missing: typeof fetch = async () => {
      calls += 1;
      return respond(404, { error: "Dataset not found" });
    };
    await expect(listSplits("nobody/does-not-exist", missing)).rejects.toThrow(/not found/i);
    // One attempt only: retrying would delay an error the user must see to act on.
    expect(calls).toBe(1);
  });

  it(
    "gives up loudly after exhausting retries, reporting the real status",
    async () => {
      let calls = 0;
      const down: typeof fetch = async () => {
        calls += 1;
        return respond(503, { error: "Service Unavailable" });
      };
      await expect(listSplits("roneneldan/TinyStories", down)).rejects.toThrow(/Unavailable/i);
      expect(calls).toBe(4); // initial + 3 retries
    },
    // The full backoff is 500 + 1500 + 4000 = 6 s of REAL waiting, past vitest's 5 s
    // default. Waiting it out is the point: this asserts the policy actually gives up
    // rather than retrying forever, so the delays must not be shortened away.
    15_000,
  );
});
