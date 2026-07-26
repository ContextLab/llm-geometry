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
