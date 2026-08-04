/**
 * REAL-network tests for the safetensors Range reader (project policy: no
 * mocks — these hit the actual HuggingFace CDN at the revisions pinned in
 * public/static-data/arch/<slug>/meta.json). If the network is unreachable the
 * suite skips LOUDLY rather than passing vacuously.
 *
 * Cross-checks:
 *  - a 3×3 exact window of gpt2's wte equals values fetched via a SECOND,
 *    independent single-element range read (byte math done in the test);
 *  - every exact value lies within the full-tensor stats bounds recorded by
 *    the real backend in tiles.json;
 *  - BF16 decode (SmolLM2) produces values within its recorded bounds.
 */

import { beforeAll, describe, expect, it } from "vitest";

import { createStaticClient } from "../../src/lib/staticClient";
import {
  BYTES_PER,
  SafetensorsFile,
  asMatrixShape,
  coalesceRanges,
  decodeScalars,
  type SafeDtype,
} from "../../src/lib/staticClient/safetensors";
import { fsStaticFetch, readStaticJson } from "./staticTestUtils";

interface Meta {
  model_id: string;
  revision: string;
  safetensors_url: string;
}

interface TilesJson {
  tiles: { param: string; stats: { min: number; max: number } }[];
}

let online = false;
let gpt2Meta: Meta;

beforeAll(async () => {
  gpt2Meta = await readStaticJson<Meta>("arch/gpt2/meta.json");
  try {
    const res = await fetch(gpt2Meta.safetensors_url, { headers: { Range: "bytes=0-7" } });
    online = res.status === 206 || res.status === 200;
  } catch {
    online = false;
  }
  if (!online) {
    // eslint-disable-next-line no-console
    console.warn(
      "\n*** OFFLINE: skipping REAL safetensors range-read tests against " +
        `${gpt2Meta?.safetensors_url}. These tests verify the exact-window reader ` +
        "against live HuggingFace CDN bytes and MUST be re-run with network access. ***\n",
    );
  }
}, 30_000);

/** Independent range read of ONE scalar, with byte math done here in the test. */
async function fetchScalar(
  url: string,
  dataStart: number,
  tensorOffset: number,
  dtype: SafeDtype,
  cols: number,
  r: number,
  c: number,
): Promise<number> {
  const bpe = BYTES_PER[dtype];
  const start = dataStart + tensorOffset + (r * cols + c) * bpe;
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${start + bpe - 1}` } });
  expect(res.status).toBe(206);
  const bytes = new Uint8Array(await res.arrayBuffer());
  return decodeScalars(dtype, bytes, 1)[0];
}

describe("coalesceRanges / decode (pure, no network)", () => {
  it("merges adjacent and gap-tolerant ranges in order", () => {
    expect(
      coalesceRanges(
        [
          { start: 100, end: 110 },
          { start: 0, end: 10 },
          { start: 10, end: 20 },
          { start: 25, end: 30 },
        ],
        4,
      ),
    ).toEqual([
      { start: 0, end: 20 },
      { start: 25, end: 30 },
      { start: 100, end: 110 },
    ]);
    expect(coalesceRanges([], 8)).toEqual([]);
  });

  it("decodes F16 including subnormals, infinities and NaN", () => {
    const bits = [0x3c00, 0x3800, 0xc000, 0x0001, 0x7c00, 0xfc00, 0x7e00, 0x8000];
    const bytes = new Uint8Array(bits.length * 2);
    const dv = new DataView(bytes.buffer);
    bits.forEach((b, i) => dv.setUint16(i * 2, b, true));
    const out = decodeScalars("F16", bytes, bits.length);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(0.5);
    expect(out[2]).toBe(-2);
    expect(out[3]).toBe(2 ** -24);
    expect(out[4]).toBe(Infinity);
    expect(out[5]).toBe(-Infinity);
    expect(Number.isNaN(out[6])).toBe(true);
    expect(Object.is(out[7], -0)).toBe(true);
  });

  it("decodes BF16 as the top half of the f32 bit pattern", () => {
    const bits = [0x3f80, 0xc0a0, 0x0000, 0x7f80];
    const bytes = new Uint8Array(bits.length * 2);
    const dv = new DataView(bytes.buffer);
    bits.forEach((b, i) => dv.setUint16(i * 2, b, true));
    const out = decodeScalars("BF16", bytes, bits.length);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(-5);
    expect(out[2]).toBe(0);
    expect(out[3]).toBe(Infinity);
  });

  /**
   * A dtype arrives as a STRING out of a remote file's JSON header, and the supported set
   * was tested with `dtype in BYTES_PER`. `in` walks the prototype chain, so `"constructor"`
   * and its seven siblings passed the gate; `BYTES_PER["constructor"]` is then `Object`
   * itself, `count * bpe` is `NaN`, the length check `bytes.byteLength < NaN` is false, and
   * the decoder's `else` branch — which meant F16 — returned a Float32Array of real-looking
   * numbers for a tensor whose dtype nothing had actually recognised. Unsupported REAL
   * dtypes (`I64`, `F64`, `U8`) were always refused correctly; only the inherited keys slipped.
   */
  it("refuses a dtype that is an inherited object key instead of decoding it as F16", () => {
    const bytes = new Uint8Array(16);
    for (const key of ["constructor", "toString", "valueOf", "hasOwnProperty", "__proto__"]) {
      expect(key in BYTES_PER, `${key} is on Object.prototype`).toBe(true);
      expect(Object.hasOwn(BYTES_PER, key), `${key} is not a supported dtype`).toBe(false);
      expect(() => decodeScalars(key as SafeDtype, bytes, 4), key).toThrow(/unsupported dtype/);
    }
    // A real unsupported dtype is refused the same way, and the three supported ones work.
    expect(() => decodeScalars("I64" as SafeDtype, bytes, 2)).toThrow(/unsupported dtype/);
    for (const dtype of ["F32", "F16", "BF16"] as const) {
      expect(decodeScalars(dtype, bytes, 2)).toHaveLength(2);
    }
  });

  it("matrixizes shapes the way the backend does", () => {
    expect(asMatrixShape([50257, 768])).toEqual([50257, 768]);
    expect(asMatrixShape([768])).toEqual([768, 1]);
    expect(asMatrixShape([12, 3, 4])).toEqual([12, 12]); // reshape(d0, -1)
  });
});

describe("safetensors range reads (REAL HuggingFace CDN)", () => {
  it("reads the gpt2 header and finds wte at its recorded shape", async (ctx) => {
    if (!online) return ctx.skip();
    const file = new SafetensorsFile(gpt2Meta.safetensors_url);
    const header = await file.header();
    const wte = header.tensors["wte.weight"]; // gpt2's file drops "transformer."
    expect(wte).toBeDefined();
    expect(wte.dtype).toBe("F32");
    expect(wte.shape).toEqual([50257, 768]);
  }, 60_000);

  it("3×3 exact window of gpt2 wte matches independent per-scalar reads AND stats bounds", async (ctx) => {
    if (!online) return ctx.skip();
    const file = new SafetensorsFile(gpt2Meta.safetensors_url);
    const header = await file.header();
    const wte = header.tensors["wte.weight"];
    const [rows, cols] = asMatrixShape(wte.shape);
    const [r0, c0] = [100, 200];
    const win = await file.readWindow("wte.weight", r0, r0 + 3, c0, c0 + 3);
    expect(win.rows).toBe(rows);
    expect(win.cols).toBe(cols);
    expect(win.values.length).toBe(9);

    // (a) independent single-scalar range reads (separate byte math, separate requests)
    for (const [dr, dc] of [
      [0, 0],
      [1, 2],
      [2, 1],
    ]) {
      const independent = await fetchScalar(
        gpt2Meta.safetensors_url,
        header.dataStart,
        wte.data_offsets[0],
        "F32",
        cols,
        r0 + dr,
        c0 + dc,
      );
      expect(win.values[dr * 3 + dc]).toBe(independent);
    }

    // (b) every value within the backend-recorded full-tensor stats bounds
    const tiles = await readStaticJson<TilesJson>("arch/gpt2/tiles.json");
    const stats = tiles.tiles.find((t) => t.param === "transformer.wte.weight")!.stats;
    for (const v of win.values) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(stats.min);
      expect(v).toBeLessThanOrEqual(stats.max);
    }
  }, 120_000);

  it("narrow multi-row windows (span fetch) equal per-row reads", async (ctx) => {
    if (!online) return ctx.skip();
    const file = new SafetensorsFile(gpt2Meta.safetensors_url);
    // 4 rows × 2 cols — exercises the non-full-width span path
    const win = await file.readWindow("wte.weight", 10, 14, 5, 7);
    const single = await file.readWindow("wte.weight", 12, 13, 5, 7);
    expect(Array.from(win.values.subarray(4, 6))).toEqual(Array.from(single.values));
  }, 120_000);

  it("decodes a REAL BF16 window (SmolLM2) matching independent per-scalar reads", async (ctx) => {
    if (!online) return ctx.skip();
    // Data-driven on meta.json only (quick exports carry it without SmolLM2 tiles):
    // per-scalar independent Range reads are a STRONGER check than tile bounds —
    // separate byte math + separate requests must agree with the window decode.
    const meta = await readStaticJson<Meta>("arch/HuggingFaceTB__SmolLM2-135M-Instruct/meta.json");
    const file = new SafetensorsFile(meta.safetensors_url);
    const header = await file.header();
    const embed = header.tensors["model.embed_tokens.weight"];
    expect(embed.dtype).toBe("BF16");
    const cols = asMatrixShape(embed.shape)[1];
    const [r0, c0] = [1000, 100];
    const win = await file.readWindow("model.embed_tokens.weight", r0, r0 + 2, c0, c0 + 3);
    expect(win.values.length).toBe(6);
    for (const [dr, dc] of [
      [0, 0],
      [0, 2],
      [1, 1],
    ]) {
      const independent = await fetchScalar(
        meta.safetensors_url,
        header.dataStart,
        embed.data_offsets[0],
        "BF16",
        cols,
        r0 + dr,
        c0 + dc,
      );
      expect(win.values[dr * 3 + dc]).toBe(independent);
    }
    for (const v of win.values) {
      expect(Number.isFinite(v)).toBe(true);
      expect(Math.abs(v)).toBeLessThan(10); // sane embedding magnitude, not garbage bits
    }
  }, 120_000);

  it("getArchWeights serves EXACT windows end-to-end (static assets + live ranges)", async (ctx) => {
    if (!online) return ctx.skip();
    const c = createStaticClient({ fetchImpl: fsStaticFetch(true) });
    const w = await c.getArchWeights({
      model_id: "gpt2",
      param: "transformer.wte.weight",
      r0: 100,
      r1: 103,
      c0: 200,
      c1: 203,
    });
    expect(w.method).toBe("exact");
    expect(w.downsampled).toBe(false);
    expect(w.grid_shape).toEqual([3, 3]);
    expect(w.shape).toEqual([50257, 768]);
    // cross-check one cell against an independent read
    const file = new SafetensorsFile(gpt2Meta.safetensors_url);
    const header = await file.header();
    const wte = header.tensors["wte.weight"];
    const independent = await fetchScalar(
      gpt2Meta.safetensors_url,
      header.dataStart,
      wte.data_offsets[0],
      "F32",
      768,
      101,
      201,
    );
    expect(w.values[1][1]).toBe(independent);
    expect(w.stats.min).toBeLessThanOrEqual(w.stats.mean);
    expect(w.stats.max).toBeGreaterThanOrEqual(w.stats.mean);
  }, 120_000);
});
