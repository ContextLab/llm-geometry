/**
 * Exact weight windows over HTTP Range reads of a safetensors file on the
 * HuggingFace CDN (CORS + 206 verified in notes/agent-reports/
 * static-site-research.md). Layout (https://github.com/huggingface/safetensors):
 *
 *   bytes 0..7   little-endian u64 N = header length
 *   bytes 8..8+N JSON header: { name: {dtype, shape, data_offsets:[b,e]}, ... }
 *   data section starts at byte 8+N; data_offsets are relative to it.
 *
 * Tensors are row-major and contiguous, so a [r0,r1)×[c0,c1) window of a [R,C]
 * matrix is (r1-r0) row segments of (c1-c0) elements. Segments are coalesced:
 * full-width windows collapse to ONE range; narrow windows whose total span is
 * still small are fetched as one span and sliced locally; genuinely scattered
 * rows fall back to per-row ranges (gap-coalesced, bounded concurrency).
 *
 * Dtypes: F32 passthrough, BF16 (u16 << 16 into the f32 bit pattern — JS has no
 * BFloat16Array), F16 (manual sign/exponent/mantissa decode incl. subnormals).
 */

import { ApiError } from "../dataClient";

export type SafeDtype = "F32" | "F16" | "BF16";

export const BYTES_PER: Record<SafeDtype, number> = { F32: 4, F16: 2, BF16: 2 };

export interface TensorEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface SafetensorsHeader {
  /** Absolute byte offset of the data section (8 + header length). */
  dataStart: number;
  tensors: Record<string, TensorEntry>;
}

/** Half-open byte range [start, end). */
export interface ByteRange {
  start: number;
  end: number;
}

/** Merge sorted ranges whose gap is at most `gapTolerance` bytes. */
export function coalesceRanges(ranges: ByteRange[], gapTolerance = 0): ByteRange[] {
  if (ranges.length === 0) return [];
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const out: ByteRange[] = [{ ...sorted[0] }];
  for (let i = 1; i < sorted.length; i++) {
    const last = out[out.length - 1];
    const r = sorted[i];
    if (r.start <= last.end + gapTolerance) {
      last.end = Math.max(last.end, r.end);
    } else {
      out.push({ ...r });
    }
  }
  return out;
}

const F32_SCRATCH = new DataView(new ArrayBuffer(4));

function f32FromBits(bits: number): number {
  F32_SCRATCH.setUint32(0, bits);
  return F32_SCRATCH.getFloat32(0);
}

function f16ToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24; // zero / subnormal
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** Decode `count` little-endian scalars of `dtype` from `bytes` into f32. */
export function decodeScalars(dtype: SafeDtype, bytes: Uint8Array, count: number): Float32Array {
  // A dtype that is not one of the three is refused HERE rather than falling into the F16
  // branch below. `dtype` is typed, but it comes from a remote header's JSON at the only
  // call site, and the `else` used to mean "F16" — so an unsupported dtype that slipped
  // past `readWindow`'s gate was decoded as half precision and returned real-looking
  // numbers. `count * bpe` is `NaN` for an unknown dtype, so the length check above cannot
  // catch it either: `n < NaN` is false.
  if (!Object.hasOwn(BYTES_PER, dtype)) {
    throw new ApiError(
      "ComputeError",
      `safetensors decode: unsupported dtype ${JSON.stringify(dtype)}; only F32/F16/BF16 are supported.`,
    );
  }
  const bpe = BYTES_PER[dtype];
  if (bytes.byteLength < count * bpe) {
    throw new ApiError(
      "ComputeError",
      `safetensors decode: need ${count * bpe} bytes for ${count}×${dtype}, got ${bytes.byteLength}`,
    );
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  if (dtype === "F32") {
    for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, true);
  } else if (dtype === "BF16") {
    for (let i = 0; i < count; i++) out[i] = f32FromBits(dv.getUint16(i * 2, true) << 16);
  } else {
    for (let i = 0; i < count; i++) out[i] = f16ToF32(dv.getUint16(i * 2, true));
  }
  return out;
}

/** Backend `_as_matrix`: 1-D → [R,1]; >2-D → [d0, prod(rest)] (row-major). */
export function asMatrixShape(shape: number[]): [number, number] {
  if (shape.length === 0) return [1, 1];
  if (shape.length === 1) return [shape[0], 1];
  let cols = 1;
  for (let i = 1; i < shape.length; i++) cols *= shape[i];
  return [shape[0], cols];
}

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** Byte span above which a narrow window falls back to per-row range reads. */
const SINGLE_SPAN_LIMIT = 4 * 1024 * 1024;
/** Accept a 200 (Range ignored) only when buffering the whole file is cheap. */
const FULL_BODY_LIMIT = 32 * 1024 * 1024;
const ROW_FETCH_CONCURRENCY = 6;
const HEADER_SANITY_LIMIT = 50_000_000;

export class SafetensorsFile {
  private headerPromise: Promise<SafetensorsHeader> | null = null;

  constructor(
    readonly url: string,
    private readonly fetchImpl: FetchLike = (i, init) => fetch(i, init),
  ) {}

  /** Fetch + cache the header (one 8-byte read + one header-sized read). */
  header(): Promise<SafetensorsHeader> {
    if (!this.headerPromise) {
      this.headerPromise = this.loadHeader().catch((e) => {
        this.headerPromise = null;
        throw e;
      });
    }
    return this.headerPromise;
  }

  /**
   * Read the exact [r0,r1)×[c0,c1) window of tensor `name` (bounds are the
   * caller's responsibility to validate against the matrixized shape).
   * Returns row-major f32 values plus the matrixized [rows, cols] of the
   * full tensor.
   */
  async readWindow(
    name: string,
    r0: number,
    r1: number,
    c0: number,
    c1: number,
  ): Promise<{ values: Float32Array; rows: number; cols: number; dtype: SafeDtype }> {
    const header = await this.header();
    // `Object.hasOwn`, not a truthiness test on the lookup. `header.tensors` used to be an
    // ordinary object literal, so `tensors["constructor"]` was `Object` itself and a
    // request for a tensor named after any `Object.prototype` member sailed past this
    // guard into `entry.dtype === undefined`, reporting a dtype problem for a tensor that
    // is not there. The map is built with a null prototype now (see `header()`), which
    // closes that door from the other side — the two are kept together on purpose, because
    // this method is also reached with names from the caller, not only from the file.
    if (!Object.hasOwn(header.tensors, name)) {
      throw new ApiError(
        "NotFoundError",
        `Tensor '${name}' is not in the safetensors index of ${this.url}.`,
      );
    }
    const entry = header.tensors[name];
    const dtype = entry.dtype as SafeDtype;
    if (!Object.hasOwn(BYTES_PER, dtype)) {
      throw new ApiError(
        "ComputeError",
        `Tensor '${name}' has dtype ${entry.dtype}; only F32/F16/BF16 are supported.`,
      );
    }
    const [rows, cols] = asMatrixShape(entry.shape);
    if (!(0 <= r0 && r0 < r1 && r1 <= rows && 0 <= c0 && c0 < c1 && c1 <= cols)) {
      throw new ApiError(
        "InvalidParamError",
        `Window [${r0}:${r1}, ${c0}:${c1}] is out of range for '${name}' with shape [${rows}, ${cols}].`,
      );
    }
    const bpe = BYTES_PER[dtype];
    const tensorStart = header.dataStart + entry.data_offsets[0];
    const nRows = r1 - r0;
    const nCols = c1 - c0;
    const rowBytes = nCols * bpe;
    const values = new Float32Array(nRows * nCols);

    if (c0 === 0 && c1 === cols) {
      // Full-width rows are one contiguous block: a single range request.
      const start = tensorStart + r0 * cols * bpe;
      const bytes = await this.range(start, start + nRows * rowBytes);
      values.set(decodeScalars(dtype, bytes, nRows * nCols));
      return { values, rows, cols, dtype };
    }

    const segments: ByteRange[] = [];
    for (let r = r0; r < r1; r++) {
      const start = tensorStart + (r * cols + c0) * bpe;
      segments.push({ start, end: start + rowBytes });
    }
    const span = segments[segments.length - 1].end - segments[0].start;
    const wanted = nRows * rowBytes;
    if (span <= Math.max(256 * 1024, 4 * wanted) && span <= SINGLE_SPAN_LIMIT) {
      // One over-fetch of the whole span beats dozens of tiny requests.
      const bytes = await this.range(segments[0].start, segments[0].start + span);
      for (let i = 0; i < nRows; i++) {
        const off = segments[i].start - segments[0].start;
        values.set(decodeScalars(dtype, bytes.subarray(off, off + rowBytes), nCols), i * nCols);
      }
      return { values, rows, cols, dtype };
    }

    // Scattered rows: coalesce near-adjacent segments, fetch with bounded
    // concurrency, then slice each row out of its covering range.
    const coalesced = coalesceRanges(segments, 2048);
    const buffers = new Map<ByteRange, Uint8Array>();
    let next = 0;
    const worker = async (): Promise<void> => {
      for (;;) {
        const i = next++;
        if (i >= coalesced.length) return;
        const cr = coalesced[i];
        buffers.set(cr, await this.range(cr.start, cr.end));
      }
    };
    await Promise.all(
      Array.from({ length: Math.min(ROW_FETCH_CONCURRENCY, coalesced.length) }, worker),
    );
    let ci = 0;
    for (let i = 0; i < nRows; i++) {
      const seg = segments[i];
      while (coalesced[ci].end < seg.end) ci++;
      const cr = coalesced[ci];
      const buf = buffers.get(cr) as Uint8Array;
      const off = seg.start - cr.start;
      values.set(decodeScalars(dtype, buf.subarray(off, off + rowBytes), nCols), i * nCols);
    }
    return { values, rows, cols, dtype };
  }

  /** One HTTP Range request for [start, end) — insists on real partial content. */
  private async range(start: number, end: number): Promise<Uint8Array> {
    const len = end - start;
    let res: Response;
    try {
      res = await this.fetchImpl(this.url, {
        headers: { Range: `bytes=${start}-${end - 1}` },
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new ApiError("NetworkError", `Range read of ${this.url} failed: ${msg}`);
    }
    if (res.status === 206) {
      const buf = new Uint8Array(await res.arrayBuffer());
      if (buf.byteLength !== len) {
        throw new ApiError(
          "ComputeError",
          `Range read returned ${buf.byteLength} bytes, expected ${len} (${this.url}).`,
        );
      }
      return buf;
    }
    if (res.status === 200) {
      // Some servers ignore Range and return the whole file; only tolerate that
      // when buffering it is cheap, and slice the requested window out.
      const buf = new Uint8Array(await res.arrayBuffer());
      if (buf.byteLength === len) return buf; // served exactly the window
      if (buf.byteLength >= end && buf.byteLength <= FULL_BODY_LIMIT) {
        return buf.subarray(start, end);
      }
      throw new ApiError(
        "ComputeError",
        `Server ignored the Range request (${buf.byteLength} bytes for a ${len}-byte window).`,
      );
    }
    if (res.status === 404) {
      throw new ApiError("NotFoundError", `Safetensors file not found: ${this.url}`);
    }
    throw new ApiError("HttpError", `Range read of ${this.url}: HTTP ${res.status}`);
  }

  private async loadHeader(): Promise<SafetensorsHeader> {
    const lenBytes = await this.range(0, 8);
    const dv = new DataView(lenBytes.buffer, lenBytes.byteOffset, 8);
    const big = dv.getBigUint64(0, true);
    if (big <= 0n || big > BigInt(HEADER_SANITY_LIMIT)) {
      throw new ApiError(
        "ComputeError",
        `Implausible safetensors header length ${big} for ${this.url}.`,
      );
    }
    const headerLen = Number(big);
    const headerBytes = await this.range(8, 8 + headerLen);
    let parsed: unknown;
    try {
      parsed = JSON.parse(new TextDecoder().decode(headerBytes));
    } catch {
      throw new ApiError("ComputeError", `Invalid safetensors header JSON in ${this.url}.`);
    }
    // `Object.create(null)`, not `{}`: the keys here are tensor names chosen by a REMOTE
    // file, and `tensors["__proto__"] = e` on an ordinary object literal does not create a
    // property at all — it invokes `Object.prototype`'s `__proto__` setter and replaces
    // the map's prototype with the attacker's entry. The named tensor then silently
    // vanishes from the map (no throw, no missing-key error), and every field of that
    // entry — `dtype`, `shape`, `data_offsets` — becomes visible on the map itself, so any
    // reader that looks a name up with a truthiness test rather than `Object.hasOwn`
    // resolves `tensors.dtype` to a string a remote host supplied. A null-prototype map
    // has no such setter and no inherited keys to confuse a lookup with.
    const tensors: Record<string, TensorEntry> = Object.create(null);
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (k === "__metadata__") continue;
      const e = v as TensorEntry;
      if (
        typeof e?.dtype !== "string" ||
        !Array.isArray(e.shape) ||
        !Array.isArray(e.data_offsets)
      ) {
        throw new ApiError("ComputeError", `Malformed header entry for tensor '${k}'.`);
      }
      tensors[k] = e;
    }
    return { dataStart: 8 + headerLen, tensors };
  }
}
