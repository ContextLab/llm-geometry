/**
 * Weight presets, weight-set editing, and content-hash tokens — TypeScript port of
 * code/backend/src/llm_geometry/geo/weights.py.
 *
 * weights_token reproduces the backend hash EXACTLY: sha256 over the name-sorted
 * weight set, feeding for each tensor `utf8(name) + utf8(repr(shape)) + float32
 * little-endian bytes`, truncated to the first 32 hex chars. Golden tests assert
 * string equality with tokens minted by the real backend.
 *
 * Preset portability: `identity`, `toeplitz_fuzzy`, `zero`, and `learned` are
 * computed natively with float32-faithful arithmetic (every numpy float32 op is
 * mirrored with Math.fround; float64 double-rounding through fround is exact for
 * +,*,/,sqrt since 53 >= 2*24+2). The seeded presets `random` / `random_autocorr`
 * use numpy's PCG64 + ziggurat standard-normal stream (plus scipy's
 * gaussian_filter for autocorr), which cannot be reproduced bit-exactly in TS
 * without porting numpy's generator tables — and the content-hash token demands
 * bit-exactness. They are therefore served from `presetFixtures.json`: real
 * matrices computed by the backend's preset_matrix() for seeds 0..2 and shipped
 * with the engine (base64 of the float32 bytes). Requesting an unavailable seed
 * throws an InvalidWeightEditError telling the UI to offer the fixture seeds.
 */

import { invalidWeightEdit, notFound } from "./errors";
import { sha256Hex, utf8Bytes } from "./hash";
import {
  D_MODEL,
  EDITABLE_MATRICES,
  N_LAYERS,
  VOCAB_SIZE,
  WEIGHT_SHAPES,
  cloneWeightSet,
  type EditableMatrix,
  type WeightSet,
} from "./model";

export const PRESETS = [
  "identity",
  "toeplitz_fuzzy",
  "random",
  "random_autocorr",
  "zero",
  "learned",
] as const;
export type PresetName = (typeof PRESETS)[number];

const TOEPLITZ_SIGMA = 0.75;
const f32 = Math.fround;

// --- Seeded-preset fixtures --------------------------------------------------------

export interface PresetFixtures {
  seeds: number[];
  /** preset -> seed(string) -> base64(float32-LE) 3x3 matrix. */
  square: Record<string, Record<string, string>>;
  /** preset -> seed(string) -> base64(float32-LE) 1003x3 matrix. */
  embedding: Record<string, Record<string, string>>;
}

// Pure-JS base64 decoder (no atob/Buffer, so it runs identically in the main
// thread, workers, and Node test processes).
const B64_LOOKUP = (() => {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const lookup = new Int16Array(128).fill(-1);
  for (let i = 0; i < alphabet.length; i++) lookup[alphabet.charCodeAt(i)] = i;
  return lookup;
})();

function b64ToF32(b64: string, expected: number, context: string): Float32Array {
  let end = b64.length;
  while (end > 0 && b64[end - 1] === "=") end--;
  const bytes = new Uint8Array(Math.floor((end * 3) / 4));
  let o = 0;
  let buffer = 0;
  let bits = 0;
  for (let i = 0; i < end; i++) {
    const v = B64_LOOKUP[b64.charCodeAt(i)] ?? -1;
    if (v < 0) continue; // tolerate whitespace
    buffer = (buffer << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes[o++] = (buffer >> bits) & 0xff;
    }
  }
  if (o !== expected * 4) {
    throw invalidWeightEdit(`${context}: fixture has ${o} bytes, expected ${expected * 4}`);
  }
  const dv = new DataView(bytes.buffer, 0, o);
  const out = new Float32Array(expected);
  for (let i = 0; i < expected; i++) out[i] = dv.getFloat32(i * 4, true); // little-endian
  return out;
}

// --- float32-faithful row normalization (backend _unit_rows) -----------------------

/**
 * Unit-normalize rows exactly like numpy on a float32 matrix: per row a sequential
 * float32 sum of squares, float32 sqrt, float32 division. Throws (like the backend)
 * if any row norm is below 1e-8.
 */
export function unitRows32(mat: Float32Array, rows: number, cols: number, context: string): Float32Array {
  const norms = new Float64Array(rows);
  let nearZero = 0;
  for (let r = 0; r < rows; r++) {
    let s = 0; // float32 accumulator (kept in f32 via fround at each step)
    for (let c = 0; c < cols; c++) {
      const x = mat[r * cols + c];
      s = f32(s + f32(x * x));
    }
    const n = f32(Math.sqrt(s));
    norms[r] = n;
    if (n < 1e-8) nearZero++;
  }
  if (nearZero > 0) {
    throw invalidWeightEdit(
      `${context}: ${nearZero} embedding row(s) have (near-)zero norm and cannot be unit-normalized`,
    );
  }
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) out[r * cols + c] = f32(mat[r * cols + c] / norms[r]);
  }
  return out;
}

function matrixShape(name: EditableMatrix): [number, number] {
  return name === "embedding" ? [VOCAB_SIZE, D_MODEL] : [D_MODEL, D_MODEL];
}

// --- Presets -----------------------------------------------------------------------

export interface PresetContext {
  /** The canonical learned weight set (for preset "learned"). */
  canonical: WeightSet;
  /** Fixture matrices for the seeded numpy presets. */
  fixtures?: PresetFixtures | null;
}

/** Build one preset matrix (backend preset_matrix; float32, bit-faithful). */
export function presetMatrix(
  preset: string,
  matrix: EditableMatrix,
  seed: number,
  ctx: PresetContext,
): Float32Array {
  if (!(PRESETS as readonly string[]).includes(preset)) {
    throw invalidWeightEdit(`Unknown preset '${preset}'; expected one of ${PRESETS.join(", ")}`);
  }
  if (!(EDITABLE_MATRICES as readonly string[]).includes(matrix)) {
    throw invalidWeightEdit(`Unknown matrix '${matrix}'; expected one of ${EDITABLE_MATRICES.join(", ")}`);
  }
  const [rows, cols] = matrixShape(matrix);
  const isEmbedding = matrix === "embedding";

  if (preset === "identity") {
    // Row i = e_(i mod cols) (np.tile of eye, truncated to rows).
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) out[r * cols + (r % cols)] = 1;
    return out;
  }

  if (preset === "toeplitz_fuzzy") {
    // T[i,j] = exp(-((i mod cols) - j)^2 / (2 sigma^2)), float64 exp then float32
    // cast, exactly like numpy; embedding rows are then unit-normalized.
    const denom = 2 * TOEPLITZ_SIGMA * TOEPLITZ_SIGMA;
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const i = r % cols;
      for (let j = 0; j < cols; j++) {
        out[r * cols + j] = f32(Math.exp(-((i - j) * (i - j)) / denom));
      }
    }
    return isEmbedding ? unitRows32(out, rows, cols, "toeplitz_fuzzy embedding") : out;
  }

  if (preset === "zero") {
    if (isEmbedding) {
      throw invalidWeightEdit(
        "preset 'zero' is invalid for the embedding: zero rows cannot satisfy the unit-norm constraint",
      );
    }
    return new Float32Array(rows * cols);
  }

  if (preset === "learned") {
    if (isEmbedding) return new Float32Array(ctx.canonical["embedding"]);
    // Mirrors the backend: layer matrices need the layer; build_weight_set handles it.
    throw invalidWeightEdit("preset 'learned' for a layer matrix requires the layer; use build_weight_set");
  }

  // random / random_autocorr: numpy PCG64+ziggurat streams — served from fixtures.
  const fixtures = ctx.fixtures;
  const table = fixtures ? (isEmbedding ? fixtures.embedding : fixtures.square)[preset] : undefined;
  const b64 = table?.[String(seed)];
  if (b64 === undefined) {
    const seeds = fixtures?.seeds?.join(", ") ?? "(none loaded)";
    throw invalidWeightEdit(
      `preset '${preset}' with seed ${seed} is not available in the static build: numpy's seeded ` +
        `RNG stream is not portable to the browser, so these matrices are precomputed by the real ` +
        `backend for seeds [${seeds}]. Pick one of those seeds (or run the full stack).`,
    );
  }
  return b64ToF32(b64, rows * cols, `preset '${preset}' seed ${seed}`);
}

// --- Edit application (backend build_weight_set) -----------------------------------

export interface WeightEditInput {
  layer?: number | null;
  matrix: string;
  preset?: string | null;
  values?: unknown;
  seed?: number | null;
}

export interface EditSummary {
  layer: number | null;
  matrix: EditableMatrix;
  source: string; // "edited" | "preset:<name>"
}

/** Validate an explicit values matrix (backend validate_values): shape + finiteness. */
export function validateValues(matrix: EditableMatrix, values: unknown): Float32Array {
  const [rows, cols] = matrixShape(matrix);
  if (!Array.isArray(values) || values.length !== rows) {
    const got = Array.isArray(values) ? `(${values.length}, ?)` : typeof values;
    throw invalidWeightEdit(`values for '${matrix}' have shape ${got}, expected (${rows}, ${cols})`);
  }
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const row = values[r];
    if (!Array.isArray(row) || row.length !== cols) {
      throw invalidWeightEdit(`values for '${matrix}' have a bad row ${r}: expected ${cols} numbers`);
    }
    for (let c = 0; c < cols; c++) {
      const x = row[c];
      if (typeof x !== "number") {
        throw invalidWeightEdit(`values for '${matrix}' are not a numeric matrix (row ${r}, col ${c})`);
      }
      if (!Number.isFinite(x)) {
        throw invalidWeightEdit(`values for '${matrix}' contain non-finite entries`);
      }
      out[r * cols + c] = f32(x);
    }
  }
  return out;
}

/**
 * Apply `edits` to a copy of `base`; return the new weight set + edit summaries.
 * Validation and float32 semantics follow the backend exactly (including the
 * embedding being re-unit-normalized after EVERY embedding edit — presets included,
 * which is why an already-normalized preset row is normalized twice, like numpy).
 */
export function buildWeightSet(
  base: WeightSet,
  edits: WeightEditInput[],
  ctx: PresetContext,
): { ws: WeightSet; summaries: EditSummary[] } {
  const ws = cloneWeightSet(base);
  const summaries: EditSummary[] = [];
  edits.forEach((edit, n) => {
    const matrix = edit.matrix as EditableMatrix;
    if (!(EDITABLE_MATRICES as readonly string[]).includes(matrix)) {
      throw invalidWeightEdit(
        `edit ${n}: unknown matrix ${JSON.stringify(edit.matrix)}; expected one of ${EDITABLE_MATRICES.join(", ")}`,
      );
    }
    const preset = edit.preset ?? null;
    const values = edit.values ?? null;
    if ((preset === null) === (values === null)) {
      throw invalidWeightEdit(`edit ${n} (${matrix}): exactly one of preset/values must be given`);
    }
    const seed = Math.trunc(Number(edit.seed ?? 0)) || 0;
    const layer = edit.layer ?? null;
    if (matrix !== "embedding") {
      if (layer === null || !Number.isInteger(layer) || layer < 0 || layer >= N_LAYERS) {
        throw invalidWeightEdit(
          `edit ${n} (${matrix}): layer must be an int in 0..${N_LAYERS - 1}, got ${JSON.stringify(edit.layer)}`,
        );
      }
    }

    let arr: Float32Array;
    let source: string;
    if (values !== null) {
      arr = validateValues(matrix, values);
      source = "edited";
    } else if (preset === "learned" && matrix !== "embedding") {
      arr = new Float32Array(ctx.canonical[`layers.${layer}.${matrix}`]);
      source = "preset:learned";
    } else {
      arr = presetMatrix(String(preset), matrix, seed, ctx);
      source = `preset:${preset}`;
    }

    if (matrix === "embedding") {
      arr = unitRows32(arr, VOCAB_SIZE, D_MODEL, `edit ${n} (embedding)`);
      ws["embedding"] = arr;
    } else {
      ws[`layers.${layer}.${matrix}`] = arr;
    }
    summaries.push({ layer, matrix, source });
  });
  return { ws, summaries };
}

// --- Content-hash tokens (backend weights_token) -----------------------------------

/** Python tuple repr of a shape: (3, 3) / (12,) / (1003, 3). */
function reprShape(shape: number[]): string {
  return shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
}

/**
 * Content hash over the full weight set: sha256 of name-sorted
 * [utf8(name), utf8(repr(shape)), float32-LE bytes], first 32 hex chars —
 * byte-identical to the backend's weights_token().
 */
export function weightsToken(ws: WeightSet): string {
  const names = Object.keys(ws).sort(); // ASCII code-unit order == Python sorted() here
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (const name of names) {
    const shape = WEIGHT_SHAPES.get(name);
    if (shape === undefined) {
      throw notFound(`weightsToken: unknown weight name '${name}'`);
    }
    const arr = ws[name];
    const nameBytes = utf8Bytes(name);
    const shapeBytes = utf8Bytes(reprShape(shape));
    const dataBytes = new Uint8Array(arr.length * 4);
    const dv = new DataView(dataBytes.buffer);
    for (let i = 0; i < arr.length; i++) dv.setFloat32(i * 4, arr[i], true); // little-endian
    chunks.push(nameBytes, shapeBytes, dataBytes);
    total += nameBytes.length + shapeBytes.length + dataBytes.length;
  }
  const all = new Uint8Array(total);
  let off = 0;
  for (const chunk of chunks) {
    all.set(chunk, off);
    off += chunk.length;
  }
  return sha256Hex(all).slice(0, 32);
}
