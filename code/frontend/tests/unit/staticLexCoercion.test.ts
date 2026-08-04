/**
 * Numbers on the wire in the STATIC build: refused, never coerced.
 *
 * `asInt`/`asFloat` in `src/lib/staticClient/lex.ts` were `Number(value)` behind a
 * finiteness check, and `Number` is the widest parser in the language. Measured against
 * this build by the red team (`notes/agent-reports/verify-007-round3.md`, item 6d):
 *
 *     "7" → 7 · true → 1 · false → 0 · null → 0 · [] → 0 · [7] → 7 · "" → 0
 *     " 7 " → 7 · "0x10" → 16
 *
 * `"0x10" → 16` is the one that names the defect class: a field documented as "an
 * integer", handed `0x10`, ran with 16 and reported 16 back, and nothing anywhere said
 * that the number used was not the number sent. The Python side answers every one of
 * these with a typed 400 (`tests/contract/test_api_lex_params.py`), so the static build —
 * the half the public site actually runs — was the lenient arm of a two-stack
 * disagreement about what a request means.
 *
 * These tests drive the REAL static client over the REAL vacancy transform and the REAL
 * bundle loader. There is no mock in this file: `fsStaticFetch` serves the committed
 * static-data directory off disk, which is what the deployed page fetches over HTTP.
 *
 * The tables below are the CLASS, not the examples. Every form `Number` accepts and a
 * JSON integer is not appears once, so a future rewrite that reaches for `Number`,
 * `parseInt`, `parseFloat` or a unary `+` fails here rather than in a user's browser.
 */

import { describe, expect, it } from "vitest";

import { createStaticClient, type StaticClient } from "../../src/lib/staticClient";
import type { FetchLike } from "../../src/lib/staticClient/assets";
import type { LexVacancyBody, LexTrainBody } from "../../src/lib/staticClient/lex";
import { fsStaticFetch } from "./staticTestUtils";

function client(fetchImpl: FetchLike = fsStaticFetch()): StaticClient {
  return createStaticClient({ baseUrl: "/", fetchImpl });
}

/** Short enough to keep the transform fast, real enough to have vacatable words. */
const TEXT = [
  "The little brown squirrel ate the pretty acorn.",
  "The squirrel ran away and the children sang loudly today.",
].join("\n");

/**
 * Every JSON value that is not a number and that `Number()` nevertheless turns into one,
 * with the number it used to become. The comment beside each is what the old code did.
 */
const NOT_A_NUMBER: [label: string, value: unknown][] = [
  ["a decimal string", "7"], //            → 7
  ["a hexadecimal string", "0x10"], //     → 16, the finding verbatim
  ["a binary string", "0b101"], //         → 5
  ["an octal string", "0o17"], //          → 15
  ["exponent notation", "1e3"], //         → 1000
  ["a leading plus", "+7"], //             → 7
  ["surrounding whitespace", " 7 "], //    → 7
  ["a newline", "\n7\n"], //               → 7
  ["the empty string", ""], //             → 0
  ["a whitespace-only string", "   "], //  → 0
  ["the string Infinity", "Infinity"], //  → Infinity (caught, but as the wrong error)
  ["true", true], //                       → 1
  ["false", false], //                     → 0
  ["null", null], //                       → the default, silently
  ["an empty array", []], //               → 0
  ["a one-element array", [7]], //         → 7
  ["a Date", new Date(0)], //              → 0
];

/** Numeric values that are not integers, or not usable at all. */
const NOT_AN_INTEGER: [label: string, value: number][] = [
  ["a fraction", 1.5],
  ["a near-integer", 2.0000000001],
  ["Infinity", Infinity],
  ["-Infinity", -Infinity],
  ["NaN", NaN],
];

async function vacancy(body: LexVacancyBody): Promise<Record<string, unknown>> {
  return (await client().lexVacancy({ text: TEXT, preview_chars: 40, ...body })) as unknown as Record<
    string,
    unknown
  >;
}

describe("an integer parameter takes a JSON integer or nothing", () => {
  it.each(NOT_A_NUMBER)("refuses %s as a seed rather than reading a number out of it", async (
    _label,
    value,
  ) => {
    await expect(vacancy({ p: 0.5, seed: value as number })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it.each(NOT_AN_INTEGER)("refuses %s as a seed rather than truncating it", async (_l, value) => {
    await expect(vacancy({ p: 0.5, seed: value })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it("names the value it refused, and does not print Infinity as null", async () => {
    // `JSON.stringify(Infinity)` is the string "null", so the old message reported the
    // one value nobody sent for the three numbers hardest to spot.
    await expect(vacancy({ p: 0.5, seed: Infinity })).rejects.toMatchObject({
      message: expect.stringContaining("Infinity"),
    });
    await expect(vacancy({ p: 0.5, seed: "0x10" as unknown as number })).rejects.toMatchObject({
      message: expect.stringContaining("0x10"),
    });
  });

  it.each([0, 7, -7, 7.0, 2 ** 53 - 1])("uses the integer %s exactly as given", async (seed) => {
    const res = await vacancy({ p: 0.5, seed });
    expect(res.seed).toBe(seed);
  });

  it("omitting the key still takes the default", async () => {
    const res = await vacancy({ p: 0.5 });
    expect(res.seed).toBe(0);
  });

  it("an explicit null is refused where an absent key defaults", async () => {
    // The backend's `_as_int(None)` raises; the static client returned the default, so
    // `{"seed": null}` meant two different runs in the two stacks.
    await expect(vacancy({ p: 0.5, seed: null as unknown as number })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it.each(NOT_A_NUMBER)("refuses %s as reveal_after too", async (_label, value) => {
    await expect(
      vacancy({ p: 0.5, reveal_after: value as number }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
  });

  it.each(NOT_A_NUMBER)("refuses %s as preview_chars too", async (_label, value) => {
    await expect(
      vacancy({ p: 0.5, preview_chars: value as number }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
  });
});

describe("a float parameter takes a finite JSON number or nothing", () => {
  it.each(NOT_A_NUMBER)("refuses %s as p", async (_label, value) => {
    await expect(vacancy({ p: value as number })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it.each([
    ["Infinity", Infinity],
    ["-Infinity", -Infinity],
    ["NaN", NaN],
  ] as [string, number][])("refuses %s as p", async (_label, value) => {
    // NaN is the quiet one: `0 <= NaN && NaN <= 1` is false, so the range check DID catch
    // it — but `lr <= 0` on the training route is false too, and there nothing caught it.
    await expect(vacancy({ p: value })).rejects.toMatchObject({ type: "InvalidParamError" });
  });

  it.each([0, 0.25, 0.5, 1])("uses the float %s exactly as given", async (p) => {
    const res = await vacancy({ p });
    expect(res.p).toBe(p);
  });
});

describe("the training route refuses the same class at the wire, not mid-run", () => {
  const train = (body: LexTrainBody) => client().lexTrain({ text: TEXT, steps: 1, ...body });

  it.each(NOT_A_NUMBER)("refuses %s as steps", async (_label, value) => {
    await expect(train({ steps: value as number })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it.each([
    ["a numeric string", "1e-3"],
    ["true", true],
    ["Infinity", Infinity],
    ["NaN", NaN],
  ] as [string, unknown][])("refuses %s as lr", async (_label, value) => {
    // `lr = NaN` passed `if (!(lr > 0))`… no: every comparison with NaN is false, so
    // `!(NaN > 0)` is TRUE and that one is caught. `lr = "1e-3"` was not: it became
    // 0.001, which is plausible, which is the whole problem.
    await expect(train({ lr: value as number })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it.each([
    ["a numeric string", "0.01"],
    ["Infinity", Infinity],
    ["NaN", NaN],
  ] as [string, unknown][])("refuses %s as weight_decay", async (_label, value) => {
    // `weight_decay = NaN` survived `if (weightDecay < 0)` — NaN < 0 is false — and
    // multiplied every parameter by NaN on step 1.
    await expect(train({ weight_decay: value as number })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it("refuses a seed the backend would have refused, rather than running with it", async () => {
    // `Number.isInteger(1e300)` is TRUE, so `asInt` alone let this through. Python
    // answers it with a typed 400 (`abs(seed) > MAX_SEED`), and the seed is echoed back
    // in the job result — so one request body produced two different documented runs.
    await expect(train({ seed: 1e300 })).rejects.toMatchObject({
      type: "InvalidParamError",
      message: expect.stringContaining("seed must lie in"),
    });
  });
});

describe("a model file's declared numbers get the same rule as the wire's", () => {
  /** The smallest bundle shape the loader looks at before it reaches the weights. */
  const bundle = (over: Record<string, unknown>) => ({
    format: "llm-geometry/lex-model",
    version: 1,
    config: { vocab_rows: 16, d_model: 16, n_layers: 1, n_heads: 2, ctx: 32, tied: true, dropout: 0 },
    vocab: { words: [], source: "dolch", budget: "custom" },
    weights: {},
    ...over,
  });

  it.each(NOT_A_NUMBER)("refuses %s as the bundle version", async (_label, value) => {
    await expect(client().lexImportModel(bundle({ version: value }))).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it.each(NOT_A_NUMBER)("refuses %s as config.d_model", async (_label, value) => {
    const b = bundle({});
    (b.config as Record<string, unknown>).d_model = value;
    await expect(client().lexImportModel(b)).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });

  it("refuses a weight entry whose shape is not a list, with a typed error", async () => {
    // `(entry.shape as unknown[]).map` on a number threw `.map is not a function`, an
    // untyped TypeError the file dialog had nothing to print.
    const b = bundle({
      config: {
        vocab_rows: 3,
        d_model: 2,
        n_layers: 1,
        n_heads: 1,
        ctx: 4,
        tied: true,
        dropout: 0,
      },
      vocab: { words: ["a"], source: "dolch", budget: "custom" },
      weights: { embed: { shape: 6, data: "" } },
    });
    await expect(client().lexImportModel(b)).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });
});

/**
 * The other half of the sweep: a lookup table indexed by a key that came from data.
 *
 * `DISPLAY_NAMES[m.model_id] ?? m.model_id` reads as a safe default and is not one.
 * `DISPLAY_NAMES` is an object literal, so `DISPLAY_NAMES["constructor"]` is the `Object`
 * function — truthy, not nullish — and `??` never fires. A model whose id is an
 * `Object.prototype` member would be listed in the tab's model menu under the
 * stringification of a JavaScript builtin instead of under its own id: the panel names a
 * model that does not exist, and nothing throws.
 *
 * The transport below serves the REAL committed `static-data` directory and renames one
 * model's `model_id` in `index.json` on the way through. Every per-model asset path is
 * keyed by `slug`, which is untouched, so this is the real catalog with one real id
 * changed — not a fabricated fixture.
 */
describe("a data-supplied key never resolves through a lookup table's prototype", () => {
  function indexRenamingFirstModel(newId: string): FetchLike {
    const disk = fsStaticFetch();
    return async (input: string, init?: RequestInit): Promise<Response> => {
      const res = await disk(input, init);
      if (!input.endsWith("/static-data/index.json")) return res;
      const idx = JSON.parse(await res.text());
      idx.arch_models[0].model_id = newId;
      return new Response(JSON.stringify(idx), { status: 200 });
    };
  }

  it.each(["constructor", "toString", "valueOf", "hasOwnProperty", "__proto__"])(
    "lists a model whose id is %s under its own id",
    async (id) => {
      const c = client(indexRenamingFirstModel(id));
      const ref = await c.resolveModel(id);
      expect(ref.model_id).toBe(id);
      expect(typeof ref.display_name).toBe("string");
      expect(ref.display_name).toBe(id);
    },
    30_000,
  );

  it("still gives the curated models their curated labels", async () => {
    const ref = await client().resolveModel("gpt2");
    expect(ref.display_name).toBe("GPT-2 124M (base — completes text)");
  }, 30_000);
});
