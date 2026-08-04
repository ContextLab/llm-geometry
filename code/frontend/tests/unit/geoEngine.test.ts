/**
 * Golden-vector tests: the TypeScript geoEngine vs the real Python backend.
 *
 * Golden data comes from BOTH available real-backend exports (see
 * geoGoldenAssets.ts): the static-site export in public/static-data/geo/ and
 * the committed fixtures in tests/fixtures/geo/. Because training is
 * platform-divergent at the bit level, every golden-driven suite runs once PER
 * SOURCE against an engine built from THAT source's own checkpoint + vocab —
 * goldens are never evaluated against the other source's engine. Numeric
 * comparisons use the spec's <=1e-5 tolerance (see RTOL/ATOL); tokens, token
 * ids, and weights_token strings are compared EXACTLY.
 *
 * Golden-independent behavior tests (rejections, detokenization rules,
 * performance, persistence round-trips) run ONCE against the fixtures engine —
 * the committed, platform-stable source.
 */

import { beforeAll, describe, expect, it } from "vitest";

import { GeoEngine, GeoEngineError, weightsToken, type WeightSet } from "../../src/lib/geoEngine";
import { clipPrompt } from "../../src/lib/geoEngine/fields";
import { sha256Hex, utf8Bytes } from "../../src/lib/geoEngine/hash";
import { GeoModel } from "../../src/lib/geoEngine/model";
import { canonicalVocabJson } from "../../src/lib/geoEngine/tokenizer";
import type {
  GeoVectorFieldData,
  GeoVectorFieldParams,
  GeoWeightsParams,
} from "../../src/lib/dataClient";
import {
  expectClose,
  expectMatrixClose,
  expectVecClose,
  goldenSources,
} from "./geoGoldenAssets";

const sources = goldenSources();
const fixtureSrc = sources.find((s) => s.name === "fixtures")!;

// --- shared helpers ----------------------------------------------------------------

/** Decode a bundle's base64 tensors back into a WeightSet (for re-hashing a file). */
function weightSetFromBundle(bundle: { weights: Record<string, { data: string }> }): WeightSet {
  const ws: WeightSet = {};
  for (const [name, payload] of Object.entries(bundle.weights)) {
    const bytes = Buffer.from(payload.data, "base64");
    ws[name] = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  return ws;
}

interface GoldenArrow {
  origin_index: number;
  vec: number[];
  weight: number;
}

function groupByOrigin(arrows: GoldenArrow[]): Map<number, GoldenArrow[]> {
  const groups = new Map<number, GoldenArrow[]>();
  for (const arrow of arrows) {
    const list = groups.get(arrow.origin_index) ?? [];
    list.push(arrow);
    groups.set(arrow.origin_index, list);
  }
  for (const list of groups.values()) list.sort((a, b) => b.weight - a.weight);
  return groups;
}

/**
 * The backend runs in float32; where two next-token logits are within float32
 * noise of each other, its argmax can legitimately land on the other candidate
 * than our float64 argmax (observed margin: 2.4e-7 — one f32 ulp at logit
 * scale). A temperature-0 arrow may therefore differ from golden ONLY if the
 * engine's own logit margin between the two competing targets is below this
 * threshold; anything larger is a real error.
 */
const TIE_LOGIT_MARGIN = 1e-5;

function targetIndexFromVec(points: number[][], origin: number, vec: number[]): number {
  const tx = points[origin][0] + vec[0];
  const ty = points[origin][1] + vec[1];
  const tz = points[origin][2] + vec[2];
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < points.length; i++) {
    const d = (points[i][0] - tx) ** 2 + (points[i][1] - ty) ** 2 + (points[i][2] - tz) ** 2;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  expect(bestD).toBeLessThan(1e-6); // the golden vec must resolve to a real vocab point
  return best;
}

/** Engine-side logits for `origin` appended to the prompt at the given layer. */
function logitsForOrigin(
  engine: GeoEngine,
  params: Record<string, any>,
  origin: number,
): Float64Array {
  const model = new GeoModel(engine.canonical);
  const promptIds = engine.tokenize(params.prompt as string).tokens.map((t) => t.id);
  const layerIdx = params.layer === "full" ? 3 : (params.layer as number);
  const seq = [...clipPrompt(promptIds, 1), origin];
  const acts = model.forwardSeq(seq, layerIdx + 1);
  return model.readoutOne(acts.layers[layerIdx].hiddenOut, (seq.length - 1) * 3);
}

// --- golden-driven suites: once per source, each against ITS OWN engine ------------
//
// The two sources are trained on different platforms (macOS fixtures vs the
// CI/Linux static export), and training is bit-level platform-sensitive, so a
// golden set only matches the checkpoint from its own run. Pairing them here is
// the point of these loops.

for (const src of sources) {
  const golden = src.golden;

  describe(`geoEngine golden [${src.name}]`, () => {
    let engine: GeoEngine;

    beforeAll(() => {
      engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    });

    // --- spec ----------------------------------------------------------------------

    describe("spec", () => {
      it("reproduces its own source's checkpoint_id exactly (hash port is bit-exact)", () => {
        const spec = engine.spec();
        expect(spec.checkpoint.status).toBe("ready");
        expect(spec.checkpoint.checkpoint_id).toBe(golden.spec.checkpoint.checkpoint_id);
      });

      it("reports the frozen model spec", () => {
        const spec = engine.spec();
        expect(spec.model).toEqual(golden.spec.model);
        expect(spec.special_tokens).toEqual(golden.spec.special_tokens);
      });
    });

    // --- tokenizer -----------------------------------------------------------------

    describe("tokenizer", () => {
      // The static export ships no tokenize goldens by design (its trace.tokens
      // are id-exact instead), so this runs only where the source provides them
      // — and the fixtures source must ALWAYS provide them.
      it.runIf(golden.tokenize.length > 0 || src.name === "fixtures")(
        "is id-exact on every golden prompt (incl. unks, curly quotes, truncation)",
        () => {
          expect(golden.tokenize.length).toBeGreaterThan(0);
          for (const { text, response } of golden.tokenize) {
            const actual = engine.tokenize(text);
            expect(actual.tokens, JSON.stringify(text)).toEqual(response.tokens);
            expect(actual.n_unk, JSON.stringify(text)).toBe(response.n_unk);
            expect(actual.truncated, JSON.stringify(text)).toBe(response.truncated);
          }
        },
      );
    });

    // --- trace ---------------------------------------------------------------------

    describe("trace", () => {
      it("matches golden traces to <=1e-5 (probs + per-layer q/k/v/attention/hidden)", () => {
        expect(golden.trace.length).toBeGreaterThan(0);
        for (const { prompt, response } of golden.trace) {
          const actual = engine.trace(prompt);
          expect(actual.tokens, prompt).toEqual(response.tokens);
          expectMatrixClose(actual.embeddings, response.embeddings, `${prompt}: embeddings`);
          expect(actual.layers.length).toBe(response.layers.length);
          for (let l = 0; l < response.layers.length; l++) {
            const a = actual.layers[l];
            const g = response.layers[l];
            expect(a.layer).toBe(g.layer);
            for (const key of [
              "attention",
              "q",
              "k",
              "v",
              "hidden_in",
              "attn_out",
              "mlp_out",
              "hidden_out",
            ] as const) {
              expectMatrixClose(a[key], g[key], `${prompt}: layers[${l}].${key}`);
            }
          }
          expectVecClose(actual.probs, response.probs, `${prompt}: probs`);
          expect(actual.next_token.id, `${prompt}: next_token`).toBe(response.next_token.id);
          expect(actual.next_token.text).toBe(response.next_token.text);
          // top-k: the same ids must carry the same probabilities (order can only
          // differ on sub-tolerance ties; the argmax itself is asserted exact above).
          for (let i = 0; i < response.logits_topk.ids.length; i++) {
            const id = response.logits_topk.ids[i];
            expectClose(actual.probs[id], response.logits_topk.probs[i], `${prompt}: topk prob of ${id}`);
          }
          expect(actual.logits_topk.ids[0]).toBe(response.logits_topk.ids[0]);
        }
      });
    });

    // --- vector fields -------------------------------------------------------------

    describe("vector_field", () => {
      it("matches golden fields (both modes) to <=1e-5", () => {
        expect(golden.vector_field.length).toBeGreaterThan(0);
        for (const { params, response } of golden.vector_field) {
          const label = JSON.stringify(params);
          const actual = engine.vectorField(params as GeoVectorFieldParams) as GeoVectorFieldData;
          expect(actual.mode, label).toBe(response.mode);
          expect(actual.layer, label).toBe(response.layer);
          expect(actual.tangent_exact, label).toBe(response.tangent_exact);
          expect(actual.token_ids, label).toEqual(response.token_ids);
          expectMatrixClose(actual.points, response.points, `${label}: points`);

          const temp0NextNext = response.mode === "next_next" && Number(params.temperature ?? 0) === 0;
          let tieBreaks = 0;
          const actualGroups = groupByOrigin(actual.arrows);
          const goldenGroups = groupByOrigin(response.arrows as GoldenArrow[]);
          expect(actualGroups.size, `${label}: arrow origins`).toBe(goldenGroups.size);
          for (const [origin, goldenList] of goldenGroups) {
            const actualList = actualGroups.get(origin);
            expect(actualList, `${label}: arrows for origin ${origin}`).toBeDefined();
            expect(actualList!.length, `${label}: arrow count for origin ${origin}`).toBe(
              goldenList.length,
            );
            for (let i = 0; i < goldenList.length; i++) {
              try {
                expectClose(
                  actualList![i].weight,
                  goldenList[i].weight,
                  `${label}: origin ${origin} arrow ${i} weight`,
                );
                expectVecClose(
                  actualList![i].vec,
                  goldenList[i].vec,
                  `${label}: origin ${origin} arrow ${i} vec`,
                );
              } catch (err) {
                if (!temp0NextNext) throw err;
                // Sub-float32 argmax tie? Verify with the engine's own logits.
                const goldenTarget = targetIndexFromVec(response.points, origin, goldenList[i].vec);
                const myTarget = targetIndexFromVec(response.points, origin, actualList![i].vec);
                const logits = logitsForOrigin(engine, params, origin);
                const margin = Math.abs(logits[myTarget] - logits[goldenTarget]);
                if (margin > TIE_LOGIT_MARGIN) throw err;
                tieBreaks++;
              }
            }
          }
          if (tieBreaks > 0) {
            console.log(`[geoEngine] ${label}: ${tieBreaks} sub-float32 argmax tie(s) tolerated`);
            expect(tieBreaks).toBeLessThan(8); // ties must stay rare or something is wrong
          }

          if (response.sequence_forces === null) {
            expect(actual.sequence_forces, label).toBeNull();
          } else {
            expect(actual.sequence_forces!.length, label).toBe(response.sequence_forces.length);
            for (let i = 0; i < response.sequence_forces.length; i++) {
              const a = actual.sequence_forces![i];
              const g = response.sequence_forces[i];
              expect(a.position).toBe(g.position);
              expectVecClose(a.vec, g.vec, `${label}: sequence_forces[${i}].vec`);
              expectClose(a.normal_residual, g.normal_residual, `${label}: sequence_forces[${i}].normal_residual`);
            }
          }
        }
      });
    });

    // --- weights: minting (exact tokens), then reads --------------------------------

    describe("weights", () => {
      it("mints EXACTLY the backend's weights_token for every golden edit set", () => {
        expect(golden.weights_post.length).toBeGreaterThan(0);
        for (const { body, response } of golden.weights_post) {
          const actual = engine.postWeights(body as never);
          expect(actual.weights_token, JSON.stringify(body)).toBe(response.weights_token);
          if (response.edited !== undefined) {
            expect(actual.edited, JSON.stringify(body)).toEqual(response.edited);
          }
        }
      });

      it("serves golden weight windows (values 6-sig-rounded; source exact)", () => {
        for (const { params, response } of golden.weights_get) {
          const actual = engine.getWeights(params as GeoWeightsParams);
          expect(actual.shape, JSON.stringify(params)).toEqual(response.shape);
          expect(actual.source, JSON.stringify(params)).toBe(response.source);
          expectMatrixClose(actual.values, response.values, `${JSON.stringify(params)}: values`);
        }
      });
    });
  });
}

// --- golden-independent behavior: once, against the fixtures engine ----------------
//
// These tests need AN engine but no golden vectors, so they run once against the
// committed fixtures source (platform-stable) rather than per source.

describe("geoEngine behavior [fixtures]", () => {
  let engine: GeoEngine;

  beforeAll(() => {
    engine = GeoEngine.fromAssets(fixtureSrc.checkpoint, fixtureSrc.vocab);
  });

  it("applies the backend's detokenization spacing rules", () => {
    // Expected string produced by the real backend tokenizer (note "( loudly":
    // the backend checks its no-space-after rule against the last APPENDED
    // string, which carries a leading space — the port reproduces that exactly).
    const enc = engine.tokenizer.encode('the queen said : " off with her head ! " ( loudly )');
    expect(engine.tokenizer.decode(enc.ids)).toBe('the queen said: " off with her head! " ( loudly)');
  });

  it("rejects an empty prompt like the backend (400)", () => {
    expect(() => engine.trace("   ")).toThrowError(GeoEngineError);
    try {
      engine.trace("");
    } catch (err) {
      expect((err as GeoEngineError).type).toBe("InvalidParamError");
    }
  });

  it('rejects layer="full" in force mode like the backend', () => {
    expect(() => engine.vectorField({ mode: "force", layer: "full", prompt: "" })).toThrowError(
      /full/,
    );
  });

  it("rejects the zero preset on the embedding (unit-norm invariant)", () => {
    try {
      engine.postWeights({
        base: "learned",
        edits: [{ layer: 0, matrix: "embedding", preset: "zero" }],
      });
      throw new Error("expected InvalidWeightEditError");
    } catch (err) {
      expect((err as GeoEngineError).type).toBe("InvalidWeightEditError");
    }
  });

  it("rejects seeded random presets outside the fixture seed list with a clear message", () => {
    try {
      engine.postWeights({
        base: "learned",
        edits: [{ layer: 0, matrix: "W_Q", preset: "random", seed: 99 }],
      });
      throw new Error("expected InvalidWeightEditError");
    } catch (err) {
      expect((err as GeoEngineError).type).toBe("InvalidWeightEditError");
      expect((err as Error).message).toMatch(/precomputed by the real backend/);
    }
  });

  it("rejects an unknown weights_token with NotFoundError", () => {
    try {
      engine.trace("alice", "feedfacefeedfacefeedfacefeedface");
      throw new Error("expected NotFoundError");
    } catch (err) {
      expect((err as GeoEngineError).type).toBe("NotFoundError");
    }
  });

  it("rejects giving both preset and values", () => {
    try {
      engine.postWeights({
        base: "learned",
        edits: [
          {
            layer: 0,
            matrix: "W_Q",
            preset: "identity",
            values: [
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
            ],
          },
        ],
      });
      throw new Error("expected InvalidWeightEditError");
    } catch (err) {
      expect((err as GeoEngineError).type).toBe("InvalidWeightEditError");
    }
  });

  // --- performance (informational; generous ceilings so CI never flakes) -----------

  it("traces and full-vocab fields complete quickly", () => {
    const prompt = fixtureSrc.golden.trace[0].prompt;
    let t0 = performance.now();
    engine.trace(prompt);
    const traceMs = performance.now() - t0;

    t0 = performance.now();
    engine.vectorField({ mode: "next_next", layer: "full", prompt, temperature: 0, top_m: 1 });
    const fieldMs = performance.now() - t0;

    t0 = performance.now();
    engine.vectorField({ mode: "force", layer: 0, prompt });
    const forceMs = performance.now() - t0;

    console.log(
      `[geoEngine perf] trace=${traceMs.toFixed(1)}ms nextNextField(full vocab)=${fieldMs.toFixed(1)}ms forceField=${forceMs.toFixed(1)}ms`,
    );
    expect(traceMs).toBeLessThan(2000);
    expect(fieldMs).toBeLessThan(10000);
  });
});

describe("minted-set persistence hooks (static reload survival) [fixtures]", () => {
  it("export -> import round-trips a minted set with token-hash validation", () => {
    const engine = GeoEngine.fromAssets(fixtureSrc.checkpoint, fixtureSrc.vocab);
    const minted = engine.postWeights({
      base: "learned",
      edits: [{ layer: 0, matrix: "W_V", preset: "identity" }],
    });
    const payload = engine.exportWeightSet(minted.weights_token);
    // A fresh engine (same assets) must accept the import and serve identical values.
    const engine2 = GeoEngine.fromAssets(fixtureSrc.checkpoint, fixtureSrc.vocab);
    expect(engine2.importWeightSet(minted.weights_token, payload)).toBe(true);
    const w = engine2.getWeights({ matrix: "W_V", layer: 0, weights_token: minted.weights_token });
    expect(w.source).toBe("preset:identity");
    expect(w.values[0][0]).toBe(1);
    // Corruption is refused: flip a byte and the hash check fails.
    const bad = { ...payload, weights: { ...payload.weights } };
    const firstKey = Object.keys(bad.weights)[0];
    bad.weights[firstKey] = bad.weights[firstKey].slice(0, -4) + "AAA=";
    const engine3 = GeoEngine.fromAssets(fixtureSrc.checkpoint, fixtureSrc.vocab);
    expect(engine3.importWeightSet(minted.weights_token, bad)).toBe(false);
  });

  /**
   * A model trained from scratch (or loaded from a file) has a vocabulary of its OWN:
   * its token id 17 is not the shipped model's token id 17. The static build persists
   * minted sets to sessionStorage and restores them after a reload — and it used to
   * persist the weights WITHOUT the word list, so the restored set fell back to the
   * shipped tokenizer. The damage was not a wrong label on screen: `exportBundle` then
   * wrote a `.llmgeo.json` pairing those weights with the shipped word list and hashed
   * THAT list into `vocab_sha256`, producing a file no integrity check can reject — the
   * exact corruption the three digests exist to prevent, committed by the writer. So
   * "save → reload → save" silently changed which words the model file described.
   */
  it("carries a loaded model's OWN vocabulary across the persistence hop", () => {
    const engine = GeoEngine.fromAssets(fixtureSrc.checkpoint, fixtureSrc.vocab);

    // A real model file whose word list is NOT the shipped one — the situation every
    // from-scratch run and every `.llmgeo.json` load produces.
    // Distinct weights (so the token is not the canonical one) AND distinct words.
    const minted = engine.postWeights({
      base: "learned",
      edits: [{ layer: 1, matrix: "W_K", preset: "identity" }],
    });
    const shipped = engine.exportBundle(minted.weights_token);
    const words = (JSON.parse(shipped.vocab) as { words: string[] }).words.map((w, i) =>
      i === 0 ? `${w}zz` : w,
    );
    const vocabJson = canonicalVocabJson(words);
    // A file naming a DIFFERENT model must declare that model's token: the content hash
    // covers the word list, so weights + someone else's words is not the model the
    // original token names. Swapping only the vocabulary (and its digest) is exactly the
    // substitution attack, and it is refused — asserted below before the real file is
    // built with the token the new word list gives it.
    const tampered = {
      ...shipped,
      vocab: vocabJson,
      vocab_sha256: sha256Hex(utf8Bytes(vocabJson)),
    };
    expect(() => engine.importBundle(tampered)).toThrow(/corrupt/);
    const file = {
      ...tampered,
      weights_token: weightsToken(weightSetFromBundle(shipped), vocabJson),
    };
    const { weights_token: token } = engine.importBundle(file);
    expect(token).not.toBe(minted.weights_token); // different words ⇒ a different model
    expect(JSON.parse(engine.exportBundle(token).vocab).words).toEqual(words);

    // The reload hop: persist, restore into a fresh engine, save again.
    const saved = engine.exportWeightSet(token);
    expect(saved.vocabWords).toEqual(words);
    const reloaded = GeoEngine.fromAssets(fixtureSrc.checkpoint, fixtureSrc.vocab);
    expect(reloaded.importWeightSet(token, saved)).toBe(true);
    const after = engine.exportBundle(token);
    expect(JSON.parse(reloaded.exportBundle(token).vocab).words).toEqual(words);
    // Byte-for-byte the same file, which is the user-visible claim.
    expect(reloaded.exportBundle(token)).toEqual(after);

    // A payload that LOST the word list is dropped, not restored half-right: the token
    // simply is not there afterwards, so the caller deletes it and the evicted-token
    // self-heal resets visibly instead of quietly relabelling the model.
    const { vocabWords: _dropped, ...withoutVocab } = saved;
    const stale = GeoEngine.fromAssets(fixtureSrc.checkpoint, fixtureSrc.vocab);
    expect(stale.importWeightSet(token, withoutVocab)).toBe(false);
    expect(() => stale.exportBundle(token)).toThrow(/unknown/);
  });
});
