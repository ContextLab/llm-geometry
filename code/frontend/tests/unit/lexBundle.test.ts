/**
 * The Lexicon Lab model file (US-8), tested against a model that really trained.
 *
 * No fixtures and no synthetic weights in the round-trip cases: every one starts from
 * `runTraining` on the real committed corpus, so "round-trips exactly" is measured on the
 * kind of file a user would actually save.
 *
 * Three things are being defended here.
 *
 *  1. **One wire format.** The tag `llm-geometry/lex-model` belongs to the contract in
 *     `specs/006-lexicon-lab-tiny/contracts/api-lex.md` (`GET|POST /api/lex/model`). The
 *     browser must write THAT payload — snake_case config, `vocab` object, PyTorch tensor
 *     names — not a dialect of it that happens to share the tag. The shape is asserted
 *     field for field against the contract, and the `model_token` is pinned against one
 *     the REAL Python produced (see PY_TOKEN below).
 *  2. **Exactness.** float32 base64 is lossless, so a reloaded model must reproduce its
 *     generation token for token (SC-607).
 *  3. **Integrity, mandatory.** A model file carries weights AND the vocabulary that gives
 *     its token ids meaning; if the two can be separated the UI mislabels every token
 *     while looking healthy. So all three digests are required and checked — including
 *     the attack that broke the Geometry Lab's first loader in feature 004, where
 *     DELETING a digest (rather than editing it) skipped the check entirely.
 */

import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  exportLexBundle,
  importLexBundle,
  lexCanonicalConfig,
  lexCanonicalVocab,
  lexModelToken,
  lexVocabDigest,
  lexWeightsToken,
  lexWeightsTokenOf,
  lexWireShapes,
  toWireConfig,
  toWireName,
  LEX_BUNDLE_FORMAT,
  LEX_BUNDLE_VERSION,
  type LexModelBundle,
} from "../../src/lib/lexEngine/bundle";
import { generate } from "../../src/lib/lexEngine/generate";
import { LexModel, defaultConfig, weightNames, type LexConfig } from "../../src/lib/lexEngine/model";
import { runTraining } from "../../src/lib/lexEngine/train";
import { LexVocab, SPECIAL_TOKENS, buildVocab } from "../../src/lib/lexEngine/vocab";
import { lexModelToken as staticLexModelToken } from "../../src/lib/staticClient/lex";

const CORPUS = path.resolve(__dirname, "../../../backend/src/llm_geometry/lex/data/real-mother-goose.txt");
const corpusText = fs.readFileSync(CORPUS, "utf-8");

/**
 * One real training run, shared by every case here: a small but genuine model — the
 * budget is a real Dolch list, the corpus is the committed nursery-rhyme text, and the
 * weights below are whatever 40 AdamW steps produced.
 */
const trained = (() => {
  const vocab = buildVocab("dolch", "pre_primer", corpusText);
  const config = defaultConfig(vocab.rows, { dModel: 16, nLayers: 1, nHeads: 1, ctx: 32, tied: true });
  const result = runTraining({
    cfg: config,
    tokens: vocab.encodeText(corpusText),
    steps: 40,
    seed: 5,
  });
  return { vocab, config, weights: result.weights, result };
})();

/** A fresh, unshared bundle per case — every test tampers with its own copy. */
function freshBundle(): LexModelBundle {
  return exportLexBundle({
    config: trained.config,
    weights: trained.weights,
    vocabWords: trained.vocab.words,
    budgetSource: trained.vocab.source,
    budgetName: trained.vocab.budgetName,
    metrics: { note: "trained from scratch · dolch pre_primer · 40 steps", steps: 40 },
  });
}

/** Round-trip through JSON text, which is what save-then-load actually does. */
function throughFile(bundle: LexModelBundle): unknown {
  return JSON.parse(JSON.stringify(bundle));
}

describe("the model really trained", () => {
  it("moved its loss, so these are not fresh weights dressed up as a model", () => {
    expect(trained.result.finalTrainLoss).toBeLessThan(trained.result.initialTrainLoss);
    expect(Number.isFinite(trained.result.valLoss)).toBe(true);
    expect(trained.vocab.words.length).toBeGreaterThan(0);
    expect(trained.config.vocabRows).toBe(trained.vocab.words.length + SPECIAL_TOKENS.length);
  });
});

/**
 * The `model_token` golden below was produced by the REAL backend (from code/backend,
 * with its venv), on the deterministic weight set `w[i] = ((i*37) % 101 - 50) / 100`:
 *
 *   python -c "
 *   import numpy as np
 *   from llm_geometry.lex.model import LexConfig, LexModel
 *   from llm_geometry.api.routes_lex import _model_token
 *   cfg = LexConfig(vocab_rows=6, d_model=4, n_layers=1, n_heads=2, ctx=8,
 *                   tied=False, dropout=0.0)
 *   m = LexModel(cfg, seed=0)
 *   m.load_weight_dict({n: np.array([((i*37)%101-50)/100 for i in range(a.size)],
 *                                   dtype=np.float32).reshape(a.shape)
 *                       for n, a in m.weight_dict().items()})
 *   print(_model_token(m.weight_dict(), cfg.as_dict(), ('cat','dog')))"
 *
 * It is the same golden `tests/unit/staticLex.test.ts` pins, and reproducing it is what
 * makes "one wire format" a measurement rather than a claim: if this module's hash drifts
 * from Python's, a file written here stops loading into the full stack, and this fails.
 */
describe("the hash really is the backend's", () => {
  const cfg: LexConfig = { vocabRows: 6, dModel: 4, nLayers: 1, nHeads: 2, ctx: 8, tied: false, dropout: 0 };
  const PY_TOKEN = "4810360a3aedbd3f5e4a2f4b1a45e81f";

  function deterministicWeights(): Record<string, Float32Array> {
    const out: Record<string, Float32Array> = {};
    for (const [name, shape] of Object.entries(lexWireShapes(cfg))) {
      const n = shape.reduce((a, d) => a * d, 1);
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = (((i * 37) % 101) - 50) / 100;
      out[name] = a;
    }
    return out;
  }

  it("reproduces a model_token computed by real Python", () => {
    const wire = deterministicWeights();
    expect(lexModelToken(wire, lexWireShapes(cfg), toWireConfig(cfg), ["cat", "dog"])).toBe(PY_TOKEN);
  });

  it("agrees with the static client's independently-written implementation", () => {
    // Two implementations of one hash is a rot risk; pinning them to each other on a REAL
    // trained model turns that risk into a failing test rather than a silent divergence.
    const shapes = lexWireShapes(trained.config);
    const wire: Record<string, Float32Array> = {};
    for (const name of weightNames(trained.config)) wire[toWireName(name)] = trained.weights[name];
    expect(lexModelToken(wire, shapes, toWireConfig(trained.config), trained.vocab.words)).toBe(
      staticLexModelToken(wire, shapes, toWireConfig(trained.config), trained.vocab.words),
    );
  });

  it("canonicalises the config the way cache/keys.py does", () => {
    // sorted keys, no whitespace, and a Python float keeps its `.0`.
    expect(lexCanonicalConfig(toWireConfig(cfg))).toBe(
      '{"ctx":8,"d_model":4,"dropout":0.0,"n_heads":2,"n_layers":1,"tied":false,"vocab_rows":6}',
    );
    expect(lexCanonicalVocab(["cat", "dog"], "dolch", "pre_primer")).toBe(
      '{"budget":"pre_primer","source":"dolch","specials":["<unk>","<bos>","<eos>","<pad>"],"words":["cat","dog"]}',
    );
  });
});

describe("the payload is the contract's, field for field", () => {
  it("matches specs/006-lexicon-lab-tiny/contracts/api-lex.md GET /api/lex/model", () => {
    const b = freshBundle();
    // The contract's documented field set, plus the two half-digests this feature adds.
    expect(Object.keys(b).sort()).toEqual(
      ["config", "format", "metrics", "model_token", "version", "vocab", "vocab_sha256", "weights", "weights_token"],
    );
    expect(b.format).toBe(LEX_BUNDLE_FORMAT);
    expect(b.version).toBe(LEX_BUNDLE_VERSION);
    expect(Object.keys(b.config).sort()).toEqual(
      ["ctx", "d_model", "dropout", "n_heads", "n_layers", "tied", "vocab_rows"],
    );
    expect(Object.keys(b.vocab).sort()).toEqual(["budget", "source", "specials", "words"]);
    expect(b.vocab.specials).toEqual([...SPECIAL_TOKENS]);
    expect(b.vocab.words.length + SPECIAL_TOKENS.length).toBe(b.config.vocab_rows);
  });

  it("names its tensors the way the PyTorch model does, not the way the engine does", () => {
    const b = freshBundle();
    const names = Object.keys(b.weights);
    expect(names).toContain("blocks.0.qkv_w");
    expect(names.some((n) => n.startsWith("layers."))).toBe(false);
    expect(names.sort()).toEqual(Object.keys(lexWireShapes(trained.config)).sort());
    for (const [name, payload] of Object.entries(b.weights)) {
      expect(payload.shape).toEqual(lexWireShapes(trained.config)[name]);
      expect(typeof payload.data).toBe("string");
    }
    // Tied: no readout tensor at all, which is how `tied` is enforced on reload.
    expect(names).not.toContain("head_w");
  });

  it("declares digests computed from its own contents", () => {
    const b = freshBundle();
    expect(b.model_token).toMatch(/^[0-9a-f]{32}$/);
    expect(b.weights_token).toMatch(/^[0-9a-f]{32}$/);
    expect(b.vocab_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(b.weights_token).toBe(lexWeightsTokenOf(trained.config, trained.weights));
    expect(b.vocab_sha256).toBe(
      lexVocabDigest(lexCanonicalVocab(trained.vocab.words, "dolch", trained.vocab.budgetName)),
    );
    // The joint hash is NOT the weights hash: it also covers the config and the word list.
    expect(b.model_token).not.toBe(b.weights_token);
  });

  it("refuses to write a file whose word list does not fit its own weights", () => {
    expect(() =>
      exportLexBundle({
        config: trained.config,
        weights: trained.weights,
        vocabWords: trained.vocab.words.slice(0, 3),
        budgetSource: trained.vocab.source,
        budgetName: trained.vocab.budgetName,
      }),
    ).toThrow(/embedding rows/);
  });
});

describe("round trip", () => {
  it("reproduces the config, the vocabulary and every weight EXACTLY", () => {
    const loaded = importLexBundle(throughFile(freshBundle()));
    expect(loaded.config).toEqual(trained.config);
    expect(loaded.vocabWords).toEqual([...trained.vocab.words]);
    expect(loaded.budgetSource).toBe(trained.vocab.source);
    expect(loaded.budgetName).toBe(trained.vocab.budgetName);
    expect(loaded.metrics.note).toBe("trained from scratch · dolch pre_primer · 40 steps");

    // Back in the ENGINE's names, ready for `new LexModel(...)`.
    expect(Object.keys(loaded.weights).sort()).toEqual(weightNames(trained.config).sort());
    for (const name of weightNames(trained.config)) {
      const before = trained.weights[name];
      const after = loaded.weights[name];
      expect(after.length).toBe(before.length);
      // float32 base64 is lossless, so this is exact equality, not a tolerance.
      for (let i = 0; i < before.length; i++) expect(after[i]).toBe(before[i]);
    }
    expect(lexWeightsTokenOf(loaded.config, loaded.weights)).toBe(freshBundle().weights_token);
    expect(loaded.modelToken).toBe(freshBundle().model_token);
  });

  it("regenerates identical text (SC-607)", () => {
    const before = generate(new LexModel(trained.config, trained.weights), trained.vocab, {
      prompt: "the little",
      temperature: 0.9,
      maxNewTokens: 30,
      seed: 11,
    });
    const loaded = importLexBundle(throughFile(freshBundle()));
    const after = generate(
      new LexModel(loaded.config, loaded.weights),
      new LexVocab(loaded.vocabWords, loaded.budgetSource, loaded.budgetName),
      { prompt: "the little", temperature: 0.9, maxNewTokens: 30, seed: 11 },
    );
    expect(after.text).toBe(before.text);
    expect(after.ids).toEqual(before.ids);
    expect(before.words.length).toBeGreaterThan(0);
    // Generation stays in budget by construction, before and after the round trip.
    const budget = new Set(trained.vocab.words);
    for (const w of after.words) expect(budget.has(w)).toBe(true);
  });

  it("survives a greedy decode too, which has no RNG to hide behind", () => {
    const loaded = importLexBundle(throughFile(freshBundle()));
    const opts = { prompt: "little boy", temperature: 0, maxNewTokens: 20 };
    expect(
      generate(
        new LexModel(loaded.config, loaded.weights),
        new LexVocab(loaded.vocabWords, loaded.budgetSource, loaded.budgetName),
        opts,
      ).text,
    ).toBe(generate(new LexModel(trained.config, trained.weights), trained.vocab, opts).text);
  });

  it("re-exports byte-identically, so a save→load→save chain is stable", () => {
    const first = freshBundle();
    const loaded = importLexBundle(throughFile(first));
    const again = exportLexBundle({
      config: loaded.config,
      weights: loaded.weights,
      vocabWords: loaded.vocabWords,
      budgetSource: loaded.budgetSource,
      budgetName: loaded.budgetName,
      metrics: loaded.metrics,
    });
    expect(JSON.stringify(again)).toBe(JSON.stringify(first));
  });
});

describe("integrity is mandatory, and a mismatch is fatal", () => {
  const DIGESTS = ["model_token", "weights_token", "vocab_sha256"] as const;

  it("refuses a bundle with ANY digest DELETED, not edited", () => {
    // The exact attack that defeated feature 004's first loader: a missing digest must
    // not be read as "nothing to verify".
    for (const field of DIGESTS) {
      const b = throughFile(freshBundle()) as Record<string, unknown>;
      delete b[field];
      expect(() => importLexBundle(b)).toThrow(new RegExp(`no usable \`${field}\``));
    }
  });

  it("refuses an empty, null or non-string digest", () => {
    for (const field of DIGESTS) {
      for (const bad of ["", null, 42, {}, "NOTHEX"]) {
        const b = throughFile(freshBundle()) as Record<string, unknown>;
        b[field] = bad;
        expect(() => importLexBundle(b)).toThrow(new RegExp(`no usable \`${field}\``));
      }
    }
  });

  it("refuses a tampered model_token", () => {
    const b = throughFile(freshBundle()) as Record<string, unknown>;
    b.model_token = "0".repeat(32);
    expect(() => importLexBundle(b)).toThrow(/declares model_token .* contents hash to/s);
  });

  it("refuses a tampered weights_token", () => {
    const b = throughFile(freshBundle()) as Record<string, unknown>;
    b.weights_token = "0".repeat(32);
    expect(() => importLexBundle(b)).toThrow(/corrupt.*weights hash to/s);
  });

  it("refuses a tampered vocab_sha256", () => {
    const b = throughFile(freshBundle()) as Record<string, unknown>;
    b.vocab_sha256 = "f".repeat(64);
    expect(() => importLexBundle(b)).toThrow(/corrupt.*vocabulary hashes to/s);
  });

  it("refuses weights that were edited after the file was written", () => {
    const b = throughFile(freshBundle()) as LexModelBundle;
    // Flip one float in the embedding and re-encode it; no length or shape check would
    // notice, and every digest field is still the one the writer wrote.
    const edited = Float32Array.from(trained.weights.embed);
    edited[0] = edited[0] + 1;
    b.weights.embed.data = exportLexBundle({
      config: trained.config,
      weights: { ...trained.weights, embed: edited },
      vocabWords: trained.vocab.words,
      budgetSource: trained.vocab.source,
      budgetName: trained.vocab.budgetName,
    }).weights.embed.data;
    expect(() => importLexBundle(b)).toThrow(/contents hash to/);
  });

  it("refuses a substituted word list, however the attacker patches the digests", () => {
    const swapped = trained.vocab.words.map((w) => `${w}x`);

    // (a) word list swapped, digests left alone.
    const naive = throughFile(freshBundle()) as LexModelBundle;
    naive.vocab.words = [...swapped];
    expect(() => importLexBundle(naive)).toThrow(/contents hash to/);

    // (b) word list swapped AND both vocabulary-facing digests recomputed — the JOINT
    //     hash still catches it, which is why it is in the file.
    const clever = throughFile(freshBundle()) as LexModelBundle;
    clever.vocab.words = [...swapped];
    clever.vocab_sha256 = lexVocabDigest(
      lexCanonicalVocab(swapped, clever.vocab.source, clever.vocab.budget),
    );
    expect(() => importLexBundle(clever)).toThrow(/contents hash to/);
  });

  it("refuses reordered or renamed special tokens", () => {
    const b = throughFile(freshBundle()) as LexModelBundle;
    b.vocab.specials = ["<bos>", "<unk>", "<eos>", "<pad>"];
    expect(() => importLexBundle(b)).toThrow(/special tokens/);
  });
});

describe("a file must describe a model this build can construct", () => {
  it("refuses a word list that does not fill the model's embedding rows", () => {
    const b = throughFile(freshBundle()) as LexModelBundle;
    b.vocab.words = b.vocab.words.slice(0, -1);
    expect(() => importLexBundle(b)).toThrow(/embedding rows/);
  });

  it("refuses a foreign format, an unknown version, and a non-object", () => {
    const wrongFormat = throughFile(freshBundle()) as Record<string, unknown>;
    wrongFormat.format = "llm-geometry/geo-model";
    expect(() => importLexBundle(wrongFormat)).toThrow(/not a Lexicon Lab model bundle/);

    const wrongVersion = throughFile(freshBundle()) as Record<string, unknown>;
    wrongVersion.version = LEX_BUNDLE_VERSION + 1;
    expect(() => importLexBundle(wrongVersion)).toThrow(/version .* is not supported/);

    expect(() => importLexBundle(null)).toThrow(/must be a JSON object/);
    expect(() => importLexBundle([freshBundle()])).toThrow(/must be a JSON object/);
    expect(() => importLexBundle("{}")).toThrow(/must be a JSON object/);
  });

  it("refuses a config this build's architecture does not offer", () => {
    const b = throughFile(freshBundle()) as LexModelBundle;
    b.config.d_model = 17; // not in D_MODEL_CHOICES
    expect(() => importLexBundle(b)).toThrow();
  });

  it("refuses missing, extra and malformed weights", () => {
    const missing = throughFile(freshBundle()) as LexModelBundle;
    delete (missing.weights as Record<string, unknown>).pos;
    expect(() => importLexBundle(missing)).toThrow(/missing pos/);

    const extra = throughFile(freshBundle()) as LexModelBundle;
    extra.weights["blocks.9.qkv_w"] = { shape: [1], data: extra.weights.lnf_b.data };
    expect(() => importLexBundle(extra)).toThrow(/no slot for/);

    const malformed = throughFile(freshBundle()) as LexModelBundle;
    (malformed.weights as Record<string, unknown>).embed = { data: "not base64!!" };
    expect(() => importLexBundle(malformed)).toThrow(/shape and data/);

    const wrongShape = throughFile(freshBundle()) as LexModelBundle;
    wrongShape.weights.embed = { shape: [1, 1], data: wrongShape.weights.embed.data };
    expect(() => importLexBundle(wrongShape)).toThrow(/declares shape/);

    const noWeights = throughFile(freshBundle()) as Record<string, unknown>;
    delete noWeights.weights;
    expect(() => importLexBundle(noWeights)).toThrow(/missing its `weights` object/);
  });

  it("refuses a missing config or vocab block", () => {
    for (const field of ["config", "vocab"]) {
      const b = throughFile(freshBundle()) as Record<string, unknown>;
      delete b[field];
      expect(() => importLexBundle(b)).toThrow(new RegExp(`missing its \`${field}\` object`));
    }
  });
});

describe("an untied model", () => {
  /** Untied readout means an extra `head_w` tensor — a different file, and `tied` must
   *  travel with it. The source project's `probe.py` dropped `tie` on reload and silently
   *  reloaded a tied checkpoint as an untied model; that cannot happen here. */
  const untied = (() => {
    const vocab = buildVocab("dolch", "pre_primer", corpusText);
    const config = defaultConfig(vocab.rows, {
      dModel: 16,
      nLayers: 1,
      nHeads: 1,
      ctx: 32,
      tied: false,
    });
    const result = runTraining({ cfg: config, tokens: vocab.encodeText(corpusText), steps: 20, seed: 3 });
    return { vocab, config, weights: result.weights };
  })();

  function bundle(): LexModelBundle {
    return exportLexBundle({
      config: untied.config,
      weights: untied.weights,
      vocabWords: untied.vocab.words,
      budgetSource: untied.vocab.source,
      budgetName: untied.vocab.budgetName,
    });
  }

  it("carries head_w and round-trips exactly", () => {
    const b = bundle();
    expect(b.weights.head_w).toBeTruthy();
    expect(b.config.tied).toBe(false);
    const loaded = importLexBundle(throughFile(b));
    expect(loaded.config.tied).toBe(false);
    for (let i = 0; i < untied.weights.head_w.length; i++) {
      expect(loaded.weights.head_w[i]).toBe(untied.weights.head_w[i]);
    }
  });

  it("cannot be reloaded as a tied model by flipping the flag", () => {
    const swapped = throughFile(bundle()) as LexModelBundle;
    swapped.config = { ...swapped.config, tied: true };
    expect(() => importLexBundle(swapped)).toThrow(/no slot for/);
  });

  it("refuses a tied bundle that has been given a readout", () => {
    const b = throughFile(freshBundle()) as LexModelBundle;
    b.weights.head_w = { shape: [b.config.vocab_rows, b.config.d_model], data: b.weights.embed.data };
    expect(() => importLexBundle(b)).toThrow(/no slot for/);
  });
});

/**
 * The weights-only token is what the Weight Lab mints for an edited weight set (US-6), so
 * it has to behave like an identity: same numbers ⇒ same token, one flipped float ⇒ a
 * different one, and it must not move when only the vocabulary does.
 */
describe("the weights-only token behaves like an identity", () => {
  it("is stable, sensitive, and independent of the word list", () => {
    const a = lexWeightsTokenOf(trained.config, trained.weights);
    expect(lexWeightsTokenOf(trained.config, { ...trained.weights })).toBe(a);

    const nudged = Float32Array.from(trained.weights.embed);
    nudged[3] += 1e-3;
    expect(lexWeightsTokenOf(trained.config, { ...trained.weights, embed: nudged })).not.toBe(a);

    const shapes = lexWireShapes(trained.config);
    const wire: Record<string, Float32Array> = {};
    for (const name of weightNames(trained.config)) wire[toWireName(name)] = trained.weights[name];
    expect(lexWeightsToken(wire, shapes)).toBe(a);
    // Different word list, same weights: the JOINT token moves, the weights token does not.
    const other = trained.vocab.words.map((w) => `${w}x`);
    expect(lexModelToken(wire, shapes, toWireConfig(trained.config), other)).not.toBe(
      lexModelToken(wire, shapes, toWireConfig(trained.config), trained.vocab.words),
    );
  });
});

describe("a weight named after a JavaScript builtin is refused, typed [round 5, F5]", () => {
  /**
   * `const shape = shapes[name]` in `readWeights` walked `Object.prototype`, so a file
   * carrying a weight called `toString` (or `constructor`, or `__proto__`) got a
   * JavaScript builtin instead of `undefined`, sailed past the "has no slot for" refusal,
   * and died two lines later as `TypeError: shape.reduce is not a function` — an UNTYPED
   * throw, outside the `ApiError` surface `viz/lex/ModelFile.svelte` prints. Measured:
   *
   *     bogus_weight   -> GeoEngineError: … has no slot for   (the correct refusal)
   *     toString       -> TypeError: shape.reduce is not a function
   *     constructor    -> TypeError: shape.join is not a function
   *     __proto__      -> TypeError: shape.join is not a function
   *
   * This is the UNFIXED MIRROR of the defect round 5 fixed one file over, in
   * `staticClient/lex.ts` — the static path got `Object.hasOwn`, the engine did not.
   */
  /** A real saved file, as TEXT, carrying one extra weight called `name`. */
  function fileWithWeight(name: string): unknown {
    const text = JSON.stringify(freshBundle());
    const entry = `${JSON.stringify(name)}:{"shape":[1],"data":"AAAAAA=="},`;
    return JSON.parse(text.replace('"weights":{', `"weights":{${entry}`));
  }

  const INHERITED = [
    "constructor",
    "toString",
    "toLocaleString",
    "valueOf",
    "hasOwnProperty",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "__proto__",
    "__defineGetter__",
    "__lookupGetter__",
  ];

  for (const name of INHERITED) {
    it(`refuses a weight called ${name} the way it refuses any other stranger`, () => {
      // Injected as JSON TEXT, which is what a file on disk is: assigning
      // `weights["__proto__"] = …` in JavaScript sets the prototype and creates no key at
      // all, so an object-literal fixture cannot reach this case. `JSON.parse` does.
      const b = fileWithWeight(name);
      let thrown: unknown;
      try {
        importLexBundle(b);
      } catch (e) {
        thrown = e;
      }
      expect(thrown, `${name} loaded`).toBeInstanceOf(Error);
      // TYPED, and by the same sentence a made-up name gets — not a TypeError from
      // arithmetic on a builtin.
      expect((thrown as { type?: string }).type).toBe("InvalidParamError");
      expect((thrown as Error).message).toMatch(/has no slot for/);
    });
  }

  it("still refuses an ordinary unknown weight, and still loads a real file", () => {
    expect(() => importLexBundle(fileWithWeight("bogus_weight"))).toThrow(/has no slot for/);
    expect(importLexBundle(throughFile(freshBundle())).vocabWords).toEqual(trained.vocab.words);
  });
});
