/**
 * Derived weight sets, honest training reports, and one canonical vocabulary
 * serialization — the static-build half of red-team 007 (findings F1, F2, F3, F6).
 *
 * Every case here reproduces something the red team observed on the DEPLOYED site:
 * a fine-tune or a weight edit of a scratch-trained model silently reverting to Alice
 * in Wonderland's word list (with self-consistent digests, so no reader could detect
 * it); a fine-tuning corpus tokenized 100 % to <unk> and reported as "loss 6.58 → 5.58
 * on your text"; a run that ended at the uniform baseline announced as a trained
 * model; and the same model getting two different `vocab_sha256` values depending on
 * which build wrote the file.
 *
 * Real engine, real training, real files. The scratch weights are produced by the real
 * TS trainer on a real (invented, structureless-on-purpose) corpus.
 */

import fs from "node:fs";
import path from "node:path";

import { describe, expect, it, beforeAll } from "vitest";

import { GeoEngine, weightsToken, type WeightSet } from "../../src/lib/geoEngine";
import {
  FINETUNE_MAX_UNK_RATE,
  VOCAB_SIZE,
  VOCAB_WORDS,
  WEIGHT_SHAPES,
} from "../../src/lib/geoEngine/model";
import {
  buildVocabWords,
  runScratchTrain,
  SCRATCH_LEARNED_MARGIN,
  uniformBaselineLoss,
} from "../../src/lib/geoEngine/scratch";
import { sha256Hex, utf8Bytes } from "../../src/lib/geoEngine/hash";
import { GeoTokenizer, canonicalVocabJson } from "../../src/lib/geoEngine/tokenizer";
import { FIXTURE_DIR, goldenSources } from "./geoGoldenAssets";

const SYLLABLES = [
  "ba", "de", "fi", "go", "hu", "ka", "le", "mo", "nu", "pa",
  "ri", "so", "tu", "va", "ze", "bo", "da", "fe", "gi", "ho",
];

/** `n` distinct pronounceable nonsense words, none of them English. */
function inventedWords(n: number): string[] {
  const out: string[] = [];
  for (const a of SYLLABLES) {
    for (const b of SYLLABLES) {
      for (const c of SYLLABLES) {
        out.push(a + b + c);
        if (out.length >= n) return out;
      }
    }
  }
  throw new Error("not enough syllable combinations");
}

/** A tiny deterministic PRNG so the corpus is identical on every run. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

const words = inventedWords(1200);
const rand = lcg(1865);
/** 13,200 i.i.d. tokens over 1,200 invented types: no structure to learn, no English. */
const inventedCorpus = Array.from({ length: 13_200 }, () => words[Math.floor(rand() * words.length)]).join(" ");

const src = goldenSources().find((s) => s.name === "fixtures")!;
/** The exact bytes `GeoTokenizer.to_json()` wrote — what F6 is about. */
const vocabRaw = fs.readFileSync(path.join(FIXTURE_DIR, "vocab.json"), "utf-8").trim();

/** ONE real from-scratch run on the invented corpus, shared by every case below. */
let scratchWords: string[];
let scratchRun: { weights: WeightSet; finalLoss: number };
beforeAll(() => {
  scratchWords = buildVocabWords(inventedCorpus);
  const tokenizer = new GeoTokenizer(scratchWords);
  scratchRun = runScratchTrain({
    tokenIds: tokenizer.encodeStream(inventedCorpus),
    epochs: 1,
    seed: 0,
  });
}, 300_000);

describe("derived weight sets keep the vocabulary they inherited [F1]", () => {
  let engine: GeoEngine;
  let scratchToken: string;

  beforeAll(() => {
    engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    scratchToken = engine.registerScratchModel(scratchRun.weights, scratchWords);
    expect(scratchWords).not.toEqual(engine.tokenizer.words);
  });

  it("carries a scratch model's words through a weight edit, into the saved file", () => {
    const edited = engine.postWeights({
      base: scratchToken,
      edits: [{ layer: 0, matrix: "W_Q", preset: "identity", seed: 0 }],
    });
    expect(engine.tokenizerFor(edited.weights_token).words).toEqual(scratchWords);
    const bundle = engine.exportBundle(edited.weights_token);
    expect(JSON.parse(bundle.vocab).words).toEqual(scratchWords);
  });

  it("carries a scratch model's words through a fine-tune, into the saved file", () => {
    const ft = engine.finetune({ text: inventedCorpus.slice(0, 4000), steps: 2, base: scratchToken });
    expect(engine.tokenizerFor(ft.weights_token!).words).toEqual(scratchWords);
    const bundle = engine.exportBundle(ft.weights_token!);
    expect(JSON.parse(bundle.vocab).words).toEqual(scratchWords);
  });

  it("keeps them through a whole chain: scratch → edit → fine-tune → edit", () => {
    const a = engine.postWeights({
      base: scratchToken,
      edits: [{ layer: 1, matrix: "W_K", preset: "identity", seed: 0 }],
    });
    const b = engine.finetune({ text: inventedCorpus.slice(0, 3000), steps: 1, base: a.weights_token });
    const c = engine.postWeights({
      base: b.weights_token!,
      edits: [{ layer: 2, matrix: "W_O", preset: "identity", seed: 0 }],
    });
    expect(engine.tokenizerFor(c.weights_token).words).toEqual(scratchWords);
  });

  it("leaves a fine-tune of the SHIPPED model on the canonical vocabulary", () => {
    const ft = engine.finetune({ text: "alice was beginning to get very tired of sitting", steps: 1 });
    expect(engine.tokenizerFor(ft.weights_token!).words).toEqual(engine.tokenizer.words);
    // ...and that file legitimately carries the shipped word list.
    expect(JSON.parse(engine.exportBundle(ft.weights_token!).vocab).words).toEqual(engine.tokenizer.words);
  });

  it("persists the ownership claim, and refuses to restore a derived set without its words", () => {
    const ft = engine.finetune({ text: inventedCorpus.slice(0, 2500), steps: 1, base: scratchToken });
    const exported = engine.exportWeightSet(ft.weights_token!);
    expect(exported.ownsVocab).toBe(true);
    expect(exported.vocabWords).toEqual(scratchWords);

    // A payload stripped of the word list it needs must be REFUSED, not restored
    // half-right — restoring it would leave `exportBundle` free to write a file
    // pairing these weights with Alice's words under a matching `vocab_sha256`.
    const fresh = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    const { vocabWords: _dropped, ...withoutVocab } = exported;
    expect(fresh.importWeightSet(ft.weights_token!, withoutVocab)).toBe(false);
    expect(fresh.importWeightSet(ft.weights_token!, exported)).toBe(true);
    expect(fresh.tokenizerFor(ft.weights_token!).words).toEqual(scratchWords);
  });
});

describe("a model's identity covers its word list [F1, third path]", () => {
  let engine: GeoEngine;

  beforeAll(() => {
    engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
  });

  it("gives two models with identical weights and different words different tokens", () => {
    // The token used to cover the weights alone, so these two collided — and the two
    // stacks resolved the collision in OPPOSITE directions: this map overwrote (last
    // write wins) where the Python store kept the first entry's word list. A saved
    // file's word list therefore depended on which build wrote it and in what order.
    const otherWords = scratchWords.map((w) => `zz${w}`);
    const a = engine.registerScratchModel(scratchRun.weights, scratchWords);
    const b = engine.registerScratchModel(scratchRun.weights, otherWords);
    expect(b).not.toBe(a);

    expect(engine.tokenizerFor(a).words).toEqual(scratchWords);
    expect(engine.tokenizerFor(b).words).toEqual(otherWords);
    expect(JSON.parse(engine.exportBundle(a).vocab).words).toEqual(scratchWords);
    expect(JSON.parse(engine.exportBundle(b).vocab).words).toEqual(otherWords);
  });

  it("hashes the vocabulary exactly as the Python backend does", () => {
    // The same two constants are pinned in
    // `tests/integration/test_geo_derived_vocab.py::test_the_token_covers_the_vocabulary_byte_for_byte`.
    // A deterministic synthetic weight set, so both stacks hash identical bytes without
    // shipping another fixture: this is what makes "the same model saved by either build
    // is the same file" checkable.
    const ws: WeightSet = {};
    for (const [name, shape] of WEIGHT_SHAPES) {
      const n = shape.reduce((a, d) => a * d, 1);
      const arr = new Float32Array(n);
      for (let i = 0; i < n; i++) arr[i] = Math.fround(i * Math.fround(0.001));
      ws[name] = arr;
    }
    const words = [...Array(VOCAB_WORDS)].map((_, i) => `w${i}`);
    expect(weightsToken(ws)).toBe("38cb99338fb6c40f022641b579a7e827");
    expect(weightsToken(ws, canonicalVocabJson(words))).toBe("50246246e336794517fcc299b505659a");
  });

  it("refuses to SAVE a set that owns a word list it no longer has", () => {
    // Unreachable through the public API by design — which is exactly why it was
    // untested, and why deleting the guard changed nothing. Reached here the way a real
    // session reaches it: the set is registered, then its vocabulary goes missing (an
    // eviction, a stale restore). Substituting the shipped word list would write a file
    // pairing these weights with Alice's words under a matching `vocab_sha256` — a file
    // no reader could ever reject.
    const token = engine.registerScratchModel(scratchRun.weights, scratchWords.map((w) => `q${w}`));
    const vocabs = (engine as unknown as { vocabs: Map<string, unknown> }).vocabs;
    expect(vocabs.delete(token)).toBe(true);
    // The refusal must come from the ownership guard by name — the re-hash check below
    // it would also throw here, and a test that accepted either would not notice this
    // guard being deleted (it did not, which is how it survived a mutation run).
    expect(() => engine.exportBundle(token)).toThrowError(
        /its ids mean its own words rather than the shipped model's/,
    );
  });
});

describe("persisted payloads that predate `ownsVocab` are refused, not decided [F2]", () => {
  let engine: GeoEngine;
  let scratchToken: string;

  beforeAll(() => {
    engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    scratchToken = engine.registerScratchModel(scratchRun.weights, scratchWords);
  });

  it("refuses a payload carrying neither an ownership flag nor a word list", () => {
    // Byte-for-byte what the pre-fix build persisted for a model derived from a scratch
    // model: weights, sources, setSource — and a token that is the WEIGHTS-ONLY hash,
    // because that is what minted it. So the content-hash check passes and cannot save
    // us here; nothing in the payload distinguishes this from a fine-tune of the shipped
    // model. Deciding it as "does not own a vocabulary" is the original corruption:
    // `tokenizerFor` falls back to Alice's words and `exportBundle` writes them into the
    // file under a matching `vocab_sha256`. Undecidable, therefore REFUSED — a storage-key
    // rename is not a defence, it only hides the payloads this build happens to have
    // written, not one copied between profiles or restored from a backup.
    const preFixToken = weightsToken(scratchRun.weights);
    const weights: Record<string, string> = {};
    for (const [name, arr] of Object.entries(scratchRun.weights)) {
      weights[name] = Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString("base64");
    }
    const preFix = { weights, sources: {}, setSource: "finetuned" };

    const fresh = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    expect(fresh.importWeightSet(preFixToken, preFix)).toBe(false);
    // Refused AND accounted for: `/unknown/` (the evicted-token wording) passed while a
    // model the user trained was being erased on boot with nothing said about it, so the
    // assertion now pins the sentence the engine actually owes them.
    expect(() => fresh.exportBundle(preFixToken)).toThrowError(/could not be restored/);
    expect(() => fresh.exportBundle(preFixToken)).toThrowError(
      /did not record whether a model's ids mean its own words/,
    );
    // ...and the same payload with the flag present is decidable, so it restores.
    expect(fresh.importWeightSet(preFixToken, { ...preFix, ownsVocab: false })).toBe(true);
  });

  it("refuses a payload whose claim and payload disagree, in either direction", () => {
    const exported = engine.exportWeightSet(scratchToken);
    const fresh = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    expect(fresh.importWeightSet(scratchToken, { ...exported, vocabWords: undefined })).toBe(false);
    expect(fresh.importWeightSet(scratchToken, { ...exported, ownsVocab: false })).toBe(false);
    // ...and a payload that pairs these weights with SOMEBODY ELSE's word list no longer
    // hashes to the token it is filed under, so the claim is checked rather than believed.
    const swapped = { ...exported, vocabWords: scratchWords.map((w) => `zz${w}`) };
    expect(fresh.importWeightSet(scratchToken, swapped)).toBe(false);
    expect(fresh.importWeightSet(scratchToken, exported)).toBe(true);
  });
});

describe("fine-tuning tokenizes with the ACTIVE model's vocabulary [F2 / issue #6]", () => {
  let engine: GeoEngine;
  let scratchToken: string;

  beforeAll(() => {
    engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    scratchToken = engine.registerScratchModel(scratchRun.weights, scratchWords);
  });

  it("reports an unk rate, and it is low when the base model knows the words", () => {
    const text = inventedCorpus.slice(0, 4000);
    // Under the CANONICAL tokenizer this text is 100 % <unk> — the stream the engine
    // used to train on while reporting "loss 6.58 → 5.58 on your text".
    const canonical = engine.tokenizer.encode(text, { truncate: false });
    expect(canonical.n_unk / canonical.ids.length).toBeGreaterThan(0.9);

    const ft = engine.finetune({ text, steps: 2, base: scratchToken });
    // The only unknowns left are the 200 word types a 1000-word vocabulary cannot
    // hold out of this corpus's 1200 — real, small, and REPORTED.
    expect(ft.unk_rate!).toBeLessThan(0.2);
    expect(ft.n_tokens!).toBeGreaterThan(100);
    expect(ft.n_unk!).toBe(Math.round(ft.unk_rate! * ft.n_tokens!));
  });

  it("refuses an almost-entirely-<unk> stream instead of reporting a loss drop", () => {
    expect(() => engine.finetune({ text: inventedCorpus.slice(0, 4000), steps: 2 })).toThrowError(
      /outside the active model's vocabulary/,
    );
    expect(FINETUNE_MAX_UNK_RATE).toBe(0.9);
  });
});

describe("a run that ended at the uniform baseline says so [F3]", () => {
  it("classifies a structureless run as not-learned", () => {
    const baseline = uniformBaselineLoss();
    expect(baseline).toBeCloseTo(Math.log(VOCAB_SIZE), 12);
    // Structureless text cannot be learned; the run is not wrong, the SILENCE was.
    expect(scratchRun.finalLoss).toBeGreaterThan(baseline - SCRATCH_LEARNED_MARGIN);
  });
});

describe("one canonical vocabulary serialization in both stacks [F6]", () => {
  it("is byte-identical to the vocab.json the Python backend wrote", () => {
    // vocabRaw is the file itself — `GeoTokenizer.to_json()` output — so this compares
    // the two serializers directly, byte for byte.
    expect(canonicalVocabJson((src.vocab as { words: string[] }).words)).toBe(vocabRaw);
  });

  it("is what exportBundle writes, so a round-trip is byte-identical", () => {
    const engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    expect(engine.exportBundle("learned").vocab).toBe(vocabRaw);
  });

  it("escapes non-ASCII words the way Python's ensure_ascii does", () => {
    const w = [...Array(VOCAB_WORDS)].map((_, i) => `w${i}`);
    w[0] = "é";
    w[1] = "—";
    const json = canonicalVocabJson(w);
    expect(/^[\x00-\x7f]*$/.test(json)).toBe(true);
    expect(json).toContain("\\u00e9");
    expect(json).toContain("\\u2014");
    expect(JSON.parse(json).words.slice(0, 2)).toEqual(["é", "—"]);
  });

  it("starts with the pinned prefix (sorted keys, compact separators)", () => {
    expect(canonicalVocabJson((src.vocab as { words: string[] }).words)).toMatch(
      /^\{"format":"geo-tokenizer-v1","specials":\{"<eos>":1,"<pad>":2,"<unk>":0\},"words":\[/,
    );
  });
});

/**
 * Round 5. Three wrong-answer paths that survived the identity fix, and one coercion.
 *
 * Each case here failed against the build these tests were added to: `tokenize` answered
 * with the shipped word list for a model this engine does not have; a payload written
 * under the old identity was deleted on boot with nothing said about it; a model file's
 * vocabulary was read by the permissive ASSET loader, so the two stacks disagreed about
 * which files are valid; and an edit's seed was coerced into a different matrix.
 */
describe("a model this engine does not have is not the shipped model [round 5, F1]", () => {
  let engine: GeoEngine;
  let scratchToken: string;

  beforeAll(() => {
    engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    scratchToken = engine.registerScratchModel(scratchRun.weights, scratchWords);
  });

  it("refuses to tokenize under an unknown token instead of using Alice's words", () => {
    // The static build LRU-drops persisted sets while the ACTIVE token is persisted
    // separately, so this is the ordinary state of a returning tab, not an exotic one.
    // `tokenize` used to answer 200 with the canonical vocabulary here — and the token
    // strip's own verification probe compares against exactly that vocabulary, so the
    // tab reported the word list VERIFIED for a model whose words it had never seen.
    expect(() => engine.tokenize("the cat sat", "deadbeefdeadbeefdeadbeefdeadbeef")).toThrowError(
      /unknown \(never minted here, or evicted\)/,
    );
    // ...and the two operations now agree, which is the actual defect: they did not.
    expect(() => engine.trace("the cat sat", "deadbeefdeadbeefdeadbeefdeadbeef")).toThrowError(
      /unknown \(never minted here, or evicted\)/,
    );
  });

  it("still answers for a model it does have, with that model's own words", () => {
    expect(engine.tokenize(scratchWords[0], scratchToken).tokens[0].unk).toBe(false);
    expect(engine.tokenizerFor(scratchToken).words).toEqual(scratchWords);
    // A canonical-descended set has no own vocabulary — absence there is the answer,
    // not a miss (this is the case a blanket throw would have broken).
    const ft = engine.finetune({ text: "alice was beginning to get very tired", steps: 1 });
    expect(engine.tokenizerFor(ft.weights_token!).words).toEqual(engine.tokenizer.words);
  });
});

describe("a persisted payload from an older build is explained, not erased [round 5, F2]", () => {
  it("names the format change for a payload written under the weights-only identity", () => {
    // Exactly what the previous build wrote for a scratch model: the word list is
    // present and correct, but the token is `weightsToken(weights)` — the hash from
    // before the word list joined the model's identity.
    const preIdentityToken = weightsToken(scratchRun.weights);
    const weights: Record<string, string> = {};
    for (const [name, arr] of Object.entries(scratchRun.weights)) {
      weights[name] = Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString("base64");
    }
    const payload = {
      weights,
      sources: {},
      setSource: "scratch",
      ownsVocab: true,
      vocabWords: scratchWords,
    };

    const fresh = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    expect(fresh.importWeightSet(preIdentityToken, payload)).toBe(false);
    // Refusing is right; refusing SILENTLY is what destroyed the model. The engine must
    // be able to say which of the two kinds of gone this is.
    let message = "";
    try {
      fresh.exportBundle(preIdentityToken);
    } catch (e) {
      message = (e as Error).message;
    }
    expect(message).toMatch(/could not be restored/);
    expect(message).toMatch(/named by its weights alone/);
    expect(message).not.toMatch(/never minted here/);
  });

  it("keeps saying 'evicted' for a token that really was never here", () => {
    const fresh = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    expect(() => fresh.exportBundle("0123456789abcdef0123456789abcdef")).toThrowError(
      /never minted here, or evicted/,
    );
  });

  it("distinguishes a tampered payload from an out-of-date one", () => {
    const engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    const token = engine.registerScratchModel(scratchRun.weights, scratchWords);
    const exported = engine.exportWeightSet(token);
    const fresh = GeoEngine.fromAssets(src.checkpoint, src.vocab);
    // Someone else's word list under these weights hashes to neither identity.
    expect(
      fresh.importWeightSet(token, { ...exported, vocabWords: scratchWords.map((w) => `zz${w}`) }),
    ).toBe(false);
    expect(() => fresh.exportBundle(token)).toThrowError(/do not hash to the token/);
  });
});

describe("both stacks agree on which model files are valid [round 5, F3]", () => {
  const engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
  const good = engine.exportBundle("learned");

  it("refuses a vocabulary whose declared specials are not the ones we use", () => {
    // Python IGNORED `specials` entirely, so this file loaded there with HTTP 200 (and
    // `<unk>` still read as 0) while the browser refused it: two builds, two answers to
    // "is this file valid?". Both refuse now.
    const vocab = JSON.stringify({
      format: "geo-tokenizer-v1",
      specials: { "<unk>": 5, "<eos>": 1, "<pad>": 2 },
      words: JSON.parse(good.vocab).words,
    });
    expect(() => GeoTokenizer.fromModelVocabJson(vocab)).toThrowError(
      /special <unk> has id 5, expected 0/,
    );
  });

  it("refuses the site's `tokens`-shaped asset export inside a model file", () => {
    // Accepted here (1000 words recovered from the 1003 entries) and an untyped HTTP 500
    // in Python — the same file, valid in one stack and a crash in the other.
    const words = JSON.parse(good.vocab).words as string[];
    const vocab = JSON.stringify({
      format: "geo-tokenizer-v1",
      specials: { unk: 0, eos: 1, pad: 2 },
      tokens: ["<unk>", "<eos>", "<pad>", ...words],
    });
    expect(() => GeoTokenizer.fromModelVocabJson(vocab)).toThrowError(/`tokens` but no `words`/);
    // Through the whole import path, with the file's own digest honestly recomputed so
    // the vocabulary check is what refuses it rather than the digest.
    const file = { ...good, vocab, vocab_sha256: sha256Hex(utf8Bytes(vocab)) };
    expect(() => engine.importBundle(file)).toThrowError(/`tokens` but no `words`/);
  });

  it("gives a typed refusal for every malformed vocabulary shape", () => {
    const cases: [string, RegExp][] = [
      ["{", /not valid JSON/],
      ["[1,2]", /must be a JSON object/],
      ['"hi"', /must be a JSON object/],
      ["null", /must be a JSON object/],
      ['{"format":"geo-tokenizer-v1","words":null}', /must be an array of strings/],
      ['{"format":"geo-tokenizer-v1","words":[["a"]]}', /must be an array of strings/],
      ['{"format":"geo-tokenizer-v1","specials":"x","words":[]}', /`specials` must be an object/],
    ];
    for (const [vocab, pattern] of cases) {
      expect(() => GeoTokenizer.fromModelVocabJson(vocab), vocab).toThrowError(pattern);
    }
  });

  it("still loads a real file", () => {
    expect(GeoTokenizer.fromModelVocabJson(good.vocab).words).toEqual(engine.tokenizer.words);
  });
});

describe("an edit's seed is the seed you asked for [round 5, item 6]", () => {
  const engine = GeoEngine.fromAssets(src.checkpoint, src.vocab);
  const edit = (seed: unknown) => () =>
    engine.postWeights({
      base: "learned",
      edits: [{ layer: 0, matrix: "W_Q", preset: "random", seed } as never],
    });

  it("refuses a non-integer, non-finite or negative seed instead of picking seed 0", () => {
    // `Math.trunc(Number(seed)) || 0` mapped 1.5 -> 1 (a DIFFERENT preset matrix,
    // reported as the one requested) and Infinity / NaN / -1 -> 0, which also swallowed
    // the "that seed is not shipped" refusal the static build owes.
    expect(edit(1.5)).toThrowError(/seed must be an integer/);
    expect(edit("7")).toThrowError(/seed must be an integer/);
    expect(edit(true)).toThrowError(/seed must be an integer/);
    expect(edit(Infinity)).toThrowError(/seed must be an integer/);
    expect(edit(NaN)).toThrowError(/seed must be an integer/);
    expect(edit(-1)).toThrowError(/seed must be in 0\.\./);
    expect(edit(Number.MAX_SAFE_INTEGER + 2)).toThrowError(/seed must be in 0\.\./);
  });

  it("accepts the seeds that really exist", () => {
    expect(edit(0)()).toHaveProperty("weights_token");
    expect(edit(1)()).toHaveProperty("weights_token");
  });
});
