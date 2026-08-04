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

import { GeoEngine, type WeightSet } from "../../src/lib/geoEngine";
import { FINETUNE_MAX_UNK_RATE, VOCAB_SIZE, VOCAB_WORDS } from "../../src/lib/geoEngine/model";
import {
  buildVocabWords,
  runScratchTrain,
  SCRATCH_LEARNED_MARGIN,
  uniformBaselineLoss,
} from "../../src/lib/geoEngine/scratch";
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
