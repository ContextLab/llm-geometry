/**
 * From-scratch training in TypeScript (feature 004, FR-421).
 *
 * Real training on the real committed corpus — no fixtures of convenience. The
 * strongest check here is the vocabulary one: the canonical vocab.json shipped by the
 * Python backend was built by GeoTokenizer.from_corpus_text from this exact file, so
 * if buildVocabWords() reproduces it word-for-word then the TS port of the frequency
 * ranking (including its alphabetical tie-break) is exact.
 */

import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { D_MODEL, VOCAB_SIZE, VOCAB_WORDS } from "../../src/lib/geoEngine/model";
import {
  buildVocabWords,
  corpusStats,
  initWeights,
  runScratchTrain,
} from "../../src/lib/geoEngine/scratch";
import { GeoTokenizer } from "../../src/lib/geoEngine/tokenizer";
import { goldenSources } from "./geoGoldenAssets";

const CORPUS = path.resolve(
  __dirname,
  "../../../backend/src/llm_geometry/geo/data/alice-in-wonderland.txt",
);

/**
 * The same body the backend trains on: geo/corpus.py strips the Project Gutenberg
 * header and footer before building the vocabulary, so comparing against the shipped
 * vocab.json requires the same trim (the license boilerplate contributes hundreds of
 * one-off word types that would otherwise crowd the tail of the ranking).
 */
function corpusBody(raw: string): string {
  const lines = raw.split(/\r?\n/);
  let start = 0;
  let end = lines.length;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes("*** START OF THE PROJECT GUTENBERG")) start = i + 1;
    else if (lines[i].includes("*** END OF THE PROJECT GUTENBERG")) {
      end = i;
      break;
    }
  }
  return lines.slice(start, end).join("\n").trim();
}

const corpusText = corpusBody(fs.readFileSync(CORPUS, "utf-8"));

describe("corpus stats", () => {
  it("counts tokens and distinct types like the backend", () => {
    const stats = corpusStats(corpusText);
    expect(stats.n_tokens).toBeGreaterThan(10_000);
    expect(stats.n_distinct).toBeGreaterThanOrEqual(VOCAB_WORDS);
    expect(stats.vocab_words_required).toBe(VOCAB_WORDS);
    expect(corpusStats("").n_distinct).toBe(0);
  });

  it("refuses text that cannot fill the vocabulary, naming the shortfall", () => {
    expect(() => buildVocabWords("alice met the rabbit ".repeat(20))).toThrowError(
      /distinct word types/,
    );
  });
});

describe("vocabulary construction matches the Python tokenizer exactly", () => {
  it("rebuilds the canonical vocabulary from the canonical corpus", () => {
    // The first golden source ships the vocab.json the backend really built.
    const src = goldenSources()[0];
    const canonical = GeoTokenizer.fromVocabJson(src.vocab);
    const rebuilt = buildVocabWords(corpusText);
    expect(rebuilt.length).toBe(VOCAB_WORDS);
    expect(rebuilt).toEqual(canonical.words);
  });
});

describe("fresh initialization", () => {
  it("puts every token embedding on the unit sphere", () => {
    const ws = initWeights(0);
    const emb = ws.embedding;
    expect(emb.length).toBe(VOCAB_SIZE * D_MODEL);
    let worst = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const o = v * D_MODEL;
      worst = Math.max(worst, Math.abs(Math.hypot(emb[o], emb[o + 1], emb[o + 2]) - 1));
    }
    expect(worst).toBeLessThan(1e-6);
  });

  it("is deterministic for a given seed and different across seeds", () => {
    expect(Array.from(initWeights(3).embedding)).toEqual(Array.from(initWeights(3).embedding));
    expect(Array.from(initWeights(3).embedding)).not.toEqual(
      Array.from(initWeights(4).embedding),
    );
  });
});

describe("training really learns", () => {
  it("drops well below the uniform-distribution loss and keeps embeddings on S²", () => {
    const tokenizer = new GeoTokenizer(buildVocabWords(corpusText));
    const ids = tokenizer.encodeStream(corpusText).slice(0, 20_000);

    const uniform = Math.log(VOCAB_SIZE); // 6.91 nats — the "learned nothing" baseline
    const result = runScratchTrain({ tokenIds: ids, epochs: 2, seed: 0 });

    expect(Number.isFinite(result.finalLoss)).toBe(true);
    expect(result.finalLoss).toBeLessThan(uniform - 1.0);
    expect(result.epochs).toBe(2);
    expect(result.nWindows).toBeGreaterThan(0);

    const emb = result.weights.embedding;
    let worst = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      const o = v * D_MODEL;
      worst = Math.max(worst, Math.abs(Math.hypot(emb[o], emb[o + 1], emb[o + 2]) - 1));
    }
    expect(worst).toBeLessThan(1e-5); // FR-103: still on the sphere after training
  }, 240_000);

  it("rejects an out-of-range epoch count and a too-short corpus", () => {
    const tokenizer = new GeoTokenizer(buildVocabWords(corpusText));
    const ids = tokenizer.encodeStream(corpusText).slice(0, 5_000);
    expect(() => runScratchTrain({ tokenIds: ids, epochs: 0 })).toThrowError(/epochs/);
    expect(() => runScratchTrain({ tokenIds: ids, epochs: 999 })).toThrowError(/epochs/);
    expect(() => runScratchTrain({ tokenIds: [1, 2, 3], epochs: 1 })).toThrowError(/too short/);
  });
});
