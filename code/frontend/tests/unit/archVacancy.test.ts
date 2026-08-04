/**
 * The pretrained arm, browser side (contract §8; FR-717…720a).
 *
 * Real tokenizers and the real transform — no fabricated pieces anywhere. What this file
 * does NOT do is download 280 MB of ONNX weights: the arithmetic on real logits is
 * asserted by the backend's suite (which runs the same algorithm at float32) and by the
 * static e2e, which drives the built site with the real quantized model. What is left
 * here is exactly what can be wrong silently:
 *
 *   - byte-level alignment, which mis-attributes rather than failing;
 *   - the passage cut, which must be the SAME six excerpts the backend scores;
 *   - the static build's reporting policy, which is what keeps a quantized number from
 *     being presented as a measurement.
 */

import { describe, expect, it } from "vitest";
import { AutoTokenizer } from "@huggingface/transformers";

import { WORD_RE } from "../../src/lib/lexEngine";
import {
  defaultVacancyPassages,
  pairedDifference,
  pooledStats,
  preservedWordIndices,
  staticVacancyDifferences,
  vacancyVariantTexts,
  VACANCY_ABSOLUTE_REFUSAL,
  VACANCY_PER_PASSAGE_REFUSAL,
} from "../../src/lib/staticClient/arch";
import {
  checkWordAlphabet,
  fragmentedWords,
  nCharsOf,
  preservedTokenIndices,
  tokenByteSpans,
  wordSpans,
} from "../../src/lib/staticClient/byteSpans";
import type { ApiError } from "../../src/lib/dataClient";
import { readStaticJson } from "./staticTestUtils";
import golden from "../fixtures/arch-vacancy-passages.json";

const TEXTS = [
  "The cow jumped over the moon, and the little dog laughed.",
  "café naïve — “owl” ≈ ç√ 東京 end",
  "  leading and\ttabbed\nnewlines  ",
  "don't good-bye o'clock",
];

async function pieces(modelId: string, text: string): Promise<string[]> {
  const tok = await AutoTokenizer.from_pretrained(modelId);
  const inner = (tok as unknown as { _tokenizer: { encode(t: string): { tokens: string[] } } })
    ._tokenizer;
  return inner.encode(text).tokens;
}

async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

describe("byte-level alignment (§8.2, FR-718)", () => {
  for (const modelId of ["gpt2", "Qwen/Qwen2.5-0.5B-Instruct"]) {
    for (const raw of TEXTS) {
      it(`tiles ${JSON.stringify(raw.slice(0, 24))} exactly on ${modelId}`, async () => {
        const text = raw.normalize("NFC");
        const spans = tokenByteSpans(await pieces(modelId, text), text);
        const bytes = new TextEncoder().encode(text);
        // A true partition: contiguous, covering, ordered — so a per-token quantity can
        // be summed over a word without double-counting.
        expect(spans[0][0]).toBe(0);
        expect(spans[spans.length - 1][1]).toBe(bytes.length);
        for (let i = 1; i < spans.length; i++) expect(spans[i][0]).toBe(spans[i - 1][1]);
        const joined = spans.flatMap(([a, b]) => Array.from(bytes.slice(a, b)));
        expect(joined).toEqual(Array.from(bytes));
      }, 60_000);
    }
  }

  it("raises rather than mis-attributing when the pieces do not rebuild the text", async () => {
    const text = TEXTS[0];
    const p = await pieces("gpt2", text);
    expect(() => tokenByteSpans(p.slice(0, -1), text)).toThrowError(/alignment failed/);
  }, 60_000);

  it("attributes a leading-space token to its word (overlap, not 'starts inside')", async () => {
    const text = "the cow and the moon";
    const spans = tokenByteSpans(await pieces("gpt2", text), text);
    const words = wordSpans(text, WORD_RE);
    expect(words.map((w) => w.word)).toEqual(["the", "cow", "and", "the", "moon"]);
    const got = preservedTokenIndices(spans, words, new Set([0, 2, 3]));
    // gpt2 gives one token per word here, so the three closed-class words are three
    // tokens — each of which starts one byte BEFORE its word, on the space.
    expect(got.length).toBe(3);
    expect(spans[got[0]][0]).toBeLessThanOrEqual(words[0].start);
  }, 60_000);

  it("refuses a token that spans a preserved and a vacated word", () => {
    const words = wordSpans("the cow", WORD_RE);
    expect(() => preservedTokenIndices([[0, 7]], words, new Set([0]))).toThrowError(
      /spans both/,
    );
  });
});

describe("the three variants (§8.3)", () => {
  it("preserves the scaffolding byte for byte and moves everything else", () => {
    const passage = "Hey diddle diddle, the cat and the fiddle,\nThe cow jumped over the moon.";
    const texts = vacancyVariantTexts(passage, { p: 1, seed: 0, matchProsody: true, keep: [] });
    const { words, preserved } = preservedWordIndices(texts);
    expect(preserved.size).toBeGreaterThan(0);
    for (const name of ["swap", "nonce"] as const) {
      const variant = wordSpans(texts[name], WORD_RE);
      expect(variant.length).toBe(words.length);
      for (const i of preserved) expect(variant[i].word).toBe(words[i].word);
    }
    // …and the two vacated variants really differ from each other: a real English word
    // where the nonce variant invented one. Without that the decomposition is vacuous.
    const swap = wordSpans(texts.swap, WORD_RE);
    const nonce = wordSpans(texts.nonce, WORD_RE);
    const moved = words.filter((w) => !preserved.has(w.index));
    expect(moved.length).toBeGreaterThan(0);
    expect(moved.some((w) => swap[w.index].word !== w.word)).toBe(true);
    expect(moved.some((w) => nonce[w.index].word !== swap[w.index].word)).toBe(true);
  });

  it("is the identity at p = 0, so every variant is the same text", () => {
    const passage = "The cow jumped over the moon and the little dog laughed.";
    const texts = vacancyVariantTexts(passage, { p: 0, seed: 0, matchProsody: true, keep: [] });
    expect(new Set(Object.values(texts)).size).toBe(1);
  });

  it("refuses intermediate p, exactly as the Lexicon Lab does (§5.2a)", () => {
    // Red team F7: the panel exposed p at 0.05 steps and scored a decomposition through
    // a swap map that is injective only at p ∈ {0, 1}. The Lexicon Lab's engine raises
    // there; this arm did not.
    const passage = "The cow jumped over the moon and the little dog laughed at the cat.";
    for (const p of [0.05, 0.5, 0.95]) {
      expect(() =>
        vacancyVariantTexts(passage, { p, seed: 0, matchProsody: true, keep: [] }),
      ).toThrowError(/§5\.2a/);
    }
    for (const p of [0, 1]) {
      expect(() =>
        vacancyVariantTexts(passage, { p, seed: 0, matchProsody: true, keep: [] }),
      ).not.toThrow();
    }
  });
});

describe("the word alphabet (§8.2) — red team F6", () => {
  it("names the words WORD_RE would split or truncate", () => {
    // `WORD_RE` is ASCII-only, so `café` is the word `caf` and `naïvely` is `na`+`vely`.
    expect(fragmentedWords("a café on the table", WORD_RE)).toEqual(["café"]);
    expect(fragmentedWords("done naïvely", WORD_RE)).toEqual(["naïvely"]);
    expect(wordSpans("a café", WORD_RE).map((w) => w.word)).toEqual(["a", "caf"]);
  });

  it("leaves alone what WORD_RE matches whole, and what it never touches", () => {
    for (const ok of ["don't good-bye o'clock", "The cat sat. 🙂🙂 The dog ran.", "the 猫が座った cat"]) {
      expect(fragmentedWords(ok, WORD_RE)).toEqual([]);
      expect(() => checkWordAlphabet(ok, WORD_RE, 0)).not.toThrow();
    }
  });

  it("refuses such a passage with a typed error naming the word", () => {
    // The full stack RETURNED a score for this one: "a café" vacated to "a washé".
    expect(() =>
      checkWordAlphabet("The dog and the cat sat with a café on the table.", WORD_RE, 0),
    ).toThrowError(/ASCII letters only/);
    const err = (() => {
      try {
        checkWordAlphabet("He did it naïvely.", WORD_RE, 2);
        return null;
      } catch (e) {
        return e as ApiError;
      }
    })();
    expect(err?.type).toBe("InvalidParamError");
    expect(err?.message).toContain("naïvely");
    expect(err?.message).toContain("passage 2");
  });
});

describe("nChars is the same count in both stacks — red team F9", () => {
  it("counts Unicode code points, never UTF-16 units", () => {
    const probe = "The cat sat on the mat. 🙂🙂 It was a very good day for the dog and the bird.";
    // Verbatim from the red team's probe: python len() = 75, JS .length = 77.
    expect(probe.length).toBe(77);
    expect(nCharsOf(probe)).toBe(75);
  });
});

describe("the default passage set is the one the backend scores", () => {
  it("cuts byte-identical excerpts from the shipped corpus", async () => {
    const corpus = await readStaticJson<{ text: string }>("lex/corpus.json");
    const cut = defaultVacancyPassages(corpus.text.normalize("NFC"));
    expect(cut.length).toBe(golden.count);
    for (const row of golden.passages) {
      expect(await sha256Hex(cut[row.index])).toBe(row.sha256);
      expect((cut[row.index].match(new RegExp(WORD_RE.source, "g")) ?? []).length).toBe(
        row.n_words,
      );
    }
  });
});

describe("what the quantized static build may say (§8.3a, FR-720a)", () => {
  const swap = { nats: 0.83, se: 0.09, nPairs: 780 };
  const nonce = { nats: 1.01, se: 0.1, nPairs: 780 };
  const diffs = staticVacancyDifferences(swap, nonce);
  const byId = Object.fromEntries(diffs.map((d) => [d.id, d]));

  it("refuses nonce − swap with a typed error that names the full stack", () => {
    const d = byId.unknown_form;
    expect(d.nats).toBeNull();
    expect(d.se).toBeNull();
    expect(d.refused?.type).toBe("StaticModeError");
    expect(d.refused?.message).toMatch(/full stack/);
    expect(d.refused?.message).toMatch(/uvicorn/);
    // It must say WHY, in measured terms — not "unavailable in this demo".
    expect(d.refused?.message).toMatch(/0\.16–0\.27/);
    expect(d.refused?.message).toMatch(/sign flip/);
  });

  it("reports the two pooled differences it has a measured bound for", () => {
    for (const id of ["wrong_content", "total"]) {
      expect(byId[id].nats).toBeTypeOf("number");
      // The stated ± was MEASURED for q8; nothing here invents one.
      expect(byId[id].quantizationUncertaintyNats).toBe(0.2);
      expect(byId[id].refused).toBeUndefined();
    }
  });

  it("never headlines the conflated difference", () => {
    expect(byId.wrong_content.headline).toBe(true);
    expect(byId.unknown_form.headline).toBe(true);
    expect(byId.total.headline).toBe(false);
    expect(byId.total.note).toMatch(/conflates/);
    expect(byId.unknown_form.upperBound).toBe(true);
  });

  it("refuses absolute NLLs and per-passage deltas by name", () => {
    expect(VACANCY_ABSOLUTE_REFUSAL.type).toBe("StaticModeError");
    expect(VACANCY_ABSOLUTE_REFUSAL.message).toMatch(/−0\.19|\+0\.40/);
    expect(VACANCY_PER_PASSAGE_REFUSAL.message).toMatch(/115 %/);
    expect(VACANCY_PER_PASSAGE_REFUSAL.message).toMatch(/full stack/);
  });

  it("withholds the absolute numbers but not the token counts", () => {
    const stats = pooledStats(
      [{ pieces: [], nll: [NaN, 1, 2, 3], nChars: 40 }],
      [[1, 3]],
    );
    expect(stats.nllPreserved).toBeNull();
    expect(stats.nllAll).toBeNull();
    expect(stats.bitsPerChar).toBeNull();
    // Counts are exact at any dtype, and they are what show a nonce variant fragmenting.
    expect(stats.nTokens).toBe(3);
    expect(stats.nPreservedTokens).toBe(2);
    expect(stats.nChars).toBe(40);
  });
});

describe("paired differences", () => {
  it("pairs preserved tokens one-for-one and refuses a mismatch", () => {
    const a = [{ pieces: [], nll: [NaN, 1, 2, 3], nChars: 10 }];
    const b = [{ pieces: [], nll: [NaN, 1.5, 2.5, 9], nChars: 10 }];
    const d = pairedDifference(a, [[1, 2]], b, [[1, 2]]);
    expect(d.nats).toBeCloseTo(0.5, 12);
    expect(d.nPairs).toBe(2);
    expect(d.se).toBeCloseTo(0, 12);
    expect(() => pairedDifference(a, [[1, 2]], b, [[1]])).toThrowError(/cannot be paired/);
  });
});
