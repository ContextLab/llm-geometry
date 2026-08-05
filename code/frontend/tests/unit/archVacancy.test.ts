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
import fs from "node:fs";
import path from "node:path";
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
  VACANCY_FP32_REFERENCE,
  VACANCY_PER_PASSAGE_REFUSAL,
  VACANCY_PRE_REWRITE_Q8,
  VACANCY_UNKNOWN_FORM_REFUSAL,
  VACANCY_MIN_POOLED_PRESERVED,
  VACANCY_ONNX_TORCH_AGREEMENT_NATS,
  assertPooledPreservedBoundable,
  onnxRepo,
  onnxRepoIds,
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

/** tests/unit -> tests -> frontend -> code -> repo root. */
const REPO_ROOT = path.resolve(__dirname, "../../../..");
/** The recording of one real gpt2 run, written by `scripts/measure_vacancy_fp32.py`. */
const FP32_RECORD = path.join(REPO_ROOT, "specs/007-vacancy-transform-field/fp32-reference.json");

/** A number as it appears in an interpolated sentence, safe to put in a RegExp. */
function esc(value: number): string {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

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

  /**
   * Round 3. F6 was closed for LETTERS and left open for JOINERS, which is the half an
   * ordinary reader actually hits: `don’t` with the curly apostrophe every browser, phone
   * keyboard and word processor produces scored HTTP 200 and swapped to `big’t`.
   *
   * This table is the CLASS, not the character that was reported, and it is duplicated
   * verbatim in the backend's `test_the_whole_joiner_class_is_refused_not_just_the_one_seen`
   * so the two stacks are pinned to the same answers.
   */
  const JOINER_CASES: [string, string][] = [
    ["The cat’s don’t stop.", "’ U+2019 right single quote — the default pasted apostrophe"],
    ["The ‘cat‘s ran.", "‘ U+2018 left single quote"],
    ["He said itʼs fine.", "ʼ U+02BC modifier letter apostrophe"],
    ["The catʹs paw.", "ʹ U+02B9 modifier letter prime"],
    ["The cat′s paw.", "′ U+2032 prime"],
    ["The co­operate dog ran.", "U+00AD soft hyphen — invisible"],
    ["The cat‍sat down.", "U+200D ZWJ — invisible"],
    ["The cat‌sat down.", "U+200C ZWNJ — invisible"],
    ["The cat⁠sat down.", "U+2060 word joiner — invisible"],
    ["The cat﻿sat down.", "U+FEFF ZWNBSP — invisible"],
    ["The cat​sat down.", "U+200B ZWSP — invisible"],
    ["A co‐operate dog.", "U+2010 hyphen"],
    ["A co‑operate dog.", "U+2011 non-breaking hyphen"],
    ["A cat‒dog.", "U+2012 figure dash"],
    ["A cat–dog.", "U+2013 en dash"],
    ["A cat—dog.", "U+2014 em dash"],
    ["A cat－dog.", "U+FF0D fullwidth hyphen-minus"],
    ["A cat﹣dog.", "U+FE63 small hyphen-minus"],
    ["A cat−dog.", "U+2212 minus sign"],
    ["The para·llel cat.", "U+00B7 middle dot"],
    ["The para‧llel cat.", "U+2027 hyphenation point"],
    ["The cab́le ran.", "U+0301 combining acute with no precomposed form on b"],
  ];

  it("refuses the whole joiner class, not just the character that was reported", () => {
    for (const [text, why] of JOINER_CASES) {
      expect(fragmentedWords(text, WORD_RE), why).not.toEqual([]);
      expect(() => checkWordAlphabet(text, WORD_RE, 0), why).toThrowError(
        /InvalidParamError|word alphabet/,
      );
    }
  });

  it("still admits every joiner WORD_RE really does handle", () => {
    // The straight apostrophe and the ASCII hyphen ARE in the transform's alphabet, so
    // widening the run definition must not start refusing ordinary English.
    for (const ok of [
      "The cat's don't stop.",
      "A co-operate good-bye o'clock dog.",
      "The cat sat on the mat and did not move at all today.",
    ]) {
      expect(fragmentedWords(ok, WORD_RE)).toEqual([]);
      expect(() => checkWordAlphabet(ok, WORD_RE, 0)).not.toThrow();
    }
  });

  it("names the joined word whole, so the reader can find it", () => {
    // Not "don" or "t": the run the reader wrote.
    expect(fragmentedWords("The cat’s don’t stop.", WORD_RE)).toEqual(["cat’s", "don’t"]);
    expect(fragmentedWords("The co­operate dog.", WORD_RE)).toEqual(["co­operate"]);
    // A trailing joiner is punctuation, not part of the word: an em dash between spaces
    // separates, and both neighbours are whole WORD_RE words.
    expect(fragmentedWords("The cat — the dog ran.", WORD_RE)).toEqual([]);
  });

  it("refuses such a passage with a typed error naming the word", () => {
    // The full stack RETURNED a score for this one: "a café" vacated to "a washé".
    expect(() =>
      checkWordAlphabet("The dog and the cat sat with a café on the table.", WORD_RE, 0),
    ).toThrowError(/ASCII letters joined by/);
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
    expect(d.refused?.message).toMatch(/sign flip/);
  });

  /**
   * Round 5, and the THIRD instance of one failure in this campaign: a number in a
   * user-facing sentence that describes a configuration which no longer exists.
   *
   * What shipped, verbatim: "Measured on this very configuration: float32 says 0.273 for
   * gpt2 and q8 says 0.235, a 14 % error". Both values are pre-rewrite — the swap rewrite
   * of 2026-08-04 moved `unknown_form` to 0.2872 — so the sentence asserted a measurement
   * of "this very configuration" taken on a different one. Round 3 fixed the *label*
   * (`QUANTIZATION_TERM`) and left the *prose*.
   *
   * The first repair was called structural and was not: the sentence interpolated the
   * constants, and this test then asserted the sentence CONTAINED them — which it must,
   * having just interpolated them. Two mutants proved it (round 5, 815/815 green each):
   * changing `unknownForm` from 0.2872 to 0.4872, and EXCHANGING the two interpolation
   * slots, which restored the exact original defect — "on the configuration that ships …
   * it is 0.2726" — because both numbers were still somewhere in the string.
   *
   * So the assertions below pin each figure to the CLAUSE it belongs to, and the constant
   * itself is pinned, in the test after this one, to a recorded real run.
   */
  it("puts each figure in the clause it belongs to, not merely somewhere in the message", () => {
    const msg = byId.unknown_form.refused!.message;
    const fp32 = VACANCY_FP32_REFERENCE;
    const pre = VACANCY_PRE_REWRITE_Q8;
    // The two values must differ, or the clause test below could not tell them apart.
    expect(fp32.unknownForm).not.toBe(pre.unknownForm.fp32);
    // THE SHIPPED CONFIGURATION -> the re-measured float32 figure, its se and its n.
    expect(msg).toMatch(
      new RegExp(
        `configuration that ships — ${fp32.model}, float32, the six default passages, ` +
          `p = 1, seed = 0 — it is ${esc(fp32.unknownForm)} ± ${esc(fp32.unknownFormSe)} ` +
          `nats over ${fp32.pairedPreserved} paired tokens`,
      ),
    );
    // HISTORY -> the pre-rewrite pair, and only there.
    expect(msg).toMatch(
      new RegExp(
        `variant texts that no longer exist: there float32 read ${esc(pre.unknownForm.fp32)} ` +
          `and q8 read ${esc(pre.unknownForm.q8)}, a ${pre.unknownForm.errorPercent} % error`,
      ),
    );
    // …and neither number appears in the other's clause. Exchanging the slots must fail.
    expect(msg).not.toMatch(
      new RegExp(`configuration that ships[^.]*${esc(pre.unknownForm.fp32)}`),
    );
    expect(msg).not.toMatch(new RegExp(`no longer exist[^.]*${esc(fp32.unknownForm)}`));
    // The stale pair, and the stale range that excluded the value that ships.
    expect(msg).not.toContain("0.16–0.27");
    expect(msg).not.toMatch(/Measured on this very configuration/);
  });

  /**
   * The other half of the pin, and the one the mutation survived: the CONSTANT.
   *
   * `VACANCY_FP32_REFERENCE` is a literal in TypeScript because the browser build cannot
   * run gpt2 at build time. Its claim to be "pinned to a real run" was, until now, a
   * Python test asserting its own separately-typed literal. The chain is now
   *
   *     the real gpt2 run -> specs/007-vacancy-transform-field/fp32-reference.json
   *                       -> VACANCY_FP32_REFERENCE
   *
   * with `test_the_fp32_arm_quoted_in_the_static_client` (backend, real model) asserting
   * the first arrow, and this test the second. Neither end restates a number.
   */
  it("ships fp32 constants that equal the recorded run of the real model", () => {
    const record = JSON.parse(fs.readFileSync(FP32_RECORD, "utf-8")) as {
      format: string;
      model: string;
      params: { p: number; seed: number; passages: string };
      preserved: Record<string, number>;
      differences: Record<string, { nats: number; se: number; nPairs: number }>;
    };
    expect(record.format).toBe("vacancy-fp32-reference-v1");
    expect(record.model).toBe(VACANCY_FP32_REFERENCE.model);
    expect(record.params).toEqual({ passages: "default", p: 1, seed: 0 });
    expect(VACANCY_FP32_REFERENCE.pairedPreserved).toBe(record.differences.unknown_form.nPairs);
    expect(VACANCY_FP32_REFERENCE.pairedPreserved).toBe(record.preserved.english);
    // Four decimal places is the precision the sentences quote; `toBeCloseTo(x, 3)` is
    // |Δ| < 5e-4, the same tolerance the backend test applies to the same numbers.
    expect(VACANCY_FP32_REFERENCE.wrongContent).toBeCloseTo(record.differences.wrong_content.nats, 3);
    expect(VACANCY_FP32_REFERENCE.unknownForm).toBeCloseTo(record.differences.unknown_form.nats, 3);
    expect(VACANCY_FP32_REFERENCE.total).toBeCloseTo(record.differences.total.nats, 3);
    expect(VACANCY_FP32_REFERENCE.unknownFormSe).toBeCloseTo(record.differences.unknown_form.se, 3);
  });

  it("labels the retained q8 figures as pre-rewrite wherever it quotes them", () => {
    const msg = byId.unknown_form.refused!.message;
    // The pre-rewrite pair may be quoted — it is the only like-for-like comparison that
    // exists — but only as history, and never as the arithmetic of a current bound.
    expect(msg).toContain(String(VACANCY_PRE_REWRITE_Q8.unknownForm.fp32));
    expect(msg).toContain(String(VACANCY_PRE_REWRITE_Q8.unknownForm.q8));
    expect(msg).toMatch(/before the swap transform was rewritten/);
    expect(msg).toMatch(/no longer exist/);
    expect(msg).toMatch(/No q8 figure exists for the configuration that ships/);
  });

  /**
   * The sweep the round-3 fix did not do: every user-facing string this module can hand a
   * reader, checked for the claim FR-720a governs. "measured" is a claim about provenance,
   * and no q8 quantity in this build has current provenance.
   */
  it("makes no unqualified q8 measurement claim on any refusal it can emit", () => {
    const surfaces = [
      VACANCY_ABSOLUTE_REFUSAL.message,
      VACANCY_PER_PASSAGE_REFUSAL.message,
      VACANCY_UNKNOWN_FORM_REFUSAL.message,
    ];
    for (const msg of surfaces) {
      expect(msg).not.toMatch(/Measured on this very configuration/);
      expect(msg).not.toMatch(/has a measured bound/);
      // Any sentence that still uses the word must say which configuration it belongs to.
      if (/(?<!re-)measured/.test(msg)) {
        expect(msg, msg).toMatch(/before the swap|pre-rewrite|retained|no longer exist/);
      }
    }
  });

  /**
   * The quantization floor, at the unit level.
   *
   * `VACANCY_MIN_POOLED_PRESERVED` could be lowered from 700 to 7 with the whole unit
   * suite green (round 5): the only gate was `tests/e2e/docs.spec.ts`, which compares it
   * to the Info tab's prose. The static build would then print pooled q8 differences on
   * samples two orders of magnitude below the size the retained bound was taken at — the
   * regime where a single-passage delta was wrong by 115 % of its own value.
   *
   * The sizes below are not invented: they come from the recorded real run (856 preserved
   * tokens over the six default passages) and from the study the bound is retained from.
   */
  it("refuses a pool smaller than the size the retained q8 bound was measured at", () => {
    const record = JSON.parse(fs.readFileSync(FP32_RECORD, "utf-8")) as {
      preserved: Record<string, number>;
    };
    // One default passage's worth of preserved tokens — what a reader who pastes a single
    // passage gets. It must NOT be enough for a q8 number.
    const onePassage = Math.round(record.preserved.english / golden.count);
    expect(onePassage).toBeLessThan(VACANCY_MIN_POOLED_PRESERVED);
    expect(() => assertPooledPreservedBoundable(onePassage)).toThrow(/below the/);
    expect(() => assertPooledPreservedBoundable(onePassage)).toThrow(
      new RegExp(`Pooled over ${onePassage} preserved tokens`),
    );
    // …and the boundary itself is where the constant says it is.
    expect(() =>
      assertPooledPreservedBoundable(VACANCY_MIN_POOLED_PRESERVED - 1),
    ).toThrow(/below the/);
    expect(() => assertPooledPreservedBoundable(VACANCY_MIN_POOLED_PRESERVED)).not.toThrow();
    // …and the floor is the size §8.3a says the bound was taken at, read from §8.3a.
    const architecture = fs.readFileSync(
      path.join(REPO_ROOT, "specs/007-vacancy-transform-field/architecture.md"),
      "utf-8",
    );
    const measuredAt = architecture.match(/~(\d+) preserved\s+closed-class tokens per condition/);
    expect(measuredAt, "§8.3a no longer states the size the bound was measured at").not.toBeNull();
    expect(VACANCY_MIN_POOLED_PRESERVED).toBe(Number(measuredAt![1]));
  });

  /**
   * "Two exported constants now hold the figures, and every sentence is interpolated from
   * them" was said while `−0.19`, `+0.40`, `0.65`, `0.28` and `5.3e-4` were still typed by
   * hand into these very messages (round 5, F5). All five were correct at the time, which
   * is exactly how the earlier stale numbers survived too.
   *
   * So the claim is checked instead of repeated: every DECIMAL a reader can be shown by
   * this module must be derivable from a constant. A new hand-typed figure fails here.
   */
  it("hand-types no decimal figure in any refusal it can emit", () => {
    const pre = VACANCY_PRE_REWRITE_Q8;
    const fp32 = VACANCY_FP32_REFERENCE;
    const allowed = new Set(
      [
        Math.abs(pre.absoluteShiftNats.gpt2).toFixed(2),
        Math.abs(pre.absoluteShiftNats.smollm2).toFixed(2),
        String(pre.study.perPassageWorstNats),
        String(pre.study.unknownFormWorstPassageNats),
        String(pre.study.pooledBoundNats),
        String(pre.unknownForm.fp32),
        String(pre.unknownForm.q8),
        String(fp32.unknownForm),
        String(fp32.unknownFormSe),
        ...pre.study.unknownFormRange.split(/[–-]/),
        VACANCY_ONNX_TORCH_AGREEMENT_NATS.toExponential(1),
      ].map((s) => s.trim()),
    );
    for (const msg of [
      VACANCY_ABSOLUTE_REFUSAL.message,
      VACANCY_PER_PASSAGE_REFUSAL.message,
      VACANCY_UNKNOWN_FORM_REFUSAL.message,
    ]) {
      for (const token of msg.match(/\d+\.\d+(?:e-?\d+)?/g) ?? []) {
        expect(allowed, `${token} in: ${msg}`).toContain(token);
      }
    }
    // …and the two figures that are not decimals but are quoted as measurements.
    expect(VACANCY_UNKNOWN_FORM_REFUSAL.message).toContain(String(fp32.pairedPreserved));
    expect(VACANCY_PER_PASSAGE_REFUSAL.message).toContain(
      `${pre.study.perPassageWorstPercent} %`,
    );
  });

  it("reports the two pooled differences it has a bound for", () => {
    for (const id of ["wrong_content", "total"]) {
      expect(byId[id].nats).toBeTypeOf("number");
      // The stated ± is the retained q8 bound; nothing here invents one, and nothing
      // here calls it a measurement of the shipped swap (architecture.md §8.3a).
      expect(byId[id].quantizationUncertaintyNats).toBe(0.2);
      expect(byId[id].refused).toBeUndefined();
    }
  });

  it("calls an identity an identity from the TEXTS, not from p", () => {
    // Red team F3. `p === 0` is only the commonest way to reach an identity: a passage of
    // nothing but closed-class scaffolding has no vacatable word at any p, so the three
    // variants come out one string there too. On the `p` test that rendered as
    // "0.000 ± 0.000 (sampling, 20 paired tokens)", an "upper bound" caption, and the
    // advice to score more text — none of which is true of an identity.
    const zero = { nats: 0, se: 0, nPairs: 20 };
    const atP1 = staticVacancyDifferences(zero, zero, { identical: true, p: 1 });
    for (const d of atP1) {
      expect(d.identity).toBe(true);
      expect(d.identityNote).toMatch(/no word the transform vacates/);
      expect(d.identityNote).toMatch(/not a measurement/);
    }
    // …and the p = 0 route keeps its own, correct, note.
    const atP0 = staticVacancyDifferences(zero, zero, { identical: true, p: 0 });
    expect(atP0[0].identityNote).toMatch(/At p = 0 no stem is vacated/);
    // A run whose variants really do differ is not an identity at p = 0 either — the
    // condition is the texts, in both directions.
    for (const d of staticVacancyDifferences(swap, nonce, { identical: false, p: 0 })) {
      expect(d.identity).toBe(false);
      expect(d.identityNote).toBeUndefined();
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
