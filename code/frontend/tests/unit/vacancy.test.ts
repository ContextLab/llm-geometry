/**
 * The vacancy transform, measured on the REAL committed corpus.
 *
 * No mocks and no toy strings standing in for the thing: every assertion below runs
 * against `public/static-data/lex/corpus.json` — *The Real Mother Goose*, the same bytes
 * the Pages build ships and the same bytes the Python backend trains on. The properties
 * being checked are the ones `specs/007-vacancy-transform-field/architecture.md` says the
 * instrument must have, and they are the reason a `p`-sweep means anything:
 *
 *   SC-701 NESTING     `{stems vacated at p} ⊆ {stems vacated at p'}` for `p < p'`,
 *                      because `u` is a function of `(seed, stem)` alone.
 *   SC-702 STABILITY   a stem's nonce is identical at every `p` where it is vacated, and
 *                      unchanged if the input type order is shuffled — the map is built
 *                      once in canonical order, which is the correction §5.2 makes to the
 *                      source's order-dependent `used` set.
 *   SC-704 INJECTIVITY the image of the real type set is the same size as the type set,
 *                      verified rather than assumed, with `remintRounds` reported.
 *   FR-706             no minted form is ever a real corpus type.
 *
 * Plus the segmentation guarantee of §1 — every output is a single complete `WORD_RE`
 * match, so `tokenize(vacated)` aligns with `tokenize(original)` element for element and
 * line structure survives — which is what the invariance theorem of §7.3 rests on.
 *
 * The final `it` prints the measured numbers (types, tokens, prosody, stress-table
 * coverage) rather than hiding them: §10 forbids transcribing the source's numbers from
 * ITS corpus, so ours have to come from a run like this one.
 */

import { describe, expect, it } from "vitest";

import { DOLCH_ORDER, LexVocab, WORD_RE, dolchBudget, splitLines, tokenize } from "../../src/lib/lexEngine";
import {
  CODAS,
  DEFAULT_VACANCY_PARAMS,
  FUNCTION_WORDS,
  NUCLEI,
  ONSETS,
  SPLIT_EXCEPTIONS,
  STRESS_TABLE,
  SUFFIXES,
  UNSTRESSED_TAILS,
  buildVacancyMap,
  effectiveKeepSet,
  isEligible,
  mapVocabWords,
  meterScore,
  stemAndSuffix,
  stress,
  syllables,
  transformWord,
  typeCounts,
  vacancyDomain,
  vacancyParams,
  vacancyStats,
  vacancyU,
  vacateText,
  type VacancyMap,
  type VacancyParams,
} from "../../src/lib/lexEngine/vacancy";
import type { LexCorpusAsset } from "../../src/lib/staticClient/lex";
import { readStaticJson } from "./staticTestUtils";

// --- the real corpus, once ------------------------------------------------------------

const corpusAsset = await readStaticJson<LexCorpusAsset>("lex/corpus.json");
const CORPUS = corpusAsset.text;
const CORPUS_TYPES = new Set(tokenize(CORPUS));
const BUDGET = dolchBudget("full");
/** §5.2: corpus types ∪ the FULL Dolch list, via the helper — never built by hand here,
 *  since building it by hand at one call site and not another is the failure the helper
 *  exists to prevent. */
const DOMAIN = vacancyDomain(CORPUS_TYPES);

const P_GRID = [0, 0.25, 0.5, 0.75, 1] as const;
const SEEDS = [0, 7] as const;

function params(partial: Partial<VacancyParams>): VacancyParams {
  return vacancyParams(partial);
}

/** One map per seed — it is `p`-independent by construction, which is the point. */
const MAPS = new Map<number, VacancyMap>(
  SEEDS.map((seed) => [seed, buildVacancyMap(DOMAIN, params({ seed }))]),
);

function mapFor(seed: number): VacancyMap {
  const m = MAPS.get(seed);
  if (m === undefined) throw new Error(`no map for seed ${seed}`);
  return m;
}

/** The map as a sorted array of pairs, so two maps compare by value. */
function mappingEntries(vmap: VacancyMap): [string, string][] {
  return [...vmap.mapping].sort((a, b) => (a[0] < b[0] ? -1 : 1));
}

/** The source types whose surface actually changed, measured from the two texts. */
function changedTypes(original: string, rewritten: string): Set<string> {
  const before = original.match(new RegExp(WORD_RE.source, "g")) ?? [];
  const after = rewritten.match(new RegExp(WORD_RE.source, "g")) ?? [];
  expect(before.length).toBe(after.length);
  const out = new Set<string>();
  for (let i = 0; i < before.length; i++) {
    if (before[i] !== after[i]) out.add(before[i].toLowerCase());
  }
  return out;
}

/** The stems the map would vacate at this `p`, as a set. */
function vacatedStems(seed: number, p: number): Set<string> {
  const out = new Set<string>();
  for (const stem of mapFor(seed).mapping.keys()) {
    if (vacancyU(stem, seed) < p) out.add(stem);
  }
  return out;
}

/** Cache the vacated corpus per (seed, p) — each pass rewrites 86 kB. */
const vacatedCache = new Map<string, string>();
function vacated(seed: number, p: number): string {
  const key = `${seed}:${p}`;
  const hit = vacatedCache.get(key);
  if (hit !== undefined) return hit;
  const out = vacateText(CORPUS, mapFor(seed), params({ seed, p }));
  vacatedCache.set(key, out);
  return out;
}

// --- the verbatim tables --------------------------------------------------------------

describe("the ported tables are the source's, unchanged", () => {
  it("keeps the curated closed class and nothing else", () => {
    expect(FUNCTION_WORDS.size).toBe(137);
    // The warning the source carries: the closed class is the curated list ONLY. Union it
    // with Dolch service words and these content verbs are silently protected.
    for (const verb of ["run", "eat", "see", "get", "let", "put"]) {
      expect(FUNCTION_WORDS.has(verb)).toBe(false);
    }
    for (const w of ["the", "not", "under", "ten"]) expect(FUNCTION_WORDS.has(w)).toBe(true);
  });

  it("keeps the phonotactic tables in their source order", () => {
    expect(ONSETS.length).toBe(47);
    expect(NUCLEI.length).toBe(19);
    // architecture.md §5.4 says 49; the source list it is copied verbatim from has 46.
    expect(CODAS.length).toBe(46);
    expect(CODAS[0]).toBe("");
    expect(ONSETS[0]).toBe("b");
    expect(ONSETS[ONSETS.length - 1]).toBe("sq");
    expect(NUCLEI[0]).toBe("a");
    expect(NUCLEI[NUCLEI.length - 1]).toBe("er");
    expect(UNSTRESSED_TAILS.length).toBe(13);
    expect(UNSTRESSED_TAILS[0]).toBe("y");
    expect(UNSTRESSED_TAILS[UNSTRESSED_TAILS.length - 1]).toBe("ing");
  });

  it("keeps the suffix list in its trial order and the audited exceptions", () => {
    expect([...SUFFIXES]).toEqual(["ing", "edly", "est", "ies", "'s", "n't", "ed", "es", "er", "ly", "s"]);
    expect(SPLIT_EXCEPTIONS.size).toBe(10);
    // Departure 9: without these, `brother -> broth+er` and `morning -> morn+ing`.
    expect(stemAndSuffix("brother")).toEqual(["brother", ""]);
    expect(stemAndSuffix("morning")).toEqual(["morning", ""]);
    // ... and the artifact that remains, honestly: this is a spelling heuristic.
    expect(stemAndSuffix("ladder")).toEqual(["ladd", "er"]);
  });

  it("keeps the 61-entry hand stress table", () => {
    expect(STRESS_TABLE.size).toBe(61);
    expect(STRESS_TABLE.get("together")).toBe("0100");
    expect(STRESS_TABLE.get("Christmas")).toBe("10");
  });
});

// --- §6 prosody -----------------------------------------------------------------------

describe("prosody", () => {
  it("uses the hand table before the rule, case-sensitively", () => {
    expect(stress("Christmas")).toBe("10");
    expect(stress("little")).toBe("100");
    expect(stress("away")).toBe("01");
    // Not in the table -> the spelling rule.
    expect(stress("cat")).toBe("1");
    expect(stress("candle")).toBe("10");
  });

  it("prefers a minted pattern over both", () => {
    const minted = new Map([["zorble", "010"]]);
    expect(stress("zorble", minted)).toBe("010");
    expect(syllables("zorble", minted)).toBe(3);
    expect(stress("zorble")).toBe("10");
  });

  it("scores a foot as the fraction of matching syllable positions", () => {
    expect(meterScore("", "anapest")).toBe(0);
    // "the little cat" scans "1" + "100" + "1" = "11001": monosyllables are stressed by
    // the rule, so the scan is not the metrist's reading — it is the table's and the
    // rule's, which is exactly what `stressTableCoverage` exists to qualify.
    expect(meterScore("the little cat", "trochee")).toBeCloseTo(3 / 5, 12);
    expect(meterScore("the little cat", "anapest")).toBeCloseTo(1 / 5, 12);
    // "away away" scans "0101": a perfect iamb, and the hand table is why.
    expect(meterScore("away away", "iamb")).toBe(1);
    expect(() => meterScore("x", "spondee")).toThrow(/unknown foot/);
  });
});

// --- §2.2 eligibility -----------------------------------------------------------------

describe("eligibility (architecture.md §2.2)", () => {
  const keep = effectiveKeepSet();

  it("never vacates good-bye: no suffix matches and the stem carries a hyphen", () => {
    expect(stemAndSuffix("good-bye")).toEqual(["good-bye", ""]);
    expect(isEligible("good-bye", keep)).toBe(false);
    // ...and it survives the real transform at p = 1, in the corpus's own spelling.
    const out = vacateText("good-bye", mapFor(0), params({ seed: 0, p: 1 }));
    expect(out).toBe("good-bye");
  });

  it("never vacates don't", () => {
    // architecture.md §2.2 says the `n't` suffix splits this to stem `do`, which then
    // fails tests 1 and 3. It does not: §3's own rule requires `len(word) - len(s) >= 3`
    // and `len("don't") - len("n't")` is 2, so NO suffix matches and the stem is the
    // whole word — which then fails test 2 on the apostrophe. Same verdict, different
    // clause; §3 is the operative rule and the source behaves this way too.
    expect(stemAndSuffix("don't")).toEqual(["don't", ""]);
    expect(isEligible("don't", keep)).toBe(false);
    expect(isEligible("do", keep)).toBe(false); // and would fail tests 1 and 3 anyway
    expect(vacateText("don't", mapFor(0), params({ seed: 0, p: 1 }))).toBe("don't");
    // A longer contraction DOES split, and its stem is what §2.2 describes.
    expect(stemAndSuffix("couldn't")).toEqual(["could", "n't"]);
  });

  it("vacates dog's as <nonce>'s", () => {
    expect(stemAndSuffix("dog's")).toEqual(["dog", "'s"]);
    expect(isEligible("dog", keep)).toBe(true);
    const out = vacateText("dog's", mapFor(0), params({ seed: 0, p: 1 }));
    expect(out).not.toBe("dog's");
    expect(out.endsWith("'s")).toBe(true);
    expect(out.slice(0, -2)).toBe(mapFor(0).mapping.get("dog"));
  });

  it("rejects non-ASCII letters where Python's isalpha() would accept them", () => {
    // Test 2 is `^[A-Za-z]+$`, not `str.isalpha()` — the difference is invisible in the
    // shipped corpus and visible the moment someone pastes one.
    expect(isEligible("café", keep)).toBe(false);
    expect(isEligible("naïve", keep)).toBe(false);
  });

  it("respects an extra keep set", () => {
    const extended = effectiveKeepSet(["Dog", "cat"]);
    expect(isEligible("dog", extended)).toBe(false);
    expect(isEligible("cat", extended)).toBe(false);
    const vmap = buildVacancyMap(["dog", "cat", "hill"], params({ keep: ["dog"] }));
    expect(vmap.mapping.has("dog")).toBe(false);
    expect(vmap.mapping.has("hill")).toBe(true);
  });
});

// --- §4 nesting (SC-701) --------------------------------------------------------------

describe("SC-701 nesting: the vacated sets grow monotonically with p", () => {
  it("nests across the p grid, for both seeds, on the real corpus", () => {
    for (const seed of SEEDS) {
      const sets = P_GRID.map((p) => vacatedStems(seed, p));
      for (let i = 0; i + 1 < sets.length; i++) {
        for (const stem of sets[i]) {
          expect(sets[i + 1].has(stem)).toBe(true);
        }
        expect(sets[i].size).toBeLessThan(sets[i + 1].size);
      }
      expect(sets[0].size).toBe(0);
      expect(sets[sets.length - 1].size).toBe(mapFor(seed).mapping.size);
    }
  });

  it("nests at the level of the rewritten text, not just the decision", () => {
    // Whatever changed at p = 0.25 must still be changed, identically, at p = 0.5.
    const low = tokenize(vacated(0, 0.25));
    const mid = tokenize(vacated(0, 0.5));
    const base = tokenize(CORPUS);
    for (let i = 0; i < base.length; i++) {
      if (low[i] !== base[i]) expect(mid[i]).toBe(low[i]);
    }
  });

  it("u depends on (seed, stem) alone, so the two seeds disagree", () => {
    expect(vacancyU("hill", 0)).toBe(vacancyU("HILL", 0));
    expect(vacancyU("hill", 0)).not.toBe(vacancyU("hill", 7));
    for (const seed of SEEDS) {
      for (const stem of ["hill", "jack", "candle"]) {
        const u = vacancyU(stem, seed);
        expect(u).toBeGreaterThanOrEqual(0);
        expect(u).toBeLessThan(1);
        // 53-bit numerator over 2**53 — exactly representable, which is departure 2.
        expect(Number.isInteger(u * 2 ** 53)).toBe(true);
      }
    }
  });
});

// --- §5 stability (SC-702) ------------------------------------------------------------

describe("SC-702 stability: a stem's nonce does not depend on p or on input order", () => {
  it("mints the same nonce at every p where the stem is vacated", () => {
    const seed = 0;
    // Real corpus types whose stem is already vacated at the bottom of the grid, so every
    // higher p must reproduce the identical surface form.
    const low = vacatedStems(seed, 0.25);
    const early = [...CORPUS_TYPES].filter((t) => low.has(stemAndSuffix(t)[0])).sort().slice(0, 300);
    expect(early.length).toBeGreaterThan(100);
    const reference = new Map<string, string>();
    for (const p of [0.25, 0.5, 0.75, 1]) {
      const surface = mapVocabWords(early, mapFor(seed), params({ seed, p }));
      early.forEach((type, i) => {
        const seen = reference.get(type);
        if (seen === undefined) reference.set(type, surface[i]);
        else expect(surface[i]).toBe(seen);
        expect(surface[i]).not.toBe(type);
      });
    }
  });

  it("is unchanged when the input type order is shuffled", () => {
    // A deterministic shuffle: the assertion is about the map, not about randomness.
    const shuffled = [...DOMAIN];
    let state = 123456789;
    for (let i = shuffled.length - 1; i > 0; i--) {
      state = (state * 1103515245 + 12345) >>> 0;
      const j = state % (i + 1);
      const tmp = shuffled[i];
      shuffled[i] = shuffled[j];
      shuffled[j] = tmp;
    }
    expect(shuffled).not.toEqual(DOMAIN);
    const rebuilt = buildVacancyMap(shuffled, params({ seed: 0 }));
    const original = mapFor(0);
    expect(rebuilt.mapping.size).toBe(original.mapping.size);
    for (const [stem, nonce] of original.mapping) expect(rebuilt.mapping.get(stem)).toBe(nonce);
    expect(rebuilt.remintRounds).toBe(original.remintRounds);
  });

  it("is unchanged when the corpus is rewritten a second time", () => {
    expect(vacateText(CORPUS, mapFor(0), params({ seed: 0, p: 0.5 }))).toBe(vacated(0, 0.5));
  });
});

// --- §7.3 injectivity (SC-704) --------------------------------------------------------

describe("§5.2 the map is a pure function of (domain, seed, matchProsody)", () => {
  it("builds the domain by the union rule, and refuses a text", () => {
    expect(DOMAIN.length).toBe(CORPUS_TYPES.size + 22);
    for (const w of BUDGET) expect(DOMAIN).toContain(w.toLowerCase());
    for (const t of CORPUS_TYPES) expect(DOMAIN).toContain(t);
    // A string is itself an iterable of characters, so this would silently yield a domain
    // of single letters — every one failing §2.2's length test, giving an empty map and a
    // transform that does nothing, with no error anywhere.
    expect(() => vacancyDomain(CORPUS)).toThrow(/expected an iterable of TYPES/);
    expect(() => vacancyDomain("hello")).toThrow(/tokenize\(text\)/);
    // Idempotent, and insensitive to case and duplicates in the input.
    expect(vacancyDomain(DOMAIN)).toEqual(DOMAIN);
    expect(vacancyDomain([...CORPUS_TYPES].map((t) => t.toUpperCase()))).toEqual(DOMAIN);
  });

  it("is byte-identical through two different call paths", () => {
    // The whole point of removing `avoid`: the map can no longer depend on what a caller
    // remembered to pass, so two call sites that build the domain differently — but to the
    // same set — must produce the same map, key for key.
    for (const seed of SEEDS) {
      const viaHelper = buildVacancyMap(vacancyDomain(CORPUS_TYPES), params({ seed }));
      // A different path to the same set: reversed, duplicated, upper-cased, budget first.
      const scrambled = [
        ...BUDGET.map((w) => w.toUpperCase()),
        ...[...CORPUS_TYPES].reverse(),
        ...BUDGET,
        ...[...CORPUS_TYPES].map((t) => t.toUpperCase()),
      ];
      const viaScrambled = buildVacancyMap(vacancyDomain(scrambled), params({ seed }));
      expect(viaScrambled.mapping.size).toBe(viaHelper.mapping.size);
      for (const [stem, nonce] of viaHelper.mapping) expect(viaScrambled.mapping.get(stem)).toBe(nonce);
      expect(viaScrambled.remintRounds).toBe(viaHelper.remintRounds);
      expect(viaScrambled.imageSize).toBe(viaHelper.imageSize);
      expect([...viaScrambled.mintedStress].sort()).toEqual([...viaHelper.mintedStress].sort());
    }
  });

  it("is identical across all five Dolch domains", () => {
    // §5.2 measured this and says to assert it rather than rely on it: the domain rule is
    // "always the FULL list" precisely so switching budgets cannot re-mint the corpus in
    // front of the reader. A future change to the canonical order could break it silently.
    const reference = mapFor(7);
    for (const name of DOLCH_ORDER) {
      const domain = vacancyDomain([...CORPUS_TYPES, ...dolchBudget(name)]);
      const vmap = buildVacancyMap(domain, params({ seed: 7 }));
      expect(vmap.mapping.get("gum")).toBe(reference.mapping.get("gum"));
      expect(vmap.mapping.get("hang")).toBe(reference.mapping.get("hang"));
    }
  });

  it("depends on seed and on matchProsody, and on nothing else", () => {
    const a = buildVacancyMap(DOMAIN, params({ seed: 0 }));
    expect(a.mapping.get("gum")).not.toBe(mapFor(7).mapping.get("gum"));
    const flat = buildVacancyMap(DOMAIN, params({ seed: 0, matchProsody: false }));
    let differ = 0;
    for (const [stem, nonce] of flat.mapping) if (a.mapping.get(stem) !== nonce) differ++;
    expect(differ).toBeGreaterThan(0);
    // p and the other knobs are NOT inputs to the map — it is built once, for all p.
    for (const p of P_GRID) {
      const atP = buildVacancyMap(DOMAIN, params({ seed: 0, p, revealAfter: 3, consistent: false }));
      for (const [stem, nonce] of a.mapping) expect(atP.mapping.get(stem)).toBe(nonce);
    }
  });
});

describe("SC-704 injectivity on the real corpus", () => {
  it("maps the type set one-to-one, verified rather than assumed", () => {
    for (const seed of SEEDS) {
      const vmap = mapFor(seed);
      expect(vmap.bijective).toBe(true);
      expect(vmap.imageSize).toBe(DOMAIN.length);
      // Seed 0 needs no re-mint; seed 7 needs exactly one, for `hang` (see the regression
      // test below). Both are measured facts about this corpus, not aspirations.
      expect(vmap.remintRounds).toBe(seed === 7 ? 1 : 0);
      // The nonces themselves are distinct — a weaker statement than the above, but the
      // one the source's `used` set was supposed to guarantee.
      expect(new Set(vmap.mapping.values()).size).toBe(vmap.mapping.size);
    }
  });

  it("stays injective at EVERY p, where vacated and English types mix", () => {
    // Conditions A and B of §5.2 are p-independent, so injectivity has to hold at every p
    // and not only at full vacancy. This is the assertion the contract's §7.3 now makes.
    for (const seed of SEEDS) {
      const types = [...DOMAIN];
      for (const p of P_GRID) {
        const image = mapVocabWords(types, mapFor(seed), params({ seed, p }));
        expect(new Set(image).size).toBe(types.length);
      }
    }
  });

  it("re-mints the seed-7 hanged/waked collision away (regression)", () => {
    // The named case that forced conditions A/B into the contract. At seed 7 the stem
    // `hang` originally minted `wak`; no corpus type equals `wak`, so a bare-nonce check
    // passed, and at p = 1 `waked` is itself vacated, so a full-vacancy check passed too.
    // At p ∈ {0.25, 0.5} `hanged` was vacated and `waked` was not, so both became `waked`.
    const vmap = mapFor(7);
    expect(vmap.remintRounds).toBe(1);
    expect(vmap.mapping.get("hang")).not.toBe("wak");
    // The re-mint is still prosody-matched — `smeeg` is monosyllabic like `hang` — which
    // is the observable proving §5.5's thresholds are on the ATTEMPT counter and not on
    // the absolute salt. On the absolute reading a re-mint from base salt 1001 would start
    // with every check relaxed and the replacement could carry any syllable count.
    expect(vmap.mapping.get("hang")).toBe("smeeg");
    expect(syllables("smeeg")).toBe(1);
    expect(transformWord("hanged", vmap, params({ seed: 7, p: 1 }))).toBe("smeeged");
    // Condition B, stated directly: no assembled surface form is a domain type.
    for (const t of DOMAIN) {
      const [stem, suffix] = stemAndSuffix(t);
      const nonce = vmap.mapping.get(stem);
      if (nonce === undefined) continue;
      const surface = mapVocabWords([t], vmap, params({ seed: 7, p: 1 }))[0];
      expect(DOMAIN.includes(surface)).toBe(false);
      expect(surface.endsWith(suffix)).toBe(true);
    }
    // And the specific pair no longer meets, at the p where it used to.
    for (const p of [0.25, 0.5]) {
      const [hanged, waked] = mapVocabWords(["hanged", "waked"], vmap, params({ seed: 7, p }));
      expect(hanged).not.toBe(waked);
      expect(waked).toBe("waked"); // still English at this p — u(wak) = 0.571179
    }
  });

  it("maps a budget onto ids the pre-images had, order preserved (§7.2)", () => {
    const seed = 0;
    for (const p of P_GRID) {
      const mapped = mapVocabWords(BUDGET, mapFor(seed), params({ seed, p }));
      expect(mapped.length).toBe(BUDGET.length);
      expect(new Set(mapped).size).toBe(BUDGET.length);
      if (p === 0) expect(mapped).toEqual([...BUDGET]);
      // Order is the contract: word i of the budget becomes word i of the mapped budget.
      BUDGET.forEach((w, i) => {
        const [stem] = stemAndSuffix(w);
        const eligible = isEligible(stem, effectiveKeepSet());
        if (!eligible || !(vacancyU(stem, seed) < p)) expect(mapped[i]).toBe(w);
        else expect(mapped[i]).not.toBe(w);
      });
    }
  });
});

// --- FR-706 no minted form is a real word --------------------------------------------

describe("FR-706: a nonce never collides with a real corpus type", () => {
  it("holds for every minted form at both seeds", () => {
    for (const seed of SEEDS) {
      for (const nonce of mapFor(seed).mapping.values()) {
        expect(CORPUS_TYPES.has(nonce)).toBe(false);
      }
    }
  });

  it("holds for the rewritten corpus: no new type is an old type in disguise", () => {
    // Every type of the vacated corpus that is NOT a type of the original must be minted,
    // and every minted surface form must be absent from the original.
    const after = new Set(tokenize(vacated(0, 1)));
    const survivors = new Set<string>();
    for (const t of after) if (CORPUS_TYPES.has(t)) survivors.add(t);
    for (const t of survivors) {
      const [stem] = stemAndSuffix(t);
      // A surviving English type must be one the transform was never allowed to touch.
      const untouchable = !isEligible(stem, effectiveKeepSet());
      expect(untouchable).toBe(true);
    }
  });
});

// --- §1 segmentation ------------------------------------------------------------------

describe("§1 segmentation: the transform is a word-for-word bijection", () => {
  it("emits a single complete WORD_RE match for every corpus type", () => {
    const whole = new RegExp(`^(?:${WORD_RE.source})$`);
    const surface = mapVocabWords([...CORPUS_TYPES], mapFor(0), params({ seed: 0, p: 1 }));
    expect(surface.length).toBe(CORPUS_TYPES.size);
    for (const w of surface) {
      expect(whole.test(w)).toBe(true);
      const re = new RegExp(WORD_RE.source, "g");
      expect(w.match(re)).toEqual([w]);
    }
  });

  it("preserves the token count and ordering on the real corpus", () => {
    const base = tokenize(CORPUS);
    expect(base.length).toBe(corpusAsset.n_tokens);
    for (const seed of SEEDS) {
      for (const p of P_GRID) {
        expect(tokenize(vacated(seed, p)).length).toBe(base.length);
      }
    }
  });

  it("preserves line structure, so the <eos>-per-line rule fires in the same places", () => {
    const baseLines = splitLines(CORPUS);
    for (const p of P_GRID) {
      const lines = splitLines(vacated(0, p));
      expect(lines.length).toBe(baseLines.length);
      lines.forEach((line, i) => {
        expect(tokenize(line).length).toBe(tokenize(baseLines[i]).length);
      });
    }
  });

  it("§5.7 commutes with lowercasing, over every type in three casings", () => {
    // `lower(transformWord(w)) === transformWord(lower(w))`, normative. The first
    // implementation sliced the suffix case-preserved and ran the seam test against it, so
    // `gums -> flels` while `GUMS -> FLESS`: one source type, two surface forms, and since
    // the tokenizer lowercases those are two different types — §7.3 false.
    for (const seed of SEEDS) {
      const vmap = mapFor(seed);
      const cfg = params({ seed, p: 1 });
      for (const t of CORPUS_TYPES) {
        const lowerImage = transformWord(t.toLowerCase(), vmap, cfg);
        for (const cased of [t.toLowerCase(), t.toUpperCase(), t[0].toUpperCase() + t.slice(1)]) {
          expect(transformWord(cased, vmap, cfg).toLowerCase()).toBe(lowerImage);
        }
      }
    }
    // The `gums` case by name, since it is the one that was wrong.
    const vmap = mapFor(0);
    const cfg = params({ seed: 0, p: 1 });
    expect(transformWord("GUMS", vmap, cfg).toLowerCase()).toBe(transformWord("gums", vmap, cfg));
    // ...and case is still carried, not discarded.
    expect(transformWord("GUMS", vmap, cfg)).toBe(transformWord("gums", vmap, cfg).toUpperCase());
    const capital = transformWord("Gums", vmap, cfg);
    expect(capital[0]).toBe(capital[0].toUpperCase());
    expect(capital.slice(1)).toBe(transformWord("gums", vmap, cfg).slice(1));
  });

  it("commutes with lowercasing over the whole real corpus", () => {
    expect(vacated(0, 1).toLowerCase()).toBe(
      vacateText(CORPUS.toLowerCase(), mapFor(0), params({ seed: 0, p: 1 })),
    );
  });

  it("passes everything that is not a word through byte for byte", () => {
    const out = vacated(0, 1);
    const strip = (s: string): string => s.replace(new RegExp(WORD_RE.source, "g"), " ");
    expect(strip(out)).toBe(strip(CORPUS));
  });
});

// --- the endpoints of the sweep -------------------------------------------------------

describe("the endpoints of the p sweep", () => {
  it("p = 0 is the identity, for both seeds", () => {
    for (const seed of SEEDS) {
      expect(vacated(seed, 0)).toBe(CORPUS);
    }
  });

  it("p = 1 vacates every eligible type", () => {
    const keep = effectiveKeepSet();
    const out = vacated(0, 1);
    const before = tokenize(CORPUS);
    const after = tokenize(out);
    let eligibleTokens = 0;
    for (let i = 0; i < before.length; i++) {
      const [stem] = stemAndSuffix(before[i]);
      if (!isEligible(stem, keep)) {
        expect(after[i]).toBe(before[i]);
        continue;
      }
      eligibleTokens++;
      expect(after[i]).not.toBe(before[i]);
    }
    expect(eligibleTokens).toBeGreaterThan(0);
  });
});

// --- §6/§7.1 the control conditions ---------------------------------------------------

describe("the control conditions really are different conditions", () => {
  const seed = 0;

  it("consistent = false destroys type identity while holding the vacancy rate", () => {
    const vmapA = buildVacancyMap(DOMAIN, params({ seed }));
    const inconsistent = vacateText(CORPUS, vmapA, params({ seed, p: 1, consistent: false }));
    const consistent = vacated(seed, 1);
    expect(inconsistent).not.toBe(consistent);
    // Same number of tokens, same tokens changed, far more distinct types.
    expect(tokenize(inconsistent).length).toBe(tokenize(consistent).length);
    const changed = (text: string): number => {
      const base = tokenize(CORPUS);
      const t = tokenize(text);
      let n = 0;
      for (let i = 0; i < base.length; i++) if (t[i] !== base[i]) n++;
      return n;
    };
    expect(changed(inconsistent)).toBe(changed(consistent));
    expect(new Set(tokenize(inconsistent)).size).toBeGreaterThan(new Set(tokenize(consistent)).size);
  });

  it("condition B applies to the per-occurrence path too — the seed-7 `tak` case", () => {
    // §5.8. Condition B — no minted form may equal a domain type — was enforced when
    // building the map and NOT on the `consistent = false` minting path. Observable at
    // seed 7, `p = 1`: the stem `tak` (of `taking`) minted the nonce `tak`, so
    // `Taking -> Taking` and one token silently failed to vacate — `corpusTypesVacated`
    // 1921 against the consistent path's 1922, `tokensVacated` 8201 against 8202.
    //
    // §7.1 denies this control a STABILITY property, which is about a nonce being reused
    // across occurrences; it does not license a word surviving the transform. A control
    // whose vacancy rate is not the stated rate is not a control, so a per-occurrence nonce
    // must equal neither a domain type nor the stem it replaces.
    //
    // `tak` is a stem, not a type, which is why the domain did not already forbid it: the
    // domain holds the corpus's TYPES (`taking`, `takes`, …) plus the Dolch list.
    expect(stemAndSuffix("taking")).toEqual(["tak", "ing"]);
    expect(DOMAIN.includes("tak")).toBe(false);
    expect(DOMAIN.includes("taking")).toBe(true);

    for (const s of SEEDS) {
      // A fresh map per condition: `consistent = false` writes to `mintedStress`.
      const cfg = params({ seed: s, p: 1, consistent: false });
      const vmap = buildVacancyMap(DOMAIN, cfg);
      const text = vacateText(CORPUS, vmap, cfg);
      const stats = vacancyStats(CORPUS, text, vmap, cfg);
      // At `p = 1` every eligible type vacates (§10) — in this control exactly as in the
      // mapped condition, which is the whole claim.
      expect(stats.corpusTypesVacated, `seed ${s}`).toBe(1922);
      expect(stats.corpusTypesEligible, `seed ${s}`).toBe(1922);
      expect(stats.tokensVacated, `seed ${s}`).toBe(8202);

      // No eligible token survives the transform as itself.
      const before = tokenize(CORPUS);
      const after = tokenize(text);
      const survivors: string[] = [];
      for (let i = 0; i < before.length; i++) {
        if (!isEligible(stemAndSuffix(before[i])[0], effectiveKeepSet([]))) continue;
        if (after[i] === before[i]) survivors.push(before[i]);
      }
      expect(survivors, `seed ${s}`).toEqual([]);
    }
  });

  it("revealAfter > 0 leaves the first occurrences in English", () => {
    const revealed = vacateText(CORPUS, mapFor(seed), params({ seed, p: 1, revealAfter: 2 }));
    const pure = vacated(seed, 1);
    expect(revealed).not.toBe(pure);
    const base = tokenize(CORPUS);
    const r = tokenize(revealed);
    const q = tokenize(pure);
    let revealedUnchanged = 0;
    let pureUnchanged = 0;
    for (let i = 0; i < base.length; i++) {
      if (r[i] === base[i]) revealedUnchanged++;
      if (q[i] === base[i]) pureUnchanged++;
    }
    expect(revealedUnchanged).toBeGreaterThan(pureUnchanged);
  });

  it("matchProsody = false drops the syllable/stress match", () => {
    const flat = buildVacancyMap(DOMAIN, params({ seed, matchProsody: false }));
    const prosodic = mapFor(seed);
    expect(flat.mapping.size).toBe(prosodic.mapping.size);
    let differ = 0;
    for (const [stem, nonce] of flat.mapping) if (prosodic.mapping.get(stem) !== nonce) differ++;
    expect(differ).toBeGreaterThan(0);
    // Every flat nonce is monosyllabic by construction; the prosodic ones are not.
    for (const pattern of flat.mintedStress.values()) expect(pattern).toBe("1");
    const polysyllabic = [...prosodic.mintedStress.values()].filter((s) => s.length > 1).length;
    expect(polysyllabic).toBeGreaterThan(0);
  });

  it("the two seeds are different assignments of the same instrument", () => {
    expect(vacated(0, 1)).not.toBe(vacated(7, 1));
    expect(tokenize(vacated(0, 1)).length).toBe(tokenize(vacated(7, 1)).length);
  });
});

// --- §10 statistics -------------------------------------------------------------------

describe("§10 statistics", () => {
  it("returns exactly the contracted field names", () => {
    const stats = vacancyStats(CORPUS, vacated(0, 0.5), mapFor(0), params({ seed: 0, p: 0.5 }));
    expect(Object.keys(stats).sort()).toEqual(
      [
        "bijective",
        "corpusTypesEligible",
        "corpusTypesTotal",
        "corpusTypesVacated",
        "domainTypesEligible",
        "domainTypesTotal",
        "domainTypesVacated",
        "imageSize",
        "meanAnapestAfter",
        "meanAnapestBefore",
        "meanSyllablesAfter",
        "meanSyllablesBefore",
        "remintRounds",
        "stemsTotal",
        "stemsVacated",
        "stressFromMintedAfter",
        "stressFromMintedBefore",
        "stressFromRuleAfter",
        "stressFromRuleBefore",
        "stressFromTableAfter",
        "stressFromTableBefore",
        "tokensTotal",
        "tokensVacated",
      ].sort(),
    );
  });

  it("forbids an unprefixed types* name, which is what let the two stacks diverge", () => {
    // §10 now requires every type count to declare its scope. An unprefixed `typesTotal`
    // read as "domain" in one stack and "corpus" in the other, and both were defensible.
    const stats = vacancyStats(CORPUS, vacated(0, 0.5), mapFor(0), params({ seed: 0, p: 0.5 }));
    for (const key of Object.keys(stats)) {
      if (!key.startsWith("types")) continue;
      throw new Error(`unprefixed type count ${JSON.stringify(key)} — §10 forbids it`);
    }
    expect(stats.domainTypesTotal).not.toBe(stats.corpusTypesTotal);
  });

  it("counts stems ACTUALLY vacated, not the size of the prebuilt map", () => {
    // Departure 11: the zip copy reports `len(self.mapping)`, which is p-independent.
    const full = mapFor(0).mapping.size;
    const half = vacancyStats(CORPUS, vacated(0, 0.5), mapFor(0), params({ seed: 0, p: 0.5 }));
    const none = vacancyStats(CORPUS, vacated(0, 0), mapFor(0), params({ seed: 0, p: 0 }));
    expect(none.stemsVacated).toBe(0);
    expect(none.domainTypesVacated).toBe(0);
    expect(none.corpusTypesVacated).toBe(0);
    expect(none.tokensVacated).toBe(0);
    expect(half.stemsVacated).toBeGreaterThan(0);
    expect(half.stemsVacated).toBeLessThan(full);
    expect(half.stemsTotal).toBe(full);
    // The vacancy rate tracks p over the eligible stems, which is what p means.
    expect(half.stemsVacated / half.stemsTotal).toBeGreaterThan(0.4);
    expect(half.stemsVacated / half.stemsTotal).toBeLessThan(0.6);
  });

  it("satisfies the p = 1 identities that exposed the counting gap", () => {
    for (const seed of SEEDS) {
      const s = vacancyStats(CORPUS, vacated(seed, 1), mapFor(seed), params({ seed, p: 1 }));
      // u ∈ [0, 1) by construction, so at p = 1 everything eligible vacates — in BOTH
      // scopes. These three identities are what caught the domain/corpus ambiguity.
      expect(s.stemsVacated).toBe(s.stemsTotal);
      expect(s.domainTypesVacated).toBe(s.domainTypesEligible);
      expect(s.corpusTypesVacated).toBe(s.corpusTypesEligible);
      // Inflected forms share a stem, so there are always at least as many types as stems.
      expect(s.domainTypesEligible).toBeGreaterThanOrEqual(s.stemsTotal);
      expect(s.domainTypesTotal).toBe(DOMAIN.length);
      expect(s.corpusTypesTotal).toBe(CORPUS_TYPES.size);
    }
  });

  it("separates the two scopes by exactly the 22 domain-only words", () => {
    // The domain-only words are budget entries the reader never meets in the text, which
    // is why the panel shows the CORPUS scope: counting them inflates the vacancy rate
    // being reported to someone looking at that text.
    const domainOnly = [...DOMAIN].filter((w) => !CORPUS_TYPES.has(w));
    expect(domainOnly.length).toBe(22);
    expect(domainOnly).toContain("funny");
    expect(domainOnly).toContain("squirrel");
    expect(domainOnly).toContain("today");
    // All 22 happen to be eligible, so the two scopes differ by exactly 22.
    const keep = effectiveKeepSet();
    expect(domainOnly.every((w) => isEligible(stemAndSuffix(w)[0], keep))).toBe(true);
    for (const seed of SEEDS) {
      const s = vacancyStats(CORPUS, vacated(seed, 1), mapFor(seed), params({ seed, p: 1 }));
      expect(s.domainTypesEligible).toBe(s.corpusTypesEligible + 22);
      expect(s.domainTypesTotal).toBe(s.corpusTypesTotal + 22);
    }
  });

  it("measures corpusTypesVacated from the texts, not from map membership, under revealAfter", () => {
    // The defect the golden fixture caught. A type whose every occurrence falls inside the
    // reveal window is STILL LISTED IN THE MAP but has changed nowhere in the text, so map
    // membership over-reports it. The two readings coincide at revealAfter = 0, which is
    // why only a control condition exposed it.
    const seed = 0;
    const cfg = params({ seed, p: 1, revealAfter: 1 });
    const revealed = vacateText(CORPUS, mapFor(seed), cfg);
    const s = vacancyStats(CORPUS, revealed, mapFor(seed), cfg);

    // The text-measured count, computed here independently of the implementation.
    const beforeToks = tokenize(CORPUS);
    const afterToks = tokenize(revealed);
    const changed = new Set<string>();
    for (let i = 0; i < beforeToks.length; i++) {
      if (beforeToks[i] !== afterToks[i]) changed.add(beforeToks[i]);
    }
    expect(s.corpusTypesVacated).toBe(changed.size);

    // ...and it is strictly smaller than what map membership would have said, which is the
    // number the domain scope still reports (deliberately — see below).
    const byMap = [...CORPUS_TYPES].filter((t) => {
      const [stem] = stemAndSuffix(t);
      return isEligible(stem, effectiveKeepSet()) && vacancyU(stem, seed) < 1;
    }).length;
    expect(s.corpusTypesVacated).toBeLessThan(byMap);
    expect(byMap).toBe(s.corpusTypesEligible);
  });

  it("pins the exact golden-fixture case the defect was found at: p = 0.7, revealAfter = 2", () => {
    // The coordinate the two stacks split on, reproduced to the type. Map membership says
    // 1337 and the texts say 665 — the 2x over-report, at the golden fixture's own p.
    const seed = 0;
    const cfg = params({ seed, p: 0.7, revealAfter: 2 });
    const s = vacancyStats(CORPUS, vacateText(CORPUS, mapFor(seed), cfg), mapFor(seed), cfg);
    const byMap = [...CORPUS_TYPES].filter((t) => {
      const [stem] = stemAndSuffix(t);
      return isEligible(stem, effectiveKeepSet()) && vacancyU(stem, seed) < 0.7;
    }).length;
    expect(byMap).toBe(1337);
    expect(s.corpusTypesVacated).toBe(665);
    // The domain scope keeps the map reading, and so is unmoved by revealAfter.
    expect(s.domainTypesVacated).toBe(1354);
  });

  it("agrees at revealAfter = 0 and diverges at revealAfter > 0", () => {
    // The property that would have caught the defect, asserted directly.
    for (const seed of SEEDS) {
      const pure = params({ seed, p: 1, revealAfter: 0 });
      const held = params({ seed, p: 1, revealAfter: 1 });
      const sPure = vacancyStats(CORPUS, vacated(seed, 1), mapFor(seed), pure);
      const sHeld = vacancyStats(CORPUS, vacateText(CORPUS, mapFor(seed), held), mapFor(seed), held);

      // At revealAfter = 0 the text reading and the map reading are the same number.
      expect(sPure.corpusTypesVacated).toBe(sPure.corpusTypesEligible);
      // At revealAfter > 0 the text-measured count is STRICTLY smaller...
      expect(sHeld.corpusTypesVacated).toBeLessThan(sPure.corpusTypesVacated);
      // ...while the domain scope, which reads map membership, does not move at all.
      expect(sHeld.domainTypesVacated).toBe(sPure.domainTypesVacated);
      // Tokens are text-measured in both, so they drop too — by one per revealed type.
      expect(sHeld.tokensVacated).toBeLessThan(sPure.tokensVacated);
    }
  });

  it("splits stress three ways, summing to 1 on each side", () => {
    for (const p of P_GRID) {
      const s = vacancyStats(CORPUS, vacated(0, p), mapFor(0), params({ seed: 0, p }));
      expect(s.stressFromTableBefore + s.stressFromMintedBefore + s.stressFromRuleBefore).toBeCloseTo(1, 12);
      expect(s.stressFromTableAfter + s.stressFromMintedAfter + s.stressFromRuleAfter).toBeCloseTo(1, 12);
      // The original corpus contains no minted form — `avoid` and condition B guarantee it.
      expect(s.stressFromMintedBefore).toBe(0);
      if (p > 0) expect(s.stressFromMintedAfter).toBeGreaterThan(0);
    }
  });

  it("agrees with the corpus manifest the backend wrote", () => {
    const stats = vacancyStats(CORPUS, CORPUS, mapFor(0), DEFAULT_VACANCY_PARAMS);
    expect(stats.tokensTotal).toBe(corpusAsset.n_tokens);
    // The manifest counts the CORPUS, so that is the scope that has to match it.
    expect(stats.corpusTypesTotal).toBe(corpusAsset.n_distinct);
    expect(stats.domainTypesTotal).toBe(DOMAIN.length);
  });

  it("reports the measured numbers on the real corpus", () => {
    const rows: string[] = [];
    for (const seed of SEEDS) {
      for (const p of P_GRID) {
        const s = vacancyStats(CORPUS, vacated(seed, p), mapFor(seed), params({ seed, p }));
        rows.push(
          [
            `seed=${seed}`,
            `p=${p.toFixed(2)}`,
            `domainTypes=${s.domainTypesTotal}/${s.domainTypesEligible}/${s.domainTypesVacated}`,
            `corpusTypes=${s.corpusTypesTotal}/${s.corpusTypesEligible}/${s.corpusTypesVacated}`,
            `stemsTotal=${s.stemsTotal}`,
            `stemsVacated=${s.stemsVacated}`,
            `tokensTotal=${s.tokensTotal}`,
            `tokensVacated=${s.tokensVacated}`,
            `syl=${s.meanSyllablesBefore.toFixed(4)}->${s.meanSyllablesAfter.toFixed(4)}`,
            `anapest=${s.meanAnapestBefore.toFixed(4)}->${s.meanAnapestAfter.toFixed(4)}`,
            `table=${s.stressFromTableBefore.toFixed(4)}->${s.stressFromTableAfter.toFixed(4)}`,
            `minted=${s.stressFromMintedBefore.toFixed(4)}->${s.stressFromMintedAfter.toFixed(4)}`,
            `rule=${s.stressFromRuleBefore.toFixed(4)}->${s.stressFromRuleAfter.toFixed(4)}`,
            `bijective=${s.bijective}`,
            `imageSize=${s.imageSize}`,
            `remintRounds=${s.remintRounds}`,
          ].join(" "),
        );
        expect(s.bijective).toBe(true);
      }
    }
    // eslint-disable-next-line no-console
    console.log(["", "vacancy transform, measured on The Real Mother Goose:", ...rows, ""].join("\n"));
    expect(rows.length).toBe(SEEDS.length * P_GRID.length);
  });
});

// --- `forbidden`, §5.8 ----------------------------------------------------------------

describe("`forbidden` is stored, and keeps superseded re-mint nonces (§5.8)", () => {
  it("carries `wak`, the nonce seed 7's re-mint of `hang` replaced", () => {
    // THE CASE THAT DISTINGUISHES a stored set from `domain ∪ mapping.values()`, and the
    // only one the shipped corpus produces: at seed 7 the stem `hang` first minted `wak`,
    // whose surface `wak` + `ed` is the real English word `waked`; condition B rejected it
    // and the re-mint returned `smeeg`. `wak` is now no stem's nonce, so a reconstruction
    // drops it — but it must stay forbidden, because it was rejected for cause and the
    // `consistent = false` control draws against this very set.
    const vmap = mapFor(7);
    expect(vmap.mapping.get("hang")).toBe("smeeg");
    expect(vmap.remintRounds).toBe(1);
    expect(vmap.forbidden.has("wak")).toBe(true);
    expect([...vmap.mapping.values()]).not.toContain("wak"); // what a rebuild would lose
    expect(vmap.domain.has("waked")).toBe(true); // ... and why it was superseded
  });

  it("contains the whole domain and every nonce, at both seeds", () => {
    for (const seed of SEEDS) {
      const vmap = mapFor(seed);
      for (const t of vmap.domain) expect(vmap.forbidden.has(t)).toBe(true);
      for (const n of vmap.mapping.values()) expect(vmap.forbidden.has(n)).toBe(true);
      expect(vmap.forbidden.size).toBe(
        vmap.domain.size + new Set(vmap.mapping.values()).size + (seed === 7 ? 1 : 0),
      );
    }
  });

  it("keeps the superseded nonce out of the inconsistent control's output", () => {
    const p = params({ p: 1, seed: 7, consistent: false });
    const text = vacateText(CORPUS, buildVacancyMap(DOMAIN, p), p);
    expect(new Set(tokenize(text)).has("wak")).toBe(false);
  });
});

// --- the swap control, §8.3 / §5.2a ---------------------------------------------------

const COUNTS = typeCounts(tokenize(CORPUS));
const SWAP_MAPS = new Map<number, VacancyMap>(
  SEEDS.map((seed) => [seed, buildVacancyMap(DOMAIN, params({ seed, mint: "swap" }), COUNTS)]),
);
function swapMapFor(seed: number): VacancyMap {
  const m = SWAP_MAPS.get(seed);
  if (m === undefined) throw new Error(`no swap map for seed ${seed}`);
  return m;
}

describe("mint = 'swap' draws a real English word (§8.3)", () => {
  it("replaces every stem with a word the corpus or the budget already had", () => {
    const real = new Set<string>([...CORPUS_TYPES, ...BUDGET.map((w) => w.toLowerCase())]);
    for (const seed of SEEDS) {
      const vmap = swapMapFor(seed);
      expect(vmap.mapping.size).toBeGreaterThan(0);
      for (const [stem, word] of vmap.mapping) {
        expect(real.has(word), `${seed}: ${stem} -> ${word}`).toBe(true);
        expect(word).not.toBe(stem); // a stem keeping its form is a word that failed to vacate
      }
    }
  });

  it("needs the frequency counts, and says so rather than ranking alphabetically", () => {
    expect(() => buildVacancyMap(DOMAIN, params({ mint: "swap" }))).toThrow(/type counts/);
  });

  it("refuses the inconsistent control — there is no supply of fresh real words", () => {
    expect(() =>
      buildVacancyMap(DOMAIN, params({ mint: "swap", consistent: false }), COUNTS),
    ).toThrow(/consistent/);
  });

  it("leaves the nonce map a pure function of (domain, seed, matchProsody)", () => {
    for (const seed of SEEDS) {
      const withCounts = buildVacancyMap(DOMAIN, params({ seed }), COUNTS);
      expect([...withCounts.mapping]).toEqual([...mapFor(seed).mapping]);
      expect(withCounts.remintRounds).toBe(mapFor(seed).remintRounds);
    }
  });

  it("registers no minted stress — the replacements are real English words", () => {
    for (const seed of SEEDS) expect(swapMapFor(seed).mintedStress.size).toBe(0);
  });

  it("is stable in (seed, stem): rebuilding gives the same map (SC-702)", () => {
    for (const seed of SEEDS) {
      const again = buildVacancyMap(DOMAIN, params({ seed, p: 1, mint: "swap" }), COUNTS);
      expect(mappingEntries(again)).toEqual(mappingEntries(swapMapFor(seed)));
    }
  });

  it("is nested in `p` — the `u(stem) < p` decision is untouched (SC-701)", () => {
    let previous = new Set<string>();
    for (const p of P_GRID) {
      const text = vacateText(CORPUS, swapMapFor(0), params({ p, seed: 0, mint: "swap" }));
      const changed = changedTypes(CORPUS, text);
      for (const t of previous) expect(changed.has(t), `p=${p}: ${t}`).toBe(true);
      previous = changed;
    }
  });

  it("is a bijection of the domain at full vacancy (A + B₁ of §5.2a)", () => {
    for (const seed of SEEDS) {
      const vmap = swapMapFor(seed);
      expect(vmap.bijective).toBe(true);
      expect(vmap.imageSize).toBe(vmap.domain.size);
      expect(vmap.remintRounds).toBe(0);
      expect(vmap.injectiveAtEveryP).toBe(false);
    }
  });
});

describe("the invariance theorem under mint = 'swap' (SC-703 / §5.2a)", () => {
  it("holds at p ∈ {0, 1}, exactly as it does for mint = 'nonce'", () => {
    const base = new LexVocab(BUDGET, "dolch", "full");
    const reference = base.encode(tokenize(CORPUS));
    for (const seed of SEEDS) {
      for (const p of [0, 1]) {
        const ps = params({ p, seed, mint: "swap" });
        const vmap = swapMapFor(seed);
        const text = vacateText(CORPUS, vmap, ps);
        const words = mapVocabWords(BUDGET, vmap, ps);
        expect(new Set(words).size).toBe(words.length);
        const mapped = new LexVocab(words, "dolch", "full");
        expect(mapped.rows).toBe(base.rows);
        expect(mapped.encode(tokenize(text))).toEqual(reference);
      }
    }
  });

  it("refuses the mapped vocabulary at intermediate p rather than duplicating a row", () => {
    // §5.2a: no `p`-stable map whose images are domain types is injective at 0 < p < 1
    // unless it is the identity. That is a theorem, not a defect to be re-drawn away, so
    // the mapped vocabulary is refused exactly as it is for the two controls.
    for (const p of [0.25, 0.5, 0.75]) {
      expect(() =>
        mapVocabWords(BUDGET, swapMapFor(0), params({ p, seed: 0, mint: "swap" })),
      ).toThrow(/full vacancy/);
    }
  });

  it("measures WHY: a vacated type lands on a word that has not moved", () => {
    // Pinned so the refusal above can never be mistaken for over-caution. If a future change
    // makes swap injective at p = 0.5, this fails and the contract is wrong.
    const vmap = swapMapFor(0);
    const ps = params({ p: 0.5, seed: 0, mint: "swap" });
    const images = new Map<string, string>();
    let collisions = 0;
    for (const t of [...vmap.domain].sort()) {
      const image = transformWord(t, vmap, ps).toLowerCase();
      if (images.has(image)) collisions++;
      images.set(image, t);
    }
    expect(collisions).toBeGreaterThan(0);

    const full = params({ p: 1, seed: 0, mint: "swap" });
    const atOne = new Set([...vmap.domain].map((t) => transformWord(t, vmap, full).toLowerCase()));
    expect(atOne.size).toBe(vmap.domain.size);
  });
});
