/**
 * The vacancy transform — field without location, at a controlled rate.
 *
 * TypeScript half of `specs/007-vacancy-transform-field/architecture.md`, which is the
 * normative document: this file implements *that contract*, not the Python module, and
 * the Python module implements it too. Feature 006 taught us why — a contract that omits
 * one sentence lets two stacks drift for a day without either being "wrong".
 *
 * What it does: rewrite a corpus in place so that closed-class scaffolding, inflectional
 * morphology, punctuation and line structure survive byte for byte, while a controlled
 * fraction `p` of open-class STEMS is replaced by phonotactically legal nonce forms that
 * carry the stem's syllable count and stress. The result is Carroll's condition — a token
 * whose distributional neighbourhood is fully specified by context and whose form carries
 * no prior — manufactured on demand rather than borrowed from the 28 words Carroll minted.
 *
 * The three properties that make a `p`-sweep interpretable, and where they come from:
 *
 *   * NESTING (§4). A stem is vacated iff `u(stem) < p`, and `u` is a hash of
 *     `(seed, stem)` alone — not of `p`, not of traversal order, not of which other words
 *     exist. So `{vacated at p} ⊆ {vacated at p'}` whenever `p < p'`.
 *   * STABILITY (§5). The nonce map is built ONCE over the whole type set in canonical
 *     (sorted) order, and the map at any `p` is its restriction to `{u < p}`. The source's
 *     minter built the map lazily while rewriting, so its `used` set — and therefore the
 *     nonce a word got — depended on `p`. That breaks the stability the source claims for
 *     itself; §5.2 of the contract is the correction, and it makes stability structural
 *     rather than hoped for.
 *   * INJECTIVITY AT EVERY `p` (§5.2 A/B). The check is over assembled, lowercased SURFACE
 *     forms, not bare nonces, and it forbids a surface form from equalling ANY domain type
 *     — eligible or not. Both weaker checks were tried and both were wrong; the comments
 *     on `buildVacancyMap` name the collision each one missed.
 *
 * All three feed the invariance theorem (§7.3): with `consistent = true` and
 * `revealAfter = 0`, the transform is a pure relabelling of the vocabulary, so a
 * word-level model trained from scratch sees the identical token id stream and trains
 * bit-identically. That is only true if the transform's idea of a word is EXACTLY the
 * tokenizer's, which is why `WORD_RE`/`tokenize` are imported from `./vocab` rather than
 * re-declared here (departure 1 from the source, whose `[A-Za-z][A-Za-z']*` split
 * `good-bye` in two).
 *
 * The tokenizer lowercases, so the transform must COMMUTE WITH LOWERCASING (§5.7):
 *
 *     lower(transformWord(w)) === transformWord(lower(w))       // normative, and a test
 *
 * Everything is therefore computed on the lowercased word — stem, suffix, seam test, seam
 * hash, assembly — and `matchCase` is applied once at the end to the whole assembled
 * surface form. Slicing the suffix case-preserved (as the source does) makes the seam test
 * branch on case: `gums -> flels` but `GUMS -> FLESS`, one source type with two surface
 * forms, and the theorem is false.
 *
 * Determinism across the TS/Python seam is bought with three deliberate departures:
 *
 *   * `u = (top64 >> 11) / 2**53`, not `top64 / 2**64`. NOT because the source's
 *     expression diverges — §4 records that it was measured over 200 006 values and does
 *     not — but because a 53-bit numerator over 2^53 is exactly representable, so the
 *     agreement is structural rather than a proof one refactor could invalidate.
 *   * `random.Random` is replaced by a sha256 counter stream (`bytesFor`), because
 *     MT19937 seeded from a string is not reproducible in TypeScript.
 *   * the seam fix and the give-up path are hashes of their inputs rather than draws from
 *     a shared RNG or counts of a mutable `used` set, so neither depends on order.
 *
 * The hash is the SYNCHRONOUS pure-JS sha256 of `../geoEngine/hash`; WebCrypto is async
 * and this transform has to be callable from a store update.
 */

import { sha256Hex, utf8Bytes } from "../geoEngine/hash";
import { dolchBudget } from "./dolch";
import { WORD_RE, splitLines } from "./vocab";

// --- §2.1 the closed class -----------------------------------------------------------

/**
 * The source's curated closed class, ported verbatim (whitespace-split, lowercased).
 *
 * The source carries a warning we keep: an earlier version unioned this with short Dolch
 * service words, which silently protected content verbs (`run`, `eat`, `see`, `get`,
 * `let`, `put`) and understated the vacancy rate. The closed class is THIS LIST ONLY.
 */
export const FUNCTION_WORDS: ReadonlySet<string> = new Set(
  `a an the this that these those my your his her its our
their some any all both each every no none i me you he she it we they him them
us who whom whose which what where when why how is am are was were be been being
do does did done have has had having will would shall should can could may might
must not and or but so if then than as of to in on at by for with from into onto
up down out off over under again once here there very too also only just even
still yet ever never always about after before while because though although
unless until since during between among against through above below near far
one two three four five six seven eight nine ten`
    .split(/\s+/)
    .filter((w) => w.length > 0),
);

/** `FUNCTION_WORDS ∪ lower(extra)` — the effective keep set (§2.1). */
export function effectiveKeepSet(extra: Iterable<string> = []): ReadonlySet<string> {
  const out = new Set(FUNCTION_WORDS);
  for (const w of extra) out.add(w.toLowerCase());
  return out;
}

// --- §3 suffix splitting -------------------------------------------------------------

/** Tried in THIS order; the first match wins. Order is part of the contract. */
export const SUFFIXES: readonly string[] = [
  "ing",
  "edly",
  "est",
  "ies",
  "'s",
  "n't",
  "ed",
  "es",
  "er",
  "ly",
  "s",
];

/**
 * Never split, whatever the spelling suggests (§3, departure 9).
 *
 * From the AUDITED copy of the source, not the copy in the zip: without it
 * `brother -> broth+er` and `morning -> morn+ing`, which the source itself flags as a
 * known artifact. This is a spelling heuristic and not a morphological analyser, so it is
 * still wrong outside the list (`ladder -> ladd+er`). That is acceptable — the nonce
 * still carries a consistent identity and an inflected-looking surface — but it must be
 * documented in the UI rather than quietly tolerated.
 */
export const SPLIT_EXCEPTIONS: ReadonlySet<string> = new Set([
  "brother",
  "father",
  "mother",
  "sister",
  "never",
  "over",
  "under",
  "morning",
  "giving",
  "thing",
]);

/**
 * Split a word into `[stem, suffix]`, preserving case.
 *
 * A suffix `s` matches iff `lower(word)` ends with it AND `len(word) - len(s) >= 3`.
 * The slice is taken from the ORIGINAL word, so `Dog's` -> `["Dog", "'s"]`.
 */
export function stemAndSuffix(word: string): [string, string] {
  const lower = word.toLowerCase();
  if (SPLIT_EXCEPTIONS.has(lower)) return [word, ""];
  for (const suffix of SUFFIXES) {
    if (lower.endsWith(suffix) && word.length - suffix.length >= 3) {
      const cut = word.length - suffix.length;
      return [word.slice(0, cut), word.slice(cut)];
    }
  }
  return [word, ""];
}

// --- §2.2 eligibility ----------------------------------------------------------------

/** ASCII letters only. Python must use `re.fullmatch(r"[A-Za-z]+", ...)`, NOT
 *  `str.isalpha()`, which is Unicode-aware and would accept what this rejects. */
const ASCII_STEM = /^[A-Za-z]+$/;

/**
 * May this stem be vacated? (§2.2) All three must hold:
 *
 *   1. `lower(stem)` is not in the keep set,
 *   2. `stem` is ASCII letters only,
 *   3. `stem` is longer than two characters.
 *
 * Test 2 is what makes hyphens and apostrophes behave, and the three cases both stacks
 * must agree on exactly:
 *
 *   * `good-bye` — no suffix matches, the stem is `good-bye`, the hyphen fails test 2,
 *     so the word is NEVER vacated;
 *   * `don't` — the `n't` suffix splits it to stem `do`, which fails test 3 (and test 1);
 *   * `dog's` — the `'s` suffix splits it to stem `dog`, which passes, so the output is
 *     `<nonce>'s`.
 *
 * `keep` is the EFFECTIVE set (`effectiveKeepSet`), not the caller's extras.
 */
export function isEligible(stem: string, keep: ReadonlySet<string>): boolean {
  if (keep.has(stem.toLowerCase())) return false;
  if (!ASCII_STEM.test(stem)) return false;
  return stem.length > 2;
}

// --- §5.4 the phonotactic tables -----------------------------------------------------
//
// Ported verbatim from `tiny-seuss/synth/jabberwockify.py`, ORDER SIGNIFICANT: the index
// into each list is what the byte stream selects, so reordering silently changes every
// nonce ever minted.

/** 47 entries. */
export const ONSETS: readonly string[] = [
  "b", "br", "bl", "d", "dr", "f", "fl", "fr", "g", "gl", "gr", "h",
  "j", "k", "kl", "kr", "l", "m", "n", "p", "pl", "pr", "r", "s", "sk",
  "sl", "sm", "sn", "sp", "st", "str", "sw", "t", "tr", "th", "thr",
  "v", "w", "wr", "y", "z", "sh", "shr", "ch", "gn", "sc", "sq",
];

/** 19 entries. */
export const NUCLEI: readonly string[] = [
  "a", "e", "i", "o", "u", "ai", "ee", "ea", "oo", "ou", "oa", "ie",
  "y", "au", "ur", "ir", "or", "ar", "er",
];

/**
 * 46 entries, beginning with the empty string.
 *
 * An early draft of §5.4 said 49. The contract now says 46 and, more usefully, states that
 * **the source lists are normative and the counts are commentary** — a nonce is a function
 * of these strings and their indices, not of a tally in a document.
 */
export const CODAS: readonly string[] = [
  "", "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v",
  "z", "sh", "ch", "th", "ck", "ff", "ll", "mp", "nd", "ng", "nk", "nt",
  "sk", "sp", "st", "ft", "lt", "lk", "rd", "rk", "rm", "rn", "rt", "ble",
  "dle", "gle", "kle", "tle", "mble", "ndle", "ffle", "zzle",
];

/** 13 entries — the tail of a non-initial unstressed syllable. */
export const UNSTRESSED_TAILS: readonly string[] = [
  "y", "le", "er", "ow", "en", "el", "ish", "ous", "id",
  "ic", "um", "ent", "ing",
];

/** The prefix an INITIAL unstressed syllable draws from. */
export const UNSTRESSED_PREFIXES: readonly string[] = ["a", "be", "re", "de", "un", "en"];

/**
 * The reduced coda set for an unstressed syllable. The duplicated empty string doubles
 * its weight — keep it (§5.4).
 *
 * §5.5's enumerated rule has exactly three branches and none of them reaches this table:
 * a stressed syllable draws from `CODAS`, an unstressed one emits a whole prefix or tail.
 * It is dead in the source too (`_syl(stressed=False)` is never called from `mint`).
 * **Do not "fix" this** — the byte stream's list indices must not shift, and drawing from
 * it would change every multi-syllable nonce in both stacks.
 */
export const REDUCED_CODAS: readonly string[] = ["", "", "l", "n", "r", "s"];

/** Replacement characters for a seam, indexed by a hash of `(stem, suffix)` (§5.7). */
const SEAM_CHARS = "lnrtk";

// --- §6 prosody ----------------------------------------------------------------------

/**
 * The 61 polysyllables of the Dolch list, hand-set. `1` = stressed, `0` = unstressed.
 * Ported verbatim from `tiny-seuss/synth/lexicon.py`.
 *
 * PROVENANCE — read this before quoting a prosody number. The source describes the table
 * as "seeded by rule and then overridden by a hand table", and its own status table lists
 * it under *not yet exercised*: "seeded by rule; wants roughly an hour of human
 * checking." So we do NOT claim exact prosody and no UI string may. The shipped corpus is
 * *The Real Mother Goose*, most of whose types are not Dolch words, so most of them fall
 * through to the spelling rule — which is exactly what `stressTableCoverage*` measures,
 * and why every prosody statistic must be shown next to it.
 */
export const STRESS_TABLE: ReadonlyMap<string, string> = new Map<string, string>([
  ["away", "01"], ["funny", "10"], ["little", "100"], ["yellow", "10"],
  ["into", "10"], ["over", "10"], ["pretty", "10"], ["under", "10"],
  ["after", "10"], ["again", "01"], ["any", "10"], ["every", "100"],
  ["giving", "10"], ["once", "1"], ["open", "10"],
  ["always", "100"], ["around", "01"], ["because", "01"], ["before", "01"],
  ["seven", "10"], ["eight", "1"], ["myself", "01"], ["never", "10"],
  ["only", "10"], ["today", "01"], ["together", "0100"], ["better", "10"],
  ["carry", "10"], ["many", "10"], ["upon", "01"], ["very", "100"],
  ["apple", "10"], ["baby", "10"], ["birthday", "100"], ["brother", "10"],
  ["chicken", "10"], ["children", "100"], ["Christmas", "10"],
  ["farmer", "10"], ["flower", "10"], ["garden", "10"], ["good-bye", "01"],
  ["horse", "1"], ["kitty", "10"], ["letter", "10"], ["money", "10"],
  ["morning", "10"], ["mother", "10"], ["paper", "100"], ["party", "10"],
  ["picture", "10"], ["rabbit", "10"], ["robin", "10"], ["squirrel", "1"],
  ["table", "10"], ["water", "10"], ["window", "10"], ["Santa Claus", "101"],
  ["father", "10"], ["sister", "10"], ["summer", "10"],
]);

/** Where a stress pattern came from — the honesty of every prosody number (§6.1). */
export type StressSource = "minted" | "table" | "rule";

/**
 * Syllable count by spelling rule (§6.2). Fallback only.
 *
 * The source has a further `if w.endswith("le") ...: pass` branch. It is a literal `pass`
 * — dead code — and is NOT ported; the behaviour here is byte-identical to the source's.
 */
export function ruleSyllables(word: string): number {
  let w = word.toLowerCase().replace(/^['-]+/, "").replace(/['-]+$/, "");
  w = w.replace(/[^a-z]/g, "");
  if (w.length === 0) return 1;
  let n = (w.match(/[aeiouy]+/g) ?? []).length;
  if (w.endsWith("e") && n > 1 && !(w.endsWith("le") || w.endsWith("ee") || w.endsWith("ye"))) {
    n -= 1;
  }
  return Math.max(1, n);
}

/**
 * The stress pattern of a word, and where it came from. Lookup order is exactly §6.3:
 *
 *   1. `minted[lower(word)]` — a form WE minted, whose pattern is intended rather than
 *      guessed, so prosody scoring on a vacated corpus is exact for the minted forms;
 *   2. `STRESS_TABLE[word]` — CASE-SENSITIVE, which is there for `Christmas`;
 *   3. `STRESS_TABLE[lower(word)]`;
 *   4. the rule.
 *
 * `minted` is passed explicitly rather than kept in a module-level registry (the source
 * used a global `MINTED_STRESS`): a global would leak one build's nonces into the next
 * one's syllable checks and make minting depend on call order, which is the class of bug
 * §5 exists to remove. Omit it and you get pure spelling+table prosody, which is what
 * minting itself uses and what the ORIGINAL corpus should be scored with.
 */
export function stressWithSource(
  word: string,
  minted?: ReadonlyMap<string, string>,
): { pattern: string; source: StressSource } {
  const lower = word.toLowerCase();
  const mintedPattern = minted?.get(lower);
  if (mintedPattern !== undefined) return { pattern: mintedPattern, source: "minted" };
  const exact = STRESS_TABLE.get(word);
  if (exact !== undefined) return { pattern: exact, source: "table" };
  const lowered = STRESS_TABLE.get(lower);
  if (lowered !== undefined) return { pattern: lowered, source: "table" };
  const n = ruleSyllables(lower);
  return { pattern: n === 1 ? "1" : "1" + "0".repeat(n - 1), source: "rule" };
}

/** §6.3. */
export function stress(word: string, minted?: ReadonlyMap<string, string>): string {
  return stressWithSource(word, minted).pattern;
}

/** §6.3: `syllables(word) := len(stress(word))`. */
export function syllables(word: string, minted?: ReadonlyMap<string, string>): number {
  return stress(word, minted).length;
}

/** The repeating feet §6.4 names. */
export const METER_FEET: Readonly<Record<string, string>> = {
  anapest: "001",
  iamb: "01",
  trochee: "10",
  dactyl: "100",
};

/** Raw `WORD_RE` matches, case preserved — `stress`'s case-sensitive step 2 needs them.
 *  A fresh RegExp per call: the shared `g`-flagged literal carries `lastIndex`. */
function wordMatches(text: string): string[] {
  const re = new RegExp(WORD_RE.source, "g");
  const out: string[] = [];
  for (let m = re.exec(text); m !== null; m = re.exec(text)) out.push(m[0]);
  return out;
}

/**
 * §6.4: the fraction of syllable positions in the line's concatenated stress string that
 * match the repeating foot. `0.0` for a line with no syllables.
 */
export function meterScore(
  line: string,
  foot: string = "anapest",
  minted?: ReadonlyMap<string, string>,
): number {
  const pat = METER_FEET[foot];
  if (pat === undefined) {
    throw new Error(`unknown foot ${JSON.stringify(foot)}; expected ${Object.keys(METER_FEET).join(", ")}`);
  }
  let scan = "";
  for (const w of wordMatches(line)) scan += stress(w, minted);
  if (scan.length === 0) return 0;
  let hits = 0;
  for (let i = 0; i < scan.length; i++) if (scan[i] === pat[i % pat.length]) hits++;
  return hits / scan.length;
}

// --- §4 the vacancy decision ---------------------------------------------------------

/** sha256 of a UTF-8 string, as bytes. */
function digestBytes(s: string): Uint8Array {
  const hex = sha256Hex(utf8Bytes(s));
  const out = new Uint8Array(32);
  for (let i = 0; i < 32; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

/**
 * Big-endian uint32 from four bytes. The `>>> 0` is REQUIRED: without it JS sign-extends
 * and `%` returns a negative index (§5.3).
 */
function u32At(bytes: Uint8Array, offset: number): number {
  return (
    ((bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]) >>> 0
  );
}

/**
 * `u(stem, seed) ∈ [0, 1)` — the vacancy coordinate (§4). A stem is vacated iff `u < p`.
 *
 * `u` depends on `(seed, lower(stem))` alone, which is what makes the vacated sets NESTED
 * as `p` grows, so a `p`-sweep varies the vacancy set and nothing else.
 *
 * The `>> 11n` is mandatory and is a departure from the source's `top64 / 2**64`: a
 * 64-bit integer divided by 2^64 is not exactly representable as a float64, so Python and
 * JavaScript can land on different doubles for the same digest and disagree about a word
 * at the boundary. Shifting to 53 bits makes the numerator exact in both — and
 * `Number()` of a BigInt below 2^53 is itself exact, so no rounding enters here either.
 */
export function vacancyU(stem: string, seed: number): number {
  if (!Number.isInteger(seed)) {
    throw new Error(`vacancyU: seed must be an integer, got ${seed}`);
  }
  const hex = sha256Hex(utf8Bytes(`${seed}:${stem.toLowerCase()}`));
  return Number(BigInt("0x" + hex.slice(0, 16)) >> 11n) / 2 ** 53;
}

// --- §5.3 the deterministic byte stream ----------------------------------------------

/**
 * `bytesFor(seed, stem, salt, counter) = sha256(f"{seed}:mint:{stem}:{salt}:{counter}")`,
 * consumed four bytes at a time as a big-endian uint32 and refilled from the next
 * `counter` when exhausted. This replaces `random.Random`, which cannot be reproduced
 * across the two languages (departure 3).
 */
class MintStream {
  private readonly prefix: string;
  private buf: Uint8Array;
  private pos = 0;
  private counter = 0;

  constructor(seed: number, stem: string, salt: number) {
    this.prefix = `${seed}:mint:${stem}:${salt}`;
    this.buf = digestBytes(`${this.prefix}:0`);
  }

  nextU32(): number {
    if (this.pos + 4 > this.buf.length) {
      this.counter += 1;
      this.buf = digestBytes(`${this.prefix}:${this.counter}`);
      this.pos = 0;
    }
    const v = u32At(this.buf, this.pos);
    this.pos += 4;
    return v;
  }

  /** `list[nextU32() % len(list)]`. Every list here is shorter than 256, so the modulo
   *  bias is aesthetic rather than statistical — but both stacks bias IDENTICALLY. */
  choice<T>(items: readonly T[]): T {
    return items[this.nextU32() % items.length];
  }
}

// --- §5.5 the mint loop --------------------------------------------------------------

/** Attempt `a >= 400` drops the syllable-count check. */
const ATTEMPT_DROP_SYLLABLES = 400;
/** Attempt `a >= 800` drops the length check. */
const ATTEMPT_DROP_LENGTH = 800;
/** Attempt `a >= 1200` raises — it has never happened and if it does we want to know. */
const ATTEMPT_GIVE_UP = 1200;

/** Collapse runs: `bbb -> bb`. */
const RUN_RE = /([bcdfghjklmnpqrstvwxz])\1{2,}/g;

/**
 * Mint one nonce (§5.5).
 *
 * Depends only on `(seed, key, pattern, forbidden, baseSalt)` — never on `p`, never on
 * the document, never on the order words are encountered while rewriting. That is what
 * makes a stem's nonce the same at every `p` (§5.6).
 *
 * TWO COUNTERS, and the distinction is load-bearing (§5.5). The byte stream is keyed on
 * `salt = baseSalt + a`, but the quality thresholds are on the ATTEMPT counter `a`, which
 * restarts at 0 in every call. Read the other way — thresholds on the absolute salt — a
 * re-mint starting at `baseSalt = 1001` would begin with every check already relaxed, so
 * the replacement would not be prosody-matched, and round 2 would blow straight past the
 * give-up bound and contradict §5.2's "raise after 8 rounds".
 *
 * The relaxations replace the source's give-up path, which returned
 * `syllable + str(len(self.used))` — a count of how many words happened to be minted
 * first, and therefore order-dependent (departure 6).
 */
function mintNonce(
  key: string,
  pattern: string,
  seed: number,
  forbidden: ReadonlySet<string>,
  baseSalt: number,
): { nonce: string; salt: number } {
  const nSyl = pattern.length;
  for (let a = 0; a < ATTEMPT_GIVE_UP; a++) {
    const salt = baseSalt + a;
    const stream = new MintStream(seed, key, salt);
    let w = "";
    for (let i = 0; i < nSyl; i++) {
      if (pattern[i] === "1") {
        w += stream.choice(ONSETS) + stream.choice(NUCLEI) + stream.choice(CODAS);
      } else if (i === 0) {
        w += stream.choice(UNSTRESSED_PREFIXES);
      } else {
        w += stream.choice(UNSTRESSED_TAILS);
      }
    }
    w = w.replace(RUN_RE, "$1$1");
    if (w.length < 3 && a < ATTEMPT_DROP_LENGTH) continue;
    if (forbidden.has(w)) continue;
    if (a < ATTEMPT_DROP_SYLLABLES && syllables(w) !== nSyl) continue;
    return { nonce: w, salt };
  }
  throw new Error(
    `vacancy: could not mint a nonce for ${JSON.stringify(key)} (pattern ${pattern}) ` +
      `in ${ATTEMPT_GIVE_UP} attempts from base salt ${baseSalt} (§5.5) — this has never ` +
      `happened; report it rather than raising the bound`,
  );
}

/**
 * §5.7. Re-attaching a suffix can produce a seam (`wee` + `er` -> `weeer`). When the
 * nonce's last character equals the suffix's first, replace it with a character chosen by
 * a hash of `(stem, suffix)` — deterministic and order-independent, where the source drew
 * from a shared RNG (departure 7).
 *
 * `suffix` and `key` are ALWAYS the lowercased forms. This is the case-commuting invariant
 * of §5.7: the seam test branches on `suffix[0]`, so a case-preserved suffix would make
 * `gums -> flels` and `GUMS -> FLESS` — one source type, two surface forms, theorem false.
 */
function seamFix(nonce: string, suffix: string, key: string, seed: number): string {
  if (suffix.length === 0 || nonce.length === 0) return nonce;
  if (nonce[nonce.length - 1] !== suffix[0]) return nonce;
  const idx = u32At(digestBytes(`${seed}:seam:${key}:${suffix}`), 0) % SEAM_CHARS.length;
  return nonce.slice(0, -1) + SEAM_CHARS[idx];
}

/**
 * Source's `match_case`: ALL CAPS stays all caps, Titlecase stays titlecase.
 *
 * §5.7: applied ONCE, to the whole assembled surface form, with the ORIGINAL WHOLE WORD as
 * the case source. It is the only step in the transform allowed to look at case, which is
 * what makes the transform commute with lowercasing.
 */
function matchCase(src: string, surface: string): string {
  if (src.length > 1 && src === src.toUpperCase() && /[A-Za-z]/.test(src)) return surface.toUpperCase();
  if (/^[A-Z]/.test(src)) return surface.slice(0, 1).toUpperCase() + surface.slice(1);
  return surface;
}

/** §1: every output is itself a single, COMPLETE `WORD_RE` match — checked, not assumed.
 *  If this ever fired, `tokenize(vacate(text))` would not align with `tokenize(text)` and
 *  the invariance theorem would be quietly false. */
const WHOLE_WORD_RE = new RegExp(`^(?:${WORD_RE.source})$`);

function assertWholeWord(output: string, input: string): string {
  if (!WHOLE_WORD_RE.test(output)) {
    throw new Error(
      `vacancy: transform of ${JSON.stringify(input)} produced ${JSON.stringify(output)}, ` +
        `which is not a single complete WORD_RE match`,
    );
  }
  return output;
}

/**
 * The assembled, LOWERCASED surface form of §5.7 — `stem` and `suffix` lowercase, the seam
 * fixed, the suffix re-attached. This is the object conditions A and B of §5.2 range over,
 * and the only thing `matchCase` is ever applied to.
 */
function surfaceForm(stem: string, suffix: string, nonce: string, seed: number): string {
  return seamFix(nonce, suffix, stem, seed) + suffix;
}

// --- §7.1 parameters -----------------------------------------------------------------

/** The knobs of §7.1. Defaults in `DEFAULT_VACANCY_PARAMS`. */
export interface VacancyParams {
  /** Fraction of eligible types vacated. Compared to `u` as given — the UI emits two
   *  decimal places and both stacks parse it as float64. */
  p: number;
  /** Selects both `u` and the nonce assignment. Must be an integer. */
  seed: number;
  /** One nonce per source type, corpus-wide. `false` is the source's "inconsistent
   *  assignment" control: same vacancy rate, no learnable identity, and DELIBERATELY no
   *  stability property. */
  consistent: boolean;
  /** Nonce carries the stem's syllable count and stress. */
  matchProsody: boolean;
  /** First N occurrences of a vacated stem keep the English form — a partial location. */
  revealAfter: number;
  /** Extra words added to the closed class. */
  keep: readonly string[];
}

export const DEFAULT_VACANCY_PARAMS: VacancyParams = {
  p: 0,
  seed: 0,
  consistent: true,
  matchProsody: true,
  revealAfter: 0,
  keep: [],
};

/** Fill in §7.1's defaults around a partial setting. */
export function vacancyParams(partial: Partial<VacancyParams> = {}): VacancyParams {
  return { ...DEFAULT_VACANCY_PARAMS, ...partial };
}

// --- §5.2 the map --------------------------------------------------------------------

/** The `p`-independent nonce assignment, plus the facts §7.3 requires be verified. */
export interface VacancyMap {
  /** lowercase stem -> nonce, over EVERY eligible stem of the domain. The map at a given
   *  `p` is this map restricted to `{stem : u(stem) < p}`. */
  mapping: ReadonlyMap<string, string>;
  /** nonce -> intended stress pattern, so prosody scoring on a vacated corpus reflects
   *  what we built. `vacateText` adds to it in the `consistent = false` control. */
  mintedStress: Map<string, string>;
  /** How many re-mint rounds conditions A/B of §5.2 needed. Reported, not assumed. */
  remintRounds: number;
  /** Conditions A and B both hold — §7.3's injectivity, measured at build time and
   *  therefore true at EVERY `p`, not only at `p = 1`. */
  bijective: boolean;
  /** `|image of the domain under the full map|`, reported in the statistics (§10). */
  imageSize: number;
  /** Every minted nonce plus the whole domain. Carried so the `consistent = false` control can
   *  keep minting fresh forms that collide with neither. */
  forbidden: ReadonlySet<string>;
  /** The lowercased domain (corpus types ∪ budget words). Condition B ranges over it and
   *  §10's type counts are defined over it. */
  domain: ReadonlySet<string>;
}

/** How many re-mint rounds §5.2 allows before raising. */
const MAX_REMINT_ROUNDS = 8;

/**
 * The §5.2 domain: `corpus types ∪ the FULL Dolch list`. Every call site uses this rather
 * than building the union itself — the asymmetry of Python having such a helper and
 * TypeScript not is exactly how two call sites end up constructing the domain two
 * different ways.
 *
 * The full list ALWAYS, never the active budget: the domain must not depend on which
 * budget the reader has selected, or switching budgets would re-mint the corpus in front
 * of them and the stability the panel is demonstrating would look false. A frequency
 * budget needs no special case, since its words are corpus types by construction.
 *
 * Takes an iterable of TYPES, not a text. The guard is not pedantry in either language: a
 * `string` is itself an iterable of characters, so `vacancyDomain(corpusText)` would
 * silently yield a domain of single letters — every one of which fails the length test of
 * §2.2, giving an empty map and a transform that does nothing, with no error anywhere.
 */
export function vacancyDomain(types: Iterable<string>): string[] {
  if (typeof types === "string") {
    throw new Error(
      "vacancyDomain: expected an iterable of TYPES, got a string. A string iterates " +
        "character by character and would yield a domain of single letters. Pass tokenize(text).",
    );
  }
  const out = new Set<string>();
  for (const t of types) out.add(t.toLowerCase());
  for (const w of dolchBudget("full")) out.add(w.toLowerCase());
  // Sorted so the helper is a pure function with a canonical order. The order does not
  // reach the map — `buildVacancyMap` sorts the stems itself — which is asserted.
  return [...out].sort();
}

/** A domain type decomposed for the surface-form check: both parts lowercase. */
interface StemSuffixPair {
  stem: string;
  suffix: string;
}

/** The full-map (i.e. `p = 1`) image of a lowercased type. */
function imageOfType(type: string, map: ReadonlyMap<string, string>, seed: number): string {
  const [stem, suffix] = stemAndSuffix(type);
  const nonce = map.get(stem);
  if (nonce === undefined) return type;
  return surfaceForm(stem, suffix, nonce, seed);
}

/**
 * Conditions A and B of §5.2, evaluated over the assembled surface forms. Returns the
 * stems that must be re-minted — empty when the map is injective at every `p`.
 *
 *   A. the surface forms are pairwise distinct
 *   B. no surface form equals any lowercased domain type, eligible or not
 *
 * The loser of an A-collision is the stem later in canonical (ASCII-ascending) order among
 * those involved; the winner keeps its nonce, so a re-mint never cascades (§5.8). A
 * B-violation has no winner — the English word is fixed — so the offending stem re-mints.
 */
function injectivityOffenders(
  pairs: readonly StemSuffixPair[],
  map: ReadonlyMap<string, string>,
  domain: ReadonlySet<string>,
  seed: number,
): Set<string> {
  const offenders = new Set<string>();
  const bySurface = new Map<string, Set<string>>();
  for (const { stem, suffix } of pairs) {
    const nonce = map.get(stem);
    if (nonce === undefined) continue;
    const surface = surfaceForm(stem, suffix, nonce, seed);
    if (domain.has(surface)) offenders.add(stem); // B
    const bucket = bySurface.get(surface);
    if (bucket === undefined) bySurface.set(surface, new Set([stem]));
    else bucket.add(stem);
  }
  for (const [, stems] of bySurface) {
    if (stems.size < 2) continue; // A: one stem, one surface — nothing to break the tie
    const sorted = [...stems].sort();
    for (let i = 1; i < sorted.length; i++) offenders.add(sorted[i]);
  }
  return offenders;
}

/**
 * Build the nonce assignment ONCE over the whole type set, in canonical order (§5.2).
 *
 *     domain := { lower(t) for t in types }
 *     stems  := sorted({ stemOf(t) for t in domain if eligible(stemOf(t)) })
 *     used   := {}
 *     for stem in stems:                  # canonical order — never p, never document order
 *         nonce := mint(stem, seed, matchProsody, forbidden = used ∪ domain)
 *         used.add(nonce); map[stem] = nonce
 *
 * `types` must be the UNION of the corpus's type set and the full Dolch list — build it
 * with `vacancyDomain`, never by hand. §7.2 pushes budget words through the same
 * transform, so a budget word absent from the corpus still needs an image.
 *
 * THERE IS NO CALLER-SUPPLIED `avoid` PARAMETER. The domain is always avoided, implicitly.
 * Both stacks first gave `avoid` a default of empty and left the caller to pass the type
 * set; both agreed with each other, so no parity test could catch it — but the map was
 * then a function of what the caller remembered to pass. Measured: the same corpus and
 * seed give different nonces, and a different `remintRounds`, depending only on whether
 * the caller passed the set. Both maps are valid, which is precisely the problem — one
 * caller passing it and another not (the panel and the golden fixture, say) is a silent
 * divergence with nothing failing. Condition B below already forbids a surface form equal
 * to any domain type, so avoiding the domain at mint time is not extra policy, only the
 * cheaper route to the same fixed point. Afterwards the map is a pure function of
 * `(domain, seed, matchProsody)` — asserted in the tests, through two call paths.
 *
 * The source accepts an `avoid` parameter and then never passes one, which lets a minted
 * form silently merge with an English type (departure 5). We do not repeat that by making
 * it optional.
 *
 * INJECTIVITY IS VERIFIED, NOT ASSUMED, AND AT EVERY `p` (§5.2 / §7.3). Two weaker checks
 * were tried first and both were wrong, each for a reason worth keeping written down:
 *
 *   * checking BARE NONCES misses the collision that arrives through the suffix;
 *   * checking `|image| == |types|` AT `p = 1` ONLY misses it too, because at full vacancy
 *     every eligible type has moved and no minted form can meet a surviving English word.
 *
 * The measured example, on the shipped corpus: at `seed = 7` the stem `hang` minted `wak`.
 * No corpus type equals `wak`, so a bare-nonce check passes; at `p = 1` `waked` is itself
 * vacated, so a full-vacancy check passes; but at `p ∈ {0.25, 0.5}` `hanged` is vacated and
 * `waked` is not, so `hanged -> wak + ed = waked` collides with the English `waked` and
 * injectivity — and with it §7.3 — fails.
 *
 * So the check is conditions A and B of `injectivityOffenders`, over assembled surface
 * forms, both `p`-independent, which is what makes injectivity hold simultaneously for
 * every `p`. Offenders are re-minted in canonical order at salt
 * `1000 * round + previousSalt + 1` (round from 1, §5.8); only the loser moves, so a
 * re-mint never cascades. Eight rounds and then it raises.
 */
export function buildVacancyMap(types: Iterable<string>, params: VacancyParams): VacancyMap {
  const keep = effectiveKeepSet(params.keep);
  const domain = new Set<string>();
  for (const t of types) domain.add(t.toLowerCase());

  // `pairs` is one (stem, suffix) per domain type with an eligible stem — the objects
  // conditions A and B range over. `stems` is their canonical, ASCII-ascending order.
  const pairs: StemSuffixPair[] = [];
  const stems = new Set<string>();
  for (const t of domain) {
    const [stem, suffix] = stemAndSuffix(t);
    if (!isEligible(stem, keep)) continue;
    pairs.push({ stem, suffix });
    stems.add(stem);
  }
  const ordered = [...stems].sort();

  // The domain is forbidden from the start — implicitly, never by caller agreement.
  const forbidden = new Set(domain);
  const map = new Map<string, string>();
  const mintedStress = new Map<string, string>();
  const patterns = new Map<string, string>();
  const salts = new Map<string, number>();
  for (const stem of ordered) {
    const pattern = params.matchProsody ? stress(stem) : "1";
    patterns.set(stem, pattern);
    const { nonce, salt } = mintNonce(stem, pattern, params.seed, forbidden, 0);
    forbidden.add(nonce);
    map.set(stem, nonce);
    mintedStress.set(nonce, pattern);
    salts.set(stem, salt);
  }

  let remintRounds = 0;
  for (;;) {
    const offenders = injectivityOffenders(pairs, map, domain, params.seed);
    if (offenders.size === 0) break;
    remintRounds += 1;
    if (remintRounds > MAX_REMINT_ROUNDS) {
      throw new Error(
        `vacancy: conditions A/B of architecture.md §5.2 still violated after ` +
          `${MAX_REMINT_ROUNDS} re-mint rounds (${offenders.size} stems outstanding, e.g. ` +
          `${JSON.stringify([...offenders].sort().slice(0, 5))})`,
      );
    }
    // Canonical order, so the result does not depend on iteration order.
    for (const stem of [...offenders].sort()) {
      const pattern = patterns.get(stem);
      const previousSalt = salts.get(stem);
      if (pattern === undefined || previousSalt === undefined) {
        throw new Error(`vacancy: re-mint offender ${JSON.stringify(stem)} was never minted`);
      }
      const startSalt = 1000 * remintRounds + previousSalt + 1;
      const { nonce, salt } = mintNonce(stem, pattern, params.seed, forbidden, startSalt);
      // The superseded nonce stays in `forbidden` (it is not handed to anyone else) but
      // leaves `mintedStress`, where a stale key would claim a pattern nothing carries.
      const superseded = map.get(stem);
      if (superseded !== undefined) mintedStress.delete(superseded);
      forbidden.add(nonce);
      map.set(stem, nonce);
      mintedStress.set(nonce, pattern);
      salts.set(stem, salt);
    }
  }

  const image = new Set<string>();
  for (const t of domain) image.add(imageOfType(t, map, params.seed));

  return {
    mapping: map,
    mintedStress,
    remintRounds,
    bijective: true,
    imageSize: image.size,
    forbidden,
    domain,
  };
}

// --- §5 / §7 the transform -----------------------------------------------------------

/** Per-`vacateText` mutable state; nothing here survives the call. */
interface TransformState {
  /** stem -> how many eligible, vacancy-decided occurrences have been seen so far. */
  counts: Map<string, number>;
  /** Growing forbidden set for the `consistent = false` control. */
  forbidden: Set<string>;
}

/**
 * The core rewrite of one `WORD_RE` match. `state` is `undefined` for the order-free
 * paths (`mapVocabWords`), which have no occurrence index and therefore no `revealAfter`.
 *
 * Everything is computed on the LOWERCASED word and `matchCase` is applied once at the
 * end, which is what makes `lower(transformWord(w)) === transformWord(lower(w))` (§5.7).
 */
function transformWordWith(
  word: string,
  vmap: VacancyMap,
  params: VacancyParams,
  keep: ReadonlySet<string>,
  state?: TransformState,
): string {
  const [stem, suffix] = stemAndSuffix(word.toLowerCase());
  if (!isEligible(stem, keep)) return word;
  if (!(vacancyU(stem, params.seed) < params.p)) return word;

  // 0-based occurrence index of this STEM in document order (§5.8).
  let index = 0;
  if (state !== undefined) {
    index = state.counts.get(stem) ?? 0;
    state.counts.set(stem, index + 1);
    if (index + 1 <= params.revealAfter) return word;
  } else if (params.revealAfter > 0) {
    throw new Error(
      "vacancy: revealAfter > 0 needs an occurrence order; use vacateText, not mapVocabWords",
    );
  }

  let nonce: string;
  if (params.consistent) {
    const mapped = vmap.mapping.get(stem);
    if (mapped === undefined) {
      throw new Error(
        `vacancy: no nonce for stem ${JSON.stringify(stem)} — the map's domain must include ` +
          `every type of the corpus AND every word of the budget (architecture.md §5.2)`,
      );
    }
    nonce = mapped;
  } else {
    if (state === undefined) {
      throw new Error("vacancy: consistent = false needs an occurrence order; use vacateText");
    }
    // The control condition: a fresh type per occurrence, so the vacancy rate is held
    // fixed while the learnable identity is destroyed. It has, deliberately, no stability
    // property — the nonce is a function of (stem, occurrenceIndex) in document order.
    // `#` is not a legal WORD_RE character, so the key can never collide with a stem.
    //
    // CONDITION B APPLIES HERE TOO (§5.8): the nonce may equal neither a domain type nor
    // THE STEM IT REPLACES. The second clause is not implied by the first — a stem need not
    // be a type. Measured: at seed 7, `p = 1`, `tak` minted `tak`, so `Taking -> Taking` and
    // a token silently failed to vacate. §7.1 denies this control a *stability* property,
    // which is about reusing a nonce across occurrences; it does not license a word
    // surviving the transform, and a control whose vacancy rate is not the stated rate is
    // not a control. Forbidding the stem routes the collision through §5.5's ordinary
    // re-mint loop, so the replacement meets the same quality bar as any other nonce.
    // The stem is forbidden for THIS mint only, then restored — the set is threaded through
    // the whole rewrite, and a stem left in it would forbid that string to every later
    // occurrence of every OTHER stem, which condition B does not ask for.
    const pattern = params.matchProsody ? stress(stem) : "1";
    const stemWasForbidden = state.forbidden.has(stem);
    state.forbidden.add(stem);
    let minted;
    try {
      minted = mintNonce(`${stem}#${index}`, pattern, params.seed, state.forbidden, 0);
    } finally {
      if (!stemWasForbidden) state.forbidden.delete(stem);
    }
    nonce = minted.nonce;
    state.forbidden.add(nonce);
    vmap.mintedStress.set(nonce, pattern);
  }
  const surface = surfaceForm(stem, suffix, nonce, params.seed);
  return assertWholeWord(matchCase(word, surface), word);
}

/**
 * Transform ONE word, in the mapped condition (`consistent = true`, `revealAfter = 0`).
 * This is `transformWord` as §1 and §7.2 name it; `vacateText` is this applied to every
 * `WORD_RE` match of a text, and `mapVocabWords` is this applied to a budget in order.
 */
export function transformWord(word: string, vmap: VacancyMap, params: VacancyParams): string {
  return transformWordWith(word, vmap, params, effectiveKeepSet(params.keep));
}

/**
 * Rewrite a text in place (§1). Every `WORD_RE` match is replaced by `transformWord`;
 * everything else — whitespace, punctuation, digits, line breaks — passes through
 * unchanged, byte for byte. Since every output is itself a single complete `WORD_RE`
 * match, `tokenize(vacateText(t))` has the same length and ordering as `tokenize(t)`, and
 * because line breaks are untouched the `<eos>`-per-line rule fires in the same places.
 */
export function vacateText(text: string, vmap: VacancyMap, params: VacancyParams): string {
  const keep = effectiveKeepSet(params.keep);
  const state: TransformState = { counts: new Map(), forbidden: new Set(vmap.forbidden) };
  // A fresh RegExp per call: the shared `g`-flagged literal carries `lastIndex`.
  return text.replace(new RegExp(WORD_RE.source, "g"), (m) =>
    transformWordWith(m, vmap, params, keep, state),
  );
}

/**
 * §7.2, MAPPED VOCABULARY. Push the budget's word list through the SAME transform,
 * PRESERVING ORDER, so `itos_p = SPECIALS ++ mapVocabWords(words, ...)` gives every word
 * the id its pre-image had. That, with the map's injectivity, is what makes the token id
 * stream identical and training bit-identical (§7.3).
 *
 * Valid only for the mapped condition (`consistent = true`, `revealAfter = 0`); every
 * other condition rebuilds the budget from the vacated corpus instead, and the resulting
 * collapse in coverage IS the measurement.
 */
export function mapVocabWords(
  words: readonly string[],
  vmap: VacancyMap,
  params: VacancyParams,
): string[] {
  if (!params.consistent) {
    throw new Error("vacancy: mapVocabWords requires consistent = true (architecture.md §7.2)");
  }
  if (params.revealAfter !== 0) {
    throw new Error("vacancy: mapVocabWords requires revealAfter = 0 (architecture.md §7.2)");
  }
  const keep = effectiveKeepSet(params.keep);
  return words.map((w) => transformWordWith(w, vmap, params, keep));
}

// --- §10 statistics ------------------------------------------------------------------

/**
 * The statistics contract of §10 — these exact names (camelCase in BOTH stacks, §5.8),
 * computed from these definitions.
 *
 * The counting definitions are pinned because the two implementations first disagreed on
 * them: `tokensVacated` agreed to the token while the type counts read 1 922 against
 * 1 665, which was a gap in the contract and not a disagreement about the transform. Types
 * are counted over the DOMAIN (corpus types ∪ budget words); tokens over the CORPUS.
 */
export interface VacancyStats {
  /** Distinct lowercased types in the §5.2 DOMAIN (corpus types ∪ budget words). The
   *  diagnostic scope: it governs the map and the mapped vocabulary. */
  domainTypesTotal: number;
  /** Domain types whose STEM is eligible per §2.2. */
  domainTypesEligible: number;
  /** Domain types the MAP would rewrite at this `p` — map membership, not text. The
   *  domain's 22 Dolch-only words never occur in the corpus, so there is nothing to
   *  measure for them; this scope is a diagnostic about the map. */
  domainTypesVacated: number;
  /** Distinct lowercased types of the CORPUS itself. **This is the scope a panel shows a
   *  reader**: the domain-only words (`funny`, `squirrel`, `today`, …) are in the budget
   *  but never appear in the text, so counting them inflates the vacancy rate being
   *  reported to someone looking at that text. §10 forbids an unprefixed `types*` for
   *  exactly this reason — the unprefixed name read either way, and the two stacks read
   *  it differently. */
  corpusTypesTotal: number;
  /** Corpus types whose STEM is eligible per §2.2. */
  corpusTypesEligible: number;
  /**
   * Corpus types MEASURED FROM THE TWO TEXTS: a type counts as vacated iff at least one of
   * its occurrences actually changed.
   *
   * NOT map membership, which is what the golden fixture caught. Under `revealAfter > 0` a
   * type whose every occurrence falls inside the reveal window is still listed in the map
   * yet has changed nowhere in the text — map membership over-reports it by ~2x on this
   * corpus. The readings coincide at `revealAfter = 0`, which is why it took a control
   * condition to expose. The text reading is what this number claims to a reader looking at
   * "N of M types vacated" about the text in front of them.
   */
  corpusTypesVacated: number;
  /** Distinct eligible stems — the size of the map. `typesEligible >= stemsTotal` always,
   *  since inflected forms share a stem. */
  stemsTotal: number;
  /** Stems with `u(stem) < p` (departure 11: the source reports `len(self.map)`, which is
   *  `p`-independent and therefore wrong below full vacancy). */
  stemsVacated: number;
  /** Over the CORPUS token stream, not the domain. */
  tokensTotal: number;
  tokensVacated: number;
  meanSyllablesBefore: number;
  meanSyllablesAfter: number;
  meanAnapestBefore: number;
  meanAnapestAfter: number;
  /** The hand table of §6.1 — the honesty number for English words, and the one that must
   *  appear beside every prosody statistic. */
  stressFromTableBefore: number;
  stressFromTableAfter: number;
  /** Forms we minted, whose intended pattern we registered. Known by construction but
   *  ASSERTED rather than verified: §5.5 accepts a candidate on syllable COUNT, so the
   *  count is checked and the pattern is not. */
  stressFromMintedBefore: number;
  stressFromMintedAfter: number;
  /** The spelling heuristic of §6.2 — i.e. a guess. */
  stressFromRuleBefore: number;
  stressFromRuleAfter: number;
  bijective: boolean;
  imageSize: number;
  remintRounds: number;
}

function mean(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** Mean anapest over the lines that produce at least one token (§6.4). */
function meanAnapest(text: string, minted?: ReadonlyMap<string, string>): number {
  const scores: number[] = [];
  for (const line of splitLines(text)) {
    if (wordMatches(line).length === 0) continue;
    scores.push(meterScore(line, "anapest", minted));
  }
  return mean(scores);
}

/** §10's three-way split: token-weighted fractions that sum to 1. */
function stressSplit(
  words: readonly string[],
  minted: ReadonlyMap<string, string>,
): { table: number; mintedFrac: number; rule: number } {
  if (words.length === 0) return { table: 0, mintedFrac: 0, rule: 0 };
  let table = 0;
  let mintedHits = 0;
  for (const w of words) {
    const source = stressWithSource(w, minted).source;
    if (source === "table") table++;
    else if (source === "minted") mintedHits++;
  }
  const n = words.length;
  return { table: table / n, mintedFrac: mintedHits / n, rule: (n - table - mintedHits) / n };
}

/** Types whose stem passes §2.2. Shared by both scopes — eligibility is a property of the
 *  stem alone, so there is only one reading of it to get wrong. */
function countEligible(types: ReadonlySet<string>, keep: ReadonlySet<string>): number {
  let eligible = 0;
  for (const t of types) if (isEligible(stemAndSuffix(t)[0], keep)) eligible++;
  return eligible;
}

/**
 * `vacated` BY MAP MEMBERSHIP: the types the map would rewrite at this `p`.
 *
 * This is the DOMAIN scope's reading, and the choice is forced rather than preferred: the
 * domain contains the 22 Dolch words that never occur in the corpus, so they have no
 * occurrences to measure. Under a text-measured reading the domain scope would silently
 * collapse onto the corpus scope and stop being a separate diagnostic. The domain number
 * answers "what does the map do", and this computes exactly that.
 *
 * It is deliberately NOT the corpus scope's reading — see `vacancyStats`, where the two
 * readings are named and contrasted rather than left to look like one mechanism.
 */
function countVacatedByMap(
  types: ReadonlySet<string>,
  vmap: VacancyMap,
  params: VacancyParams,
  keep: ReadonlySet<string>,
): number {
  let vacated = 0;
  for (const t of types) {
    const [stem, suffix] = stemAndSuffix(t);
    if (!isEligible(stem, keep)) continue;
    if (!(vacancyU(stem, params.seed) < params.p)) continue;
    const nonce = vmap.mapping.get(stem);
    if (nonce !== undefined && surfaceForm(stem, suffix, nonce, params.seed) !== t) vacated++;
  }
  return vacated;
}

/**
 * §10. Both sides are scored with the SAME minted map, so the split is symmetric; on the
 * `Before` side `stressFromMinted` comes out 0 because the implicitly-forbidden domain
 * keeps every bare nonce off the corpus's type list, and conditions A/B keep every
 * assembled surface form off it too.
 */
export function vacancyStats(
  original: string,
  vacated: string,
  vmap: VacancyMap,
  params: VacancyParams,
): VacancyStats {
  const before = wordMatches(original);
  const after = wordMatches(vacated);
  if (before.length !== after.length) {
    throw new Error(
      `vacancy: the transform changed the token count (${before.length} -> ${after.length}); ` +
        `architecture.md §1 requires a word-for-word bijection`,
    );
  }

  const keep = effectiveKeepSet(params.keep);
  const corpusTypes = new Set(before.map((w) => w.toLowerCase()));

  // TWO SCOPES, TWO DELIBERATELY DIFFERENT READINGS — stated here rather than left to be
  // inferred, because the golden fixture caught them being silently different mechanisms.
  //
  //   domain: MAP MEMBERSHIP. A diagnostic about the map. The 22 Dolch-only words have no
  //           occurrences in the text, so a text reading cannot see them at all.
  //   corpus: MEASURED FROM THE TWO TEXTS. A type counts as vacated iff at least one of its
  //           occurrences actually changed — which is what the number claims to a reader,
  //           who is looking at "N of M types vacated" about the text in front of them.
  //
  // The readings coincide at `revealAfter = 0`, which is why only a control condition
  // exposed the difference. At `revealAfter > 0` a type whose every occurrence falls inside
  // the reveal window is still in the map but has changed nowhere in the text; counting it
  // over-reports by roughly 2x on this corpus.
  const domainVacated = countVacatedByMap(vmap.domain, vmap, params, keep);

  let tokensVacated = 0;
  const changedTypes = new Set<string>();
  for (let i = 0; i < before.length; i++) {
    const was = before[i].toLowerCase();
    if (was === after[i].toLowerCase()) continue;
    tokensVacated++;
    changedTypes.add(was);
  }

  let stemsVacated = 0;
  for (const stem of vmap.mapping.keys()) if (vacancyU(stem, params.seed) < params.p) stemsVacated++;

  const splitBefore = stressSplit(before, vmap.mintedStress);
  const splitAfter = stressSplit(after, vmap.mintedStress);

  return {
    domainTypesTotal: vmap.domain.size,
    domainTypesEligible: countEligible(vmap.domain, keep),
    domainTypesVacated: domainVacated,
    corpusTypesTotal: corpusTypes.size,
    corpusTypesEligible: countEligible(corpusTypes, keep),
    corpusTypesVacated: changedTypes.size,
    stemsTotal: vmap.mapping.size,
    stemsVacated,
    tokensTotal: before.length,
    tokensVacated,
    meanSyllablesBefore: mean(before.map((w) => syllables(w, vmap.mintedStress))),
    meanSyllablesAfter: mean(after.map((w) => syllables(w, vmap.mintedStress))),
    meanAnapestBefore: meanAnapest(original, vmap.mintedStress),
    meanAnapestAfter: meanAnapest(vacated, vmap.mintedStress),
    stressFromTableBefore: splitBefore.table,
    stressFromTableAfter: splitAfter.table,
    stressFromMintedBefore: splitBefore.mintedFrac,
    stressFromMintedAfter: splitAfter.mintedFrac,
    stressFromRuleBefore: splitBefore.rule,
    stressFromRuleAfter: splitAfter.rule,
    bijective: vmap.bijective,
    imageSize: vmap.imageSize,
    remintRounds: vmap.remintRounds,
  };
}
