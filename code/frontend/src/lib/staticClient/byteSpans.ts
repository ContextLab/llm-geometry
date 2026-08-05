/**
 * Token→word attribution in UTF-8 BYTE coordinates (contract §8.2, FR-718).
 *
 * The mirror of `llm_geometry/arch/vacancy_score.py`. It is byte-based rather than
 * character-based for reasons that were measured, not assumed:
 *
 *  - transformers.js exposes NO token offsets at all (`return_offsets_mapping` is
 *    ignored; there is not one `offset` key anywhere in the tokenizer's object graph),
 *    so HF's Python-side offsets are not a mechanism the two stacks can share;
 *  - decoding tokens one at a time and concatenating emits U+FFFD and destroys the text
 *    on any split multi-byte character, in BOTH stacks;
 *  - Python indexes code points where JavaScript indexes UTF-16 units — the two genuinely
 *    disagree about the same string (31 vs 32 on one probe text), so "character" is not a
 *    unit a cross-language contract can be written in;
 *  - HF's own offsets OVERLAP on multi-byte characters, so per-token quantities summed
 *    over a word would be double-counted. Byte spans are a true partition.
 *
 * The reconstruction assertion in `tokenByteSpans` is the guard rail: if the concatenated
 * pieces are not `utf8(text)` byte for byte, this raises rather than mis-attributing.
 */

import { computeError, invalidParamError } from "./errors";
import { JOINER_CLASS, LETTER_CLASS, MARK_CLASS } from "./wordClasses";

/** GPT-2's unicode→byte table: the inverse of `bytes_to_unicode`. */
function buildByteDecoder(): Map<string, number> {
  const bs: number[] = [];
  for (let b = "!".charCodeAt(0); b <= "~".charCodeAt(0); b++) bs.push(b);
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b);
  for (let b = 0xae; b <= 0xff; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const out = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) out.set(String.fromCharCode(cs[i]), bs[i]);
  return out;
}

const BYTE_DECODER = buildByteDecoder();

export type ByteSpan = readonly [number, number];

/**
 * The `[start, end)` UTF-8 byte range each token owns, verified by reconstruction.
 *
 * The spans tile `utf8(text)` exactly once. A token that is a bare continuation byte gets
 * a degenerate EMPTY span — the honest answer for it — and still resolves to the right
 * word, because its start byte lies strictly inside the character it continues.
 */
export function tokenByteSpans(pieces: readonly string[], text: string): ByteSpan[] {
  const spans: ByteSpan[] = [];
  const parts: number[] = [];
  let cursor = 0;
  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i];
    let width = 0;
    for (const ch of piece) {
      const b = BYTE_DECODER.get(ch);
      if (b === undefined) {
        throw computeError(
          `token ${i} (${JSON.stringify(piece)}) contains a character outside the ` +
            "byte-level BPE table, so its byte width cannot be determined",
        );
      }
      parts.push(b);
      width += 1;
    }
    spans.push([cursor, cursor + width]);
    cursor += width;
  }

  const expected = new TextEncoder().encode(text);
  let same = parts.length === expected.length;
  if (same) {
    for (let i = 0; i < expected.length; i++) {
      if (parts[i] !== expected[i]) {
        same = false;
        break;
      }
    }
  }
  if (!same) {
    throw computeError(
      "token→text alignment failed: the concatenated byte-level pieces do not reproduce " +
        `the passage (${parts.length} bytes rebuilt vs ${expected.length} expected). ` +
        "Refusing to attribute tokens to words rather than mis-attribute them.",
    );
  }
  return spans;
}

/**
 * `nChars` for the contract's §8.1 stats: **Unicode code points**, matching Python's
 * `len(str)`.
 *
 * Never `text.length`. JavaScript's `.length` counts UTF-16 units, so it reported 77 for
 * the same probe string the backend reported 75 for — two astral emoji, two surrogate
 * pairs, two extra units. One field, one name, one contract, two stacks: they have to be
 * counting the same thing, and `bitsPerChar` is derived from it in the full stack.
 * (Attribution stays in BYTES — see the header — because that is a unit both stacks can
 * partition text with. This is a reporting count, not an index.)
 */
export function nCharsOf(text: string): number {
  return [...text].length;
}

export interface WordSpan {
  index: number;
  word: string;
  start: number;
  end: number;
}

/** Every `WORD_RE` match of `text`, in UTF-8 byte coordinates. */
export function wordSpans(text: string, wordRe: RegExp): WordSpan[] {
  const re = new RegExp(wordRe.source, wordRe.flags.includes("g") ? wordRe.flags : `${wordRe.flags}g`);
  const encoder = new TextEncoder();
  const out: WordSpan[] = [];
  let m: RegExpExecArray | null;
  let index = 0;
  while ((m = re.exec(text)) !== null) {
    const start = encoder.encode(text.slice(0, m.index)).length;
    out.push({ index, word: m[0], start, end: start + encoder.encode(m[0]).length });
    index += 1;
    if (m[0] === "") re.lastIndex += 1; // defensive: never loop on an empty match
  }
  return out;
}

/**
 * Characters that BIND two letters into one written word without being letters.
 *
 * MIRROR of `WORD_JOINER_CHARS` + `WORD_JOINER_CATEGORIES` (Python). This is a class, not
 * the one character that was reported: `WORD_RE` accepts exactly two joiners (ASCII `'`
 * and `-`), and every other one used to END a wordlike run, leaving two runs `WORD_RE`
 * matched in full — so `fragmentedWords` found nothing, the transform rewrote half a word
 * and the score came back 200. Observed: `don’t` → `big’t` (U+2019, what every smart-quotes
 * editor emits), `co<SHY>operate` → `co<SHY>wood`, `cat<ZWJ>sat` → `want<ZWJ>wish`.
 *
 *  - `Pd` — dash punctuation: hyphens of every width (U+002D, U+2010 hyphen, U+2011
 *    non-breaking hyphen, U+2012–U+2015, the fullwidth and small forms);
 *  - `Cf` — invisible format characters: U+00AD soft hyphen, U+200B–U+200F (ZWSP,
 *    ZWNJ, ZWJ, the bidi marks), U+2060 word joiner, U+FEFF;
 *  - `Pc` — connector punctuation: `_`, U+203F undertie, U+2040, U+2054, U+FE33–U+FE34,
 *    U+FE4D–U+FE4F, U+FF3F. **Missing from both stacks until 2026-08-04**, which is why
 *    `don‿t` scored 200 in the full stack and swapped to `warm‿t`;
 *  - the apostrophes and word-internal points the table names, which carry no property
 *    that separates them from ordinary quotation marks.
 *
 * Combining marks are not joiners — a mark belongs to the letter it sits on — so they are
 * part of the letter atom `L M*` below. NFC composes most of them away, but only most:
 * `k` + U+0301 has no precomposed form, so `wor`+U+0301+`d` stayed two `WORD_RE` words and
 * was rewritten as a fragment.
 *
 * The three classes come from `wordClasses.ts` — a PINNED table both stacks share — and no
 * longer from `\p{...}`, which is this runtime's Unicode and not Python's. See that file
 * for the 9 993-character disagreement that made the two stacks refuse different passages.
 */

/**
 * A run of characters a READER would call one word.
 *
 * Grammar: `(L M*)+ ( J+ (L M*)+ )*` — letters with the marks that sit on them, then any number of
 * joiner-separated continuations. A trailing joiner is punctuation and is left out (the
 * group requires a letter after it); a trailing mark is part of its letter. MIRROR of
 * `wordlike_runs` (Python), which scans rather than matches because Python's `re` has no
 * character-class syntax for a range table. Both suites carry the same case table.
 */
const LETTER_RUN = `(?:[${LETTER_CLASS}][${MARK_CLASS}]*)+`;
const WORDLIKE_RE = new RegExp(
  `${LETTER_RUN}(?:[${JOINER_CLASS}]+${LETTER_RUN})*`,
  "gu",
);

/**
 * The Gutenberg em-dash convention: two or more ASCII hyphens, which a reader reads as a
 * dash BETWEEN two words rather than as a joiner inside one. `WORDLIKE_RE` cannot tell the
 * two apart (its grammar accepts any run of joiners), so `fragmentedWords` cuts a run here
 * before asking whether `wordRe` matches each piece whole. MIRROR of `EM_DASH_RE`
 * (`llm_geometry/arch/vacancy_score.py`).
 */
const EM_DASH_RE = /-{2,}/;

/** `wordRe`, anchored, for "does it match this piece WHOLE?" — never a `.test` on flags. */
function wholeWordRe(wordRe: RegExp): RegExp {
  return new RegExp(`^(?:${wordRe.source})$`, wordRe.flags.replace(/[gy]/g, ""));
}

/**
 * Words of `text` that `wordRe` splits or truncates, in order of appearance.
 *
 * MIRROR of `fragmented_words` (Python). The transform's `WORD_RE` is
 * `[A-Za-z]+(?:['-][A-Za-z]+)*` — ASCII letters joined by the ASCII apostrophe and hyphen
 * only — so `café` is the word `caf` to it, `naïvely` is the two words `na` + `vely`, and
 * `don’t` is `don` + `t`. Vacating those rewrites a fragment: "a café" became "a washé"
 * and "don’t" became "big’t", both of which the full stack RETURNED as a score. Runs
 * `wordRe` matches entirely (`don't`, `good-bye`) are fine, and so are runs it never
 * touches (CJK, emoji): those are never vacated, are byte-identical in all three variants,
 * and are attributed to no word — the same treatment punctuation gets.
 *
 * ONE exemption, and it is narrow: the Gutenberg em-dash convention `legs--upon`. Two or
 * more ASCII hyphens between letters are a DASH — punctuation a reader reads as a gap
 * between two words — not a joiner inside one word. `WORDLIKE_RE` cannot know that (`J+`
 * accepts any run of joiners), so it yields one run, `wordRe` finds two matches, and this
 * used to refuse — while the refusal message told the reader to "use a passage written in
 * the ASCII alphabet, with straight apostrophes and hyphens", which `legs--upon` already
 * is. There was no way to comply, and this project's own corpus contains `ba--are`,
 * `hea--art`, `Lady--loves` and `legs--upon`. So the run is cut at each dash and the
 * exemption holds only when `wordRe` matches EVERY resulting piece WHOLE.
 *
 * The test is that whole-match, NOT "the run is written in ASCII characters". Those two
 * differ, and the difference is the original defect: `don''t` is pure ASCII, and pure
 * `WORD_RE` alphabet, and it is still `don` + `t` with a character `wordRe` cannot see
 * stranded between two halves of a word it rewrote — it vacated to `little''t` and scored
 * 200, character-for-character the `don’t` → `big’t` defect. An ASCII-alphabet test
 * admitted it (2026-08-04, while closing the `--` false positive); the whole-match test
 * refuses it, and refuses `don-'t` and `co<SHY>operate` with it.
 */
export function fragmentedWords(text: string, wordRe: RegExp): string[] {
  const runs = new RegExp(WORDLIKE_RE.source, "gu");
  const inner = new RegExp(wordRe.source, wordRe.flags.includes("g") ? wordRe.flags : `${wordRe.flags}g`);
  const whole = wholeWordRe(wordRe);
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = runs.exec(text)) !== null) {
    const run = m[0];
    const parts = run.match(inner) ?? [];
    if (parts.length === 0 || (parts.length === 1 && parts[0] === run)) continue;
    const pieces = run.split(EM_DASH_RE).filter((piece) => piece !== "");
    if (pieces.length > 0 && pieces.every((piece) => whole.test(piece))) continue;
    out.push(run);
  }
  return out;
}

/**
 * Refuse a passage whose words the transform would mangle. MIRROR of
 * `check_word_alphabet` (Python), including its message, so a reader who runs both stacks
 * is told the same thing.
 */
export function checkWordAlphabet(text: string, wordRe: RegExp, index?: number): void {
  const bad = [...new Set(fragmentedWords(text, wordRe))];
  if (bad.length === 0) return;
  const where = index === undefined ? "" : `passage ${index}: `;
  throw invalidParamError(
    `${where}the vacancy transform's word alphabet is ASCII letters joined by the ASCII ` +
      `apostrophe and hyphen only (WORD_RE = [A-Za-z]+(?:['-][A-Za-z]+)*), so it does not ` +
      `see these words as words and would rewrite a fragment of each instead: ` +
      `${bad.map((w) => JSON.stringify(w)).join(", ")}. ` +
      "Refusing rather than scoring text the transform mangles — 'a café' vacates to " +
      "'a washé' and 'don’t' (curly apostrophe) to 'big’t', and a single BPE piece can " +
      "then cover both a preserved and a vacated fragment. Use a passage written in the " +
      "ASCII alphabet, with ONE straight apostrophe or hyphen between letters — \"don't\" " +
      "and \"good-bye\" are matched whole, and so is the Gutenberg dash \"legs--upon\", " +
      "which is two words with punctuation between them; \"don''t\" is not, and is the " +
      "same stranded-character defect as the curly apostrophe. (Emoji and CJK are fine: " +
      "they are never vacated and are identical in all three variants.) Widening the " +
      "alphabet is a change to the shared transform and its contract, not to this panel.",
  );
}

/**
 * Indices of the tokens belonging to a PRESERVED word.
 *
 * A token belongs to a word when their byte ranges OVERLAP — not "starts inside". A
 * byte-level BPE folds a word's leading space into the word's own token, so the token
 * that *is* the function word starts one byte before the word does, and the start rule
 * would drop nearly all of them.
 *
 * A token overlapping both a preserved and a vacated word cannot be attributed, and that
 * raises. For a passage inside the transform's ASCII word alphabet it cannot happen — the
 * curated models' pretokenizers never merge across a word boundary — but it CAN happen,
 * and did as an opaque 500, when `WORD_RE` splits one written word in two: `naïvely` is
 * `na` + `vely` and one BPE piece covers both halves, so one may be preserved while the
 * other is vacated. `checkWordAlphabet` refuses such passages up front, naming the word
 * instead of a token index. This stays as the last line of defence: if it fires, the
 * assumption behind this attribution has changed and no number may be reported from it.
 */
export function preservedTokenIndices(
  spans: readonly ByteSpan[],
  words: readonly WordSpan[],
  preserved: ReadonlySet<number>,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < spans.length; i++) {
    const [a, b] = spans[i];
    const hits = words.filter((w) => a < w.end && b > w.start);
    if (hits.length === 0) continue; // punctuation, whitespace, line breaks
    const flags = new Set(hits.map((w) => preserved.has(w.index)));
    if (flags.size > 1) {
      throw computeError(
        `token ${i} spans both a preserved and a vacated word ` +
          `(${hits.map((w) => JSON.stringify(w.word)).join(", ")}); refusing to attribute it`,
      );
    }
    if (flags.has(true)) out.push(i);
  }
  return out;
}
