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
 * A run of characters a READER would call one word — Unicode letters plus the internal
 * apostrophes and hyphens `WORD_RE` allows. Only used to find runs `WORD_RE` mangles.
 */
const WORDLIKE_RE = /\p{L}+(?:['\-]\p{L}+)*/gu;

/**
 * Words of `text` that `wordRe` splits or truncates, in order of appearance.
 *
 * MIRROR of `fragmented_words` (Python). The transform's `WORD_RE` is
 * `[A-Za-z]+(?:['-][A-Za-z]+)*` — ASCII letters only — so `café` is the word `caf` to it
 * and `naïvely` is the two words `na` + `vely`. Vacating those rewrites a fragment: "a
 * café" became "a washé", which the full stack RETURNED as a score. Runs `wordRe` matches
 * entirely (`don't`, `good-bye`) are fine, and so are runs it never touches (CJK, emoji):
 * those are never vacated, are byte-identical in all three variants, and are attributed
 * to no word — the same treatment punctuation gets.
 */
export function fragmentedWords(text: string, wordRe: RegExp): string[] {
  const runs = new RegExp(WORDLIKE_RE.source, "gu");
  const inner = new RegExp(wordRe.source, wordRe.flags.includes("g") ? wordRe.flags : `${wordRe.flags}g`);
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = runs.exec(text)) !== null) {
    const run = m[0];
    const parts = run.match(inner) ?? [];
    if (parts.length > 0 && (parts.length > 1 || parts[0] !== run)) out.push(run);
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
    `${where}the vacancy transform's word alphabet is ASCII letters only ` +
      `(WORD_RE = [A-Za-z]+(?:['-][A-Za-z]+)*), so it does not see these words as words ` +
      `and would rewrite a fragment of each instead: ${bad.map((w) => JSON.stringify(w)).join(", ")}. ` +
      "Refusing rather than scoring text the transform mangles — 'a café' vacates to " +
      "'a washé', and a single BPE piece can then cover both a preserved and a vacated " +
      "fragment. Use a passage written in the ASCII alphabet (emoji and CJK are fine: " +
      "they are never vacated and are identical in all three variants). Widening the " +
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
