/**
 * Deterministic word-level tokenizer — exact TypeScript port of
 * code/backend/src/llm_geometry/geo/tokenizer.py.
 *
 * Lowercases, normalizes typographic punctuation (curly quotes -> ASCII, nbsp ->
 * space), then splits into word / number / single-punctuation tokens with the same
 * regex as the backend: `[a-z]+(?:'[a-z]+)*|[0-9]+|[^\sa-z0-9]`. The `u` flag makes
 * `[^\s...]` match by code point like Python (an astral character is one token, not
 * two surrogate halves). Known residual divergence (documented, not observed in the
 * corpus/goldens): JS `\s` also treats U+FEFF as whitespace while Python's does not.
 *
 * Vocabulary: 1000 corpus word types + specials <unk>=0 <eos>=1 <pad>=2, loaded from
 * the backend-exported vocab.json (GeoTokenizer.to_json()).
 */

import { invalidParam } from "./errors";
import {
  CONTEXT_WINDOW,
  EOS_ID,
  EOS_TOKEN,
  PAD_ID,
  PAD_TOKEN,
  UNK_ID,
  UNK_TOKEN,
  VOCAB_SIZE,
  VOCAB_WORDS,
} from "./model";

// Typographic -> ASCII normalization applied before splitting (backend _NORMALIZE).
const NORMALIZE_RE = /[’‘“” ]/g;
const NORMALIZE_MAP: Record<string, string> = {
  "’": "'",
  "‘": "'",
  "“": '"',
  "”": '"',
  " ": " ",
};

// Words (with internal apostrophes), numbers, or any single non-space symbol.
const TOKEN_RE = /[a-z]+(?:'[a-z]+)*|[0-9]+|[^\sa-z0-9]/gu;

const SPECIAL_TEXTS: ReadonlyMap<number, string> = new Map([
  [UNK_ID, UNK_TOKEN],
  [EOS_ID, EOS_TOKEN],
  [PAD_ID, PAD_TOKEN],
]);

// Tokens that should not be preceded by a space when detokenizing.
const NO_SPACE_BEFORE = new Set([...".,;:!?)]}'…", "''"]);
const NO_SPACE_AFTER = new Set([..."([{"]);

/**
 * The CANONICAL vocabulary serialization — byte-identical to the backend's
 * `GeoTokenizer.to_json()`.
 *
 * `vocab_sha256` is a digest of these exact bytes, so the spelling is part of the
 * format, not a formatting preference. This build used to emit `JSON.stringify`'s
 * output in its own key order while Python used `", "`/`": "` separators and sorted
 * keys, so the SAME model had two different digests depending on which build wrote the
 * file. Both stacks now emit: keys sorted, compact separators, and every non-ASCII
 * character escaped (Python's `ensure_ascii=True`) so the bytes never depend on the
 * encoding. The token regex admits any non-space symbol, so accented letters and
 * em-dashes really do reach the word list.
 */
export function canonicalVocabJson(words: readonly string[]): string {
  // Key order: "format" < "specials" < "words", and inside specials "<eos>" < "<pad>"
  // < "<unk>" — written out rather than sorted at runtime so the order is reviewable.
  const raw = JSON.stringify({
    format: "geo-tokenizer-v1",
    specials: { [EOS_TOKEN]: EOS_ID, [PAD_TOKEN]: PAD_ID, [UNK_TOKEN]: UNK_ID },
    words: [...words],
  });
  // JSON.stringify escapes control characters, quotes and backslashes exactly as
  // Python does; it leaves non-ASCII literal, which Python does not. Escape per
  // UTF-16 CODE UNIT (not code point), so an astral character becomes its two
  // surrogate escapes — which is exactly what `ensure_ascii` emits for one.
  let out = "";
  for (let i = 0; i < raw.length; i++) {
    const code = raw.charCodeAt(i);
    out += code < 0x80 ? raw[i] : "\\u" + code.toString(16).padStart(4, "0");
  }
  return out;
}

/** Deterministically split `text` into lowercase word/punctuation tokens. */
export function splitWords(text: string): string[] {
  const normalized = text.toLowerCase().replace(NORMALIZE_RE, (ch) => NORMALIZE_MAP[ch]);
  return normalized.match(TOKEN_RE) ?? [];
}

export interface EncodedText {
  ids: number[];
  texts: string[];
  unk: boolean[];
  n_unk: number;
  truncated: boolean;
}

/** Contract-shaped token list: [{id, text, unk}, ...]. */
export function encodedTokens(enc: EncodedText): { id: number; text: string; unk: boolean }[] {
  return enc.ids.map((id, i) => ({ id, text: enc.texts[i], unk: enc.unk[i] }));
}

export class GeoTokenizer {
  readonly words: string[];
  readonly idToText: Map<number, string>;
  readonly textToId: Map<string, number>;

  constructor(words: string[]) {
    if (words.length !== VOCAB_WORDS) {
      throw invalidParam(`GeoTokenizer requires exactly ${VOCAB_WORDS} words, got ${words.length}`);
    }
    if (new Set(words).size !== words.length) {
      throw invalidParam("GeoTokenizer vocabulary contains duplicates");
    }
    this.words = [...words];
    this.idToText = new Map(SPECIAL_TEXTS);
    this.words.forEach((word, offset) => {
      this.idToText.set(SPECIAL_TEXTS.size + offset, word);
    });
    this.textToId = new Map();
    for (const [tid, text] of this.idToText) {
      if (!SPECIAL_TEXTS.has(tid)) this.textToId.set(text, tid);
    }
    if (this.idToText.size !== VOCAB_SIZE) {
      throw invalidParam(`GeoTokenizer vocabulary resolves to ${this.idToText.size} ids, expected ${VOCAB_SIZE}`);
    }
  }

  /**
   * Defensive loader for the exported vocab.json. Accepts either export shape:
   *  - the backend tokenizer's `to_json()`: {"format": "geo-tokenizer-v1",
   *    "specials": {"<unk>": 0, ...}, "words": [1000 words]}, or
   *  - the static-site export: {"format": "geo-vocab-v1", "specials":
   *    {"unk": 0, ...}, "tokens": [all 1003 token texts, specials first]}.
   */
  static fromVocabJson(data: unknown): GeoTokenizer {
    if (typeof data === "string") data = JSON.parse(data);
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
      throw invalidParam("vocab.json: expected a JSON object with a `words` or `tokens` array");
    }
    const obj = data as Record<string, unknown>;
    if (
      "format" in obj &&
      obj.format !== "geo-tokenizer-v1" &&
      obj.format !== "geo-vocab-v1"
    ) {
      throw invalidParam(`vocab.json: unknown tokenizer format ${JSON.stringify(obj.format)}`);
    }
    if ("specials" in obj && obj.specials !== null && typeof obj.specials === "object") {
      const sp = obj.specials as Record<string, unknown>;
      // Both spellings of the special map are validated when present.
      const expected: Record<string, number> = {
        [UNK_TOKEN]: UNK_ID,
        [EOS_TOKEN]: EOS_ID,
        [PAD_TOKEN]: PAD_ID,
        unk: UNK_ID,
        eos: EOS_ID,
        pad: PAD_ID,
      };
      for (const [token, id] of Object.entries(expected)) {
        if (sp[token] !== undefined && sp[token] !== id) {
          throw invalidParam(`vocab.json: special ${token} has id ${String(sp[token])}, expected ${id}`);
        }
      }
    }
    let words: unknown = obj.words;
    if (words === undefined && Array.isArray(obj.tokens)) {
      const tokens = obj.tokens;
      if (tokens.length !== VOCAB_SIZE) {
        throw invalidParam(`vocab.json: \`tokens\` has ${tokens.length} entries, expected ${VOCAB_SIZE}`);
      }
      const specialOrder = [UNK_TOKEN, EOS_TOKEN, PAD_TOKEN];
      specialOrder.forEach((tok, i) => {
        if (tokens[i] !== tok) {
          throw invalidParam(`vocab.json: tokens[${i}] is ${JSON.stringify(tokens[i])}, expected ${tok}`);
        }
      });
      words = tokens.slice(specialOrder.length);
    }
    if (!Array.isArray(words) || !words.every((w) => typeof w === "string")) {
      throw invalidParam("vocab.json: `words` (or `tokens`) must be an array of strings");
    }
    return new GeoTokenizer(words as string[]);
  }

  encode(
    text: string,
    opts: { truncate?: boolean; truncateSide?: "right" | "left"; maxTokens?: number } = {},
  ): EncodedText {
    const { truncate = true, truncateSide = "right", maxTokens = CONTEXT_WINDOW } = opts;
    let pieces = splitWords(text);
    let truncated = false;
    if (truncate && pieces.length > maxTokens) {
      truncated = true;
      pieces = truncateSide === "right" ? pieces.slice(0, maxTokens) : pieces.slice(-maxTokens);
    }
    const ids: number[] = [];
    const unk: boolean[] = [];
    let nUnk = 0;
    for (const piece of pieces) {
      const tid = this.textToId.get(piece);
      if (tid === undefined) {
        ids.push(UNK_ID);
        unk.push(true);
        nUnk++;
      } else {
        ids.push(tid);
        unk.push(false);
      }
    }
    return { ids, texts: pieces, unk, n_unk: nUnk, truncated };
  }

  /** Encode without truncation (fine-tuning token streams). */
  encodeStream(text: string): number[] {
    return this.encode(text, { truncate: false }).ids;
  }

  decode(ids: number[]): string {
    const out: string[] = [];
    for (const tid of ids) {
      const text = this.idToText.get(tid);
      if (text === undefined) {
        throw invalidParam(`Token id ${tid} is out of range (0..${VOCAB_SIZE - 1})`);
      }
      if (out.length === 0) out.push(text);
      else if (NO_SPACE_BEFORE.has(text) || NO_SPACE_AFTER.has(out[out.length - 1])) out.push(text);
      else out.push(" " + text);
    }
    return out.join("");
  }
}
