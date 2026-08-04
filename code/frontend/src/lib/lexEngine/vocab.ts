/**
 * Vocabulary budgets — the independent variable of the Lexicon Lab.
 *
 * TypeScript port of `code/backend/src/llm_geometry/lex/vocab.py` (plus the vocabulary
 * constants of `lex/config.py`). Everything here must agree with Python EXACTLY, because
 * the budget determines the row layout of every saved model:
 *
 *   * the tokenizer regex is behaviourally identical to Python's
 *     `[A-Za-z]+(?:['\-][A-Za-z]+)*`, so `don't` and `good-bye` — both real Dolch
 *     entries — survive as single tokens, and punctuation is dropped rather than
 *     tokenized (a budget of N *words* should spend all N rows on words);
 *   * frequency budgets rank by count and break ties ALPHABETICALLY, so a budget is a
 *     deterministic function of its corpus in either language;
 *   * `SPECIAL_TOKENS` occupy rows 0..3 in the same order, so `<unk>`..`<pad>` mean the
 *     same thing in every model and a saved model's ids stay meaningful.
 *
 * Out-of-budget tokens become `<unk>` during training and are BANNED at generation, so a
 * model can only ever speak in-budget. The `<unk>` rate is reported rather than hidden —
 * it is the measurable form of what a budget cannot say.
 */

import { DOLCH_ORDER, dolchBudget, isDolchBudgetName } from "./dolch";

// --- vocabulary constants (lex/config.py) --------------------------------------------

/** Reserved rows, always present whatever the budget. `<unk>` is index 0 so a missing
 *  entry can never silently alias a real word. */
export const SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>", "<pad>"] as const;
export const UNK_ID = 0;
export const BOS_ID = 1;
export const EOS_ID = 2;
export const PAD_ID = 3;

/** Never sampled during generation: `<unk>` would print a hole, `<bos>`/`<pad>` are
 *  structural. `<eos>` IS sampleable — it is how the model ends a line. */
export const GENERATION_BANNED_IDS = [UNK_ID, BOS_ID, PAD_ID] as const;

export const BUDGET_SOURCES = ["dolch", "frequency"] as const;
export type BudgetSource = (typeof BUDGET_SOURCES)[number];
export const DEFAULT_BUDGET_SOURCE: BudgetSource = "dolch";
export const DEFAULT_BUDGET = "full";

/**
 * Letters plus internal apostrophes/hyphens. Behaviourally identical to the Python
 * `re` pattern: both engines are leftmost-longest-per-alternative with greedy `+`/`*`
 * over an ASCII-only character class, and neither pattern uses a feature the other
 * lacks. `\-` inside the class is an escaped literal hyphen in both.
 */
export const WORD_RE = /[A-Za-z]+(?:['\-][A-Za-z]+)*/g;

/** Lower-cased word tokens. The only tokenizer in this feature. */
export function tokenize(text: string): string[] {
  // A fresh RegExp per call: the shared `g`-flagged literal carries `lastIndex` state.
  const re = new RegExp(WORD_RE.source, "g");
  const out: string[] = [];
  for (let m = re.exec(text); m !== null; m = re.exec(text)) out.push(m[0].toLowerCase());
  return out;
}

/** True when a line contains at least one word token (Python's `WORD_RE.search`). */
export function hasWord(line: string): boolean {
  return new RegExp(WORD_RE.source).test(line);
}

/** Python `str.splitlines()`: every Unicode line boundary, not just `\n`. */
const LINE_BREAK = new RegExp("\\r\\n|[\\n\\r\\v\\f\\u001c\\u001d\\u001e\\u0085\\u2028\\u2029]");

export function splitLines(text: string): string[] {
  return text.split(LINE_BREAK);
}

// --- coverage ------------------------------------------------------------------------

/** What a budget can and cannot express about a specific corpus. */
export interface Coverage {
  total_tokens: number;
  in_budget_tokens: number;
  distinct_types: number;
  oov_types: number;
  total_lines: number;
  whole_lines_in_budget: number;
  token_coverage: number;
  unk_rate: number;
}

// --- the resolved vocabulary ---------------------------------------------------------

/**
 * A budget, resolved into ids. Row layout is `SPECIAL_TOKENS` first, then the budget
 * words in a stable order.
 */
export class LexVocab {
  /** Budget words only, WITHOUT the specials. */
  readonly words: readonly string[];
  readonly source: BudgetSource;
  /** e.g. "full", or "top314" for frequency budgets. */
  readonly budgetName: string;
  readonly itos: readonly string[];
  private readonly stoiMap: Map<string, number>;

  constructor(words: readonly string[], source: BudgetSource, budgetName: string) {
    this.words = [...words];
    this.source = source;
    this.budgetName = budgetName;
    this.itos = [...SPECIAL_TOKENS, ...this.words];
    this.stoiMap = new Map(this.itos.map((w, i) => [w, i]));
  }

  /** |V| as the user set it — the number of real WORDS. */
  get budgetSize(): number {
    return this.words.length;
  }

  /** Embedding rows, which is |V| plus the specials. Displayed separately. */
  get rows(): number {
    return this.itos.length;
  }

  stoi(word: string): number {
    return this.stoiMap.get(word) ?? UNK_ID;
  }

  has(word: string): boolean {
    return this.stoiMap.has(word);
  }

  encode(tokens: Iterable<string>): number[] {
    const out: number[] = [];
    for (const t of tokens) out.push(this.stoi(t));
    return out;
  }

  decode(ids: Iterable<number>): string[] {
    const n = this.itos.length;
    const out: string[] = [];
    for (const i of ids) out.push(i >= 0 && i < n ? this.itos[i] : "<unk>");
    return out;
  }

  /**
   * Encode raw text: tokenize, then map. Out-of-budget types become `<unk>` (FR-604).
   *
   * NOT the training stream — this is a flat encoding with no line structure. Training
   * data is built by `train.ts::tokenStream`, which closes every non-blank line with
   * `<eos>`, matching `lex/train.py::token_stream`.
   */
  encodeText(text: string): number[] {
    return this.encode(tokenize(text));
  }

  /** Measure this budget against a corpus. Populates the UI's counters (FR-606). */
  coverage(text: string): Coverage {
    const vocab = new Set(this.words);
    const toks = tokenize(text);
    const types = new Set(toks);
    let oov = 0;
    for (const t of types) if (!vocab.has(t)) oov++;
    let inBudget = 0;
    for (const t of toks) if (vocab.has(t)) inBudget++;

    let totalLines = 0;
    let whole = 0;
    for (const line of splitLines(text)) {
      if (!hasWord(line)) continue;
      totalLines++;
      if (tokenize(line).every((t) => vocab.has(t))) whole++;
    }

    const token_coverage = toks.length > 0 ? inBudget / toks.length : 0;
    return {
      total_tokens: toks.length,
      in_budget_tokens: inBudget,
      distinct_types: types.size,
      oov_types: oov,
      total_lines: totalLines,
      whole_lines_in_budget: whole,
      token_coverage,
      unk_rate: 1 - token_coverage,
    };
  }
}

/**
 * The `size` most frequent types of `text`.
 *
 * Ties are broken alphabetically so the budget is a deterministic function of the
 * corpus. JS compares strings by UTF-16 code unit and Python by code point; the token
 * alphabet here is ASCII letters plus `'` and `-`, where the two orders coincide.
 */
export function frequencyBudget(text: string, size: number): string[] {
  const counts = new Map<string, number>();
  for (const t of tokenize(text)) counts.set(t, (counts.get(t) ?? 0) + 1);
  const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return ordered.slice(0, Math.max(0, Math.trunc(size))).map(([w]) => w);
}

/**
 * Resolve a (source, budget) pair against a corpus into a concrete vocabulary.
 *
 * For `dolch` the budget name selects a graded list and `size` is ignored — the list IS
 * the budget. For `frequency` the budget is the top-`size` types of this corpus; when
 * `size` is omitted it defaults to the matching Dolch budget's size, which is what makes
 * the two comparable at the same |V| (FR-601/FR-602).
 */
export function buildVocab(
  source: string,
  budget: string,
  corpusText: string,
  size?: number,
): LexVocab {
  if (!(BUDGET_SOURCES as readonly string[]).includes(source)) {
    throw new Error(`unknown budget source ${JSON.stringify(source)}; expected ${BUDGET_SOURCES.join(", ")}`);
  }
  if (source === "dolch") {
    if (!isDolchBudgetName(budget)) {
      throw new Error(`unknown Dolch budget ${JSON.stringify(budget)}; expected ${DOLCH_ORDER.join(", ")}`);
    }
    return new LexVocab(dolchBudget(budget), "dolch", budget);
  }
  const n = size ?? dolchBudget(isDolchBudgetName(budget) ? budget : "full").length;
  return new LexVocab(frequencyBudget(corpusText, n), "frequency", `top${n}`);
}
