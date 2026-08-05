/**
 * The PINNED Unicode word classes both stacks classify code points from.
 *
 * `WORDLIKE_RE` (`byteSpans.ts`) and `wordlike_runs` (`llm_geometry/arch/vacancy_score.py`)
 * implement one grammar over three classes — letter, combining mark, joiner. Each used to
 * read those classes out of its own runtime: `\p{L}`/`\p{M}`/`\p{Pd}` here, `unicodedata`
 * there.
 *
 * That is agreement by coincidence, and the coincidence does not hold. Python 3.10 — the
 * version CI pins — carries Unicode 13.0; Node 22 carries Unicode 16.0. Measured across the
 * whole code space, the two disagree about **9 993 letters and marks and 11 joiners**, every
 * one a character the newer table knows and the older does not. Two of the eleven are U+0890
 * and U+0891, which are genuinely `Cf` — the joiner category both stacks declare — so
 * `don\u{890}t` was a wordlike run here (refused) and two separate words in Python (scored,
 * and rewritten as `warm\u{890}t`). A silent wrong answer produced by nothing worse than the
 * two stacks being built at different times.
 *
 * So neither stack asks its runtime any more. `wordClasses.json` is a committed enumeration
 * of the three classes at one pinned Unicode version; `llm_geometry/arch/data/` carries a
 * byte-identical copy; both are compared to the normative copy in
 * `specs/007-vacancy-transform-field/` by a test in each suite. Regenerate all three
 * together with `node scripts/export_word_classes.mjs`.
 *
 * This table is not an optimization and must not be replaced by `\p{...}` that "means the
 * same thing" — it is the only reason the two stacks mean the same thing.
 */

import table from "./wordClasses.json";

type WordClass = { categories: string[]; ranges: string; named?: Record<string, string> };

if (table.format !== "word-classes-v1") {
  throw new Error(`wordClasses.json: unknown format ${String(table.format)}`);
}

/** The Unicode version the committed table was enumerated at. */
export const PINNED_UNICODE_VERSION: string = table.unicodeVersion;

/** `"41-5a,61"` -> `[[0x41, 0x5a], [0x61, 0x61]]`, ascending and non-overlapping. */
function parseRanges(spec: string): [number, number][] {
  const out: [number, number][] = [];
  for (const part of spec.split(",")) {
    if (part === "") continue;
    const dash = part.indexOf("-");
    const lo = parseInt(dash === -1 ? part : part.slice(0, dash), 16);
    const hi = dash === -1 ? lo : parseInt(part.slice(dash + 1), 16);
    const last = out[out.length - 1];
    if (!(hi >= lo) || (last && lo <= last[1])) {
      throw new Error(`wordClasses.json: ranges must ascend and not overlap, got "${part}"`);
    }
    out.push([lo, hi]);
  }
  return out;
}

function classRanges(name: "letter" | "mark" | "joiner"): [number, number][] {
  // `Object.hasOwn`, not truthiness: `table` is parsed JSON, and a lookup table read with a
  // bare `[key]` answers with `Object.prototype`'s members. `name` is a compile-time union
  // today, so this is prevention rather than a fix — it is the shape the same defect took in
  // four other places this month.
  const classes = table.classes as Record<string, WordClass>;
  if (!Object.hasOwn(classes, name)) throw new Error(`wordClasses.json: no class "${name}"`);
  const cls = classes[name];
  const ranges = parseRanges(cls.ranges);
  for (const hex of Object.keys(cls.named ?? {})) {
    const cp = parseInt(hex, 16);
    if (!ranges.some(([lo, hi]) => cp >= lo && cp <= hi)) ranges.push([cp, cp]);
  }
  return ranges.sort((a, b) => a[0] - b[0]);
}

/**
 * The body of a regex character class matching exactly this word class — `\u{...}` escapes
 * throughout, so the result is safe inside `[...]` and needs the `u` flag.
 */
function charClass(name: "letter" | "mark" | "joiner"): string {
  return classRanges(name)
    .map(([lo, hi]) =>
      lo === hi ? `\\u{${lo.toString(16)}}` : `\\u{${lo.toString(16)}}-\\u{${hi.toString(16)}}`,
    )
    .join("");
}

/** Character-class bodies, built once. Use inside `[...]` with the `u` flag. */
export const LETTER_CLASS = charClass("letter");
export const MARK_CLASS = charClass("mark");
export const JOINER_CLASS = charClass("joiner");
