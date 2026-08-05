/**
 * Regenerate the PINNED Unicode word-class table shared by both stacks.
 *
 *     node scripts/export_word_classes.mjs
 *
 * Writes three byte-identical copies of one table:
 *
 *   specs/007-vacancy-transform-field/word-classes.json   the normative copy
 *   code/backend/src/llm_geometry/arch/data/word-classes.json
 *   code/frontend/src/lib/staticClient/wordClasses.json
 *
 * WHY THIS EXISTS. `wordlike_runs` (Python) and `WORDLIKE_RE` (TypeScript) implement the
 * same grammar over the same three classes — letter, combining mark, joiner — and used to
 * read those classes out of whichever Unicode tables their own runtime happened to carry.
 * Python 3.10's `unicodedata` is Unicode 13.0; Node 22's regex engine is Unicode 16.0. The
 * two therefore disagreed about 9 993 letters and marks and 11 joiners — including U+0890
 * and U+0891, which really are `Cf`, the backend's own declared joiner category, and which
 * the backend accepted while the browser refused. A passage can be scored by one stack and
 * refused by the other, and the stack that scores it rewrites a fragment.
 *
 * A committed table removes the runtime from the question: both stacks classify a code
 * point by looking it up here, so they agree by construction rather than by both runtimes
 * happening to ship the same ICU. Upgrading the pin is a deliberate edit — rerun this
 * script, and both stacks move together in one commit.
 *
 * The generator reads Node's `\p{...}` because that is the newest table available in this
 * repo's toolchain; `unicodeVersion` records which one it read, and a test in each stack
 * asserts its copy is byte-identical to the normative one.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** Code points matching `\p{<cat>}` in this Node's regex engine. */
function codePoints(categories) {
  const res = categories.map((c) => new RegExp(`\\p{${c}}`, "u"));
  const out = [];
  for (let cp = 0; cp <= 0x10ffff; cp++) {
    if (cp >= 0xd800 && cp <= 0xdfff) continue; // lone surrogates are not characters
    const ch = String.fromCodePoint(cp);
    if (res.some((re) => re.test(ch))) out.push(cp);
  }
  return out;
}

/** `"41-5a,61-7a"` — ascending, non-overlapping, lowercase hex, inclusive bounds. */
function compact(cps) {
  const ranges = [];
  for (const cp of cps) {
    const last = ranges[ranges.length - 1];
    if (last && cp === last[1] + 1) last[1] = cp;
    else ranges.push([cp, cp]);
  }
  return ranges.map(([a, b]) => (a === b ? a.toString(16) : `${a.toString(16)}-${b.toString(16)}`)).join(",");
}

/**
 * Joiners that carry no Unicode property separating them from ordinary quotation marks or
 * accents, so they cannot come from a category and are named. Every one of them BINDS two
 * letters into a single written word in some orthography: the apostrophes of English and
 * Armenian, the Catalan `l·l` point, the Hebrew geresh, the Greek koronis, the Japanese
 * middle dot, the Cyrillic kavyka. `'` (U+0027) is one WORD_RE itself accepts and is
 * listed so the class is complete rather than "the ones WORD_RE misses".
 */
const NAMED = [
  ["0027", "apostrophe — the one WORD_RE accepts, listed so the class is complete"],
  ["00b4", "acute accent, typed as an apostrophe on many keyboard layouts"],
  ["00b7", "middle dot — Catalan l·l"],
  ["02b9", "modifier letter prime"],
  ["02bc", "modifier letter apostrophe — the Unicode-recommended word-internal one"],
  ["0375", "Greek lower numeral sign"],
  ["055a", "Armenian apostrophe"],
  ["05f3", "Hebrew punctuation geresh"],
  ["1fbd", "Greek koronis"],
  ["1fbf", "Greek psili"],
  ["2018", "left single quotation mark"],
  ["2019", "right single quotation mark — the default apostrophe of pasted text"],
  ["2027", "hyphenation point"],
  ["2032", "prime"],
  ["2035", "reversed prime"],
  ["2212", "minus sign — category Sm rather than Pd"],
  ["30fb", "katakana middle dot"],
  ["a67e", "Cyrillic kavyka"],
  ["ff07", "fullwidth apostrophe"],
  ["ff65", "halfwidth katakana middle dot"],
];

const table = {
  format: "word-classes-v1",
  contract: "specs/007-vacancy-transform-field/architecture.md",
  command: "node scripts/export_word_classes.mjs",
  unicodeVersion: process.versions.unicode,
  note:
    "PINNED. Both stacks classify code points from this table instead of from their own " +
    "runtime's Unicode data, so `wordlike_runs` (Python) and `WORDLIKE_RE` (TypeScript) " +
    "cannot disagree because one runtime is newer than the other. Ranges are inclusive, " +
    "ascending, non-overlapping, lowercase hex.",
  classes: {
    letter: { categories: ["L"], ranges: compact(codePoints(["L"])) },
    mark: { categories: ["M"], ranges: compact(codePoints(["M"])) },
    joiner: {
      categories: ["Pd", "Cf", "Pc"],
      ranges: compact(codePoints(["Pd", "Cf", "Pc"])),
      named: Object.fromEntries(NAMED),
    },
  },
};

const json = `${JSON.stringify(table, null, 1)}\n`;
for (const rel of [
  "specs/007-vacancy-transform-field/word-classes.json",
  "code/backend/src/llm_geometry/arch/data/word-classes.json",
  "code/frontend/src/lib/staticClient/wordClasses.json",
]) {
  const dest = path.join(ROOT, rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, json);
  console.log(`wrote ${rel} (${json.length} bytes)`);
}
console.log(`Unicode ${table.unicodeVersion} via node ${process.version}`);
