/**
 * The word-alphabet class, and the Unicode-version skew that used to split the stacks.
 *
 * The browser half of `code/backend/tests/unit/test_arch_word_classes.py`. Both suites read
 * the same two files out of `specs/007-vacancy-transform-field/` — the pinned class table
 * and the shared case table — so "the two stacks agree" is asserted against one artifact
 * rather than against two prose lists that can drift apart.
 *
 * What shipped, and is pinned here:
 *
 *  - `\p{Pc}` (connector punctuation) was missing from BOTH stacks, so `don‿t` scored
 *    HTTP 200 and swapped to `warm‿t` — character for character the `don’t` → `big’t`
 *    defect the joiner class had been introduced to close;
 *  - this engine's Unicode is 16.0 and the backend's is 13.0, and the two disagreed about
 *    9 993 letters and marks and 11 joiners, U+0890 among them (a real `Cf`). One stack
 *    refused a passage the other scored;
 *  - `legs--upon` was refused although it is written entirely in `WORD_RE`'s own alphabet,
 *    so the refusal's own advice could not be followed.
 */

import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";

import { fragmentedWords, checkWordAlphabet } from "../../src/lib/staticClient/byteSpans";
import {
  JOINER_CLASS,
  LETTER_CLASS,
  MARK_CLASS,
  PINNED_UNICODE_VERSION,
} from "../../src/lib/staticClient/wordClasses";
import { WORD_RE } from "../../src/lib/lexEngine/vocab";

/** tests/unit -> tests -> frontend -> code -> repo root. */
const REPO_ROOT = path.resolve(__dirname, "../../../..");
const SPEC_DIR = path.join(REPO_ROOT, "specs/007-vacancy-transform-field");
const NORMATIVE_TABLE = path.join(SPEC_DIR, "word-classes.json");
const BROWSER_TABLE = path.join(REPO_ROOT, "code/frontend/src/lib/staticClient/wordClasses.json");
const CASES_FILE = path.join(SPEC_DIR, "word-alphabet-cases.json");

type ClassSpec = { categories: string[]; ranges: string; named?: Record<string, string> };
const table = JSON.parse(fs.readFileSync(NORMATIVE_TABLE, "utf-8")) as {
  format: string;
  unicodeVersion: string;
  classes: Record<string, ClassSpec>;
};

function members(name: string): number[] {
  const cls = table.classes[name];
  const cps = new Set<number>();
  for (const part of cls.ranges.split(",")) {
    if (part === "") continue;
    const dash = part.indexOf("-");
    const lo = parseInt(dash === -1 ? part : part.slice(0, dash), 16);
    const hi = dash === -1 ? lo : parseInt(part.slice(dash + 1), 16);
    for (let cp = lo; cp <= hi; cp++) cps.add(cp);
  }
  for (const hex of Object.keys(cls.named ?? {})) cps.add(parseInt(hex, 16));
  return [...cps].sort((a, b) => a - b);
}

/** The two joiners `WORD_RE` itself accepts: `don'` + them + `t` is one match, not a fragment. */
const WORD_RE_JOINERS = new Set(["'", "-"]);

describe("the pinned Unicode word classes", () => {
  it("carries a copy byte-identical to the normative one", () => {
    // The backend's copy is checked against the same file by `test_arch_word_classes.py`.
    // If either drifts the two stacks are back to agreeing by coincidence.
    // Both files are written by one script in UTF-8, and the table is pure ASCII, so a
    // UTF-8 read is a lossless view of the bytes.
    expect(fs.readFileSync(BROWSER_TABLE, "utf-8")).toBe(fs.readFileSync(NORMATIVE_TABLE, "utf-8"));
    expect(table.format).toBe("word-classes-v1");
    expect(PINNED_UNICODE_VERSION).toBe(table.unicodeVersion);
    expect(table.classes.joiner.categories).toEqual(["Pd", "Cf", "Pc"]);
  });

  it("builds its character classes from the table, not from \\p{...}", () => {
    // Every member of the table has to be reachable through the regex fragment the run
    // scanner is built from. A revert to `\p{Pd}\p{Cf}` drops all ten Pc code points here,
    // and on a Node older than the pin it drops the newer Pd and Cf ones too.
    for (const [name, klass] of [
      ["letter", LETTER_CLASS],
      ["mark", MARK_CLASS],
      ["joiner", JOINER_CLASS],
    ] as const) {
      const re = new RegExp(`^[${klass}]$`, "u");
      const missed = members(name).filter((cp) => !re.test(String.fromCodePoint(cp)));
      expect(missed.map((cp) => cp.toString(16)), `${name} members outside its class`).toEqual([]);
    }
    expect(new RegExp(`^[${JOINER_CLASS}]$`, "u").test(" ")).toBe(false);
    expect(new RegExp(`^[${LETTER_CLASS}]$`, "u").test("-")).toBe(false);
  });

  it("emits the table's ranges and nothing else — no \\p{...} can satisfy this", () => {
    // The membership check above cannot distinguish the table from `\p{Pd}\p{Cf}\p{Pc}`
    // whenever THIS engine's Unicode happens to equal the pin, which is exactly the
    // coincidence the table exists to stop relying on. So the emitted class is parsed back
    // and compared to the file range for range: a `\p{...}` escape does not parse as one.
    const emitted = [...JOINER_CLASS.matchAll(/\\u\{([0-9a-f]+)\}(?:-\\u\{([0-9a-f]+)\})?/g)].map(
      (m) => [parseInt(m[1], 16), parseInt(m[2] ?? m[1], 16)] as [number, number],
    );
    // Nothing in the class body that is not one of those escapes.
    expect(
      JOINER_CLASS.replace(/\\u\{[0-9a-f]+\}(?:-\\u\{[0-9a-f]+\})?/g, ""),
      "JOINER_CLASS contains something other than \\u{...} escapes",
    ).toBe("");
    const covered = new Set<number>();
    for (const [lo, hi] of emitted) for (let cp = lo; cp <= hi; cp++) covered.add(cp);
    expect([...covered].sort((a, b) => a - b)).toEqual(members("joiner"));
  });

  it("treats every joiner in the class as binding two letters", () => {
    const joiners = members("joiner");
    expect(joiners.length).toBeGreaterThan(200);
    const unflagged = joiners.filter((cp) => {
      const ch = String.fromCodePoint(cp);
      if (WORD_RE_JOINERS.has(ch)) return false;
      return fragmentedWords(`don${ch}t`, WORD_RE).join("") !== `don${ch}t`;
    });
    expect(unflagged.map((cp) => `U+${cp.toString(16).toUpperCase()}`)).toEqual([]);
  });

  it("includes connector punctuation, whose absence was a live wrong answer", () => {
    // `don‿t` scored HTTP 200 in the full stack and swapped to `warm‿t`.
    for (const ch of "_‿⁀⁔︳︴﹍﹎﹏＿") {
      expect(fragmentedWords(`don${ch}t`, WORD_RE), `U+${ch.codePointAt(0)!.toString(16)}`).toEqual([
        `don${ch}t`,
      ]);
      expect(() => checkWordAlphabet(`the cat don${ch}t sit`, WORD_RE)).toThrow(/word alphabet/);
    }
  });

  it("does not refuse a run written entirely in WORD_RE's own alphabet", () => {
    // The Gutenberg em-dash convention, in this project's own corpus. The refusal told the
    // reader to use straight ASCII hyphens, which is what they had done.
    for (const text of ["legs--upon", "ba--are", "hea--art", "Lady--loves", "don''t", "a---b"]) {
      expect(fragmentedWords(text, WORD_RE), text).toEqual([]);
      expect(() => checkWordAlphabet(text, WORD_RE)).not.toThrow();
    }
    // …and the escape hatch is ASCII-only: repeated INVISIBLE joiners still refuse.
    expect(fragmentedWords("co­­operate", WORD_RE)).toEqual(["co­­operate"]);
    expect(fragmentedWords("don’’t", WORD_RE)).toEqual(["don’’t"]);
    expect(fragmentedWords("don’t", WORD_RE)).toEqual(["don’t"]);
  });

  it("answers the shared case table exactly, as the backend does", () => {
    const cases = JSON.parse(fs.readFileSync(CASES_FILE, "utf-8")) as {
      format: string;
      cases: { text: string; fragmented: string[]; why: string }[];
    };
    expect(cases.format).toBe("word-alphabet-cases-v1");
    expect(cases.cases.length).toBeGreaterThanOrEqual(25);
    for (const c of cases.cases) {
      expect(fragmentedWords(c.text, WORD_RE), `${JSON.stringify(c.text)}: ${c.why}`).toEqual(
        c.fragmented,
      );
    }
  });
});
