/**
 * The Lexicon Lab's seed box: what it says it accepts, and what it does with the rest.
 *
 * Red-team finding F4 (`notes/agent-reports/redteam-007-lex.md`): the control declared
 * `max="9999"`, accepted anything, and rewrote what it could not represent —
 * `9007199254740993` was applied as `9007199254740992`, with no error anywhere. Two
 * separate lies in one control: a declared range that is not the accepted range, and a
 * seed used that is not the seed asked for.
 *
 * These tests drive the REAL `VacancyPanel` over the REAL transform (a real map built by
 * `buildVacancyMap`, a real vacated text) in jsdom, and read the answer off the DOM and off
 * the `onSeed` callback — the two things a reader and the tab actually see.
 */
import { describe, expect, it } from "vitest";
import { flushSync, mount, unmount } from "svelte";

import VacancyPanel from "../../src/viz/lex/VacancyPanel.svelte";
import { LexVocab } from "../../src/lib/lexEngine";
import {
  buildVacancyMap,
  vacancyDomain,
  vacancyParams,
  vacateText,
} from "../../src/lib/lexEngine/vacancy";
import { tokenize } from "../../src/lib/lexEngine";

const TEXT = [
  "The little brown squirrel ate the pretty acorn.",
  "The squirrel ran away and the children sang loudly today.",
  "Little Bo-Peep has lost her sheep and cannot tell where to find them.",
].join("\n");

const SEED_IN_USE = 3;

interface Mounted {
  root: HTMLElement;
  seeds: number[];
  input: HTMLInputElement;
  error: () => HTMLElement | null;
  dispose: () => void;
}

function mountPanel(): Mounted {
  const params = vacancyParams({ p: 0.5, seed: SEED_IN_USE });
  const map = buildVacancyMap(vacancyDomain(tokenize(TEXT)), vacancyParams({ seed: SEED_IN_USE }));
  const vacated = vacateText(TEXT, map, params);
  const words = [...new Set(tokenize(TEXT))].sort();
  const vocab = new LexVocab(words, "frequency", "full");

  const target = document.createElement("div");
  document.body.appendChild(target);
  const seeds: number[] = [];
  const app = mount(VacancyPanel, {
    target,
    props: {
      corpusText: TEXT,
      vacatedText: vacated,
      map,
      params,
      baseVocab: vocab,
      vocab,
      condition: "consistent",
      revealAfter: 1,
      mint: "nonce",
      refusal: "",
      onP: () => {},
      onSeed: (v: number) => seeds.push(v),
      onCondition: () => {},
      onRevealAfter: () => {},
      onProsody: () => {},
      onMint: () => {},
    },
  });
  flushSync();
  const input = target.querySelector<HTMLInputElement>('[data-testid="lex-vacancy-seed"]');
  if (input === null) throw new Error("the seed input did not render");
  return {
    root: target,
    seeds,
    input,
    error: () => target.querySelector<HTMLElement>('[data-testid="lex-vacancy-seed-error"]'),
    dispose: () => {
      unmount(app);
      target.remove();
    },
  };
}

/** Type a value the way a person does: set it, then let the control hear about it. */
function type(m: Mounted, raw: string): void {
  m.input.value = raw;
  m.input.dispatchEvent(new Event("input", { bubbles: true }));
  flushSync();
}

describe("the seed box's declared range is its accepted range", () => {
  it("accepts its own declared maximum", () => {
    const m = mountPanel();
    try {
      const max = m.input.getAttribute("max");
      expect(max).not.toBeNull();
      expect(/^\d+$/.test(max!)).toBe(true);
      type(m, max!);
      expect(m.seeds).toEqual([Number(max)]);
      expect(m.error()).toBeNull();
    } finally {
      m.dispose();
    }
  });

  it("refuses one above its declared maximum instead of accepting it", () => {
    const m = mountPanel();
    try {
      const max = BigInt(m.input.getAttribute("max")!);
      type(m, String(max + 1n));
      // F4: `max="9999"` with `10000` accepted — the declared range meant nothing.
      expect(m.seeds).toEqual([]);
      expect(m.error()?.textContent ?? "").toContain("still in use");
    } finally {
      m.dispose();
    }
  });
});

describe("the seed box never applies a seed other than the one typed", () => {
  it("refuses 2^53 + 1 rather than silently applying 2^53", () => {
    const m = mountPanel();
    try {
      // Verbatim from F4: typing this left the field reading 9007199254740992.
      type(m, "9007199254740993");
      expect(m.seeds).toEqual([]);
      const message = m.error()?.textContent ?? "";
      expect(message).toContain("9,007,199,254,740,991");
      expect(message).toContain(`seed ${SEED_IN_USE} is still in use`);
    } finally {
      m.dispose();
    }
  });

  it.each([
    ["a negative seed", "-5"],
    ["a fractional seed", "3.5"],
    ["exponent notation", "1e3"],
    ["hexadecimal", "0x10"],
  ])("refuses %s rather than rewriting it", (_what, raw) => {
    const m = mountPanel();
    try {
      type(m, raw);
      expect(m.seeds).toEqual([]);
      expect(m.error()).not.toBeNull();
    } finally {
      m.dispose();
    }
  });

  it("applies an ordinary seed and clears the error it showed before", () => {
    const m = mountPanel();
    try {
      type(m, "999999999999");
      expect(m.seeds).toEqual([999999999999]);
      expect(m.error()).toBeNull();
      type(m, "-1");
      expect(m.error()).not.toBeNull();
      type(m, "7");
      expect(m.seeds).toEqual([999999999999, 7]);
      expect(m.error()).toBeNull();
    } finally {
      m.dispose();
    }
  });
});
