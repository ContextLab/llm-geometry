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

/**
 * The `reveal first` box: the same control, the same rule, and the box the F4 fix missed.
 *
 * It read `Math.trunc(Number(value))` and then `Math.max(1, v)`, which is three silent
 * rewrites in one line — `2.5` applied as 2, `1e3` applied as 1000 under `max="99"`, and
 * `0` (or anything the number input sanitizes away) applied as 1. `reveal_after` is a
 * boundary in the vacancy map, so a substituted value produces a different corpus, a
 * different `vacated_sha256` and a different loss curve, with the box still showing what
 * was typed. Round 3 fixed the seed box beside it and left this one.
 */
const REVEAL_IN_USE = 4;

interface MountedReveal {
  reveals: number[];
  input: HTMLInputElement;
  error: () => HTMLElement | null;
  dispose: () => void;
}

function mountRevealPanel(): MountedReveal {
  const params = vacancyParams({ p: 0.5, seed: SEED_IN_USE, revealAfter: REVEAL_IN_USE });
  const map = buildVacancyMap(vacancyDomain(tokenize(TEXT)), vacancyParams({ seed: SEED_IN_USE }));
  const vacated = vacateText(TEXT, map, params);
  const words = [...new Set(tokenize(TEXT))].sort();
  const vocab = new LexVocab(words, "frequency", "full");

  const target = document.createElement("div");
  document.body.appendChild(target);
  const reveals: number[] = [];
  const app = mount(VacancyPanel, {
    target,
    props: {
      corpusText: TEXT,
      vacatedText: vacated,
      map,
      params,
      baseVocab: vocab,
      vocab,
      condition: "reveal",
      revealAfter: REVEAL_IN_USE,
      mint: "nonce",
      refusal: "",
      onP: () => {},
      onSeed: () => {},
      onCondition: () => {},
      onRevealAfter: (v: number) => reveals.push(v),
      onProsody: () => {},
      onMint: () => {},
    },
  });
  flushSync();
  const input = target.querySelector<HTMLInputElement>('[data-testid="lex-vacancy-reveal"]');
  if (input === null) throw new Error("the reveal input did not render");
  return {
    reveals,
    input,
    error: () => target.querySelector<HTMLElement>('[data-testid="lex-vacancy-reveal-error"]'),
    dispose: () => {
      unmount(app);
      target.remove();
    },
  };
}

function typeReveal(m: MountedReveal, raw: string): void {
  m.input.value = raw;
  m.input.dispatchEvent(new Event("input", { bubbles: true }));
  flushSync();
}

describe("the reveal box never applies a number other than the one typed", () => {
  it.each([
    ["a fraction, which was truncated", "2.5"],
    ["exponent notation, which sailed past max", "1e3"],
    ["hexadecimal", "0x10"],
    ["a leading plus", "+2"],
    ["surrounding whitespace around a valid value", " 2 "],
    ["a negative number, which became 1", "-5"],
    ["zero, which became 1", "0"],
    ["one above the declared maximum", "100"],
    ["the empty box, which became 1", ""],
    ["not a number at all", "abc"],
  ])("refuses %s rather than rewriting it", (_what, raw) => {
    const m = mountRevealPanel();
    try {
      typeReveal(m, raw);
      expect(m.reveals).toEqual([]);
      expect(m.error()).not.toBeNull();
      expect(m.error()?.textContent ?? "").toContain(String(REVEAL_IN_USE));
    } finally {
      m.dispose();
    }
  });

  it("accepts its own declared bounds and everything between them", () => {
    const m = mountRevealPanel();
    try {
      const max = Number(m.input.getAttribute("max"));
      const min = Number(m.input.getAttribute("min"));
      expect(Number.isInteger(max) && max > 0).toBe(true);
      typeReveal(m, String(min));
      typeReveal(m, String(max));
      typeReveal(m, "7");
      expect(m.reveals).toEqual([min, max, 7]);
      expect(m.error()).toBeNull();
    } finally {
      m.dispose();
    }
  });

  it("clears the refusal once a usable number is typed", () => {
    const m = mountRevealPanel();
    try {
      typeReveal(m, "2.5");
      expect(m.error()).not.toBeNull();
      typeReveal(m, "3");
      expect(m.reveals).toEqual([3]);
      expect(m.error()).toBeNull();
    } finally {
      m.dispose();
    }
  });
});
