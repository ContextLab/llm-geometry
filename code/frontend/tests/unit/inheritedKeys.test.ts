/**
 * Lookups that must not answer with something off `Object.prototype`.
 *
 * This defect class has now produced instances in three successive rounds — two, then four,
 * then three more — so "we found them all" is a hypothesis. What they share: a plain object
 * used as a lookup table, read with `[key]`, `in`, or a truthiness test, where the key comes
 * from outside. Every inherited property then answers as if it were data, and the caller
 * gets a plausible value instead of the refusal it asked for.
 *
 * The instance pinned here is `meterScore`. `METER_FEET` was an object literal, so
 * `METER_FEET["constructor"]` was `Object` — not `undefined` — and the `pat === undefined`
 * guard did not fire. `meterScore(line, "constructor")` then scanned the line against
 * `String(Object)` as the target pattern and returned **`0`**: a number, from a foot that
 * does not exist, with nothing thrown. `"bogus"` threw correctly, which is exactly why a
 * one-example test would have passed.
 *
 * Python's mirror was never wrong — `foot not in FEET` on a dict is a real key test — so
 * this was also a silent TS↔Python divergence, and `test_lex_vacancy.py` asserts the same
 * keys raise there.
 *
 * The keys below are the class, not the example: every property `Object.prototype` actually
 * carries, plus `__proto__`, whose getter makes it the strangest of them.
 */

import { describe, expect, it } from "vitest";

import { METER_FEET, meterScore } from "../../src/lib/lexEngine/vacancy";

/** Everything a plain `{}` inherits, and the accessor that is not a plain property. */
const INHERITED_KEYS = [
  "constructor",
  "toString",
  "toLocaleString",
  "valueOf",
  "hasOwnProperty",
  "isPrototypeOf",
  "propertyIsEnumerable",
  "__proto__",
  "__defineGetter__",
  "__defineSetter__",
  "__lookupGetter__",
  "__lookupSetter__",
];

describe("meterScore refuses every key it does not own", () => {
  it("still answers the feet it does own", () => {
    // The guard must refuse more, not everything: §6.4's four feet still work.
    expect(Object.keys(METER_FEET).sort()).toEqual(["anapest", "dactyl", "iamb", "trochee"]);
    for (const foot of Object.keys(METER_FEET)) {
      expect(meterScore("the cat sat on the mat", foot)).toBeTypeOf("number");
    }
    // …and the default is unchanged, which is what every shipped statistic uses.
    expect(meterScore("the cat sat on the mat")).toBe(meterScore("the cat sat on the mat", "anapest"));
  });

  it("throws for an inherited key exactly as it does for a nonsense one", () => {
    expect(() => meterScore("the cat sat", "bogus")).toThrow(/unknown foot/);
    for (const key of INHERITED_KEYS) {
      expect(() => meterScore("the cat sat", key), key).toThrow(/unknown foot/);
    }
  });

  it("never returns a number for an inherited key — the actual symptom", () => {
    // The shipped defect did not throw; it returned 0. A test that only asserts "throws"
    // would pass on a fallback that silently substituted the default foot, so the return is
    // asserted too.
    for (const key of INHERITED_KEYS) {
      let value: number | string = "threw";
      try {
        value = meterScore("the cat sat on the mat", key);
      } catch {
        /* expected */
      }
      expect(value, `meterScore(..., ${JSON.stringify(key)}) returned ${String(value)}`).toBe(
        "threw",
      );
    }
  });

  it("carries no prototype at all, so the guard is not the only lock", () => {
    expect(Object.getPrototypeOf(METER_FEET)).toBeNull();
    for (const key of INHERITED_KEYS) {
      expect((METER_FEET as Record<string, unknown>)[key], key).toBeUndefined();
    }
    // Frozen: a table read from many places must not be extended from one of them.
    expect(Object.isFrozen(METER_FEET)).toBe(true);
  });
});
