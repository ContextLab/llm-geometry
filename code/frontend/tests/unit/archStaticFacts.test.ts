/**
 * Two things the Architecture Explorer's static half can get wrong SILENTLY.
 *
 * 1. **The ONNX table lookup.** `ONNX_REPOS[modelId]` on a plain object literal answers
 *    with an inherited property for `constructor`, `toString`, `valueOf`, `__proto__` …
 *    Every one of those is truthy, so the `if (!repo)` guard passes it through and the
 *    ONNX runtime is handed the `Object` FUNCTION as a repository id. Nothing throws at
 *    the lookup; the failure surfaces somewhere else entirely, or not at all. A model id
 *    reaches this from the URL hash, so it is user input.
 *
 * 2. **The numbers the Info tab prints about the static build.** They are prose, and prose
 *    does not fail when a constant moves — that is how a sentence went on asserting a
 *    measurement of a configuration that had been rewritten. The Info tab's figures are
 *    therefore read out of the component and compared with the constants they describe.
 *    (`tests/e2e/docs.spec.ts` checks the RENDERED page for the same facts; this file
 *    catches the same drift in unit CI, where the earlier mutation of
 *    `VACANCY_MIN_POOLED_PRESERVED` from 700 to 7 survived.)
 */

import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";

import {
  VACANCY_MIN_POOLED_PRESERVED,
  VACANCY_PRE_REWRITE_Q8,
  onnxRepo,
  onnxRepoIds,
} from "../../src/lib/staticClient/arch";

/** tests/unit -> tests -> frontend. */
const FRONTEND = path.resolve(__dirname, "../..");
const INFO_TAB = path.join(FRONTEND, "src/viz/info/InfoTab.svelte");

describe("the ONNX export table is a lookup, not a prototype chain", () => {
  it("answers only for the curated ids it actually carries", () => {
    for (const id of onnxRepoIds()) {
      expect(onnxRepo(id), id).toMatch(/^onnx-community\//);
    }
    expect(onnxRepoIds()).toContain("gpt2");
    expect(onnxRepo("gpt2")).toBe("onnx-community/gpt2-ONNX");
  });

  it("returns undefined for inherited properties instead of an Object", () => {
    // Each of these is truthy on a plain `{}` and would sail past `if (!repo)`.
    for (const key of [
      "constructor",
      "toString",
      "valueOf",
      "hasOwnProperty",
      "__proto__",
      "__defineGetter__",
      "isPrototypeOf",
      "propertyIsEnumerable",
    ]) {
      const repo = onnxRepo(key);
      expect(repo, key).toBeUndefined();
      expect(typeof repo, key).not.toBe("function");
    }
    expect(onnxRepo("not/a-model")).toBeUndefined();
    // …and the same for a plain missing id, which is the ordinary case.
    expect(onnxRepoIds()).not.toContain("constructor");
  });
});

describe("the Info tab's static-build figures are the constants they describe", () => {
  const source = fs.readFileSync(INFO_TAB, "utf-8");

  it("prints the pooled floor the static client enforces", () => {
    const stated = source.match(/A pool below (\d[\d,]*) preserved tokens/);
    expect(stated, "the Info tab no longer states the pooled floor").not.toBeNull();
    expect(Number(stated![1].replace(/,/g, ""))).toBe(VACANCY_MIN_POOLED_PRESERVED);
  });

  it("prints the three q8-versus-float32 gaps the ±0.2 bound was set from", () => {
    const stated = source.match(
      /([\d.]+) nats in the six-passage study that preceded the feature, and ([\d.]+)\s+and ([\d.]+) in this build's own two-stack comparison/,
    );
    expect(stated, "the Info tab no longer states where ±0.2 came from").not.toBeNull();
    const [, study, ownWrongContent, ownTotal] = stated!;
    expect(Number(study)).toBe(VACANCY_PRE_REWRITE_Q8.study.pooledBoundNats);
    // The other two are DIFFERENCES of the retained pair, so they are checked as
    // arithmetic on it rather than as two more literals.
    const gap = (pair: { fp32: number; q8: number }) => Number((pair.fp32 - pair.q8).toFixed(3));
    expect(Number(ownWrongContent)).toBe(gap(VACANCY_PRE_REWRITE_Q8.wrongContent));
    expect(Number(ownTotal)).toBe(gap(VACANCY_PRE_REWRITE_Q8.total));
  });
});
