/**
 * What the vacancy panel is allowed to SAY about the numbers it was handed
 * (`src/viz/arch/vacancyVerdict.ts`), asserted with the numbers that were actually
 * observed on the running app — not invented ones.
 *
 * Every payload below is transcribed from the red team's 2026-08-04 run:
 *
 *   - full stack, gpt2, p = 0.5, seed = 4: `unknown_form = -0.3548 ± 0.1359` over 22
 *     paired tokens, `wrong_content = 0.871`, `total = 0.516`. The panel rendered
 *     `-0.355 nats ± 0.136`, called it an "upper bound", printed `-69% of the total
 *     damage`, and closed with the finding stated as fact.
 *   - deployed static site, gpt2, defaults: `total = 0.879 ± 0.074` with
 *     `quantizationUncertaintyNats = 0.2` attached and ignored. The stated interval
 *     [0.805, 0.953] excludes the float32 value 0.9892 the backend measures.
 */

import { describe, expect, it } from "vitest";

import type { ArchVacancyDifference } from "../../src/lib/dataClient";
import {
  errorBarTerms,
  formSharePercent,
  secondaryLine,
  upperBoundLabel,
  verdictKind,
} from "../../src/viz/arch/vacancyVerdict";
import { tokenTip } from "../../src/viz/arch/archShared";

function diff(over: Partial<ArchVacancyDifference>): ArchVacancyDifference {
  return {
    id: "unknown_form",
    label: "the cost of unknown form",
    expr: "nll(nonce) − nll(swap)",
    headline: true,
    upperBound: true,
    nats: 0.111,
    se: 0.02,
    nPairs: 847,
    ...over,
  };
}

// gpt2, p = 0.5, seed = 4, the pasted "The cat sat on the mat…" passage.
const BACKWARDS = diff({ nats: -0.3548, se: 0.1359, nPairs: 22 });
const WRONG_CONTENT_THERE = diff({ id: "wrong_content", nats: 0.871, se: 0.2, upperBound: false });
const TOTAL_THERE = diff({ id: "total", nats: 0.5162, se: 0.2, headline: false, upperBound: false });

describe("F3 — a negative cost is never promoted to the conclusion", () => {
  it("classifies the observed -0.355 ± 0.136 as backwards, not as a result", () => {
    // It passes the panel's own |nats| > 2·se test — that is exactly why the magnitude
    // test alone was the bug.
    expect(Math.abs(BACKWARDS.nats!) > 2 * BACKWARDS.se!).toBe(true);
    expect(verdictKind(BACKWARDS)).toBe("backwards");
  });

  it("still calls a genuinely unresolved effect unresolved, and a positive one a result", () => {
    expect(verdictKind(diff({ nats: 0.05, se: 0.09 }))).toBe("unresolved");
    expect(verdictKind(diff({ nats: -0.05, se: 0.09 }))).toBe("unresolved");
    expect(verdictKind(diff({ nats: 0.273, se: 0.03 }))).toBe("conclusion");
  });

  it("calls p = 0 an identity rather than an unresolvable measurement", () => {
    // Every difference is exactly 0 there, so |0| > 2·0 is false and the old code fell
    // into "an effect this sample does not resolve" — which is not what 0 ± 0 means.
    expect(verdictKind(diff({ nats: 0, se: 0, identity: true }))).toBe("identity");
  });

  it("reports a refusal as a refusal", () => {
    expect(verdictKind(null)).toBe("refused");
    expect(
      verdictKind(
        diff({
          nats: null,
          se: null,
          nPairs: 0,
          refused: { type: "StaticModeError", message: "…" },
        }),
      ),
    ).toBe("refused");
  });

  it("does not print a negative percentage as a share of the total damage", () => {
    // The live panel printed "-69% of the total damage".
    expect(Math.round((BACKWARDS.nats! / TOTAL_THERE.nats!) * 100)).toBe(-69);
    expect(formSharePercent(BACKWARDS, TOTAL_THERE)).toBeNull();
    // A real positive run still gets its share.
    expect(formSharePercent(diff({ nats: 0.273 }), diff({ id: "total", nats: 0.989 }))).toBe("28%");
  });

  it("does not call a negative value an upper bound", () => {
    expect(upperBoundLabel(BACKWARDS)).toMatch(/cannot have a negative upper bound/);
    expect(upperBoundLabel(diff({ nats: 0.111 }))).toBe("upper bound — see below");
  });

  it("keeps wrong_content available so the closing ordering can be checked", () => {
    // The conclusion sentence claims form < content; the panel only states it when the
    // payload says so, and here it does not.
    expect(BACKWARDS.nats! < WRONG_CONTENT_THERE.nats!).toBe(true);
  });
});

describe("F4 — every measured term of an error bar is rendered", () => {
  // The row as the deployed site served it.
  const STATIC_TOTAL = diff({
    id: "total",
    label: "both costs together",
    expr: "nll(nonce) − nll(english)",
    headline: false,
    upperBound: false,
    nats: 0.879,
    se: 0.0741,
    nPairs: 847,
    quantizationUncertaintyNats: 0.2,
  });

  it("states the measured quantization uncertainty on a SECONDARY row too", () => {
    const line = secondaryLine(STATIC_TOTAL);
    expect(line).toContain("0.879");
    expect(line).toContain("± 0.074 (sampling)");
    expect(line).toContain("± 0.2 (quantization, measured)");
    // The interval the old row stated excluded the float32 truth; the term that fixes
    // that is the one that was dropped.
    expect(0.879 + 0.0741).toBeLessThan(0.9892);
    expect(0.879 + 0.2).toBeGreaterThan(0.9892);
  });

  it("states nothing it was not given: no quantization term at float32", () => {
    const backendTotal = diff({ id: "total", nats: 0.9892, se: 0.0595, nPairs: 847 });
    expect(secondaryLine(backendTotal)).not.toContain("quantization");
    expect(errorBarTerms(backendTotal)).toEqual(["± 0.059 (sampling, 847 paired tokens)"]);
  });

  it("headline cards carry both terms, unchanged", () => {
    expect(errorBarTerms(STATIC_TOTAL)).toEqual([
      "± 0.074 (sampling, 847 paired tokens)",
      "± 0.2 (quantization, measured)",
    ]);
  });
});

describe("F5 — the reply tooltip names the temperature the reply was drawn at", () => {
  const token = {
    text: " the",
    id: 262,
    prob: 1,
    topk: { ids: [262, 783], texts: [" the", " now"], probs: [0.0845, 0.0478] },
  };

  it("labels a greedy draw greedy, whatever a slider elsewhere says", () => {
    expect(tokenTip(token, 0)).toContain("greedy pick (chosen with certainty)");
    expect(tokenTip(token, 0)).not.toContain("T=");
  });

  it("names the temperature it is given, and only that one", () => {
    expect(tokenTip({ ...token, prob: 0.42 }, 1.2)).toContain(
      "chance of being drawn at T=1.20: 42.0%",
    );
  });

  it("always reports the top-5 under the plain softmax, not the sampling one", () => {
    // The two distributions in one line: 100% chosen vs 8.5% under plain softmax. That
    // juxtaposition is correct at T = 0 and nonsense when the two come from different runs.
    const tip = tokenTip(token, 0);
    expect(tip).toContain('" the" 8.5%');
    expect(tip).toContain("greedy pick");
  });
});
