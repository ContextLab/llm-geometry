/**
 * What the vacancy panel is ALLOWED to say about the numbers it was handed.
 *
 * This lives outside the component because it is the part that can be wrong while
 * everything renders: the panel's job is to draw the response, and every decision about
 * which sentence that response licenses is made here, against real payloads, in tests.
 * Three of those decisions were wrong on the live site:
 *
 *  - `resolved` tested |nats| > 2·se, which is the right test for "is this above the
 *    noise" and the wrong test for "did the effect run the way the label claims". A
 *    resolved NEGATIVE `nll(nonce) − nll(swap)` was therefore PROMOTED into the
 *    conclusion branch, which asserts "losing [the form] costs far less than losing the
 *    content" unconditionally;
 *  - the share was printed as `Math.round(nats/total*100)%` with no sign guard, so the
 *    same run read `-69% of the total damage` — a share of nothing;
 *  - the secondary row rendered `± se` and dropped `quantizationUncertaintyNats`, so the
 *    static site stated `0.879 ± 0.074` for a difference whose measured quantization
 *    uncertainty is ±0.2, and the interval excluded the float32 value 0.9892 (FR-720a:
 *    a ± that was never measured is a fabrication — and a measured one that was dropped
 *    is the same failure with the sign reversed).
 */

import type { ArchVacancyDifference } from "../../lib/dataClient";

/** `—` for anything that is not a real number: null, undefined, NaN. */
export function nats(v: number | null | undefined): string {
  return v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(3);
}

export function num(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

/**
 * Which closing sentence a run licenses.
 *
 *  - `refused`    — the stack would not report this difference at all;
 *  - `identity`   — p = 0: the three variants are one string, so 0 is by construction;
 *  - `unresolved` — |effect| is inside twice its own standard error; the SIGN is noise;
 *  - `backwards`  — resolved, but negative: a cost that came out below zero, which means
 *                   the contrast ran the other way on this sample. No conclusion.
 *  - `conclusion` — resolved and positive: the only case the finding may be stated in.
 *
 * Order matters: `identity` outranks `unresolved` (0 ± 0 is not "too noisy to see"), and
 * `unresolved` outranks `backwards` (the sign of noise is not a direction).
 */
export type VerdictKind = "refused" | "identity" | "unresolved" | "backwards" | "conclusion";

export function verdictKind(unknownForm: ArchVacancyDifference | null): VerdictKind {
  if (!unknownForm || unknownForm.refused) return "refused";
  if (unknownForm.identity) return "identity";
  const { nats: value, se } = unknownForm;
  if (value === null || se === null || Number.isNaN(value) || Number.isNaN(se)) return "refused";
  if (!(Math.abs(value) > 2 * se)) return "unresolved";
  return value < 0 ? "backwards" : "conclusion";
}

/**
 * The form's share of the total damage, or `null` when that phrase means nothing.
 *
 * A percentage of a total is only a share when both are positive; `-69% of the total
 * damage` is not a quantity, and neither is a percentage of a total that is itself ≤ 0.
 */
export function formSharePercent(
  unknownForm: ArchVacancyDifference | null,
  total: ArchVacancyDifference | null,
): string | null {
  const a = unknownForm?.nats;
  const b = total?.nats;
  if (a === null || a === undefined || b === null || b === undefined) return null;
  if (!(a > 0) || !(b > 0)) return null;
  return `${Math.round((a / b) * 100)}%`;
}

/**
 * Whether this run supports the closing claim "losing [the form] costs far less than
 * losing the content". The sentence was unconditional; it is an ordering, so it gets
 * checked against the two numbers it orders, and `false` when either is missing.
 */
export function formCostsLessThanContent(
  unknownForm: ArchVacancyDifference | null,
  wrongContent: ArchVacancyDifference | null,
): boolean {
  const a = unknownForm?.nats;
  const b = wrongContent?.nats;
  if (a === null || a === undefined || b === null || b === undefined) return false;
  return a < b;
}

/** The caption under an `upperBound` difference. A cost has no negative upper bound. */
export function upperBoundLabel(d: ArchVacancyDifference): string {
  return d.nats !== null && d.nats !== undefined && d.nats < 0
    ? "negative — a cost cannot have a negative upper bound; see below"
    : "upper bound — see below";
}

/**
 * Every term of a difference's error bar, in the order they are rendered.
 *
 * The sampling standard error always; the MEASURED quantization uncertainty whenever the
 * stack attached one. Nothing here invents a term, and nothing here drops one.
 */
export function errorBarTerms(d: ArchVacancyDifference): string[] {
  const terms = [`± ${num(d.se)} (sampling, ${d.nPairs.toLocaleString()} paired tokens)`];
  if (d.quantizationUncertaintyNats) {
    terms.push(`± ${d.quantizationUncertaintyNats} (quantization, measured)`);
  }
  return terms;
}

/** The same, compact, for the secondary rows: `expr = value ± se · ± q`. */
export function secondaryLine(d: ArchVacancyDifference): string {
  const terms = [`± ${num(d.se)} (sampling)`];
  if (d.quantizationUncertaintyNats) {
    terms.push(`± ${d.quantizationUncertaintyNats} (quantization, measured)`);
  }
  return `${d.expr} = ${nats(d.nats)} ${terms.join(" · ")}`;
}
