// Small pure helpers shared by the vector-field and Sankey views (unit-tested).

/**
 * Robust normalisation ceiling for probability→colour/opacity mapping: the q-th
 * quantile (default 95th) of `vals`, floored at 1e-6. Using a quantile instead of
 * the max keeps one outlier arrow from washing out the rest of the field at high
 * temperature (redteam-vector F3): callers clamp `p / robustMax(vals)` to [0, 1].
 */
export function robustMax(vals: number[], q = 0.95): number {
  if (!vals.length) return 1e-6;
  const s = [...vals].sort((a, b) => a - b);
  const pos = q * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const v = s[lo] + (s[hi] - s[lo]) * (pos - lo);
  return Math.max(v, 1e-6);
}

/** Absolute cap (px) on a Sankey ribbon's stroke width (redteam-sankey S1). */
export const LINK_WIDTH_CAP = 28;

/**
 * Sankey ribbon stroke width: proportional to the link's share of the busiest
 * link, scaled by the row pitch, but never thinner than 1px and never wider than
 * `cap` px — so a low-diversity swarm (2 rows × 1000 particles) stays readable
 * instead of collapsing into a single giant blob.
 */
export function capLinkWidth(value: number, maxVal: number, rowH: number, cap = LINK_WIDTH_CAP): number {
  const w = (value / Math.max(maxVal, 1e-9)) * (rowH * 0.8);
  return Math.max(1, Math.min(w, cap));
}

/** "1 transition", "2 transitions" — caption pluralisation (redteam-sankey S7). */
export function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}
