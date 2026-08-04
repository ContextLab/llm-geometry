<script lang="ts">
  /**
   * FR-623 — a PCA PROJECTION of the embedding matrix, labelled as one, everywhere.
   *
   * This is the opposite of the Geometry Lab's sphere. There, `d_model = 3` and the
   * picture IS the representation: nothing is discarded, and a distance on screen is a
   * distance in the model. Here `d_model` is 16 to 128 and you are looking at the top
   * components of the centred matrix — a shadow. Two tokens can sit on top of each other
   * in this view and be far apart in the model.
   *
   * So the explained variance is printed next to the picture rather than buried: it is
   * exactly the fraction of the geometry that survived the projection, and when it is
   * small the picture deserves less trust, not more.
   *
   * The coordinates arrive on the engine's `SpectrumResult`: `coords` is row-major
   * `(rows, components)`, and `explainedVarianceRatio` is the fraction of the centred
   * matrix's variance each retained component carries.
   */
  import { PCA_COMPONENTS, SPECIAL_TOKENS, type SpectrumResult } from "../../lib/lexEngine";
  import Explain from "../../lib/Explain.svelte";
  import { view } from "../../lib/stores";
  import type { Provenance } from "./provenance";

  interface Props {
    spectrumResult: SpectrumResult | null;
    words: readonly string[];
    /** What these weights are: untrained, trained, or either of those hand-edited. */
    provenance: Provenance;
  }
  let { spectrumResult, words, provenance }: Props = $props();

  let query = $state("");
  let hovered = $state<number | null>(null);

  const W = 320;
  const H = 240;
  const PAD = 14;

  const points = $derived.by(() => {
    const s = spectrumResult;
    if (!s || s.components < 2 || s.coords.length === 0) return null;
    const k = s.components;
    const n = Math.floor(s.coords.length / k);
    const at = (row: number, comp: number) => (comp < k ? s.coords[row * k + comp] : 0);
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (let r = 0; r < n; r++) {
      xs.push(at(r, 0));
      ys.push(at(r, 1));
      zs.push(at(r, 2));
    }
    const sx = scale(xs, PAD, W - PAD);
    const sy = scale(ys, H - PAD, PAD); // SVG y grows downward
    const zMin = Math.min(...zs);
    const zSpan = Math.max(...zs) - zMin || 1;
    return Array.from({ length: n }, (_, i) => ({
      i,
      x: sx(xs[i]),
      y: sy(ys[i]),
      // The third component is real information; encoding it as size keeps it visible
      // without pretending the flat picture is three-dimensional.
      r: 1.8 + 2.4 * ((zs[i] - zMin) / zSpan),
      word: words[i] ?? `row ${i}`,
      special: i < SPECIAL_TOKENS.length,
    }));
  });

  function scale(vals: number[], lo: number, hi: number): (v: number) => number {
    const min = Math.min(...vals);
    const span = Math.max(...vals) - min || 1;
    return (v: number) => lo + ((v - min) / span) * (hi - lo);
  }

  /** Rows whose word matches the search box — highlighted and labelled. */
  const matched = $derived.by(() => {
    const q = query.trim().toLowerCase();
    if (!q || !points) return new Set<number>();
    return new Set(points.filter((p) => p.word.toLowerCase().includes(q)).map((p) => p.i));
  });

  /** Labels drawn on the plot: the search matches, plus whatever is hovered. */
  const labelled = $derived.by(() => {
    if (!points) return [];
    const ids = new Set(matched);
    if (hovered != null) ids.add(hovered);
    // Beyond a handful the labels overlap into an unreadable mat; the count is reported
    // rather than the extras silently dropped.
    return points.filter((p) => ids.has(p.i)).slice(0, 24);
  });

  /**
   * Hover is handled on the SVG rather than on ~300 circles. Per-dot handlers would force
   * every dot to carry an ARIA role, filling the accessibility tree with three hundred
   * announced graphics for what is one picture — the plot's own label already describes
   * it, and the search box is the keyboard route to any particular word.
   *
   * The mapping is linear because the element's aspect ratio is pinned to the viewBox's
   * (see `.chart` in the styles), so there is no letterboxing to undo.
   */
  function onPointerMove(e: MouseEvent): void {
    const pts = points;
    if (!pts) return;
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const y = ((e.clientY - rect.top) / rect.height) * H;
    let best = -1;
    let bestD = 100; // squared viewBox units — ~10 units is a comfortable grab radius
    for (const p of pts) {
      const d = (p.x - x) ** 2 + (p.y - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = p.i;
      }
    }
    hovered = best < 0 ? null : best;
  }

  const explained = $derived<readonly number[]>(spectrumResult?.explainedVarianceRatio ?? []);
  const explainedTotal = $derived(explained.reduce((a, b) => a + b, 0));
  const pctOf = (x: number) => `${(x * 100).toFixed(1)}%`;
</script>

<div class="panel-body" data-testid="lex-cloud">
  <div class="head">
    <h3>Token cloud <span class="tag">PCA projection</span></h3>
    <span class="hint">a shadow of the embedding, not the embedding</span>
  </div>

  {#if !points}
    <p class="waiting">building the model at this shape…</p>
  {:else}
    <div class="explained" data-testid="lex-cloud-explained">
      <span class="k">explained variance</span>
      <span class="terms">
        {#each explained as ev, i (i)}
          <span class="term">PC{i + 1} <b>{pctOf(ev)}</b></span>
        {/each}
      </span>
      <span class="d">
        the top {explained.length} component{explained.length === 1 ? "" : "s"} carry
        <b>{pctOf(explainedTotal)}</b> of the centred embedding's variance — the rest of the
        geometry is not in this picture
      </span>
    </div>

    <label class="find">
      <span class="ctl-label">find a word</span>
      <input
        type="text"
        data-testid="lex-cloud-find"
        bind:value={query}
        placeholder="e.g. little, moon, and"
      />
    </label>

    <div class="chart-wrap">
      <svg
        class="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`PCA projection of ${points.length} embedding rows onto components 1 and 2, carrying ${pctOf(explainedTotal)} of the variance across ${explained.length} components`}
        data-testid="lex-cloud-svg"
        onmousemove={onPointerMove}
        onmouseleave={() => (hovered = null)}
      >
        {#each points as p (p.i)}
          <circle
            class="dot"
            class:special={p.special}
            class:match={matched.has(p.i)}
            class:hovered={hovered === p.i}
            cx={p.x.toFixed(2)}
            cy={p.y.toFixed(2)}
            r={p.r.toFixed(2)}
          />
        {/each}
        {#each labelled as p (p.i)}
          <text class="label" x={(p.x + 5).toFixed(2)} y={(p.y - 4).toFixed(2)}>{p.word}</text>
        {/each}
      </svg>
    </div>

    <p class="caption">
      Axes are <b>PC1</b> (horizontal) and <b>PC2</b> (vertical) of the column-mean-centred
      embedding; dot size is <b>PC3</b>. The {SPECIAL_TOKENS.length} reserved rows
      ({SPECIAL_TOKENS.join(" ")}) are drawn in amber — they are trained like any other row
      and often sit far out, because they occur in positions no real word does.
      {#if matched.size > 0}
        <span class="found">{matched.size} match{matched.size === 1 ? "" : "es"} highlighted{matched.size > 24 ? ", 24 labelled" : ""}.</span>
      {/if}
      {#if provenance === "untrained"}
        <span class="untrained">
          Nothing has been trained yet, so this is the projection of a random Gaussian
          matrix — a featureless blob is the correct picture, and a good thing to remember
          the shape of.
        </span>
      {:else if provenance === "edited-untrained"}
        <span class="untrained">
          Nothing has been trained yet <b>and the weights have been hand-edited</b>, so this
          is the projection of the random initialization with your edit applied — neither a
          Gaussian matrix any more nor anything that has learned. If the explained variance
          above collapsed, the edit is why.
        </span>
      {:else if provenance === "edited-trained"}
        <span class="untrained">
          <b>These weights have been hand-edited</b>, so this is the projection of the
          trained embedding plus your edit, not of the model you trained. Restore in the
          Weight Lab to see that one again.
        </span>
      {/if}
    </p>
  {/if}

  <Explain
    title="What a projection can and cannot tell you"
    hint="why this is not the Geometry Lab's sphere"
    testid="lex-explain-cloud"
  >
    <p>
      The coordinates are <code>Ac · E[:, :{PCA_COMPONENTS}]</code>, where <code>Ac</code> is
      the column-mean-centred embedding and <code>E</code> holds the leading eigenvectors of
      its Gram matrix — the same eigendecomposition the spectrum panel uses, so the two
      views are of one object.
    </p>
    <p>
      Because it is a projection, the safe readings are narrow: <b>clusters</b> that
      survive at high explained variance, and <b>outliers</b> so far out they cannot be an
      artefact of discarding components. Unsafe readings: fine distances, empty regions,
      and anything about two nearby dots. Points can collide here and be orthogonal in the
      model.
    </p>
    <p>
      The <button class="linklike" onclick={() => view.set("geometry")}>Geometry Lab</button>'s
      sphere is a different kind of picture. There the model is built with
      <code>d_model = 3</code> so that the embedding has nowhere else to be — no
      information is discarded and no explained-variance number is needed, because it is
      always 100%. Here, the number is printed because it is not.
    </p>
  </Explain>
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-width: 0;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.95rem;
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .tag {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #ffb454;
    background: rgba(255, 180, 84, 0.12);
    border: 1px solid rgba(255, 180, 84, 0.35);
    border-radius: 999px;
    padding: 0.1rem 0.45rem;
  }
  .hint {
    font-size: 0.72rem;
    color: var(--text-dim);
    line-height: 1.4;
  }
  .waiting {
    margin: 0;
    font-size: 0.74rem;
    font-family: var(--mono);
    color: var(--text-dim);
  }
  .explained {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.45rem 0.6rem;
  }
  .explained .k {
    font-size: 0.64rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .terms {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
    font-size: 0.72rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .terms b {
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .explained .d {
    font-size: 0.67rem;
    color: var(--text-dim);
    line-height: 1.5;
  }
  .explained .d b {
    color: var(--text);
    font-family: var(--mono);
  }
  .find {
    display: flex;
    flex-direction: column;
    gap: 0.22rem;
  }
  .find input {
    font-family: var(--mono);
    font-size: 0.8rem;
  }
  .ctl-label {
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .chart-wrap {
    min-width: 0;
    overflow-x: auto;
  }
  .chart {
    width: 100%;
    min-width: 240px;
    /* Pinned to the viewBox so screen coordinates map linearly onto plot coordinates —
       `onPointerMove` relies on there being no letterboxing to undo. */
    aspect-ratio: 320 / 240;
    display: block;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .dot {
    fill: var(--accent);
    opacity: 0.55;
    transition: opacity 0.15s ease;
  }
  .dot.hovered {
    opacity: 1;
    stroke: var(--text);
    stroke-width: 0.8;
  }
  .dot.special {
    fill: #ffb454;
    opacity: 0.9;
  }
  .dot.match {
    fill: var(--good);
    opacity: 1;
  }
  .label {
    font-family: var(--mono);
    font-size: 7px;
    fill: var(--text);
    paint-order: stroke;
    stroke: var(--bg-elev);
    stroke-width: 2.5px;
    stroke-linejoin: round;
  }
  .caption {
    margin: 0;
    font-size: 0.7rem;
    line-height: 1.55;
    color: var(--text-dim);
  }
  .caption b {
    color: var(--text);
  }
  .found {
    color: var(--good);
  }
  .untrained {
    color: #ffb454;
  }
  .linklike {
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: var(--accent);
    text-decoration: underline;
    cursor: pointer;
  }
</style>
