<script lang="ts">
  /**
   * US-4 / FR-620..622 — the honest centrepiece.
   *
   * Effective rank is NEVER shown on its own here. Two things are drawn with it, always:
   *
   *   1. the `min(|V|−1, d)` ceiling — centring the matrix costs one dimension, so this
   *      is the largest rank the spectrum could possibly have;
   *   2. an UNTRAINED random-init model at the same shape — because effective rank climbs
   *      with |V| for random matrices too, and a staircase that looks like learning is
   *      often just that mechanical bound being approached.
   *
   * If the trained curve sits on the random one, that is the finding, and this panel is
   * built so you cannot miss it rather than so you cannot see it.
   *
   * Tied models report ONE spectrum, labelled tied — the source project logs the
   * embedding and the readout as separate spectra when tied, which is the same matrix
   * counted twice. Untied models genuinely have two, and both are drawn.
   */
  import { SPECTRUM_DISPLAY_K, type SpectrumResult } from "../../lib/lexEngine";
  import Explain from "../../lib/Explain.svelte";
  import type { Provenance } from "./provenance";

  /** The engine's own result type — `ceiling` is `min(V−1, d)`, computed there, not here. */
  type SpectrumView = SpectrumResult;

  interface Props {
    embedding: SpectrumView | null;
    readout: SpectrumView | null;
    baseline: SpectrumView | null;
    tied: boolean;
    /** What these weights are: untrained, trained, or either of those hand-edited. */
    provenance: Provenance;
    dModel: number;
    budgetSize: number;
    vocabRows: number;
  }
  let { embedding, readout, baseline, tied, provenance, dModel, budgetSize, vocabRows }: Props =
    $props();

  const W = 320;
  const H = 110;

  /**
   * How many bars to draw. SPECTRUM_DISPLAY_K is the intended count, but the ceiling must
   * always be inside the drawn range or the line FR-622 requires would point off the
   * chart — so the range is widened to include it when necessary, never narrowed.
   * `ceiling = min(V−1, d) ≤ d = sigma.length`, so this can never exceed the data.
   */
  function barCount(s: SpectrumView): number {
    return Math.min(s.singularValues.length, Math.max(SPECTRUM_DISPLAY_K, Math.ceil(s.ceiling)));
  }

  /** Bars as a fraction of each matrix's OWN leading singular value: this compares the
   *  shape of two spectra, not their scale, which is what makes the overlay meaningful. */
  function normalized(s: SpectrumView, n: number): number[] {
    const top = s.singularValues[0] || 1;
    return Array.from({ length: n }, (_, i) => (s.singularValues[i] ?? 0) / top);
  }

  function baselinePath(s: SpectrumView, n: number): string {
    const vals = normalized(s, n);
    const step = W / n;
    // A stepped outline, so it reads as a reference curve rather than a second set of bars.
    return vals
      .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(2)},${(H - v * H).toFixed(2)} L${((i + 1) * step).toFixed(2)},${(H - v * H).toFixed(2)}`)
      .join(" ");
  }

  const fmt2 = (x: number) => x.toFixed(2);

  /** Everything FR-622 insists on is present, or the rank numbers stay hidden. */
  const complete = $derived(embedding !== null && baseline !== null);
</script>

<div class="panel-body" data-testid="lex-spectrum">
  <div class="head">
    <h3>Embedding spectrum</h3>
    <span class="hint">
      {tied ? "one matrix — the readout is tied to it" : "two matrices — embedding and readout are separate"}
    </span>
  </div>

  <!-- Declared at the top level, and taking the baseline as an ARGUMENT: a snippet body is
       a closure, so a `baseline !== null` narrowing from an enclosing block would not
       survive into it. Passing it explicitly keeps the null check where it belongs — at
       the single render site, which is inside the guard. -->
  {#snippet chart(s: SpectrumView, base: SpectrumView, label: string, testid: string)}
    {@const n = barCount(s)}
    {@const vals = normalized(s, n)}
    {@const step = W / n}
    <div class="chart-block" data-testid={testid}>
      <span class="chart-label">{label}</span>
      <div class="chart-wrap">
        <svg
          class="chart"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`${label}: ${n} leading singular values, effective rank ${fmt2(s.effectiveRank)} against a ceiling of ${s.ceiling} and a random-init baseline of ${fmt2(base.effectiveRank)}`}
        >
          {#each vals as v, i (i)}
            <rect
              class="bar"
              x={(i * step + step * 0.1).toFixed(2)}
              width={(step * 0.8).toFixed(2)}
              y={(H - v * H).toFixed(2)}
              height={(v * H).toFixed(2)}
            />
          {/each}
          <!-- The random-init model at the same shape (FR-622), never optional. -->
          <path class="baseline-path" d={baselinePath(base, n)} />
          <!-- min(|V|-1, d): the largest rank centring leaves attainable. -->
          <line
            class="ceiling"
            x1={((s.ceiling / n) * W).toFixed(2)}
            x2={((s.ceiling / n) * W).toFixed(2)}
            y1="0"
            y2={H}
            stroke-dasharray="5 4"
          />
          <line
            class="effrank"
            x1={((s.effectiveRank / n) * W).toFixed(2)}
            x2={((s.effectiveRank / n) * W).toFixed(2)}
            y1="0"
            y2={H}
          />
          <line
            class="effrank base"
            x1={((base.effectiveRank / n) * W).toFixed(2)}
            x2={((base.effectiveRank / n) * W).toFixed(2)}
            y1="0"
            y2={H}
            stroke-dasharray="2 3"
          />
        </svg>
      </div>
      <p class="axis">
        singular values 1…{n}, each as a fraction of that matrix's own σ₁
        {#if n < s.singularValues.length}<span class="why"> · {s.singularValues.length - n} smaller values not drawn</span>{/if}
      </p>
      <div class="stats">
        <div class="stat wide">
          <span class="k">effective rank</span>
          <span class="v" data-testid={`${testid}-effrank`}>{fmt2(s.effectiveRank)}</span>
          <span class="d">
            against a ceiling of <b>{s.ceiling}</b> = min(|V|−1, d) = min({vocabRows - 1}, {dModel}),
            and <b>{fmt2(base.effectiveRank)}</b> for an untrained model at this exact shape
            {#if provenance === "unrecorded"}
              — and what you are looking at came from a file that does not record whether
              it was ever trained, so the distance between these two numbers is a real
              measurement of an unknown history
            {:else if provenance === "edited-unrecorded"}
              — and what you are looking at is a loaded model of unrecorded history with
              your edits applied, so part of the distance is the edit and the rest is
              whatever the file's weights already were
            {:else if provenance === "untrained"}
              — which is what you are looking at: nothing has been trained yet, so these
              two are the same model
            {:else if provenance === "edited-untrained"}
              — which is what your edits started from: nothing has been trained, so the
              whole distance between these two numbers is the edit, not learning
            {:else if provenance === "edited-trained"}
              — and what you are looking at is the trained model with your edits applied,
              so the distance is training and editing together
            {/if}
          </span>
        </div>
        <div class="stat">
          <span class="k">stable rank</span>
          <span class="v">{fmt2(s.stableRank)}</span>
          <span class="d">Σλⱼ / λ₁</span>
        </div>
        <div class="stat">
          <span class="k">participation</span>
          <span class="v">{fmt2(s.participationRatio)}</span>
          <span class="d">1 / Σpᵢ²</span>
        </div>
        <div class="stat">
          <span class="k">var in top 2</span>
          <span class="v">{(s.fracVarTop2 * 100).toFixed(1)}%</span>
          <span class="d">top 10: {(s.fracVarTop10 * 100).toFixed(1)}%</span>
        </div>
        <div class="stat">
          <span class="k">dims for 90%</span>
          <span class="v">{s.nDimsFor90pct}</span>
          <span class="d">of the total variance</span>
        </div>
      </div>
      <!-- A matrix with no leading singular value has no spectrum to describe, and every
           statistic above is then a division by nothing. Reachable in one click — zero the
           embedding in the Weight Lab — so it is named rather than drawn as a flat chart
           with plausible-looking numbers beside it. -->
      {#if !(s.singularValues[0] > 0) || !Number.isFinite(s.effectiveRank)}
        <p class="degenerate" data-testid={`${testid}-degenerate`}>
          <b>This matrix is degenerate</b> — σ₁ = {fmt2(s.singularValues[0] ?? 0)}. There is
          no variance for the statistics above to distribute, so they describe an absence,
          and the bars are drawn against a leading value that is not positive. That is a
          real property of these weights, not a rendering artefact.
        </p>
      {/if}
    </div>
  {/snippet}

  {#if !complete}
    <p class="waiting" data-testid="lex-spectrum-waiting">
      building the model at this shape…
    </p>
  {:else if embedding && baseline}
    <div class="legend" data-testid="lex-spectrum-legend">
      <span class="key"><i class="sw bar"></i>this model</span>
      <span class="key"><i class="sw base"></i>random init, same shape</span>
      <span class="key"><i class="sw ceil"></i>min(|V|−1, d) ceiling</span>
      <span class="key"><i class="sw eff"></i>effective rank</span>
    </div>

    {#if tied}
      {@render chart(embedding, baseline, "embedding (tied — this IS the readout)", "lex-spectrum-embedding")}
    {:else}
      {@render chart(embedding, baseline, "embedding", "lex-spectrum-embedding")}
      {#if readout}
        {@render chart(readout, baseline, "readout (untied — a second matrix)", "lex-spectrum-readout")}
      {/if}
    {/if}

    {#if provenance === "unrecorded" || provenance === "edited-unrecorded"}
      <p class="untrained" data-testid="lex-spectrum-unrecorded">
        These weights came from a file that does not record whether they were ever
        trained{provenance === "edited-unrecorded"
          ? ", and they have since been hand-edited"
          : ""}. The bars are their real spectrum and the outline is a random
        initialization at the same shape, so the distance between them is measured rather
        than assumed — but a gap here is not evidence of training, and no gap is not
        evidence of its absence.
      </p>
    {:else if provenance === "untrained"}
      <p class="untrained" data-testid="lex-spectrum-untrained">
        This model has not been trained. The bars, the baseline and both rank markers
        coincide because they are the same random initialization — train it and watch them
        separate, or fail to.
      </p>
    {:else if provenance === "edited-untrained"}
      <p class="untrained" data-testid="lex-spectrum-edited-untrained">
        This model has not been trained, <b>and its weights have been hand-edited</b>. The
        bars are that random initialization with your edit applied; the outline is the same
        initialization untouched. They no longer coincide — and the whole difference is the
        edit, because nothing here has learned anything yet.
      </p>
    {:else if provenance === "edited-trained"}
      <p class="untrained" data-testid="lex-spectrum-edited">
        <b>These weights have been hand-edited</b>, so the bars are not the spectrum of the
        model you trained — the Weight Lab's restore button brings that one back. The
        outline is still an untrained model at this shape.
      </p>
    {/if}
  {/if}

  <Explain
    title="Why a ceiling and a baseline are not optional"
    hint="rank rises with |V| even when nothing is learned"
    testid="lex-explain-spectrum"
  >
    <p>
      The matrix is column-mean-centred before anything else, <code>Ac = A − mean(A, axis=0)</code>,
      and its spectrum comes from the symmetric eigendecomposition of the <code>d×d</code>
      Gram matrix <code>G = Acᵀ Ac</code>: the eigenvalues <b>are</b> <code>σᵢ²</code>, so
      no SVD runs. That is why this recomputes in milliseconds as you move a control — and
      it sidesteps the crash in the source project, whose <code>svdvals</code> call has no
      kernel on the accelerator it selects.
    </p>
    <p>
      Centring subtracts one degree of freedom, so the attainable rank is
      <code>min(|V|−1, d)</code>, currently <b>min({vocabRows - 1}, {dModel})</b> — the
      {budgetSize} budget words plus their reserved rows, against
      <code>d_model = {dModel}</code>. Approaching that bound is not evidence of learning. A random
      Gaussian matrix at the same shape approaches it too, and the more rows it has, the
      closer it gets — which is exactly the staircase somebody might mistake for a
      vocabulary "using more of its space".
    </p>
    <p>
      So the panel refuses to print effective rank alone. Read the <b>gap</b> between the
      solid marker and the dotted one, not the absolute number: that gap is what training
      did, at this budget, at this <code>d</code>.
    </p>
    <p>
      The three rank summaries answer different questions.
      <b>Effective rank</b> <code>exp(−Σ pᵢ ln pᵢ)</code> is the perplexity of the
      eigenvalue distribution — how many directions carry variance, weighted by how much.
      <b>Stable rank</b> <code>Σλⱼ / λ₁</code> is dominated by the leading direction and
      falls sharply when one direction runs away. <b>Participation ratio</b>
      <code>1 / Σpᵢ²</code> weights by squared share, so it is the harshest of the three on
      long tails. When they disagree, the shape of the spectrum is the reason.
    </p>
  </Explain>
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
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
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    font-size: 0.68rem;
    color: var(--text-dim);
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .sw {
    width: 14px;
    height: 3px;
    border-radius: 2px;
    display: inline-block;
  }
  .sw.bar {
    background: var(--accent);
  }
  .sw.base {
    background: var(--text-dim);
  }
  .sw.ceil {
    background: repeating-linear-gradient(90deg, #ffb454 0 4px, transparent 4px 7px);
  }
  .sw.eff {
    background: var(--good);
  }
  .chart-block {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    min-width: 0;
  }
  .chart-label {
    font-size: 0.7rem;
    letter-spacing: 0.05em;
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
    height: 110px;
    display: block;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .bar {
    fill: var(--accent);
    opacity: 0.8;
  }
  .baseline-path {
    fill: none;
    stroke: var(--text-dim);
    stroke-width: 1.4;
    vector-effect: non-scaling-stroke;
    opacity: 0.9;
  }
  .ceiling {
    stroke: #ffb454;
    stroke-width: 1.4;
    vector-effect: non-scaling-stroke;
  }
  .effrank {
    stroke: var(--good);
    stroke-width: 1.6;
    vector-effect: non-scaling-stroke;
  }
  .effrank.base {
    stroke: var(--text-dim);
    stroke-width: 1.4;
  }
  .axis {
    margin: 0;
    font-size: 0.67rem;
    color: var(--text-dim);
  }
  .why {
    color: #ffb454;
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 8.5rem), 1fr));
    gap: 0.4rem;
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.4rem 0.55rem;
    min-width: 0;
  }
  .stat.wide {
    grid-column: 1 / -1;
  }
  .stat .k {
    font-size: 0.64rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .stat .v {
    font-family: var(--mono);
    font-size: 1rem;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .stat.wide .v {
    color: var(--good);
  }
  .stat .d {
    font-size: 0.66rem;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .stat .d b {
    color: var(--text);
    font-family: var(--mono);
  }
  .degenerate {
    margin: 0.15rem 0 0;
    font-size: 0.7rem;
    line-height: 1.5;
    color: var(--bad);
    background: rgba(255, 122, 144, 0.1);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 9px;
    padding: 0.4rem 0.55rem;
  }
  .degenerate b {
    color: var(--bad);
  }
  .untrained {
    margin: 0;
    font-size: 0.72rem;
    line-height: 1.55;
    color: #ffb454;
    background: rgba(255, 180, 84, 0.08);
    border: 1px solid rgba(255, 180, 84, 0.3);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
  }
</style>
