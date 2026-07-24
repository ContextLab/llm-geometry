<script lang="ts">
  import type { GeoTrace } from "../../lib/dataClient";
  import MatrixHeatmap from "../../lib/MatrixHeatmap.svelte";

  // Attention view: the real (row-stochastic, causal) T×T attention matrix of each
  // layer for the current prompt, plus the model's top-10 next-token predictions.
  interface Props {
    trace: GeoTrace | null;
  }
  let { trace }: Props = $props();

  let layer = $state(0);
  const tokenLabels = $derived(trace ? trace.tokens.map((t, i) => `${i}: "${t.text}"`) : undefined);
  const att = $derived(trace?.layers[layer]?.attention ?? null);
  const topk = $derived(trace?.logits_topk ?? null);
  const maxProb = $derived(topk ? Math.max(...topk.probs, 1e-9) : 1);
</script>

<div class="panel-body" data-testid="geo-attention">
  <div class="head">
    <h3>Attention &amp; predictions</h3>
    {#if trace}
      <div class="tabs" role="tablist">
        {#each trace.layers as l (l.layer)}
          <button class:active={layer === l.layer} role="tab" aria-selected={layer === l.layer} onclick={() => (layer = l.layer)}>layer {l.layer}</button>
        {/each}
      </div>
    {/if}
  </div>

  {#if trace && att}
    <div class="cols">
      <div class="att">
        <span class="hint">who attends to whom (rows = query token, cols = key token; causal)</span>
        <div class="att-scroll">
          <MatrixHeatmap values={att} rowLabels={tokenLabels} colLabels={tokenLabels} maxCanvasPx={420} />
        </div>
      </div>
      <div class="preds">
        <span class="hint">next token after “…{trace.tokens[trace.tokens.length - 1]?.text}”</span>
        {#if topk}
          <ol class="bars">
            {#each topk.ids as id, i (id)}
              <li>
                <span class="tok">{topk.texts[i]}</span>
                <span class="bar-track"><span class="bar" style={`width:${(topk.probs[i] / maxProb) * 100}%`}></span></span>
                <span class="prob">{(topk.probs[i] * 100).toFixed(1)}%</span>
              </li>
            {/each}
          </ol>
        {/if}
      </div>
    </div>
  {:else}
    <p class="hint">type a prompt to trace the model…</p>
  {/if}
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.92rem;
  }
  .tabs {
    display: inline-flex;
    gap: 0.25rem;
    padding: 0.2rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
  }
  .tabs button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.24rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 500;
  }
  .tabs button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
  }
  .cols {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, 0.8fr);
    gap: 1rem;
    align-items: start;
  }
  @media (max-width: 760px) {
    .cols {
      grid-template-columns: 1fr;
    }
  }
  .att,
  .preds {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    min-width: 0;
  }
  .att-scroll {
    max-width: 100%;
    overflow: auto;
  }
  .hint {
    font-size: 0.72rem;
    color: var(--text-dim);
  }
  .bars {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .bars li {
    display: grid;
    grid-template-columns: 5.4rem 1fr 3.2rem;
    align-items: center;
    gap: 0.5rem;
  }
  .tok {
    font-family: var(--mono);
    font-size: 0.74rem;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: pre;
  }
  .bar-track {
    height: 9px;
    border-radius: 999px;
    background: var(--bg-elev-2);
    overflow: hidden;
  }
  .bar {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: var(--accent-grad);
  }
  .prob {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
</style>
