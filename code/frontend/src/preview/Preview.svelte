<script lang="ts">
  import * as d3 from "d3";
  import { modelId, prefixText, temperature, layer } from "../lib/stores";
  import { client, type Distribution, type Reduction2D } from "../lib/dataClient";
  import Progress from "../lib/Progress.svelte";

  // Minimal verification surface (FR-017): proves model -> compute -> cache -> serve ->
  // display end to end. It is NOT one of the three production visualizations; it renders
  // the real next-token distribution and a 2D embedding scatter so every control has a
  // visible effect.
  let loading = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let dist = $state<Distribution | null>(null);
  let red = $state<Reduction2D | null>(null);
  let lastUpdated = $state("");

  let scatterEl: SVGSVGElement | undefined;
  const REF = 256; // embedding reference-set size for the preview (snappy, cached)

  let debounceTimer: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;

  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const lyr = $layer;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => void load(m, pfx, temp, lyr), 300);
    return () => {
      if (debounceTimer) clearTimeout(debounceTimer);
    };
  });

  async function load(m: string, pfx: string, temp: number, lyr: number) {
    const myRun = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = "starting…";
    try {
      const d = await client.getDistribution(m, pfx, temp, 40);
      if (myRun !== runId) return;
      dist = d;

      const params = {
        method: "pca",
        with_grid: false,
        source: "contextual",
        layer: lyr,
        reference_set_size: REF,
        seed: 0,
      };
      await client.ensureArtifact("reduction_2d", m, params, {}, (p, msg) => {
        if (myRun === runId) {
          progress = p;
          progressMsg = msg;
        }
      });
      if (myRun !== runId) return;
      red = await client.getReduction2d(m, params);
      if (myRun !== runId) return;
      drawScatter();
      lastUpdated = new Date().toLocaleTimeString();
    } catch (e: any) {
      if (myRun === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (myRun === runId) loading = false;
    }
  }

  function drawScatter() {
    if (!scatterEl || !red) return;
    const w = scatterEl.clientWidth || 520;
    const h = 340;
    const svg = d3.select(scatterEl).attr("viewBox", `0 0 ${w} ${h}`);
    svg.selectAll("*").remove();
    if (!red.coords.length) return;
    const xs = red.coords.map((c) => c[0]);
    const ys = red.coords.map((c) => c[1]);
    const x = d3.scaleLinear().domain(d3.extent(xs) as [number, number]).range([24, w - 24]);
    const y = d3.scaleLinear().domain(d3.extent(ys) as [number, number]).range([h - 24, 24]);
    svg
      .append("g")
      .selectAll("circle")
      .data(red.coords)
      .join("circle")
      .attr("cx", (d) => x(d[0]))
      .attr("cy", (d) => y(d[1]))
      .attr("r", 0)
      .attr("fill", "#6ea8fe")
      .attr("opacity", 0.5)
      .transition()
      .duration(400)
      .attr("r", 2.6);
  }

  const maxProb = $derived(dist?.top?.length ? Math.max(...dist.top.map((t) => t.prob)) : 1);
  function showToken(s: string): string {
    if (s === "") return "∅";
    return s.replace(/\n/g, "⏎").replace(/ /g, "␣");
  }
</script>

<section class="preview panel" data-testid="preview">
  <header class="phead">
    <div>
      <h2>Live preview</h2>
      <p class="sub">Minimal verification surface — real model output through the cache pipeline.</p>
    </div>
    {#if lastUpdated}<span class="stamp" data-testid="updated">updated {lastUpdated}</span>{/if}
  </header>

  {#if loading}
    <div class="loading" data-testid="loading"><Progress {progress} message={progressMsg} /></div>
  {/if}

  {#if error}
    <div class="error" data-testid="preview-error">{error}</div>
  {/if}

  <div class="grid">
    <div class="card">
      <h3>Next-token distribution</h3>
      {#if dist?.top?.length}
        <ul class="bars" data-testid="dist-bars">
          {#each dist.top.slice(0, 12) as t (t.token_id)}
            <li>
              <span class="tok" title={t.token_str}>{showToken(t.token_str)}</span>
              <span class="track"><span class="fill" style="width: {(t.prob / maxProb) * 100}%"></span></span>
              <span class="pct">{(t.prob * 100).toFixed(1)}%</span>
            </li>
          {/each}
        </ul>
      {:else if !loading}
        <p class="empty">No distribution yet.</p>
      {/if}
    </div>

    <div class="card">
      <h3>Embedding space (2D, layer {$layer})</h3>
      <svg bind:this={scatterEl} class="scatter" data-testid="scatter" height="340"></svg>
      {#if red}<p class="caption">{red.coords.length} tokens · PCA of contextual embeddings</p>{/if}
    </div>
  </div>
</section>

<style>
  .preview { padding: 1.2rem 1.4rem; display: flex; flex-direction: column; gap: 1rem; }
  .phead { display: flex; justify-content: space-between; align-items: flex-start; }
  h2 { margin: 0; font-size: 1.1rem; }
  .sub { margin: 0.2rem 0 0; color: var(--text-dim); font-size: 0.82rem; }
  .stamp { color: var(--text-dim); font-size: 0.74rem; font-family: var(--mono); }
  .loading { padding: 0.4rem 0; }
  .error {
    background: rgba(255, 122, 144, 0.12);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-family: var(--mono);
    font-size: 0.85rem;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--bg-elev-2); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }
  .card h3 { margin: 0 0 0.7rem; font-size: 0.92rem; color: var(--text-dim); font-weight: 600; }
  .bars { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; }
  .bars li { display: grid; grid-template-columns: 5.5rem 1fr 3rem; align-items: center; gap: 0.5rem; }
  .tok { font-family: var(--mono); font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .track { height: 10px; background: var(--bg-elev); border-radius: 999px; overflow: hidden; }
  .fill { display: block; height: 100%; background: var(--accent-grad); transition: width 0.35s ease; }
  .pct { font-family: var(--mono); font-size: 0.76rem; color: var(--text-dim); text-align: right; }
  .scatter { width: 100%; display: block; }
  .caption { margin: 0.4rem 0 0; color: var(--text-dim); font-size: 0.76rem; }
  .empty { color: var(--text-dim); font-size: 0.85rem; }
</style>
