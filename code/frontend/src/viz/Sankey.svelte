<script lang="ts">
  import * as d3 from "d3";
  import { sankey as d3sankey, sankeyLinkHorizontal } from "d3-sankey";
  import { modelId, prefixText, temperature } from "../lib/stores";
  import { client, type SankeyData } from "../lib/dataClient";
  import Progress from "../lib/Progress.svelte";

  // Visualization 2 — particle-swarm next-token sampling across positions, as a Sankey
  // diagram (project_description.md §2).
  let loading = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let data = $state<SankeyData | null>(null);
  let svgEl: SVGSVGElement | undefined;

  const N_PARTICLES = 28;
  const N_STEPS = 8;
  const SEED = 0;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;

  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => void load(m, pfx, temp), 350);
    return () => {
      if (debounce) clearTimeout(debounce);
    };
  });

  async function load(m: string, pfx: string, temp: number) {
    const my = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = "starting…";
    try {
      const params = { temperature: temp, n_particles: N_PARTICLES, n_steps: N_STEPS, seed: SEED };
      await client.ensureArtifact("sankey", m, params, { prefix_text: pfx }, (p, msg) => {
        if (my === runId) {
          progress = p;
          progressMsg = msg;
        }
      });
      if (my !== runId) return;
      data = await client.getSankey(m, { prefix_text: pfx, ...params });
      if (my !== runId) return;
      draw();
    } catch (e: any) {
      if (my === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (my === runId) loading = false;
    }
  }

  function draw() {
    if (!svgEl || !data) return;
    const w = svgEl.clientWidth || 680;
    const h = 480;
    const svg = d3.select(svgEl).attr("viewBox", `0 0 ${w} ${h}`);
    svg.selectAll("*").remove();
    const d = data;

    const key = (pos: number, token: number) => `${pos}:${token}`;
    const indexOf = new Map<string, number>();
    const nodes = d.nodes.map((n, i) => {
      indexOf.set(key(n.pos, n.token), i);
      return { name: d.token_strs[String(n.token)] ?? String(n.token), pos: n.pos };
    });
    const links = d.links
      .map((l) => ({
        source: indexOf.get(key(l.pos, l.source_token)),
        target: indexOf.get(key(l.pos + 1, l.target_token)),
        value: l.value,
      }))
      .filter((l) => l.source !== undefined && l.target !== undefined) as
      { source: number; target: number; value: number }[];

    if (!links.length) {
      svg
        .append("text")
        .attr("x", w / 2)
        .attr("y", h / 2)
        .attr("fill", "var(--text-dim)")
        .attr("text-anchor", "middle")
        .text("Not enough transitions to draw flows — try a longer prompt or higher temperature.");
      return;
    }

    const layout = d3sankey<any, any>()
      .nodeWidth(14)
      .nodePadding(9)
      .extent([[8, 12], [w - 8, h - 12]]);
    const graph = layout({
      nodes: nodes.map((n) => ({ ...n })),
      links: links.map((l) => ({ ...l })),
    });

    const color = d3.scaleSequential(d3.interpolateCool).domain([0, N_STEPS]);

    svg
      .append("g")
      .attr("fill", "none")
      .selectAll("path")
      .data(graph.links)
      .join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", (l: any) => color(l.source.pos))
      .attr("stroke-width", (l: any) => Math.max(1, l.width))
      .attr("stroke-opacity", 0.4);

    const node = svg.append("g").selectAll("g").data(graph.nodes).join("g");
    node
      .append("rect")
      .attr("x", (n: any) => n.x0)
      .attr("y", (n: any) => n.y0)
      .attr("width", (n: any) => n.x1 - n.x0)
      .attr("height", (n: any) => Math.max(1, n.y1 - n.y0))
      .attr("fill", (n: any) => color(n.pos))
      .attr("rx", 2);
    node
      .append("text")
      .attr("x", (n: any) => (n.x0 < w / 2 ? n.x1 + 5 : n.x0 - 5))
      .attr("y", (n: any) => (n.y0 + n.y1) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", (n: any) => (n.x0 < w / 2 ? "start" : "end"))
      .attr("fill", "#cdd6ec")
      .attr("font-size", "10px")
      .attr("font-family", "var(--mono)")
      .text((n: any) => n.name)
      .filter((n: any) => n.y1 - n.y0 < 9)
      .remove();
  }
</script>

<section class="viz panel" data-testid="viz-sankey" data-ready={data ? 1 : 0}>
  <header>
    <div>
      <h2>Token sequences as a Sankey diagram</h2>
      <p class="sub">A particle swarm samples next tokens across positions; flow width = particle count.</p>
    </div>
  </header>
  {#if loading}<div class="loading"><Progress {progress} message={progressMsg} /></div>{/if}
  {#if error}<div class="error" data-testid="viz-sankey-error">{error}</div>{/if}
  <svg bind:this={svgEl} class="canvas" height="480" data-testid="sankey-svg"></svg>
  {#if data}<p class="caption">{data.nodes.length} nodes · {data.links.length} transitions · {N_PARTICLES} particles</p>{/if}
</section>

<style>
  .viz { padding: 1.2rem 1.4rem; display: flex; flex-direction: column; gap: 0.8rem; }
  header h2 { margin: 0; font-size: 1.1rem; }
  .sub { margin: 0.2rem 0 0; color: var(--text-dim); font-size: 0.82rem; }
  .loading { padding: 0.3rem 0; }
  .error { background: rgba(255,122,144,0.12); color: var(--bad); border: 1px solid rgba(255,122,144,0.3); border-radius: 10px; padding: 0.6rem 0.8rem; font-family: var(--mono); font-size: 0.85rem; }
  .canvas { width: 100%; display: block; background: rgba(0,0,0,0.15); border-radius: 12px; }
  .caption { margin: 0; color: var(--text-dim); font-size: 0.76rem; }
</style>
