<script lang="ts">
  import * as d3 from "d3";
  import { sankey as d3sankey, sankeyLinkHorizontal } from "d3-sankey";
  import { onDestroy } from "svelte";
  import { modelId, prefixText, temperature, nParticles, nSteps, refreshNonce } from "../lib/stores";
  import { client, type SankeyData } from "../lib/dataClient";
  import { showTip, hideTip } from "../lib/tooltip";
  import Progress from "../lib/Progress.svelte";
  import ExportBar from "../controls/ExportBar.svelte";

  // Visualization 2 — particle-swarm next-token sampling across positions, as a Sankey
  // diagram (project_description.md §2).
  let loading = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let data = $state<SankeyData | null>(null);
  let svgEl: SVGSVGElement | undefined;

  const SEED = 0;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;
  let lastRefresh = 0;
  let revealStep = $state(999); // sequence columns revealed (999 = all); ▶ Play sweeps it
  let maxPosNow = $state(0);
  let playing = $state(false);
  let playTimer: ReturnType<typeof setInterval> | undefined;

  // The Sankey samples a fresh swarm from the PROMPT and uses only the final-layer token
  // probabilities — so the response trajectory and layer selection do not apply to it.
  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const np = $nParticles;
    const ns = $nSteps;
    const rn = $refreshNonce;
    const force = rn !== lastRefresh;
    lastRefresh = rn;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => void load(m, pfx, temp, np, ns, force), force ? 0 : 350);
    return () => {
      if (debounce) clearTimeout(debounce);
    };
  });

  async function load(m: string, pfx: string, temp: number, np: number, ns: number, force = false) {
    const my = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = force ? "recomputing…" : "starting…";
    try {
      const params = { temperature: temp, n_particles: np, n_steps: ns, seed: SEED };
      const inputs = { prefix_text: pfx };
      if (!force) {
        await client.ensureArtifact("sankey", m, params, inputs, (p, msg) => {
          if (my === runId) {
            progress = p;
            progressMsg = msg;
          }
        });
      }
      if (my !== runId) return;
      data = await client.getSankey(m, { prefix_text: pfx, ...params, ...(force ? { force: true } : {}) });
      if (my !== runId) return;
      draw();
    } catch (e: any) {
      if (my === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (my === runId) loading = false;
    }
  }

  // ▶ Play: reveal sequence columns one position per frame (client-side; no recompute).
  function stopPlay() {
    playing = false;
    if (playTimer) { clearInterval(playTimer); playTimer = undefined; }
  }
  function togglePlay() {
    if (playing) { stopPlay(); revealStep = 999; draw(); return; }
    playing = true; revealStep = 0; draw();
    playTimer = setInterval(() => {
      revealStep += 1; draw();
      if (revealStep >= maxPosNow) stopPlay();
    }, 700);
  }
  onDestroy(stopPlay);

  const exportAnim = {
    total: () => maxPosNow,
    renderFrame: async (i: number) => { revealStep = i; draw(); await new Promise((r) => requestAnimationFrame(() => r(null))); },
    restore: async () => { revealStep = 999; draw(); },
  };

  function draw() {
    if (!svgEl || !data) return;
    const w = svgEl.clientWidth || 680;
    const h = 560;
    const svg = d3.select(svgEl).attr("viewBox", `0 0 ${w} ${h}`);
    svg.selectAll("*").remove();
    const d = data;
    const color = d3.scaleSequential(d3.interpolateCool).domain([0, d.n_steps]);
    const label = (tok: number) => d.token_strs[String(tok)] ?? String(tok);

    const key = (pos: number, token: number) => `${pos}:${token}`;
    const indexOf = new Map<string, number>();
    const nodes = d.nodes.map((n, i) => {
      indexOf.set(key(n.pos, n.token), i);
      return { name: label(n.token), pos: n.pos, token: n.token, count: n.count };
    });
    const links = d.links
      .map((l) => ({
        source: indexOf.get(key(l.pos, l.source_token)),
        target: indexOf.get(key(l.pos + 1, l.target_token)),
        value: l.value,
        st: l.source_token,
        tt: l.target_token,
      }))
      .filter((l) => l.source !== undefined && l.target !== undefined) as any[];

    if (!links.length) {
      svg
        .append("text")
        .attr("x", w / 2)
        .attr("y", h / 2)
        .attr("fill", "var(--text-dim)")
        .attr("text-anchor", "middle")
        .text("Not enough transitions to draw flows — try a longer prompt or higher temperature.");
    } else {
      const layout = d3sankey<any, any>()
        .nodeWidth(13)
        .nodePadding(9)
        .extent([[8, 16], [w - 8, h - 34]]);
      const graph = layout({
        nodes: nodes.map((n) => ({ ...n })),
        links: links.map((l) => ({ ...l })),
      });

      // TIME AXIS: place every node at x ∝ its sequence position (step), so a length-k
      // sequence reaches column k and flows terminate where their particles stop — d3-sankey's
      // default depth layout doesn't encode time. (y comes from the layout.)
      const maxPos = Math.max(...graph.nodes.map((n: any) => n.pos), 1);
      maxPosNow = maxPos;
      const rev = revealStep; // reveal columns up to `rev` (▶ Play / GIF sweep)
      const shownNodes = graph.nodes.filter((n: any) => n.pos <= rev);
      const shownLinks = graph.links.filter((l: any) => l.target.pos <= rev);
      const tx = d3.scaleLinear().domain([0, maxPos]).range([24, w - 70]);
      graph.nodes.forEach((n: any) => { n.x0 = tx(n.pos); n.x1 = n.x0 + 13; });

      const axis = svg.append("g");
      for (let p = 0; p <= maxPos; p++) {
        axis.append("line").attr("x1", tx(p)).attr("x2", tx(p)).attr("y1", 14).attr("y2", h - 30)
          .attr("stroke", "var(--border)").attr("stroke-opacity", 0.35);
        axis.append("text").attr("x", tx(p)).attr("y", h - 14).attr("text-anchor", "middle")
          .attr("fill", "var(--text-dim)").attr("font-size", "10px").attr("font-family", "var(--mono)")
          .text(p === 0 ? "prompt" : `+${p}`);
      }

      // The combined next-token distribution is shown DIRECTLY via transparency at each
      // timepoint: a token's opacity = its share of the particles at that position; a flow's
      // opacity = its particle count. (No separate distribution plot.)
      const maxAtPos = new Map<number, number>();
      graph.nodes.forEach((n: any) => maxAtPos.set(n.pos, Math.max(maxAtPos.get(n.pos) ?? 0, n.count)));
      const nodeAlpha = (n: any) => 0.22 + 0.78 * (n.count / (maxAtPos.get(n.pos) || 1));
      const maxVal = Math.max(...graph.links.map((l: any) => l.value), 1);
      const linkAlpha = (l: any) => 0.08 + 0.5 * (l.value / maxVal);

      const linkSel = svg
        .append("g")
        .attr("fill", "none")
        .selectAll("path")
        .data(shownLinks)
        .join("path")
        .attr("d", sankeyLinkHorizontal())
        .attr("stroke", (l: any) => color(l.source.pos))
        .attr("stroke-width", (l: any) => Math.max(1, l.width))
        .attr("stroke-opacity", (l: any) => linkAlpha(l));

      // The dominant trajectory through a link: walk the max-flow chain backward from its
      // source and forward from its target, collecting the ordered token sequence.
      const tracePath = (link: any) => {
        const set = new Set<any>([link]);
        let n = link.source;
        const back = [n.name];
        for (let g = 0; g < 64 && n.targetLinks?.length; g++) {
          const best = n.targetLinks.reduce((a: any, b: any) => (b.value > a.value ? b : a));
          set.add(best); n = best.source; back.unshift(n.name);
        }
        n = link.target;
        const fwd = [n.name];
        for (let g = 0; g < 64 && n.sourceLinks?.length; g++) {
          const best = n.sourceLinks.reduce((a: any, b: any) => (b.value > a.value ? b : a));
          set.add(best); n = best.target; fwd.push(n.name);
        }
        return { set, tokens: [...back, ...fwd] };
      };
      const restoreLinks = () =>
        linkSel.attr("stroke-opacity", (l: any) => linkAlpha(l)).attr("stroke-width", (l: any) => Math.max(1, l.width));
      linkSel
        .on("mousemove", (event, l: any) => {
          const { set, tokens } = tracePath(l);
          linkSel
            .attr("stroke-opacity", (ll: any) => (set.has(ll) ? 0.92 : 0.05))
            .attr("stroke-width", (ll: any) => Math.max(1, ll.width) * (set.has(ll) ? 1.5 : 1));
          showTip(event, `trajectory (${l.value} particles):\n${tokens.join(" → ")}`);
        })
        .on("mouseleave", () => { restoreLinks(); hideTip(); });

      const node = svg.append("g").selectAll("g").data(shownNodes).join("g");
      const rectSel = node
        .append("rect")
        .attr("x", (n: any) => n.x0)
        .attr("y", (n: any) => n.y0)
        .attr("width", (n: any) => n.x1 - n.x0)
        .attr("height", (n: any) => Math.max(1, n.y1 - n.y0))
        .attr("fill", (n: any) => color(n.pos))
        .attr("rx", 2)
        .attr("stroke", "none")
        .attr("opacity", (n: any) => nodeAlpha(n)) // transparency = probability at this timepoint
        .on("mousemove", (event, n: any) => {
          // highlight the hovered token + the links that touch it
          rectSel.attr("opacity", (m: any) => (m === n ? 1 : nodeAlpha(m) * 0.4));
          rectSel.filter((m: any) => m === n).attr("stroke", "#eaf0ff").attr("stroke-width", 2);
          const touch = new Set<any>([...(n.sourceLinks ?? []), ...(n.targetLinks ?? [])]);
          linkSel.attr("stroke-opacity", (l: any) => (touch.has(l) ? 0.92 : 0.05));
          showTip(event, `position ${n.pos}: ${n.name}   ${n.count} particles`);
        })
        .on("mouseleave", () => {
          rectSel.attr("opacity", (m: any) => nodeAlpha(m)).attr("stroke", "none");
          restoreLinks();
          hideTip();
        });
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
  }
</script>

<section class="viz panel" data-testid="viz-sankey" data-ready={data ? 1 : 0}>
  <header>
    <div>
      <h2>Token sequences as a Sankey diagram</h2>
      <p class="sub">A particle swarm samples next tokens across positions. <b>A token's opacity = its share of the combined distribution at that timepoint; flow width = particle count.</b> Every flow starts at the prompt and ends where its particles stop. Hover a flow for its trajectory; hover a token to highlight it.</p>
    </div>
    <div class="tools">
      {#if data}<button class="play" onclick={togglePlay} data-testid="sankey-play">{playing ? "⏸ Pause" : "▶ Play"}</button>{/if}
      <ExportBar name="sankey" svg={() => svgEl} anim={exportAnim} />
    </div>
  </header>
  {#if loading}<div class="loading"><Progress {progress} message={progressMsg} /></div>{/if}
  {#if error}<div class="error" data-testid="viz-sankey-error">{error}</div>{/if}
  <svg bind:this={svgEl} class="canvas" height="560" data-testid="sankey-svg"></svg>
  {#if data}<p class="caption">{data.nodes.length} token nodes · {data.links.length} transitions · {$nParticles} particles · up to {$nSteps} steps · opacity = probability at each position</p>{/if}
</section>

<style>
  .viz { padding: 1.2rem 1.4rem; display: flex; flex-direction: column; gap: 0.8rem; }
  header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
  header > div:first-child { min-width: 0; }
  .tools { display: flex; flex-direction: column; align-items: flex-end; gap: 0.4rem; flex-shrink: 0; }
  .play { background: var(--accent-grad); color: #0b0e14; border: none; border-radius: 8px; padding: 0.3rem 0.7rem; font-size: 0.8rem; font-weight: 600; cursor: pointer; white-space: nowrap; }
  header h2 { margin: 0; font-size: 1.1rem; }
  .sub { margin: 0.2rem 0 0; color: var(--text-dim); font-size: 0.82rem; }
  .loading { padding: 0.3rem 0; }
  .error { background: rgba(255,122,144,0.12); color: var(--bad); border: 1px solid rgba(255,122,144,0.3); border-radius: 10px; padding: 0.6rem 0.8rem; font-family: var(--mono); font-size: 0.85rem; }
  .canvas { width: 100%; display: block; background: rgba(0,0,0,0.15); border-radius: 12px; }
  .caption { margin: 0; color: var(--text-dim); font-size: 0.76rem; }
</style>
