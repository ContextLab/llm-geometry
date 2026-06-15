<script lang="ts">
  import * as d3 from "d3";
  import { sankey as d3sankey, sankeyLinkHorizontal } from "d3-sankey";
  import { modelId, prefixText, temperature, responseText, responseStep, layerFrom, layerTo } from "../lib/stores";
  import { client, type SankeyData } from "../lib/dataClient";
  import { showTip, hideTip } from "../lib/tooltip";
  import Progress from "../lib/Progress.svelte";

  // Visualization 2 — particle-swarm next-token sampling across positions, as a Sankey
  // diagram (project_description.md §2).
  let loading = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let data = $state<SankeyData | null>(null);
  let svgEl: SVGSVGElement | undefined;

  const N_PARTICLES = 600;
  const N_STEPS = 8;
  const SEED = 0;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;

  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const resp = $responseText;
    const step = $responseStep;
    void $layerFrom; void $layerTo; // refresh on ANY control change (layers don't alter the swarm)
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => void load(m, pfx, temp, resp, step), 350);
    return () => {
      if (debounce) clearTimeout(debounce);
    };
  });

  async function load(m: string, pfx: string, temp: number, resp: string, step: number) {
    const my = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = "starting…";
    try {
      const params = { temperature: temp, n_particles: N_PARTICLES, n_steps: N_STEPS, seed: SEED };
      const inputs = { prefix_text: pfx, response_text: resp, response_step: step };
      await client.ensureArtifact("sankey", m, params, inputs, (p, msg) => {
        if (my === runId) {
          progress = p;
          progressMsg = msg;
        }
      });
      if (my !== runId) return;
      data = await client.getSankey(m, { prefix_text: pfx, response_text: resp, response_step: step, ...params });
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
    const h = 560;
    const STRIP = 104; // top band for the per-position combined distribution
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

    const xByPos = new Map<number, number>();

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
        .nodeWidth(14)
        .nodePadding(9)
        .extent([[8, STRIP + 12], [w - 8, h - 12]]);
      const graph = layout({
        nodes: nodes.map((n) => ({ ...n })),
        links: links.map((l) => ({ ...l })),
      });

      const linkSel = svg
        .append("g")
        .attr("fill", "none")
        .selectAll("path")
        .data(graph.links)
        .join("path")
        .attr("d", sankeyLinkHorizontal())
        .attr("stroke", (l: any) => color(l.source.pos))
        .attr("stroke-width", (l: any) => Math.max(1, l.width))
        .attr("stroke-opacity", 0.4);

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
        linkSel.attr("stroke-opacity", 0.4).attr("stroke-width", (l: any) => Math.max(1, l.width));
      linkSel
        .on("mousemove", (event, l: any) => {
          const { set, tokens } = tracePath(l);
          linkSel
            .attr("stroke-opacity", (ll: any) => (set.has(ll) ? 0.92 : 0.05))
            .attr("stroke-width", (ll: any) => Math.max(1, ll.width) * (set.has(ll) ? 1.5 : 1));
          showTip(event, `trajectory (${l.value} particles):\n${tokens.join(" → ")}`);
        })
        .on("mouseleave", () => { restoreLinks(); hideTip(); });

      const node = svg.append("g").selectAll("g").data(graph.nodes).join("g");
      const rectSel = node
        .append("rect")
        .attr("x", (n: any) => n.x0)
        .attr("y", (n: any) => n.y0)
        .attr("width", (n: any) => n.x1 - n.x0)
        .attr("height", (n: any) => Math.max(1, n.y1 - n.y0))
        .attr("fill", (n: any) => color(n.pos))
        .attr("rx", 2)
        .attr("stroke", "none")
        .on("mousemove", (event, n: any) => {
          // highlight the hovered token + the links that touch it
          rectSel.attr("opacity", (m: any) => (m === n ? 1 : 0.35));
          rectSel.filter((m: any) => m === n).attr("stroke", "#eaf0ff").attr("stroke-width", 2);
          const touch = new Set<any>([...(n.sourceLinks ?? []), ...(n.targetLinks ?? [])]);
          linkSel.attr("stroke-opacity", (l: any) => (touch.has(l) ? 0.92 : 0.05));
          showTip(event, `position ${n.pos}: ${n.name}   ${n.count} particles`);
        })
        .on("mouseleave", () => {
          rectSel.attr("opacity", 1).attr("stroke", "none");
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

      graph.nodes.forEach((n: any) => {
        if (!xByPos.has(n.pos)) xByPos.set(n.pos, (n.x0 + n.x1) / 2);
      });
    }

    // Per-position combined next-token distribution (violin-style stacked bars) on top.
    const pp = d.per_position ?? [];
    if (pp.length) {
      const positions = pp.map((e) => e.pos);
      if (xByPos.size === 0) {
        positions.forEach((p, i) => xByPos.set(p, 24 + (i + 0.5) * ((w - 48) / positions.length)));
      }
      const colW =
        positions.length > 1
          ? Math.abs(
              (xByPos.get(positions[positions.length - 1])! - xByPos.get(positions[0])!) /
                (positions.length - 1),
            )
          : w / 2;
      const barMax = Math.max(24, colW * 0.82);
      const rowH = 16;
      const strip = svg.append("g");
      strip
        .append("text")
        .attr("x", 8)
        .attr("y", 12)
        .attr("fill", "var(--text-dim)")
        .attr("font-size", "11px")
        .text("combined next-token distribution per position");
      pp.forEach((entry) => {
        const cx = xByPos.get(entry.pos);
        if (cx === undefined) return;
        const top = entry.top.slice(0, 5);
        const maxp = Math.max(...top.map((t) => t.prob), 1e-6);
        top.forEach((t, i) => {
          const bw = (t.prob / maxp) * barMax;
          const y = 24 + i * rowH;
          strip
            .append("rect")
            .attr("x", cx - bw / 2)
            .attr("y", y)
            .attr("width", bw)
            .attr("height", rowH - 4)
            .attr("rx", 2)
            .attr("fill", color(entry.pos))
            .attr("opacity", 0.85)
            .on("mousemove", (event) => showTip(event, `position ${entry.pos}: ${label(t.token)}   ${(t.prob * 100).toFixed(1)}%`))
            .on("mouseleave", hideTip);
          if (i === 0) {
            strip
              .append("text")
              .attr("x", cx)
              .attr("y", y - 2)
              .attr("text-anchor", "middle")
              .attr("fill", "#cdd6ec")
              .attr("font-size", "9px")
              .attr("font-family", "var(--mono)")
              .text(label(t.token).slice(0, 9));
          }
        });
      });
    }
  }
</script>

<section class="viz panel" data-testid="viz-sankey" data-ready={data ? 1 : 0}>
  <header>
    <div>
      <h2>Token sequences as a Sankey diagram</h2>
      <p class="sub">A particle swarm samples next tokens across positions. <b>Top strip = the full combined distribution per position; flow width = particle count.</b> Hover a flow to light up its full trajectory + token sequence; hover a token to highlight it.</p>
    </div>
  </header>
  {#if loading}<div class="loading"><Progress {progress} message={progressMsg} /></div>{/if}
  {#if error}<div class="error" data-testid="viz-sankey-error">{error}</div>{/if}
  <svg bind:this={svgEl} class="canvas" height="560" data-testid="sankey-svg"></svg>
  {#if data}<p class="caption">top: combined distribution per position · below: {data.nodes.length} nodes · {data.links.length} transitions · {N_PARTICLES} particles</p>{/if}
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
