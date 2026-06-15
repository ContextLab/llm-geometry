<script lang="ts">
  import * as d3 from "d3";
  import { modelId, prefixText, temperature, layer, responseText } from "../lib/stores";
  import { client, type VectorField } from "../lib/dataClient";
  import Progress from "../lib/Progress.svelte";

  // Visualization 1 — each grid reference token's predicted next token, drawn as an
  // arrow in the 2D-reduced embedding space (project_description.md §1).
  let loading = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let data = $state<VectorField | null>(null);
  let svgEl: SVGSVGElement | undefined;

  const GRID_N = 14;
  const REF = 120;
  const FANOUT = 4;
  const SEED = 0;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;

  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const lyr = $layer;
    const resp = $responseText;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => void load(m, pfx, temp, lyr, resp), 350);
    return () => {
      if (debounce) clearTimeout(debounce);
    };
  });

  async function load(m: string, pfx: string, temp: number, lyr: number, resp: string) {
    const my = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = "starting…";
    try {
      const params = { temperature: temp, layer: lyr, grid_n: GRID_N, fanout: FANOUT, reference_set_size: REF, seed: SEED };
      const inputs = { prefix_text: pfx, response_text: resp };
      await client.ensureArtifact("vector_field", m, params, inputs, (p, msg) => {
        if (my === runId) {
          progress = p;
          progressMsg = msg;
        }
      });
      if (my !== runId) return;
      data = await client.getVectorField(m, { prefix_text: pfx, response_text: resp, ...params });
      if (my !== runId) return;
      draw();
    } catch (e: any) {
      if (my === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (my === runId) loading = false;
    }
  }

  function robust(vals: number[]): [number, number] {
    const s = [...vals].sort((a, b) => a - b);
    let lo = d3.quantileSorted(s, 0.02) ?? s[0];
    let hi = d3.quantileSorted(s, 0.98) ?? s[s.length - 1];
    if (hi <= lo) {
      lo -= 1;
      hi += 1;
    }
    return [lo, hi];
  }

  function draw() {
    if (!svgEl || !data) return;
    const w = svgEl.clientWidth || 640;
    const h = 480;
    const svg = d3.select(svgEl).attr("viewBox", `0 0 ${w} ${h}`);
    svg.selectAll("*").remove();
    const d = data;

    const xs = [...d.starts.map((s) => s[0]), ...d.ends.map((e) => e[0])];
    const ys = [...d.starts.map((s) => s[1]), ...d.ends.map((e) => e[1])];
    const x = d3.scaleLinear().domain(robust(xs)).range([30, w - 30]).clamp(true);
    const y = d3.scaleLinear().domain(robust(ys)).range([h - 30, 30]).clamp(true);

    svg
      .append("defs")
      .append("marker")
      .attr("id", "vf-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 8)
      .attr("refY", 5)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M0,0L10,5L0,10z")
      .attr("fill", "#8fb3ff");

    const maxp = Math.max(...d.probs, 1e-6);
    const rows = d.starts.map((s, i) => ({ s, e: d.ends[i], p: d.probs[i], i }));

    const g = svg.append("g");
    g.selectAll("line")
      .data(rows)
      .join("line")
      .attr("x1", (r) => x(r.s[0]))
      .attr("y1", (r) => y(r.s[1]))
      .attr("x2", (r) => x(r.e[0]))
      .attr("y2", (r) => y(r.e[1]))
      .attr("stroke", "#6ea8fe")
      .attr("stroke-width", 1.2)
      .attr("opacity", (r) => 0.12 + 0.8 * (r.p / maxp))
      .attr("marker-end", "url(#vf-arrow)")
      .append("title")
      .text((r) => `${d.start_token_strs[r.i]} → ${d.end_token_strs[r.i]}  (${(r.p * 100).toFixed(1)}%)`);

    g.selectAll("circle")
      .data(d.starts)
      .join("circle")
      .attr("cx", (s) => x(s[0]))
      .attr("cy", (s) => y(s[1]))
      .attr("r", 1.7)
      .attr("fill", "#b794f6")
      .attr("opacity", 0.6);

    // Response trajectory: a path through the same space, points colored by probability.
    if (d.trajectory && d.trajectory.length) {
      const traj = d.trajectory;
      const tp = d.trajectory_probs ?? [];
      const tl = svg.append("g");
      for (let i = 0; i < traj.length - 1; i++) {
        tl.append("line")
          .attr("x1", x(traj[i][0])).attr("y1", y(traj[i][1]))
          .attr("x2", x(traj[i + 1][0])).attr("y2", y(traj[i + 1][1]))
          .attr("stroke", "#5be0b0").attr("stroke-width", 2.5).attr("opacity", 0.85);
      }
      const cscale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
      tl.selectAll("circle.tp")
        .data(traj)
        .join("circle")
        .attr("class", "tp")
        .attr("cx", (t) => x(t[0]))
        .attr("cy", (t) => y(t[1]))
        .attr("r", 5)
        .attr("fill", (_t, i) => cscale(tp[i] ?? 0))
        .attr("stroke", "#fff")
        .attr("stroke-width", 1)
        .append("title")
        .text((_t, i) => `${d.trajectory_token_strs?.[i] ?? ""} (${((tp[i] ?? 0) * 100).toFixed(1)}%)`);
    }
  }
</script>

<section class="viz panel" data-testid="viz-vector" data-ready={data ? 1 : 0}>
  <header>
    <div>
      <h2>Transformer layers as a vector field</h2>
      <p class="sub">Each reference token points to its most-likely next token, in PCA embedding space.</p>
    </div>
  </header>
  {#if loading}<div class="loading"><Progress {progress} message={progressMsg} /></div>{/if}
  {#if error}<div class="error" data-testid="viz-vector-error">{error}</div>{/if}
  <svg bind:this={svgEl} class="canvas" height="480" data-testid="vector-svg"></svg>
  {#if data}<p class="caption">{data.reference_points} reference points · {data.starts.length} arrows · layer {data.layer}/{data.num_layers} · fan-out {data.fanout}{#if data.trajectory} · trajectory {data.trajectory.length} tokens{/if}</p>{/if}
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
