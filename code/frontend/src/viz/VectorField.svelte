<script lang="ts">
  import * as d3 from "d3";
  import { untrack } from "svelte";
  import { get } from "svelte/store";
  import { modelId, prefixText, temperature, layerFrom, layerTo, responseText, responseStep, responseTokenCount, refreshNonce, fanout, modelError } from "../lib/stores";
  import { robustMax } from "../lib/vizMath";
  import { client, type VectorField, type VectorFieldAnimation } from "../lib/dataClient";
  import { showTip, hideTip } from "../lib/tooltip";
  import Progress from "../lib/Progress.svelte";
  import ExportBar from "../controls/ExportBar.svelte";

  // Visualization 1 (project_description.md §1) — a macOS "Drift"-style flow field. A regular
  // grid of fixed origins; each origin's arrow points (uniform length) from its nearest
  // reference token toward the token the model predicts comes next (read out at layer m).
  // Only the grid dots, arrows, and the response trajectory are drawn — orientations rotate
  // smoothly as the prompt changes (the origins are prompt-independent). Hover any arrow or
  // origin; add a response + ▶ Play to trace it.
  let loading = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let data = $state<VectorField | null>(null);
  let anim = $state<VectorFieldAnimation | null>(null); // multi-key-frame data when a response is set
  let animTime = $state(0); // continuous key-frame index (interpolated); tweens between steps
  let tweenRaf = 0;
  let svgEl: SVGSVGElement | undefined;
  let stageEl: HTMLDivElement | undefined;

  const H = 520;
  const GRID_N = 24; // ~4× the previous resolution
  const REF = 400;
  const SEED = 0;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;
  let resizeObs: ResizeObserver | undefined;
  let lastW = 0;
  let lastRefresh = 0;

  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const lf = $layerFrom;
    const lt = $layerTo;
    const resp = $responseText;
    const fo = $fanout;
    const rn = $refreshNonce;
    const force = rn !== lastRefresh; // the Recompute button bypasses the cache
    lastRefresh = rn;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => void load(m, pfx, temp, lf, lt, resp, fo, force), force ? 0 : 320);
    return () => {
      if (debounce) clearTimeout(debounce);
    };
  });

  // In animation mode, the response step just tweens the (already-fetched) key frames — no
  // re-fetch — so playback is smooth. tweenTo() reads (and its rAF loop writes) `animTime`,
  // so it MUST run untracked: otherwise every rAF write re-runs this effect and restarts
  // the tween, which then never completes (redteam-vector F1). Depend ONLY on step + anim.
  $effect(() => {
    const step = $responseStep;
    const a = anim;
    if (a) untrack(() => tweenTo(Math.min(step, a.n_frames - 1)));
  });

  $effect(() => {
    if (!stageEl) return;
    resizeObs = new ResizeObserver(() => {
      const w = stageEl?.clientWidth ?? 0;
      if (w && Math.abs(w - lastW) > 1) render();
    });
    resizeObs.observe(stageEl);
    return () => resizeObs?.disconnect();
  });

  async function load(m: string, pfx: string, temp: number, lf: number, lt: number, resp: string, fo: number, force = false) {
    const my = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = force ? "recomputing…" : "starting…";
    try {
      if (resp.trim()) {
        // Animation mode: all key frames over ONE static grid (only the per-vertex token
        // assignment + arrow direction change as the context unfolds).
        const ap = { temperature: temp, layer_to: lt, reference_set_size: 576, grid_n: GRID_N, seed: SEED };
        const ainputs = { prefix_text: pfx, response_text: resp };
        if (!force) {
          await client.ensureArtifact("vector_field_animation", m, ap, ainputs, (p, msg) => {
            if (my === runId) { progress = p; progressMsg = msg; }
          });
        }
        if (my !== runId) return;
        const a = await client.getVectorFieldAnimation(m, { ...ainputs, ...ap, ...(force ? { force: true } : {}) });
        if (my !== runId) return;
        anim = a;
        data = null;
        animTime = Math.min(get(responseStep), a.n_frames - 1); // rest = current step (full = last frame)
        render();
        return;
      }
      const params = {
        temperature: temp, layer_from: lf, layer_to: lt, grid_n: GRID_N,
        fanout: fo, reference_set_size: REF, seed: SEED,
      };
      const inputs = { prefix_text: pfx, response_text: "", response_step: 0 };
      if (!force) {
        await client.ensureArtifact("vector_field", m, params, inputs, (p, msg) => {
          if (my === runId) { progress = p; progressMsg = msg; }
        });
      }
      if (my !== runId) return;
      const vf = await client.getVectorField(m, { ...inputs, ...params, ...(force ? { force: true } : {}) });
      if (my !== runId) return;
      data = vf;
      anim = null;
      render();
    } catch (e: any) {
      if (my === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (my === runId) loading = false;
    }
  }

  // Smoothly tween the continuous key-frame index toward `target` (live ▶ Play).
  function tweenTo(target: number) {
    cancelAnimationFrame(tweenRaf);
    const from = animTime;
    const dur = 650; // ms per key-frame transition
    let t0 = 0;
    const stepFn = (ts: number) => {
      if (!t0) t0 = ts;
      const k = Math.min(1, (ts - t0) / dur);
      const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; // easeInOutQuad
      animTime = from + (target - from) * e;
      render();
      if (k < 1) tweenRaf = requestAnimationFrame(stepFn);
    };
    tweenRaf = requestAnimationFrame(stepFn);
  }

  // Export: interpolate SUB sub-frames per key-frame transition for a smooth, high-frame-rate
  // video (each response step is a key frame, not a literal movie frame).
  const SUB = 12;
  const exportAnim = {
    total: () => (anim ? (anim.n_frames - 1) * SUB : 0),
    fps: 24,
    renderFrame: async (i: number) => {
      if (anim) { cancelAnimationFrame(tweenRaf); animTime = Math.min(i / SUB, anim.n_frames - 1); render(); }
      await new Promise((r) => requestAnimationFrame(() => r(null)));
    },
    restore: async () => { if (anim) { animTime = Math.min(get(responseStep), anim.n_frames - 1); render(); } },
  };

  function robust(vals: number[]): [number, number] {
    const s = [...vals].sort((a, b) => a - b);
    let lo = d3.quantileSorted(s, 0.005) ?? s[0];
    let hi = d3.quantileSorted(s, 0.995) ?? s[s.length - 1];
    if (hi <= lo) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.04;
    return [lo - pad, hi + pad];
  }

  // Scale over the grid + arrows + trajectory. In animation mode the extent spans ALL key
  // frames so the view stays fixed (tokens move within a stable frame).
  function scales(w: number) {
    let xs: number[], ys: number[];
    if (anim) {
      // STATIC grid extent (+ one arrow length of headroom) + the trajectory.
      xs = []; ys = [];
      const L = anim.arrow_len;
      for (const v of anim.grid) { xs.push(v[0] - L, v[0] + L); ys.push(v[1] - L, v[1] + L); }
      for (const t of anim.trajectory) { xs.push(t[0]); ys.push(t[1]); }
    } else {
      const d = data!;
      xs = [...d.starts.map((s) => s[0]), ...d.ends.map((e) => e[0]), ...(d.trajectory ?? []).map((t) => t[0])];
      ys = [...d.starts.map((s) => s[1]), ...d.ends.map((e) => e[1]), ...(d.trajectory ?? []).map((t) => t[1])];
    }
    const x = d3.scaleLinear().domain(robust(xs)).range([28, w - 28]);
    const y = d3.scaleLinear().domain(robust(ys)).range([H - 28, 28]);
    return { x, y };
  }

  function render() {
    if (!svgEl || !stageEl || (!data && !anim)) return;
    const w = stageEl.clientWidth || 640;
    lastW = w;
    const { x, y } = scales(w);
    if (anim) drawAnimation(w, x, y);
    else drawField(w, x, y);
  }

  // Persistent SVG structure so keyed joins can TRANSITION arrow orientations between
  // renders (the "drift" rotation). Origins are a fixed grid → indices align across prompts.
  function ensure() {
    const svg = d3.select(svgEl!);
    if (svg.select("defs").empty()) {
      // Small, fixed-size arrowhead (userSpaceOnUse so it doesn't scale with stroke width).
      svg.append("defs").append("marker")
        .attr("id", "vf-arrow").attr("viewBox", "0 0 10 10")
        .attr("refX", 8).attr("refY", 5).attr("markerWidth", 4.5).attr("markerHeight", 4.5)
        .attr("markerUnits", "userSpaceOnUse").attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M0,1.5 L9,5 L0,8.5 z").attr("fill", "context-stroke");
      const z = svg.append("g").attr("class", "zoom");
      z.append("g").attr("class", "arrows");
      z.append("g").attr("class", "points");   // animation token dots (moving)
      z.append("g").attr("class", "traj");
      z.append("g").attr("class", "origins");  // drawn on top so the hover halos catch the cursor
      // Scroll to zoom, drag to pan; the default view fits the whole field.
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.6, 14])
        .on("zoom", (e) => z.attr("transform", e.transform.toString()));
      svg.call(zoom).on("dblclick.zoom", null);
    }
    return svg;
  }

  function drawField(w: number, x: d3.ScaleLinear<number, number>, y: d3.ScaleLinear<number, number>) {
    const d = data!;
    const svg = ensure().attr("viewBox", `0 0 ${w} ${H}`);
    svg.select("g.points").selectAll("*").remove(); // drift mode has no moving token dots

    // Robust (95th-percentile) normalisation so one outlier arrow at high temperature
    // can't wash out the other ~1000 (redteam-vector F3); values above clamp to full.
    const norm = robustMax(d.probs);
    const pcolor = d3.scaleSequential(d3.interpolatePlasma).domain([-0.15 * norm, norm]).clamp(true);
    const rel = (p: number) => Math.min(1, p / norm);
    const rows = d.starts.map((s, i) => ({ s, e: d.ends[i], p: d.probs[i], i }));

    // Grid origins (one per fixed grid coordinate; dedup since fan-out shares origins).
    // Each is a visible dot plus a larger transparent halo that makes it easy to hover.
    const seen = new Set<string>();
    const origins: { s: number[]; i: number }[] = [];
    d.starts.forEach((s, i) => {
      const k = `${s[0]},${s[1]}`;
      if (!seen.has(k)) { seen.add(k); origins.push({ s, i }); }
    });
    const og = svg.select("g.origins").selectAll<SVGGElement, any>("g.o")
      .data(origins, (o: any) => `${o.s[0]},${o.s[1]}`)
      .join(
        (enter) => {
          const g = enter.append("g").attr("class", "o")
            .attr("transform", (o) => `translate(${x(o.s[0])},${y(o.s[1])})`);
          g.append("circle").attr("class", "hit").attr("r", 6).attr("fill", "transparent");
          g.append("circle").attr("class", "dot").attr("r", 1.8).attr("fill", "#9fb1e6")
            .attr("opacity", 0.75).attr("pointer-events", "none");
          return g;
        },
        (update) => update,
        (exit) => exit.remove(),
      );
    og.select<SVGCircleElement>("circle.hit")
      .on("mousemove", (event, o: any) => showTip(event, `grid vertex · nearest token: "${d.start_token_strs[o.i]}" → predicts "${d.end_token_strs[o.i]}"`))
      .on("mouseleave", hideTip);
    og.transition().duration(500).attr("transform", (o) => `translate(${x(o.s[0])},${y(o.s[1])})`);

    const lyr = d.layer_from === d.layer_to ? `layer ${d.layer_to}` : `layer ${d.layer_from}→${d.layer_to}`;
    const onMove = (event: any, r: any) =>
      showTip(event, `${d.start_token_strs[r.i]} → ${d.end_token_strs[r.i]}\n${lyr} · ${(r.p * 100).toFixed(1)}%`);

    const arrows = svg.select("g.arrows").selectAll<SVGLineElement, any>("line")
      .data(rows, (r: any) => r.i)
      .join(
        (enter) => enter.append("line")
          .attr("x1", (r) => x(r.s[0])).attr("y1", (r) => y(r.s[1]))
          .attr("x2", (r) => x(r.s[0])).attr("y2", (r) => y(r.s[1])) // grow from the origin
          .attr("marker-end", "url(#vf-arrow)")
          .attr("opacity", 0),
        (update) => update,
        (exit) => exit.remove(),
      );
    arrows.on("mousemove", onMove).on("mouseleave", hideTip);
    arrows.transition().duration(600).ease(d3.easeCubicOut)
      .attr("x1", (r) => x(r.s[0])).attr("y1", (r) => y(r.s[1]))
      .attr("x2", (r) => x(r.e[0])).attr("y2", (r) => y(r.e[1]))
      .attr("stroke", (r) => pcolor(r.p))
      .attr("stroke-width", (r) => 1.0 + 1.4 * rel(r.p))
      .attr("opacity", (r) => 0.45 + 0.5 * rel(r.p));

    drawTrajectory(svg, x, y);
  }

  function drawTrajectory(svg: any, x: d3.ScaleLinear<number, number>, y: d3.ScaleLinear<number, number>) {
    const d = data!;
    const g = svg.select("g.traj");
    g.selectAll("*").remove();
    if (!d.trajectory || !d.trajectory.length) return;
    const tp = d.trajectory_probs ?? [];
    // Reveal exactly `response_step` tokens — one new token per frame; the full path only
    // appears on the last frame (at rest, step = full count → the whole trajectory shows).
    const shown = Math.min(d.response_step, d.trajectory.length);
    if (shown <= 0) return;
    const traj = d.trajectory.slice(0, shown);
    const cur = shown - 1;
    for (let i = 0; i < traj.length - 1; i++) {
      g.append("line")
        .attr("x1", x(traj[i][0])).attr("y1", y(traj[i][1]))
        .attr("x2", x(traj[i + 1][0])).attr("y2", y(traj[i + 1][1]))
        .attr("stroke", "#5be0b0").attr("stroke-width", 2.4).attr("opacity", 0.9);
    }
    const cscale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    g.selectAll("circle.tp")
      .data(traj)
      .join("circle")
      .attr("class", "tp")
      .attr("cx", (t: any) => x(t[0])).attr("cy", (t: any) => y(t[1]))
      .attr("r", (_t: any, i: number) => (i === cur ? 9 : 5))
      .attr("fill", (_t: any, i: number) => cscale(tp[i] ?? 0))
      .attr("stroke", (_t: any, i: number) => (i === cur ? "#5be0b0" : "#fff"))
      .attr("stroke-width", (_t: any, i: number) => (i === cur ? 3 : 1))
      .attr("opacity", 1)
      .each(function (this: any, _t: any, i: number) {
        d3.select(this)
          .on("mousemove", (event) =>
            showTip(event, `${d.trajectory_token_strs?.[i] ?? ""}  ${((tp[i] ?? 0) * 100).toFixed(1)}%`))
          .on("mouseleave", hideTip);
      });
  }

  // Animation mode: the grid is STATIC. Per (interpolated) frame, each fixed vertex casts an
  // arrow in the direction of the local flow, and the token it "refers to" (its nearest
  // reference token) is revealed on hover. The arrow direction interpolates smoothly between
  // key frames; the discrete token assignment snaps at the nearest key frame.
  function drawAnimation(w: number, x: d3.ScaleLinear<number, number>, y: d3.ScaleLinear<number, number>) {
    const a = anim!;
    const svg = ensure().attr("viewBox", `0 0 ${w} ${H}`);
    const t = Math.max(0, Math.min(animTime, a.n_frames - 1));
    const f0 = Math.floor(t), f1 = Math.min(f0 + 1, a.n_frames - 1), fr = t - f0;
    const nf = Math.min(a.n_frames - 1, Math.round(t)); // nearest key frame for discrete labels
    const mix = (u: number, v: number) => u + (v - u) * fr;
    const dA = a.dirs[f0], dB = a.dirs[f1], qA = a.probs[f0], qB = a.probs[f1];
    const L = a.arrow_len;
    const tokStr = (id: number) => a.token_strs[String(id)] ?? "";
    const rows = a.grid.map((v, i) => {
      const p = mix(qA[i], qB[i]);
      // interpolate the unit direction, then renormalise so the arrow keeps a fixed length
      let dx = mix(dA[i][0], dB[i][0]), dy = mix(dA[i][1], dB[i][1]);
      const n = Math.hypot(dx, dy) || 1;
      dx /= n; dy /= n;
      return { i, vx: v[0], vy: v[1], ex: v[0] + dx * L, ey: v[1] + dy * L, p };
    });
    // Robust normalisation (see drawField): a single outlier must not blank the field.
    const norm = robustMax(rows.map((r) => r.p));
    const pcolor = d3.scaleSequential(d3.interpolatePlasma).domain([-0.15 * norm, norm]).clamp(true);
    const rel = (p: number) => Math.min(1, p / norm);
    const arrowTip = (r: any) =>
      `"${tokStr(a.from_tokens[nf][r.i])}" → "${tokStr(a.to_tokens[nf][r.i])}"\nlayer ${a.layer_to}/${a.num_layers} · ${(r.p * 100).toFixed(1)}%`;

    svg.select("g.arrows").selectAll<SVGLineElement, any>("line").data(rows, (r: any) => r.i).join("line")
      .attr("x1", (r) => x(r.vx)).attr("y1", (r) => y(r.vy))
      .attr("x2", (r) => x(r.ex)).attr("y2", (r) => y(r.ey))
      .attr("marker-end", "url(#vf-arrow)")
      .attr("stroke", (r) => pcolor(r.p)).attr("stroke-width", (r) => 1.0 + 1.3 * rel(r.p))
      .attr("opacity", (r) => 0.4 + 0.5 * rel(r.p))
      .on("mousemove", (e, r: any) => showTip(e, arrowTip(r)))
      .on("mouseleave", hideTip);

    // STATIC grid vertices: a small dot + a transparent hit halo; hover shows the current token.
    svg.select("g.points").selectAll<SVGGElement, any>("g.o").data(rows, (r: any) => r.i).join(
      (enter) => {
        const g = enter.append("g").attr("class", "o");
        g.append("circle").attr("class", "hit").attr("r", 6).attr("fill", "transparent");
        g.append("circle").attr("class", "dot").attr("r", 1.8).attr("pointer-events", "none");
        return g;
      },
      (update) => update,
      (exit) => exit.remove(),
    )
      .attr("transform", (r) => `translate(${x(r.vx)},${y(r.vy)})`)
      .each(function (this: any, r: any) {
        d3.select(this).select("circle.dot").attr("fill", "#9fb1e6").attr("opacity", 0.75);
        d3.select(this).select<SVGCircleElement>("circle.hit")
          .on("mousemove", (event) => showTip(event, `grid vertex · refers to "${tokStr(a.from_tokens[nf][r.i])}" → predicts "${tokStr(a.to_tokens[nf][r.i])}"  ${(r.p * 100).toFixed(1)}%`))
          .on("mouseleave", hideTip);
      });

    // Trajectory: dots laid down once each token is reached; the line builds in gradually.
    const g = svg.select("g.traj");
    g.selectAll("*").remove();
    const traj = a.trajectory;
    const reached = Math.floor(t);
    const cscale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    const dots = traj.slice(0, Math.min(reached, traj.length));
    const linePts = dots.map((d) => [x(d[0]), y(d[1])]);
    if (reached >= 1 && reached < traj.length && fr > 0) {
      const a0 = traj[reached - 1], a1 = traj[reached];
      linePts.push([x(a0[0] + (a1[0] - a0[0]) * fr), y(a0[1] + (a1[1] - a0[1]) * fr)]);
    }
    if (linePts.length >= 2) {
      g.append("path").attr("d", "M" + linePts.map((p) => p.join(",")).join("L"))
        .attr("fill", "none").attr("stroke", "#5be0b0").attr("stroke-width", 2.4).attr("opacity", 0.9);
    }
    g.selectAll("circle.tp").data(dots).join("circle").attr("class", "tp")
      .attr("cx", (d: any) => x(d[0])).attr("cy", (d: any) => y(d[1]))
      .attr("r", (_d: any, i: number) => (i === reached - 1 ? 7 : 5))
      .attr("fill", (_d: any, i: number) => cscale(a.trajectory_probs[i] ?? 0))
      .attr("stroke", (_d: any, i: number) => (i === reached - 1 ? "#5be0b0" : "#fff")).attr("stroke-width", 1.5)
      .each(function (this: any, _d: any, i: number) {
        d3.select(this).on("mousemove", (event) => showTip(event, `${a.trajectory_token_strs?.[i] ?? ""}  ${((a.trajectory_probs[i] ?? 0) * 100).toFixed(1)}%`)).on("mouseleave", hideTip);
      });
  }
</script>

<section class="viz panel" data-testid="viz-vector" data-ready={data || anim ? 1 : 0}>
  <header>
    <div>
      <h2>Transformer layers as a vector field</h2>
      <p class="sub">Positions are <b>contextual</b> prediction-layer embeddings — a token's representation given the prompt, just before it becomes next-token probabilities. Each grid arrow runs from a reference token (layer <i>n</i>) toward the token it predicts next (layer <i>m</i>); a response trajectory and the whole field shift as the prompt changes. <b>Colour/opacity = probability.</b> Hover any arrow or grid vertex; scroll to zoom, drag to pan; add a response + ▶ Play to trace it.</p>
    </div>
    <ExportBar name="vector-field" svg={() => svgEl} anim={exportAnim} />
  </header>
  {#if loading}<div class="loading"><Progress {progress} message={progressMsg} /></div>{/if}
  {#if error}<div class="error" data-testid="viz-vector-error" title={error}>{error.split("\n")[0].slice(0, 200)}</div>{/if}
  <div bind:this={stageEl} class="stage" class:stale={!!$modelError && !!(data || anim)} style="height:{H}px">
    <svg bind:this={svgEl} class="arrow-layer" data-testid="vector-svg"></svg>
  </div>
  {#if $modelError && (data || anim)}
    <p class="caption stale-note" data-testid="vector-stale-note">⚠ showing previous model — the selected model failed to load</p>
  {/if}
  {#if anim}
    <p class="caption">
      {anim.reference_points} static grid arrows · {anim.n_frames} key frames (response steps) ·
      layer {anim.layer_to}/{anim.num_layers} · the lattice stays fixed; each vertex's token + arrow re-organise as the context unfolds
    </p>
  {:else if data}
    <p class="caption">
      {data.starts.length} grid arrows ·
      layer {data.layer_from}{#if data.layer_to !== data.layer_from}→{data.layer_to}{/if}/{data.num_layers} ·
      fan-out {data.fanout}
    </p>
  {/if}
</section>

<style>
  .viz { padding: 1.2rem 1.4rem; display: flex; flex-direction: column; gap: 0.8rem; }
  header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
  header > div { min-width: 0; }
  header h2 { margin: 0; font-size: 1.1rem; }
  .sub { margin: 0.2rem 0 0; color: var(--text-dim); font-size: 0.82rem; }
  .loading { padding: 0.3rem 0; }
  .error { background: rgba(255,122,144,0.12); color: var(--bad); border: 1px solid rgba(255,122,144,0.3); border-radius: 10px; padding: 0.6rem 0.8rem; font-family: var(--mono); font-size: 0.85rem; }
  .stage { position: relative; width: 100%; background: rgba(0,0,0,0.22); border-radius: 12px; overflow: hidden; }
  .stage.stale { opacity: 0.45; filter: saturate(0.55); transition: opacity 0.2s ease; }
  .arrow-layer { position: absolute; inset: 0; width: 100%; height: 100%; display: block; cursor: crosshair; }
  .caption { margin: 0; color: var(--text-dim); font-size: 0.76rem; }
  .stale-note { color: #ffd166; }
</style>
