<script lang="ts">
  import * as d3 from "d3";
  import { onDestroy } from "svelte";
  import { modelId, prefixText, temperature, nParticles, nSteps, responseText, refreshNonce } from "../lib/stores";
  import { client, type SankeyData, type SankeyNode, type SankeyLink, type SankeyHighlight } from "../lib/dataClient";
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
  let highlight = $state<SankeyHighlight[]>([]); // the user's response path (separate cheap fetch)
  let highlightStrs: Record<string, string> = {};
  let svgEl: SVGSVGElement | undefined;

  const SEED = 0;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;
  let hlRunId = 0;
  let lastRefresh = 0;
  let reveal = $state(9999); // CONTINUOUS column index revealed (9999 = all); ▶ Play sweeps it
  let maxPosNow = $state(0);
  let playing = $state(false);
  let playRaf = 0;

  // The SWARM is prompt-conditioned and RESPONSE-INDEPENDENT, so it reloads only when the prompt
  // / particle settings change — NOT when the response is edited.
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

  // The response highlight is a SEPARATE, cheap overlay (one forward pass) — editing the response
  // refetches only this, so it updates instantly without recomputing the swarm. Deferred via a
  // timer so its draw() (which touches reactive state) runs OUTSIDE the effect flush.
  let hlDebounce: ReturnType<typeof setTimeout> | undefined;
  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const ns = $nSteps;
    const resp = $responseText;
    void $refreshNonce;
    if (hlDebounce) clearTimeout(hlDebounce);
    hlDebounce = setTimeout(() => void loadHighlight(m, pfx, resp, temp, ns), 120);
    return () => { if (hlDebounce) clearTimeout(hlDebounce); };
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
      data = await client.getSankey(m, { ...inputs, ...params, ...(force ? { force: true } : {}) });
      if (my !== runId) return;
      draw();
    } catch (e: any) {
      if (my === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (my === runId) loading = false;
    }
  }

  async function loadHighlight(m: string, pfx: string, resp: string, temp: number, ns: number) {
    const my = ++hlRunId;
    if (resp.trim()) {
      try {
        const r = await client.getSankeyHighlight(m, { prefix_text: pfx, response_text: resp, temperature: temp, n_steps: ns });
        if (my !== hlRunId) return;
        highlight = r.highlight;
        highlightStrs = r.token_strs;
      } catch {
        if (my !== hlRunId) return;
        highlight = [];
        highlightStrs = {};
      }
    } else {
      highlight = [];
      highlightStrs = {};
    }
    if (data) draw(); // redraw the overlay once (the swarm itself didn't change)
  }

  // ▶ Play: sweep the reveal CONTINUOUSLY (rAF) so flows grow in and columns fade up smoothly,
  // matching how the diagram renders in the page (no abrupt column pops, no recompute).
  function stopPlay() {
    playing = false;
    if (playRaf) { cancelAnimationFrame(playRaf); playRaf = 0; }
  }
  function togglePlay() {
    if (playing) { stopPlay(); reveal = 9999; draw(); return; }
    playing = true; reveal = 0; draw();
    const perCol = 650; // ms to grow one column of flows
    let start = -1;
    const step = (ts: number) => {
      if (start < 0) start = ts;
      reveal = Math.min(maxPosNow, (ts - start) / perCol);
      draw();
      if (reveal >= maxPosNow) { stopPlay(); return; }
      playRaf = requestAnimationFrame(step);
    };
    playRaf = requestAnimationFrame(step);
  }
  onDestroy(stopPlay);

  // Export: SUB interpolated sub-frames per column so the GIF/MP4 morph is watchable.
  const SUB = 12;
  const exportAnim = {
    total: () => maxPosNow * SUB,
    fps: 24,
    renderFrame: async (i: number) => { reveal = Math.min(maxPosNow, i / SUB); draw(); await new Promise((r) => requestAnimationFrame(() => r(null))); },
    restore: async () => { reveal = 9999; draw(); },
  };

  // Fixed-row layout: every column shows the SAME ordered set of tokens (token_order) at the
  // SAME y, so a token can be read horizontally across time. A cell's opacity = its share of the
  // swarm at that position; flows connect cells; the user's response is a single gold trace.
  function draw() {
    if (!svgEl || !data) return;
    const w = svgEl.clientWidth || 680;
    const h = 560;
    const svg = d3.select(svgEl).attr("viewBox", `0 0 ${w} ${h}`);
    svg.selectAll("*").remove();
    const d = data;
    const label = (tok: number) => d.token_strs[String(tok)] ?? highlightStrs[String(tok)] ?? String(tok);
    const GOLD = "#ffd166";

    // rows = the swarm's fixed token order, plus any response tokens not already among them (so
    // the gold trace always has a row to sit on, even for an unlikely response).
    const rows = [...(d.token_order ?? [])];
    const inRows = new Set(rows);
    for (const hh of highlight) if (!inRows.has(hh.token)) { inRows.add(hh.token); rows.push(hh.token); }
    const nRows = rows.length;
    if (!nRows) {
      svg.append("text").attr("x", w / 2).attr("y", h / 2).attr("fill", "var(--text-dim)")
        .attr("text-anchor", "middle")
        .text("Not enough transitions to draw flows — try a longer prompt or higher temperature.");
      maxPosNow = 0;
      return;
    }
    const hlPos = highlight.length ? Math.max(...highlight.map((x) => x.pos)) : 0;
    const maxPos = Math.max(d.max_pos ?? 0, hlPos, 1);
    maxPosNow = maxPos;
    const rankOf = new Map<number, number>(rows.map((t, i) => [t, i]));
    const hlTokens = new Set(highlight.map((x) => x.token));
    const hlCell = new Set(highlight.map((x) => `${x.pos}:${x.token}`));

    // node lookup by (pos, token); also the busiest cell per position (for opacity)
    const nodeAt = new Map<string, SankeyNode>();
    const maxAtPos = new Map<number, number>();
    for (const n of d.nodes) {
      nodeAt.set(`${n.pos}:${n.token}`, n);
      maxAtPos.set(n.pos, Math.max(maxAtPos.get(n.pos) ?? 0, n.count));
    }

    // CONTINUOUS reveal: columns ≤ full are solid; column full+1 grows in with fraction fr.
    const rev = Math.min(reveal, maxPos);
    const full = Math.floor(rev);
    const fr = rev - full;
    const revFactor = (pos: number) => (pos <= full ? 1 : pos === full + 1 ? fr : 0);

    // geometry
    const labelW = 78, top = 18, bottom = h - 30;
    const tx = d3.scaleLinear().domain([0, maxPos]).range([labelW + 16, w - 18]);
    const rowH = (bottom - top) / nRows;
    const yC = (rank: number) => top + (rank + 0.5) * rowH;
    const colGap = maxPos > 0 ? tx(1) - tx(0) : w - labelW;
    const nodeW = Math.max(7, Math.min(15, colGap * 0.26));
    const cellH = Math.min(rowH * 0.72, 20);
    const color = d3.scaleSequential(d3.interpolateCool).domain([0, maxPos]);

    // axis: column guides + position labels
    const axis = svg.append("g");
    for (let p = 0; p <= maxPos; p++) {
      axis.append("line").attr("x1", tx(p)).attr("x2", tx(p)).attr("y1", top - 5).attr("y2", bottom + 5)
        .attr("stroke", "var(--border)").attr("stroke-opacity", 0.16);
      axis.append("text").attr("x", tx(p)).attr("y", h - 12).attr("text-anchor", "middle")
        .attr("fill", "var(--text-dim)").attr("font-size", "10px").attr("font-family", "var(--mono)")
        .attr("opacity", revFactor(p) ? 1 : 0.3).text(p === 0 ? "prompt" : `+${p}`);
    }

    // row labels (left gutter) + faint guide line — identical token order for every column
    const rowG = svg.append("g");
    rows.forEach((tok, r) => {
      rowG.append("line").attr("x1", labelW + 8).attr("x2", w - 12).attr("y1", yC(r)).attr("y2", yC(r))
        .attr("stroke", hlTokens.has(tok) ? GOLD : "var(--border)").attr("stroke-opacity", hlTokens.has(tok) ? 0.18 : 0.07);
      rowG.append("text").attr("x", labelW).attr("y", yC(r)).attr("dy", "0.32em").attr("text-anchor", "end")
        .attr("fill", hlTokens.has(tok) ? GOLD : "#cdd6ec").attr("font-size", "10px").attr("font-family", "var(--mono)")
        .text(label(tok));
    });

    // flow graph (for the hover trace): out/in links keyed by (pos, token)
    const outL = new Map<string, SankeyLink[]>(), inL = new Map<string, SankeyLink[]>();
    const push = (m: Map<string, SankeyLink[]>, k: string, v: SankeyLink) => { const a = m.get(k); if (a) a.push(v); else m.set(k, [v]); };
    for (const l of d.links) { push(outL, `${l.pos}:${l.source_token}`, l); push(inL, `${l.pos + 1}:${l.target_token}`, l); }
    const tracePath = (link: SankeyLink) => {
      const set = new Set<SankeyLink>([link]);
      const toks = [label(link.source_token), label(link.target_token)];
      let cur = link;
      for (let g = 0; g < 64; g++) { const ins = inL.get(`${cur.pos}:${cur.source_token}`); if (!ins?.length) break; const best = ins.reduce((a, b) => (b.value > a.value ? b : a)); if (set.has(best)) break; set.add(best); toks.unshift(label(best.source_token)); cur = best; }
      cur = link;
      for (let g = 0; g < 64; g++) { const outs = outL.get(`${cur.pos + 1}:${cur.target_token}`); if (!outs?.length) break; const best = outs.reduce((a, b) => (b.value > a.value ? b : a)); if (set.has(best)) break; set.add(best); toks.push(label(best.target_token)); cur = best; }
      return { set, tokens: toks };
    };

    // flows: cubic ribbons between consecutive cells, width/opacity ∝ particle count
    const maxVal = Math.max(...d.links.map((l) => l.value), 1);
    const linkAlpha = (l: SankeyLink) => 0.05 + 0.5 * (l.value / maxVal);
    const growF = (l: SankeyLink) => (l.pos + 1 === full + 1 ? fr : 1);
    const linkData = d.links.filter((l) => rankOf.has(l.source_token) && rankOf.has(l.target_token) && revFactor(l.pos + 1) > 0);
    const linkPath = (l: SankeyLink) => {
      const x0 = tx(l.pos) + nodeW / 2, x1 = tx(l.pos + 1) - nodeW / 2;
      const y0 = yC(rankOf.get(l.source_token)!), y1 = yC(rankOf.get(l.target_token)!);
      const mx = (x0 + x1) / 2;
      return `M${x0},${y0}C${mx},${y0} ${mx},${y1} ${x1},${y1}`;
    };
    const linkW = (l: SankeyLink) => Math.max(1, (l.value / maxVal) * (rowH * 0.8));
    const linkSel = svg.append("g").attr("fill", "none").selectAll("path").data(linkData).join("path")
      .attr("d", linkPath as any)
      .attr("stroke", (l) => color(l.pos))
      .attr("stroke-width", linkW as any)
      .attr("stroke-linecap", "round")
      .attr("pathLength", 1)
      .attr("stroke-dasharray", (l) => (l.pos + 1 === full + 1 ? `${fr} 1` : null))
      .attr("stroke-opacity", (l) => linkAlpha(l) * growF(l));
    const restoreLinks = () => linkSel.attr("stroke-opacity", (l) => linkAlpha(l) * growF(l)).attr("stroke-width", linkW as any);
    linkSel
      .on("mousemove", (event, l) => {
        const { set, tokens } = tracePath(l);
        let logp = 0, nseg = 0;
        set.forEach((ll) => { if (typeof ll.cond === "number" && ll.cond > 0) { logp += Math.log(ll.cond); nseg++; } });
        const pPct = nseg ? `${(Math.exp(logp) * 100).toPrecision(2)}%  ·  log p = ${logp.toFixed(2)}` : "";
        linkSel.attr("stroke-opacity", (ll) => (set.has(ll) ? 0.95 : 0.04)).attr("stroke-width", (ll) => linkW(ll) * (set.has(ll) ? 1.4 : 1));
        showTip(event, `${tokens.join(" → ")}\nstep ${l.pos}→${l.pos + 1}: P=${(l.cond * 100).toFixed(1)}% · ${l.value} particles\ntrajectory ${pPct}`);
      })
      .on("mouseleave", () => { restoreLinks(); hideTip(); });

    // cells: every (row token × position), opacity = share at that position (faint floor so the
    // full token×time grid is always visible — "the same tokens at every timestep").
    type Cell = { pos: number; token: number; rank: number; count: number; prob: number; highlight: boolean };
    const cells: Cell[] = [];
    for (const tok of rows) {
      const r = rankOf.get(tok)!;
      for (let p = 0; p <= maxPos; p++) {
        if (revFactor(p) <= 0) continue;
        const n = nodeAt.get(`${p}:${tok}`);
        cells.push({ pos: p, token: tok, rank: r, count: n?.count ?? 0, prob: n?.prob ?? 0, highlight: hlCell.has(`${p}:${tok}`) });
      }
    }
    const cellAlpha = (c: Cell) => (c.count > 0 ? 0.14 + 0.86 * (c.count / (maxAtPos.get(c.pos) || 1)) : 0.05) * revFactor(c.pos);
    const rectSel = svg.append("g").selectAll<SVGRectElement, Cell>("rect").data(cells).join("rect")
      .attr("x", (c) => tx(c.pos) - nodeW / 2).attr("y", (c) => yC(c.rank) - cellH / 2)
      .attr("width", nodeW).attr("height", cellH).attr("rx", 2)
      .attr("fill", (c) => color(c.pos))
      .attr("opacity", cellAlpha)
      .on("mousemove", (event, c) => {
        rectSel.attr("opacity", (m) => (m === c ? Math.max(cellAlpha(m), 0.95) : cellAlpha(m) * 0.5));
        const touch = new Set(d.links.filter((l) => (l.pos === c.pos && l.source_token === c.token) || (l.pos + 1 === c.pos && l.target_token === c.token)));
        linkSel.attr("stroke-opacity", (l) => (touch.has(l) ? 0.95 : 0.04));
        const tag = c.highlight ? " · ✦ your response" : "";
        showTip(event, `position ${c.pos}: "${label(c.token)}"\nP = ${(c.prob * 100).toFixed(2)}%${tag} · ${c.count} particles`);
      })
      .on("mouseleave", () => { rectSel.attr("opacity", cellAlpha); restoreLinks(); hideTip(); });

    // the user's response: a SINGLE gold trace through the fixed rows (built in with the columns)
    if (highlight.length) {
      const pts = highlight
        .filter((hh) => rankOf.has(hh.token) && revFactor(hh.pos) > 0)
        .map((hh) => ({ x: tx(hh.pos), y: yC(rankOf.get(hh.token)!), h: hh }));
      const hg = svg.append("g").attr("data-testid", "sankey-highlight");
      if (pts.length >= 2) {
        hg.append("path").attr("d", "M" + pts.map((p) => `${p.x},${p.y}`).join("L"))
          .attr("fill", "none").attr("stroke", GOLD).attr("stroke-width", 2.5).attr("stroke-opacity", 0.92).attr("stroke-linejoin", "round");
      }
      hg.selectAll("circle").data(pts).join("circle")
        .attr("cx", (p) => p.x).attr("cy", (p) => p.y).attr("r", 4.5)
        .attr("fill", GOLD).attr("stroke", "#1a1408").attr("stroke-width", 1)
        .on("mousemove", (event, p) => showTip(event, `✦ your response · "${label(p.h.token)}"\nP = ${(p.h.prob * 100).toPrecision(3)}% (teacher-forced)`))
        .on("mouseleave", hideTip);
    }
  }
</script>

<section class="viz panel" data-testid="viz-sankey" data-ready={data ? 1 : 0}>
  <header>
    <div>
      <h2>Token sequences as a Sankey diagram</h2>
      <p class="sub">A swarm of particles samples <b>actual responses to the prompt</b> — each starts from the context and samples its own continuation until it stops. <b>Token opacity = its share of the swarm; flow width = particle count.</b> Add a response to <b>highlight that exact path in gold</b>. Hover any node or flow for probabilities.</p>
    </div>
    <div class="tools">
      {#if data}<button class="play" onclick={togglePlay} data-testid="sankey-play">{playing ? "⏸ Pause" : "▶ Play"}</button>{/if}
      <ExportBar name="sankey" svg={() => svgEl} anim={exportAnim} />
    </div>
  </header>
  {#if loading}<div class="loading"><Progress {progress} message={progressMsg} /></div>{/if}
  {#if error}<div class="error" data-testid="viz-sankey-error">{error}</div>{/if}
  <svg bind:this={svgEl} class="canvas" height="560" data-testid="sankey-svg"></svg>
  {#if data}<p class="caption">{data.token_order.length} token rows · {data.links.length} transitions · {data.n_particles} particles · up to {$nSteps} steps{#if highlight.length} · <span class="gold">✦ {highlight.length}-token response highlighted</span>{/if}</p>{/if}
</section>

<style>
  .viz { padding: 1.2rem 1.4rem; display: flex; flex-direction: column; gap: 0.8rem; }
  header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
  header > div:first-child { min-width: 0; }
  .tools { display: flex; flex-direction: column; align-items: flex-end; gap: 0.4rem; flex-shrink: 0; }
  .play { background: var(--accent-grad); color: #0b0e14; border: none; border-radius: 8px; padding: 0.3rem 0.7rem; font-size: 0.8rem; font-weight: 600; cursor: pointer; white-space: nowrap; }
  .gold { color: #ffd166; }
  header h2 { margin: 0; font-size: 1.1rem; }
  .sub { margin: 0.2rem 0 0; color: var(--text-dim); font-size: 0.82rem; }
  .loading { padding: 0.3rem 0; }
  .error { background: rgba(255,122,144,0.12); color: var(--bad); border: 1px solid rgba(255,122,144,0.3); border-radius: 10px; padding: 0.6rem 0.8rem; font-family: var(--mono); font-size: 0.85rem; }
  .canvas { width: 100%; display: block; background: rgba(0,0,0,0.15); border-radius: 12px; }
  .caption { margin: 0; color: var(--text-dim); font-size: 0.76rem; }
</style>
