<script lang="ts">
  import { onDestroy } from "svelte";
  import MatrixHeatmap from "../../lib/MatrixHeatmap.svelte";
  import StaticNotice from "../../lib/StaticNotice.svelte";
  import type { ArchGraph, ArchNode, ArchTrace } from "../../lib/dataClient";
  import { showTip, hideTip } from "../../lib/tooltip";

  // "Processing breakdown": tokenization strip + an animated walk through the REAL
  // forward pass (trace.node_activations, in execution order — nothing canned) +
  // per-layer detail (per-head attention heatmap, residual-stream norms, top-10
  // logits). The playhead reports the active node upward so the diagram highlights it.
  interface Props {
    trace: ArchTrace | null;
    graph: ArchGraph | null;
    loading?: boolean;
    error?: string;
    /** Static build (feature 003): "this prompt isn't among the precomputed example
     * traces" — rendered as a designed affordance (not an error) while any previous
     * trace stays visible underneath (FR-203). */
    staticNote?: string;
    onHighlight?: (nodeId: string | null) => void;
    onRetry?: () => void;
  }
  let {
    trace,
    graph,
    loading = false,
    error = "",
    staticNote = "",
    onHighlight,
    onRetry,
  }: Props = $props();

  onDestroy(() => {
    hideTip();
    cancelAnimationFrame(raf);
    onHighlight?.(null);
  });

  const order = $derived(trace?.node_activations ?? []);
  const nodeById = $derived.by(() => {
    const m = new Map<string, ArchNode>();
    for (const n of graph?.nodes ?? []) m.set(n.id, n);
    return m;
  });

  // --- playback (play/pause + scrub + speed over the traced execution order) ------
  let idx = $state(0);
  let playing = $state(false);
  let speed = $state(1);
  let engaged = $state(false); // no highlight until the user plays or scrubs
  let layerSel = $state(0);
  let headSel = $state(0);
  // The playhead used to overwrite layerSel unconditionally, so choosing a layer while
  // the trace played was impossible — it snapped back on the next op. Auto-follow is now
  // an explicit mode that touching the layer control turns off.
  let followLayer = $state(true);
  let raf = 0;
  let lastTs = 0;
  let acc = 0;

  // A new trace (fresh tensors) resets the playhead.
  let lastTrace: ArchTrace | null = null;
  $effect(() => {
    const t = trace;
    if (t !== lastTrace) {
      lastTrace = t;
      idx = 0;
      playing = false;
      engaged = false;
      acc = 0;
      lastTs = 0;
      layerSel = Math.min(layerSel, Math.max(0, (t?.layers.length ?? 1) - 1));
    }
  });

  const NODES_PER_SEC = 14; // at 1× — a 30-layer trace plays in ~30 s; 4× ≈ 7 s
  $effect(() => {
    if (!playing) {
      cancelAnimationFrame(raf);
      lastTs = 0;
      return;
    }
    const step = (ts: number) => {
      if (lastTs) {
        acc += ((ts - lastTs) / 1000) * NODES_PER_SEC * speed;
        const n = Math.floor(acc);
        if (n > 0) {
          acc -= n;
          idx = Math.min(idx + n, order.length - 1);
          if (idx >= order.length - 1) {
            playing = false;
            return;
          }
        }
      }
      lastTs = ts;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  });

  const current = $derived(engaged ? order[idx] : undefined);
  const currentNode = $derived(current ? nodeById.get(current.node_id) : undefined);

  // Report the active node up (diagram highlight) + follow its layer when asked to.
  $effect(() => {
    onHighlight?.(current ? current.node_id : null);
    const l = currentNode?.layer;
    if (followLayer && current && l !== null && l !== undefined) layerSel = l;
  });

  /** Any manual layer change means "I want this layer" — stop following the playhead. */
  function pickLayer(v: number): void {
    followLayer = false;
    layerSel = v;
  }

  function togglePlay(): void {
    if (order.length === 0) return;
    engaged = true;
    if (!playing && idx >= order.length - 1) {
      idx = 0;
      acc = 0;
    }
    lastTs = 0;
    playing = !playing;
  }

  function scrub(v: number): void {
    engaged = true;
    playing = false;
    idx = v;
  }

  // --- per-layer detail ------------------------------------------------------------
  const layer = $derived(trace?.layers[Math.min(layerSel, (trace?.layers.length ?? 1) - 1)]);
  const heads = $derived(layer?.attention.length ?? 0);
  const attn = $derived(layer ? layer.attention[Math.min(headSel, heads - 1)] : undefined);
  const tokenTexts = $derived((trace?.tokens ?? []).map((t) => t.text));
  const attnLabels = $derived(layer && !layer.attention_downsampled ? tokenTexts : undefined);
  const maxNorm = $derived.by(() => {
    let m = 0;
    for (const v of layer?.hidden_norm ?? []) m = Math.max(m, v);
    return m > 0 ? m : 1;
  });
  const maxLogit = $derived(Math.max(...(trace?.logits_topk.probs ?? [1]), 1e-9));

  function displayTok(s: string): string {
    return s.replace(/\n/g, "↵");
  }

  // The trace contract is gaining an additive `truncated: bool` (long prompts are
  // left-truncated to the last 64 tokens with no visible sign — F9). Read it
  // defensively so the UI works both before and after the backend field lands.
  const truncated = $derived(
    (trace as (ArchTrace & { truncated?: boolean }) | null)?.truncated === true,
  );
</script>

<div class="breakdown" data-testid="arch-breakdown">
  <div class="bd-head">
    <h3>Processing breakdown</h3>
    <span class="bd-sub">the real forward pass on your prompt — tokenize → {order.length || "…"} traced ops → next-token distribution</span>
    {#if loading && trace}<span class="tracing-chip">re-tracing…</span>{/if}
  </div>

  {#if staticNote}
    <StaticNotice message={staticNote} testid="arch-static-note" />
  {/if}

  {#if error}
    <div class="err" data-testid="arch-error">
      <span>{error}</span>
      {#if onRetry}<button class="retry" onclick={() => onRetry?.()}>Retry</button>{/if}
    </div>
  {:else if !trace}
    {#if loading}
      <div class="pending">
        <div class="indet"><div class="indet-bar"></div></div>
        <span>tracing the forward pass on your prompt…</span>
      </div>
    {:else}
      <p class="empty">Type a prompt on the left — 400 ms after you stop, the model runs it and every tensor lands here.</p>
    {/if}
  {:else}
    <!-- tokenization strip -->
    <div class="strip" data-testid="arch-trace-strip">
      {#each trace.tokens as t, i (i)}
        <span
          class="tokchip"
          role="note"
          aria-label={`token ${i}: ${t.text} (id ${t.id})`}
          onmousemove={(e) => showTip(e, `position ${i} · token id ${t.id}`)}
          onmouseleave={hideTip}
        ><span class="ttext">{displayTok(t.text)}</span><span class="tid">{t.id}</span></span>
      {/each}
      {#if trace.chat_template_used}
        <span
          class="tmpl"
          role="note"
          aria-label="the model's chat template wrapped your prompt"
          onmousemove={(e) => showTip(e, "the model's chat template wrapped your prompt")}
          onmouseleave={hideTip}
        >chat template</span>
      {/if}
      {#if truncated}
        <span
          class="tmpl trunc"
          data-testid="arch-truncated-chip"
          role="note"
          aria-label="the prompt was longer than the trace window — only its last 64 tokens were traced"
          onmousemove={(e) =>
            showTip(e, "the prompt was longer than the trace window — only its last 64 tokens were traced")}
          onmouseleave={hideTip}
        >⋯ prompt truncated to the last 64 tokens</span>
      {/if}
    </div>

    <!-- playback over the traced execution order -->
    <div class="player">
      <button
        class="play"
        data-testid="arch-play"
        onclick={togglePlay}
        aria-label={playing ? "pause the trace animation" : "play the trace animation"}
      >{playing ? "❚❚" : "▶"}</button>
      <input
        class="scrub"
        type="range"
        min="0"
        max={Math.max(0, order.length - 1)}
        value={idx}
        aria-label="scrub through traced ops"
        oninput={(e) => scrub(Number(e.currentTarget.value))}
      />
      <select class="speedsel" bind:value={speed} aria-label="playback speed">
        <option value={0.5}>0.5×</option>
        <option value={1}>1×</option>
        <option value={2}>2×</option>
        <option value={4}>4×</option>
      </select>
    </div>
    <div class="playhead" class:idle={!current}>
      {#if current}
        <span class="ph-step">op {idx + 1}/{order.length}</span>
        <span class="ph-label">{currentNode?.label ?? current.node_id}</span>
        <code class="ph-id">{current.node_id}</code>
        <span class="ph-norm">‖out‖ {current.out_norm.toFixed(2)} · [{current.out_shape.join(" × ")}]</span>
      {:else}
        <span class="ph-hint">▶ walks the diagram through all {order.length} traced ops in execution order; drag to scrub</span>
      {/if}
    </div>

    <!-- per-layer detail -->
    {#if layer}
      <div class="detail">
        <div class="detail-controls">
          <label class="lay">
            <span class="dlabel">layer <b>{layerSel}</b> / {trace.layers.length - 1}</span>
            <input
              type="range"
              min="0"
              max={trace.layers.length - 1}
              value={layerSel}
              oninput={(e) => pickLayer(Number(e.currentTarget.value))}
              aria-label="detail layer"
            />
          </label>
          <label class="follow" title="while the trace plays, jump the detail to whichever layer the current op belongs to">
            <input
              type="checkbox"
              data-testid="arch-follow-layer"
              checked={followLayer}
              onchange={(e) => (followLayer = e.currentTarget.checked)}
            />
            <span class="dlabel">follow playhead</span>
          </label>
        </div>
        <div class="detail-grid">
          <div class="cell">
            <span class="cell-label">
              attention · layer {layerSel} · all {heads} head{heads === 1 ? "" : "s"}
              {#if layer.attention_downsampled}· downsampled{/if}
            </span>
            <!-- Every head of the layer at once: comparing heads was impossible when
                 only one could be shown at a time. Click a tile to enlarge it below. -->
            <div class="heads" data-testid="arch-head-grid">
              {#each layer.attention as m, h (h)}
                <button
                  class="headtile"
                  class:sel={h === Math.min(headSel, heads - 1)}
                  data-testid={`arch-head-tile-${h}`}
                  aria-label={`attention head ${h} of layer ${layerSel}`}
                  aria-pressed={h === Math.min(headSel, heads - 1)}
                  onclick={() => (headSel = h)}
                >
                  <MatrixHeatmap values={m} maxCanvasPx={74} />
                  <span class="headnum">{h}</span>
                </button>
              {/each}
            </div>
            <span class="cell-label">head {Math.min(headSel, heads - 1)} · rows attend to columns</span>
            {#if attn}
              <MatrixHeatmap values={attn} rowLabels={attnLabels} colLabels={attnLabels} maxCanvasPx={252} />
            {/if}
          </div>
          <div class="cell">
            <span class="cell-label">‖residual stream‖ per token · layer {layerSel} out</span>
            <div class="bars">
              {#each layer.hidden_norm as v, i (i)}
                <div
                  class="bar"
                  role="img"
                  aria-label={`${tokenTexts[i] ?? `#${i}`}: norm ${v.toFixed(2)}`}
                  style:height={`${Math.max(4, (v / maxNorm) * 100)}%`}
                  onmousemove={(e) =>
                    showTip(e, `${displayTok(tokenTexts[i] ?? `#${i}`)} · ‖h‖ = ${v.toFixed(2)}`)}
                  onmouseleave={hideTip}
                ></div>
              {/each}
            </div>
            <span class="cell-label">next-token top-10</span>
            <div class="logits">
              {#each trace.logits_topk.ids as id, i (id)}
                <div class="lrow">
                  <code class="ltok">{displayTok(trace.logits_topk.texts[i])}</code>
                  <div class="lbarwrap">
                    <div class="lbar" style:width={`${(trace.logits_topk.probs[i] / maxLogit) * 100}%`}></div>
                  </div>
                  <span class="lpct">{(trace.logits_topk.probs[i] * 100).toFixed(1)}%</span>
                </div>
              {/each}
            </div>
          </div>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .breakdown {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    border-top: 1px solid var(--border);
    padding-top: 0.9rem;
  }
  .bd-head {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.95rem;
  }
  .bd-sub {
    font-size: 0.74rem;
    color: var(--text-dim);
  }
  .tracing-chip {
    font-size: 0.68rem;
    font-family: var(--mono);
    color: var(--accent);
    background: rgba(110, 168, 254, 0.12);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
    animation: fade 1s ease-in-out infinite alternate;
  }
  @keyframes fade {
    from {
      opacity: 0.45;
    }
    to {
      opacity: 1;
    }
  }
  .err {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.7rem;
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
  }
  .retry {
    background: transparent;
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.5);
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.74rem;
    flex-shrink: 0;
  }
  .pending {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    font-size: 0.78rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .indet {
    height: 6px;
    border-radius: 999px;
    background: var(--bg-elev-2);
    overflow: hidden;
  }
  .indet-bar {
    width: 38%;
    height: 100%;
    border-radius: 999px;
    background: var(--accent-grad);
    animation: slide 1.2s ease-in-out infinite;
  }
  @keyframes slide {
    from {
      transform: translateX(-110%);
    }
    to {
      transform: translateX(380%);
    }
  }
  .empty {
    margin: 0;
    color: var(--text-dim);
    font-size: 0.82rem;
  }
  .strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.28rem;
    align-items: center;
  }
  .tokchip {
    display: inline-flex;
    align-items: baseline;
    gap: 0.3rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.16rem 0.45rem;
    cursor: default;
    transition: border-color 0.12s ease;
  }
  .tokchip:hover {
    border-color: var(--accent);
  }
  .ttext {
    font-family: var(--mono);
    font-size: 0.76rem;
    white-space: pre;
  }
  .tid {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-dim);
  }
  .tmpl {
    font-size: 0.66rem;
    font-family: var(--mono);
    color: var(--accent-2);
    background: rgba(183, 148, 246, 0.12);
    border: 1px solid rgba(183, 148, 246, 0.35);
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
  }
  .tmpl.trunc {
    color: #ffb454;
    background: rgba(255, 180, 84, 0.12);
    border-color: rgba(255, 180, 84, 0.35);
  }
  .player {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .play {
    width: 40px;
    height: 32px;
    padding: 0;
    font-size: 0.8rem;
    flex-shrink: 0;
  }
  .scrub {
    flex: 1;
  }
  .speedsel {
    width: auto;
    padding: 0.3rem 0.5rem;
    font-size: 0.76rem;
    font-family: var(--mono);
    flex-shrink: 0;
  }
  .playhead {
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
    flex-wrap: wrap;
    font-size: 0.74rem;
    min-height: 1.2em;
  }
  .ph-step {
    font-family: var(--mono);
    color: var(--accent);
  }
  .ph-label {
    color: var(--text);
    font-weight: 600;
  }
  .ph-id {
    font-family: var(--mono);
    font-size: 0.66rem;
    color: var(--text-dim);
    word-break: break-all;
  }
  .ph-norm {
    font-family: var(--mono);
    color: var(--text-dim);
  }
  .ph-hint {
    color: var(--text-dim);
  }
  .detail {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
  }
  .detail-controls {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
  }
  .lay {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    max-width: 340px;
  }
  .follow {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    cursor: pointer;
    padding-bottom: 0.15rem;
  }
  .follow input {
    width: auto;
    margin: 0;
    accent-color: var(--accent);
  }
  .heads {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    max-height: 13rem;
    overflow-y: auto;
  }
  .headtile {
    position: relative;
    display: block;
    padding: 2px;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    line-height: 0;
    cursor: pointer;
  }
  .headtile:hover {
    border-color: var(--accent);
  }
  .headtile.sel {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }
  .headnum {
    position: absolute;
    bottom: 2px;
    right: 3px;
    font-family: var(--mono);
    font-size: 0.58rem;
    line-height: 1;
    color: var(--text);
    background: rgba(11, 14, 20, 0.72);
    border-radius: 3px;
    padding: 0 2px;
  }
  .dlabel {
    font-size: 0.74rem;
    color: var(--text-dim);
  }
  .dlabel b {
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .detail-grid {
    display: grid;
    /* Bounded on BOTH sides: with `auto` the all-heads grid grew until the residual /
       top-10 column was a few characters wide and unreadable. */
    grid-template-columns: minmax(0, 1fr) minmax(15rem, 21rem);
    gap: 1.1rem;
    align-items: start;
  }
  @media (max-width: 900px) {
    .detail-grid {
      grid-template-columns: 1fr;
    }
  }
  .cell {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    min-width: 0;
  }
  .cell-label {
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 68px;
    background: rgba(0, 0, 0, 0.22);
    border-radius: 8px;
    padding: 4px;
  }
  .bar {
    flex: 1;
    min-width: 2px;
    border-radius: 2px 2px 0 0;
    background: linear-gradient(180deg, var(--accent) 0%, rgba(110, 168, 254, 0.35) 100%);
    transition: height 0.25s ease;
  }
  .bar:hover {
    background: var(--accent-2);
  }
  .logits {
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
  }
  .lrow {
    display: grid;
    grid-template-columns: minmax(56px, auto) minmax(0, 1fr) 48px;
    align-items: center;
    gap: 0.5rem;
  }
  .ltok {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text);
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .lbarwrap {
    height: 9px;
    border-radius: 999px;
    background: var(--bg-elev-2);
    overflow: hidden;
  }
  .lbar {
    height: 100%;
    border-radius: 999px;
    background: var(--accent-grad);
    transition: width 0.3s ease;
  }
  .lpct {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    text-align: right;
  }
</style>
