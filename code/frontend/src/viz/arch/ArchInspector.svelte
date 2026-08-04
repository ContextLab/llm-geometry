<script lang="ts">
  import { onDestroy } from "svelte";
  import MatrixHeatmap from "../../lib/MatrixHeatmap.svelte";
  import { client, type ArchNode, type ArchWeightsData } from "../../lib/dataClient";
  import { KIND_EXPLAINER, formatCount, paramCount, plainError } from "./archShared";
  import { STATIC_MODE } from "../../lib/staticUx";
  import { hideTip } from "../../lib/tooltip";

  // Node inspector drawer: kind/label/shape/param list (tied_to badge for aliased
  // tensors) + a MatrixHeatmap of the selected parameter. Default view = whole matrix
  // (server-downsampled ≤4096 cells); clicking the map fetches an EXACT sub-window
  // centered on the click. Rapid zooms abort the stale tile fetch (FR-108).
  // Parameterless (functional) ops get a plain-language explainer instead.
  interface Props {
    node: ArchNode;
    modelId: string;
    onClose?: () => void;
  }
  let { node, modelId, onClose }: Props = $props();

  interface Zoom {
    r0: number;
    c0: number;
    rows: number;
    cols: number;
  }

  let selPath = $state<string | null>(null);
  let zoom = $state<Zoom | null>(null);
  let weights = $state<ArchWeightsData | null>(null);
  let loading = $state(false);
  let error = $state("");
  let retryNonce = $state(0);

  // A different node ⇒ fresh param selection + overview zoom.
  $effect(() => {
    const n = node;
    const w = n.params.find((p) => p.name === "weight") ?? n.params[0] ?? null;
    selPath = w ? w.param_path : null;
    zoom = null;
  });

  const selParam = $derived(node.params.find((p) => p.param_path === selPath) ?? null);

  let ctl: AbortController | null = null;
  onDestroy(() => ctl?.abort());

  $effect(() => {
    const path = selPath;
    const z = zoom;
    const m = modelId;
    void retryNonce; // re-fetch on Retry
    ctl?.abort();
    if (!path) {
      weights = null;
      return;
    }
    const c = new AbortController();
    ctl = c;
    loading = true;
    error = "";
    // 1-D params (bias/norm, C == 1): the whole [N,1] tensor is ≤4096 cells, so the
    // default overview would come back exact with N rows and render as an N-px-tall
    // hairline (F3). Ask for a ≤128-row downsampled overview instead; zooming pages
    // through exact 128-row windows (windowFor caps 1-D zooms at 128 rows).
    const oneD = (selParam?.shape[1] ?? 1) <= 1;
    const params = z
      ? { model_id: m, param: path, r0: z.r0, r1: z.r0 + z.rows, c0: z.c0, c1: z.c0 + z.cols }
      : oneD
        ? { model_id: m, param: path, max_cells: 128 }
        : { model_id: m, param: path };
    client
      .getArchWeights(params, c.signal)
      .then((w) => {
        if (ctl !== c) return;
        weights = w;
        loading = false;
      })
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        if (ctl !== c) return;
        error = plainError(e);
        loading = false;
      });
  });

  function clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v));
  }

  // Largest exact window (≤4096 cells) for a given full shape; 1-D params (C=1)
  // window rows only so the strip stays readable.
  function windowFor(shape: number[]): { rows: number; cols: number } {
    const R = shape[0] ?? 1;
    const C = shape[1] ?? 1;
    if (C <= 1) return { rows: Math.min(R, 128), cols: 1 };
    const cols = Math.min(C, 64);
    const rows = Math.min(R, Math.floor(4096 / cols));
    return { rows, cols };
  }

  const canZoom = $derived.by(() => {
    if (!selParam) return false;
    const R = selParam.shape[0] ?? 1;
    const C = selParam.shape[1] ?? 1;
    const w = windowFor(selParam.shape);
    return w.rows < R || w.cols < C;
  });

  function zoomAt(fr: number, fc: number): void {
    if (!selParam || !canZoom) return;
    // The cursor is stationary at the instant of the zoom click, so no mousemove fires
    // to refresh the global tooltip — it kept showing the OVERVIEW's row/col range and
    // value over the new exact window until the mouse was nudged.
    hideTip();
    const R = selParam.shape[0] ?? 1;
    const C = selParam.shape[1] ?? 1;
    const { rows, cols } = windowFor(selParam.shape);
    zoom = {
      r0: clamp(Math.round(fr * R - rows / 2), 0, Math.max(0, R - rows)),
      c0: clamp(Math.round(fc * C - cols / 2), 0, Math.max(0, C - cols)),
      rows,
      cols,
    };
  }

  // Overview click → exact window centered there; zoomed click → re-center.
  function onStageClick(e: MouseEvent): void {
    const el = e.currentTarget as HTMLElement;
    const rect = el.getBoundingClientRect();
    let fr = (e.clientY - rect.top) / rect.height;
    let fc = (e.clientX - rect.left) / rect.width;
    if (zoom && selParam) {
      const R = selParam.shape[0] ?? 1;
      const C = selParam.shape[1] ?? 1;
      fr = (zoom.r0 + fr * zoom.rows) / R;
      fc = (zoom.c0 + fc * zoom.cols) / C;
    }
    zoomAt(fr, fc);
  }

  function pan(dr: number, dc: number): void {
    if (!zoom || !selParam) return;
    const R = selParam.shape[0] ?? 1;
    const C = selParam.shape[1] ?? 1;
    zoom = {
      ...zoom,
      r0: clamp(zoom.r0 + dr * Math.ceil(zoom.rows / 2), 0, Math.max(0, R - zoom.rows)),
      c0: clamp(zoom.c0 + dc * Math.ceil(zoom.cols / 2), 0, Math.max(0, C - zoom.cols)),
    };
  }

  function fmt(v: number): string {
    return String(Number(v.toPrecision(4)));
  }

  // GLOBAL index labels for the hover tooltip (F6): the served window starts at
  // (r0, c0) of the full matrix, so window-local cell indices are misleading. Each
  // display cell of a downsampled grid covers a range of true indices.
  function idxLabels(prefix: string, lo: number, hi: number, n: number): string[] {
    const span = (hi - lo) / Math.max(1, n);
    return Array.from({ length: n }, (_, i) => {
      const a = lo + Math.floor(i * span);
      const b = Math.min(hi - 1, lo + Math.ceil((i + 1) * span) - 1);
      return b > a ? `${prefix}s ${a}–${b}` : `${prefix} ${a}`;
    });
  }
  const rowLabels = $derived(
    weights ? idxLabels("row", weights.r0, weights.r1, weights.grid_shape[0]) : undefined,
  );
  const colLabels = $derived(
    weights ? idxLabels("col", weights.c0, weights.c1, weights.grid_shape[1]) : undefined,
  );

  // STATIC build: an over-budget window comes from ONE precomputed whole-tensor overview
  // tile, so it always reports r0..c1 = the FULL tensor (a sub-window overview can't be
  // honestly derived from it — 003-C). Label it for what it is, keyed off the response
  // DATA, never the request: `quantized` is set only by the static client and only on
  // that tile path.
  const fullOverview = $derived(
    STATIC_MODE &&
      weights !== null &&
      weights.quantized === "uint8" &&
      weights.r0 === 0 &&
      weights.c0 === 0 &&
      weights.r1 === (weights.shape[0] ?? 1) &&
      weights.c1 === (weights.shape[1] ?? 1),
  );
  // Two independent facts about that tile, and it used to assert the wrong one: every
  // 1-D parameter's tile is a FULL-RESOLUTION strip (no cells averaged) that is
  // nonetheless 8-bit. The caption now says which of the two applies.
  const overviewNote = $derived(
    weights === null
      ? ""
      : weights.downsampled
        ? "strided mean, 8-bit"
        : "full resolution, 8-bit",
  );
</script>

<aside class="inspector" data-testid="arch-inspector" aria-label="node inspector">
  <header>
    <div class="titles">
      <h3>{node.label}</h3>
      <code class="path">{node.id}</code>
    </div>
    <button class="close" aria-label="close inspector (Esc)" title="close (Esc)" onclick={() => onClose?.()}>✕</button>
  </header>

  <div class="chips">
    <span class="chip kind">{node.kind}</span>
    <span class="chip">{node.op}</span>
    {#if node.layer !== null}<span class="chip">layer {node.layer}</span>{/if}
    {#if node.params.length > 0}<span class="chip">{formatCount(paramCount(node))} params</span>{/if}
  </div>

  <p class="explain">{KIND_EXPLAINER[node.kind]}</p>

  {#if node.params.length === 0}
    <div class="fnnote">
      This op has <b>no learned weights</b> — it transforms activations on the fly during the
      forward pass, so there is no matrix to plot. Watch it light up in the processing
      breakdown below instead.
    </div>
  {:else}
    <div class="params">
      {#each node.params as p (p.param_path)}
        <button
          class="param"
          class:active={selPath === p.param_path}
          onclick={() => {
            selPath = p.param_path;
            zoom = null;
          }}
        >
          <span class="pname">{p.name}</span>
          <span class="pshape">[{p.shape.join(" × ")}]</span>
          {#if p.tied_to}
            <span class="tied" title={`shares its tensor with ${p.tied_to}`}>tied → {p.tied_to}</span>
          {/if}
        </button>
      {/each}
    </div>

    {#if error}
      <div class="err" data-testid="arch-error">
        <span>{error}</span>
        <button class="retry" onclick={() => (retryNonce += 1)}>Retry</button>
      </div>
    {:else if weights}
      <div class="viewmeta">
        {#if fullOverview}
          {weights.shape.join(" × ")} · <b>overview (whole tensor, {overviewNote})</b>
        {:else}
          {#if zoom}
            rows {weights.r0}–{weights.r1 - 1} · cols {weights.c0}–{weights.c1 - 1}
            of {weights.shape.join(" × ")}
          {:else}
            whole matrix {weights.shape.join(" × ")}
          {/if}
          ·
          <b class:exact={!weights.downsampled}>
            {weights.downsampled ? "downsampled (strided mean)" : "exact values"}
          </b>
        {/if}
      </div>
      <div
        class="stage"
        class:busy={loading}
        class:zoomable={canZoom}
        role="button"
        tabindex="0"
        aria-label={zoom ? "re-center the exact weight window" : "zoom into the weights"}
        title={canZoom ? (zoom ? "click to re-center the exact window" : "click to fetch an exact window there") : ""}
        onclick={onStageClick}
        onkeydown={(e) => (e.key === "Enter" || e.key === " ") && zoomAt(0.5, 0.5)}
      >
        <MatrixHeatmap values={weights.values} {rowLabels} {colLabels} maxCanvasPx={280} />
      </div>
      <div class="zoomrow">
        {#if zoom}
          <button class="mini" title="pan up" onclick={() => pan(-1, 0)}>▲</button>
          <button class="mini" title="pan down" onclick={() => pan(1, 0)}>▼</button>
          {#if (selParam?.shape[1] ?? 1) > 1}
            <button class="mini" title="pan left" onclick={() => pan(0, -1)}>◀</button>
            <button class="mini" title="pan right" onclick={() => pan(0, 1)}>▶</button>
          {/if}
          <button class="mini reset" onclick={() => (zoom = null)}>reset zoom</button>
        {:else if canZoom}
          <span class="zoomhint">click the map for an exact ≤4096-cell window · hover cells for values</span>
        {:else}
          <span class="zoomhint">small enough to show exactly · hover cells for values</span>
        {/if}
      </div>
      <div class="stats">
        min {fmt(weights.stats.min)} · max {fmt(weights.stats.max)} · μ {fmt(weights.stats.mean)} ·
        σ {fmt(weights.stats.std)}
      </div>
    {:else if loading}
      <div class="skeleton"></div>
    {/if}
  {/if}
</aside>

<style>
  .inspector {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(340px, 94%);
    background: rgba(19, 24, 38, 0.97);
    border-left: 1px solid var(--border);
    border-radius: 0 12px 12px 0;
    box-shadow: -18px 0 42px rgba(0, 0, 0, 0.5);
    padding: 0.9rem 1rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    overflow-y: auto;
    animation: slidein 0.18s ease;
    z-index: 3;
  }
  @keyframes slidein {
    from {
      transform: translateX(24px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.6rem;
  }
  .titles {
    min-width: 0;
  }
  h3 {
    margin: 0;
    font-size: 0.98rem;
  }
  .path {
    display: block;
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    word-break: break-all;
    margin-top: 0.15rem;
  }
  .close {
    background: var(--bg-elev-2);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.15rem 0.5rem;
    font-size: 0.8rem;
    line-height: 1.4;
    flex-shrink: 0;
  }
  .close:hover {
    color: var(--text);
    border-color: var(--accent);
    filter: none;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .chip {
    font-size: 0.68rem;
    font-family: var(--mono);
    color: var(--text-dim);
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.12rem 0.5rem;
  }
  .chip.kind {
    color: var(--accent);
    border-color: rgba(110, 168, 254, 0.4);
  }
  .explain {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .fnnote {
    background: var(--bg-elev-2);
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 0.65rem 0.75rem;
    font-size: 0.8rem;
    color: var(--text-dim);
    line-height: 1.5;
  }
  .fnnote b {
    color: var(--text);
  }
  .params {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .param {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    background: var(--bg-elev-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.55rem;
    font-size: 0.76rem;
    text-align: left;
    font-weight: 500;
  }
  .param:hover {
    filter: none;
    border-color: var(--accent);
  }
  .param.active {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(110, 168, 254, 0.2);
  }
  .pname {
    font-family: var(--mono);
  }
  .pshape {
    font-family: var(--mono);
    color: var(--text-dim);
  }
  .tied {
    font-size: 0.64rem;
    font-family: var(--mono);
    color: var(--accent-2);
    background: rgba(183, 148, 246, 0.12);
    border: 1px solid rgba(183, 148, 246, 0.35);
    border-radius: 999px;
    padding: 0.08rem 0.45rem;
    word-break: break-all;
  }
  .viewmeta {
    font-size: 0.7rem;
    font-family: var(--mono);
    color: var(--text-dim);
  }
  .viewmeta .exact {
    color: var(--good);
  }
  .stage {
    align-self: flex-start;
    border-radius: 8px;
    outline: none;
    transition: opacity 0.15s ease;
  }
  .stage.zoomable {
    cursor: zoom-in;
  }
  .stage.busy {
    opacity: 0.55;
  }
  .stage:focus-visible {
    box-shadow: 0 0 0 2px var(--accent);
  }
  .zoomrow {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    flex-wrap: wrap;
  }
  .mini {
    background: var(--bg-elev-2);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.14rem 0.5rem;
    font-size: 0.7rem;
    font-family: var(--mono);
  }
  .mini:hover {
    color: var(--text);
    border-color: var(--accent);
    filter: none;
  }
  .zoomhint {
    font-size: 0.68rem;
    color: var(--text-dim);
  }
  .stats {
    font-size: 0.7rem;
    font-family: var(--mono);
    color: var(--text-dim);
  }
  .err {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    font-size: 0.78rem;
    line-height: 1.4;
  }
  .retry {
    align-self: flex-start;
    background: transparent;
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.5);
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.74rem;
  }
  .skeleton {
    height: 180px;
    border-radius: 8px;
    background: linear-gradient(100deg, var(--bg-elev-2) 40%, #232c44 50%, var(--bg-elev-2) 60%);
    background-size: 200% 100%;
    animation: shimmer 1.1s linear infinite;
  }
  @keyframes shimmer {
    to {
      background-position: -200% 0;
    }
  }
</style>
