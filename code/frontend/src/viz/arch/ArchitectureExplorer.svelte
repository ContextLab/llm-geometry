<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { get } from "svelte/store";
  import PipelineDiagram from "../../lib/PipelineDiagram.svelte";
  import {
    archModelId,
    archPrompt,
    archSystemPrompt,
    archSelectedNode,
  } from "../../lib/explorerStores";
  import {
    ApiError,
    client,
    debounced,
    type ArchGraph,
    type ArchTrace,
  } from "../../lib/dataClient";
  import ArchModelPicker from "./ArchModelPicker.svelte";
  import ArchChat from "./ArchChat.svelte";
  import Explain from "../../lib/Explain.svelte";
  import { view } from "../../lib/stores";
  import ArchInspector from "./ArchInspector.svelte";
  import ArchTracePanel from "./ArchTracePanel.svelte";
  import { evictArchGraph, fetchArchGraph, formatCount, plainError } from "./archShared";
  import { STATIC_MODE, isStaticMiss, staticExtras } from "../../lib/staticUx";
  import type { TraceIndexEntry } from "../../lib/staticClient/arch";

  // Architecture Explorer (specs/002, User Story 1 — the "ELIZA-demo" loop):
  // chat → trace → inspect → re-test, against a REAL open-weights model.
  // Left rail = model + prompt + chat; center = the traced architecture diagram
  // (inspector slides in over its right edge); below = the processing breakdown.

  // --- architecture graph (cached per model id in archShared) ---------------------
  let graph = $state<ArchGraph | null>(null);
  let graphForModel = $state("");
  let graphLoading = $state(false);
  let graphError = $state("");
  let pickerError = $state("");
  let lastGood = ""; // revert target when a picked model is rejected server-side
  // Id currently being restored after a server-side rejection. The revert re-runs
  // loadGraph for the previous model; that load's success must NOT clear pickerError
  // (it would wipe the rejection one tick after it was shown — F1). Only a load of
  // the id the user most recently picked clears the error.
  let revertingTo: string | null = null;
  let graphSeq = 0;

  function loadGraph(id: string, force = false): void {
    const my = ++graphSeq;
    graphLoading = true;
    graphError = "";
    if (force) evictArchGraph(id);
    fetchArchGraph(id)
      .then((g) => {
        if (my !== graphSeq) return;
        if (graphForModel !== id) {
          archSelectedNode.set(null); // stale node ids
          traceCtl?.abort(); // the old model's tensors are meaningless now
          trace = null;
        }
        graph = g;
        graphForModel = id;
        lastGood = id;
        graphLoading = false;
        if (id === revertingTo) {
          revertingTo = null; // revert finished — keep the rejection visible
        } else {
          revertingTo = null;
          pickerError = ""; // a user-picked model really loaded — clear the rejection
        }
      })
      .catch((e) => {
        if (my !== graphSeq) return;
        graphLoading = false;
        const msg = plainError(e);
        const gated =
          e instanceof ApiError &&
          (e.type === "ModelTooLargeError" || e.type === "UnsupportedModelError");
        if (gated && lastGood && lastGood !== id) {
          // Rejected at graph time (e.g. over the parameter ceiling): surface the
          // plain-language error at the picker and keep the previous model (FR-107).
          pickerError = msg;
          revertingTo = lastGood;
          archModelId.set(lastGood);
        } else {
          graphError = msg;
        }
      });
  }

  // Gate solely on the (non-reactive) last-requested id: reading graphLoading/
  // graphError here made the effect re-run when loadGraph wrote them (double-fetch
  // on mount, saved only by the module-level graph cache).
  let requestedFor = "";
  $effect(() => {
    const id = $archModelId;
    if (id !== requestedFor) {
      requestedFor = id;
      loadGraph(id);
    }
  });

  // --- static mode: curated models + precomputed example traces (feature 003) -----
  // In the static build per-prompt traces exist only for a labeled set of example
  // prompts (ONNX exposes no hidden states — spec US-2). The dropdown applies one;
  // the free-prompt textarea stays live for chat/tokenize, and an off-example trace
  // shows the designed "full stack only" note instead of a red error (FR-203).
  let tracePresets = $state<TraceIndexEntry[]>([]);
  let traceStaticNote = $state("");

  function applyTraceExample(n: number): void {
    const t = tracePresets.find((x) => x.n === n);
    if (!t) return;
    archPrompt.set(t.prompt);
    archSystemPrompt.set(t.system_prompt ?? "");
  }

  onMount(() => {
    const sc = staticExtras();
    if (!sc) return;
    const initialPrompt = get(archPrompt);
    const initialModel = get(archModelId);
    void (async () => {
      // The static catalog may not include the backend default (e.g. --quick exports
      // ship only gpt2): steer an untouched session to a model that exists.
      try {
        const { models } = await sc.listModels();
        if (
          models.length > 0 &&
          !models.some((m) => m.model_id === get(archModelId)) &&
          get(archModelId) === initialModel
        ) {
          archModelId.set(models[0].model_id);
        }
      } catch {
        // catalog unreachable — the graph loader will surface the real error
      }
    })();
  });

  // Keep the example list in sync with the active model; on first load, land an
  // untouched default prompt on example 1 so the tab opens with a real trace
  // instead of a miss note (judgment call documented in notes/agent-reports/003-D.md).
  const DEFAULT_ARCH_PROMPT = "What is the capital of France?"; // explorerStores initial value
  let presetsFor = "";
  $effect(() => {
    const sc = staticExtras();
    const m = $archModelId;
    if (!sc || m === presetsFor) return;
    presetsFor = m;
    sc.staticArchTracePresets(m)
      .then((list) => {
        if (presetsFor !== m) return;
        tracePresets = list;
        const p = get(archPrompt);
        const covered = list.some(
          (t) => t.prompt === p && (t.system_prompt ?? "") === get(archSystemPrompt),
        );
        if (!covered && p === DEFAULT_ARCH_PROMPT && list.length > 0) {
          applyTraceExample(list[0].n);
        }
      })
      .catch(() => {
        if (presetsFor === m) tracePresets = [];
      });
  });

  // --- forward-pass trace (debounce 400 ms + abort stale requests, FR-108) --------
  let trace = $state<ArchTrace | null>(null);
  let traceLoading = $state(false);
  let traceError = $state("");
  let traceCtl: AbortController | null = null;
  let traceArgs: { m: string; p: string; sp: string } | null = null;

  function runTrace(m: string, p: string, sp: string): void {
    traceCtl?.abort();
    const c = new AbortController();
    traceCtl = c;
    traceArgs = { m, p, sp };
    traceLoading = true;
    traceError = "";
    client
      .getArchTrace({ model_id: m, prompt: p, system_prompt: sp.trim() ? sp : undefined }, c.signal)
      .then((t) => {
        if (traceCtl !== c) return;
        trace = t;
        traceLoading = false;
        traceStaticNote = "";
      })
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        if (traceCtl !== c) return;
        traceLoading = false;
        if (isStaticMiss(e)) {
          // Static build, non-example prompt: a designed affordance, not an error —
          // the previous trace (if any) stays on screen underneath it.
          traceStaticNote = e.message;
          traceError = "";
        } else {
          traceError = plainError(e);
        }
      });
  }

  const fireTrace = debounced((m: string, p: string, sp: string) => runTrace(m, p, sp), 400);

  // Fire on (model, prompt, system prompt) changes only — keyed so a landing trace
  // (also reactive state) can never re-trigger its own request.
  let lastFireKey = "";
  $effect(() => {
    const m = $archModelId;
    const p = $archPrompt;
    const sp = $archSystemPrompt;
    const ready = graph !== null && graphForModel === m; // graph first: gates size + reuses the download
    if (!ready) {
      fireTrace.cancel();
      return;
    }
    if (!p.trim()) {
      fireTrace.cancel();
      traceCtl?.abort();
      trace = null;
      traceLoading = false;
      traceError = "";
      traceStaticNote = "";
      lastFireKey = "";
      return;
    }
    const key = JSON.stringify([m, p, sp]);
    if (key === lastFireKey) return;
    lastFireKey = key;
    traceLoading = true; // instant feedback while the debounce window runs
    fireTrace(m, p, sp);
  });

  onDestroy(() => {
    fireTrace.cancel();
    traceCtl?.abort();
  });

  /**
   * True while a DIFFERENT model is loading and the previous model's graph/trace are
   * still on screen. The diagram was already dimmed for this, but the header meta
   * ("494.0M params"), the traced-op count and the token strip were not — so mid-switch
   * the tab showed Qwen's numbers under SmolLM2's name at full opacity.
   */
  const staleModel = $derived(
    graphLoading && graphForModel !== "" && graphForModel !== $archModelId,
  );

  // --- diagram highlight (playback) + node inspector ------------------------------
  let highlightId = $state<string | null>(null);
  const diagramSelected = $derived(highlightId ?? $archSelectedNode);
  const selectedNode = $derived(
    graph?.nodes.find((n) => n.id === $archSelectedNode) ?? null,
  );

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === "Escape" && $archSelectedNode !== null) archSelectedNode.set(null);
  }

  // (A pointer-capture click workaround lived here; PipelineDiagram now arms its pan
  // capture only after a 4 px drag, so real clicks reach node handlers directly —
  // verified live with a trusted click.)
</script>

<svelte:window onkeydown={onKeydown} />

<section class="viz panel arch" data-testid="arch-view">
  <header class="head">
    <div class="head-text">
      <h2>Architecture Explorer</h2>
      <p class="sub">
        A real open-weights model, unfolded into the operations it actually performs. The diagram
        below is <b>traced from a genuine forward pass</b>, not drawn from the config: every step
        that transforms the hidden state is a node, including the parameterless ones — rotary
        embeddings, the attention softmax, residual adds. Click any block to inspect its weights;
        ▶ play the trace to walk the ops in execution order; generate a reply and hover a token for
        its probability and the alternatives it beat.
        {#if STATIC_MODE}
          Chat and tokenization run live in your browser via the model's ONNX export; the
          op-by-op trace is precomputed by the real backend for a set of example prompts, because
          browser ONNX exports do not expose hidden states.
        {/if}
      </p>
    </div>
    {#if graph}
      <div class="meta" class:stale={staleModel} data-testid="arch-meta">
        <span>{graph.meta.n_layers} layers</span>
        <span>hidden {graph.meta.hidden}</span>
        <span>
          {graph.meta.heads} heads{graph.meta.kv_heads !== graph.meta.heads
            ? ` · ${graph.meta.kv_heads} KV`
            : ""}
        </span>
        <span>vocab {graph.meta.vocab.toLocaleString()}</span>
        <span class="strong">{formatCount(graph.meta.total_params)} params</span>
        {#if staleModel}
          <span class="stale-note" data-testid="arch-stale-note">↑ still {graphForModel} — loading {$archModelId}</span>
        {/if}
      </div>
    {/if}
  </header>

  <div class="explainers">
    <Explain
      title="How to read the diagram"
      hint="traced ops, tied weights, and what the inspector is showing you"
      testid="arch-explain-diagram"
    >
      <p>
        The graph comes from running the model once on a short fixed sentence with hooks on every
        tensor operation. A node is <b>any step that transforms the hidden state</b> — so the ops
        that architecture diagrams usually leave out (rotary position embedding, the attention
        softmax, residual adds, activations) are first-class here. Edges are real dataflow,
        recovered from tensor identity and storage aliasing, with execution order as a fallback
        where views and copies break identity.
      </p>
      <p>
        <b>Tied weights appear once.</b> When a model shares one tensor between its embedding and
        its output projection, the graph detects that by storage pointer and shows a
        <code>tied_to</code> badge on the alias instead of counting the parameters twice.
      </p>
      <p>
        Clicking a node opens the inspector. The heat map you get first is an <b>overview</b>: a
        strided mean of the whole matrix within a 4096-cell budget (computed live by the backend,
        or precomputed and 8-bit quantized on the static site). Clicking into the map fetches the
        <b>exact</b> sub-window at full precision rather than magnifying pixels you already have.
        Ops with no learned weights say so — there is no matrix to plot, so watch them light up in
        the breakdown instead.
      </p>
      <p>
        Models are capped at <b>1.5B parameters</b>, decided from hub metadata <i>before</i> any
        weights are downloaded.
      </p>
    </Explain>
    <Explain
      title="What can I change here?"
      hint="model, prompt, system prompt, and the decoding controls in Chat"
      testid="arch-explain-controls"
    >
      <ul>
        <li>
          <b>Model</b> — swaps the whole tab to a different real model. Every number on screen
          comes from the one you have selected.
          {#if STATIC_MODE}
            Its graph and example traces were produced by the real backend at build time; its
            weights are read live from HuggingFace's CDN.
          {:else}
            The first load of a new one downloads and traces it, which takes as long as your
            network and CPU take; after that it is cached.
          {/if}
        </li>
        <li>
          <b>Prompt</b> and <b>system prompt</b> — rendered through the model's own chat template
          when it has one, and truncated to the last 64 tokens.
          {#if STATIC_MODE}
            They stay live for chat and tokenization, but only the labelled example prompts have a
            precomputed trace; anything else says so rather than inventing one.
          {:else}
            Retraced 400 ms after you stop typing.
          {/if}
        </li>
        <li>
          <b>Temperature</b> (in Chat) — 0 is greedy argmax; above 0 samples the temperature
          softmax restricted to top-k 50 ∩ top-p 0.9, with a 1.1 repetition penalty applied first.
          The filtering affects only which token is <i>drawn</i>: no percentage shown to you is
          computed from the truncated distribution. (The Info tab spells out which distribution
          each number comes from — they are not all the same one.)
        </li>
        <li>
          <b>Base vs instruct models</b> — <code>gpt2</code> has no chat template, so it continues
          your text rather than answering it. The Chat panel says so when a base model is selected.
        </li>
      </ul>
      <p>
        For the full notation and the mathematics, see the
        <button class="linklike" onclick={() => view.set("info")}>Info tab</button>.
      </p>
    </Explain>
  </div>

  <div class="body">
    <div class="rail">
      <ArchModelPicker externalError={pickerError} />
      <div class="divider"></div>
      <span class="group-label">Prompt</span>
      {#if STATIC_MODE && tracePresets.length > 0}
        <!-- Precomputed example traces (spec US-2): each option re-runs a trace the real
             backend recorded at build time. The free-form prompt below stays fully live
             for chat + tokenization. -->
        <select
          class="examples"
          data-testid="arch-trace-presets"
          value={String(tracePresets.find((t) => t.prompt === $archPrompt && (t.system_prompt ?? "") === $archSystemPrompt)?.n ?? 0)}
          onchange={(e) => applyTraceExample(Number(e.currentTarget.value))}
        >
          <option value="0" disabled>Example prompts (precomputed traces)…</option>
          {#each tracePresets as t (t.n)}
            <option value={String(t.n)}>{t.label}</option>
          {/each}
        </select>
      {/if}
      <textarea
        rows="3"
        data-testid="arch-prompt"
        bind:value={$archPrompt}
        placeholder="Ask the model something…"
      ></textarea>
      <input
        type="text"
        data-testid="arch-system-prompt"
        bind:value={$archSystemPrompt}
        placeholder="system prompt (optional)"
      />
      <p class="railhint">
        {#if STATIC_MODE}
          Chat and tokenization run live in your browser; the per-layer trace below is
          precomputed for the example prompts (per-layer tensors need the full stack).
        {:else}
          Tracing re-runs 400 ms after you stop typing; the breakdown below is driven by the
          real tensors of that forward pass.
        {/if}
      </p>
      <div class="divider"></div>
      <span class="group-label">Chat</span>
      <ArchChat chatTemplate={trace ? trace.chat_template_used : null} />
    </div>

    <div class="stagecol">
      <div class="stage">
        {#if graphError}
          <div class="gerr" data-testid="arch-error">
            <b>Couldn't build the architecture graph.</b>
            <span>{graphError}</span>
            <button onclick={() => loadGraph($archModelId, true)}>Retry</button>
          </div>
        {:else if graph}
          <!-- keyboard interaction lives on the diagram's own focusable nodes -->
          <div class="diag" class:dim={graphLoading} data-testid="arch-diagram">
            <PipelineDiagram
              nodes={graph.nodes}
              edges={graph.edges}
              selected={diagramSelected}
              focus={highlightId}
              onSelect={(id) => archSelectedNode.set(id)}
            />
          </div>
        {/if}
        {#if graphLoading}
          <div class="gload" class:overlay={graph !== null}>
            <div class="glogo"></div>
            <p class="phase">
              {STATIC_MODE ? "loading" : "downloading + tracing"} <b>{$archModelId}</b>…
            </p>
            <p class="phase-sub">
              {#if STATIC_MODE}
                fetching the graph the real backend traced for this model at build time
              {:else}
                the first load of a new model takes a while — the graph is built from a real traced
                forward pass, then cached
              {/if}
            </p>
            <div class="indet"><div class="indet-bar"></div></div>
          </div>
        {/if}
        {#if selectedNode && graphForModel}
          <ArchInspector
            node={selectedNode}
            modelId={graphForModel}
            onClose={() => archSelectedNode.set(null)}
          />
        {/if}
      </div>
      <p class="caption">
        scroll to zoom · drag to pan · ▸/▾ headers expand layers · click any block to inspect it · Esc closes the inspector
      </p>
    </div>
  </div>

  <ArchTracePanel
    {trace}
    {graph}
    loading={traceLoading}
    error={traceError}
    staticNote={traceStaticNote}
    hasPresets={tracePresets.length > 0}
    onHighlight={(id) => (highlightId = id)}
    onRetry={() => traceArgs && runTrace(traceArgs.m, traceArgs.p, traceArgs.sp)}
  />
</section>

<style>
  .arch {
    padding: 1.2rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
  }
  .head-text {
    min-width: 0;
  }
  h2 {
    margin: 0;
    font-size: 1.1rem;
  }
  .sub {
    margin: 0.2rem 0 0;
    color: var(--text-dim);
    font-size: 0.82rem;
    line-height: 1.45;
  }
  .meta.stale {
    opacity: 0.42;
    filter: grayscale(0.5);
  }
  .stale-note {
    color: #ffb454;
    font-style: italic;
  }
  .meta {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.12rem;
    font-size: 0.7rem;
    font-family: var(--mono);
    color: var(--text-dim);
    white-space: nowrap;
    flex-shrink: 0;
  }
  .meta .strong {
    color: var(--accent);
    font-weight: 600;
  }
  .body {
    display: grid;
    grid-template-columns: 280px minmax(0, 1fr);
    gap: 1.1rem;
    align-items: start;
  }
  @media (max-width: 900px) {
    .body {
      grid-template-columns: 1fr;
    }
    .meta {
      display: none;
    }
  }
  .rail {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    min-width: 0;
  }
  .divider {
    height: 1px;
    background: var(--border);
  }
  .group-label {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
  }
  .railhint {
    margin: 0;
    color: var(--text-dim);
    font-size: 0.72rem;
    line-height: 1.45;
  }
  .stagecol {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-width: 0;
  }
  .stage {
    position: relative;
    min-height: 522px;
    border-radius: 12px;
    overflow: hidden;
  }
  .diag {
    transition: opacity 0.2s ease;
  }
  .diag.dim {
    opacity: 0.35;
    pointer-events: none;
  }
  .gload {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    min-height: 520px;
    padding: 2rem;
    text-align: center;
  }
  .gload.overlay {
    position: absolute;
    inset: 0;
    min-height: 0;
    background: rgba(11, 14, 20, 0.55);
    backdrop-filter: blur(2px);
    z-index: 2;
  }
  .glogo {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: var(--accent-grad);
    box-shadow: 0 0 26px rgba(110, 168, 254, 0.55);
    animation: breathe 1.6s ease-in-out infinite;
  }
  @keyframes breathe {
    0%,
    100% {
      transform: scale(0.92) rotate(0deg);
      opacity: 0.75;
    }
    50% {
      transform: scale(1.06) rotate(6deg);
      opacity: 1;
    }
  }
  .phase {
    margin: 0;
    font-size: 0.86rem;
    color: var(--text);
  }
  .phase b {
    font-family: var(--mono);
    font-weight: 600;
  }
  .phase-sub {
    margin: 0;
    font-size: 0.74rem;
    color: var(--text-dim);
    max-width: 420px;
    line-height: 1.45;
  }
  .indet {
    width: min(320px, 80%);
    height: 7px;
    border-radius: 999px;
    background: var(--bg-elev-2);
    overflow: hidden;
  }
  .indet-bar {
    width: 38%;
    height: 100%;
    border-radius: 999px;
    background: var(--accent-grad);
    box-shadow: 0 0 12px rgba(110, 168, 254, 0.5);
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
  .gerr {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    min-height: 520px;
    padding: 2rem;
    text-align: center;
    background: rgba(255, 122, 144, 0.06);
    border: 1px solid rgba(255, 122, 144, 0.25);
    border-radius: 12px;
    color: var(--bad);
    font-size: 0.85rem;
  }
  .gerr span {
    color: var(--text-dim);
    max-width: 460px;
    line-height: 1.5;
  }
  .gerr button {
    margin-top: 0.3rem;
  }
  .caption {
    margin: 0;
    color: var(--text-dim);
    font-size: 0.72rem;
  }
  .explainers {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  /* A button that reads as a link: the tab switch is an action, not navigation, so it
     must stay a <button> for assistive tech. */
  .linklike {
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: var(--accent);
    text-decoration: underline;
    cursor: pointer;
  }
</style>
