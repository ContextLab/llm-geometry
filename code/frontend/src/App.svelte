<script lang="ts">
  import { onMount } from "svelte";
  import ModelSelector from "./controls/ModelSelector.svelte";
  import PromptPrefix from "./controls/PromptPrefix.svelte";
  import ResponseAnimator from "./controls/ResponseAnimator.svelte";
  import Temperature from "./controls/Temperature.svelte";
  import LayerSlider from "./controls/LayerSlider.svelte";
  import SwarmControls from "./controls/SwarmControls.svelte";
  import ManifoldControls from "./controls/ManifoldControls.svelte";
  import VectorField from "./viz/VectorField.svelte";
  import Sankey from "./viz/Sankey.svelte";
  import Manifold from "./viz/Manifold.svelte";
  import ArchitectureExplorer from "./viz/arch/ArchitectureExplorer.svelte";
  import GeometryLab from "./viz/geo/GeometryLab.svelte";
  import Tooltip from "./lib/Tooltip.svelte";
  import StaticBadge from "./lib/StaticBadge.svelte";
  import StaticPresetPicker from "./lib/StaticPresetPicker.svelte";
  import { STATIC_MODE, applyStaticDefaults } from "./lib/staticUx";
  import { view, refreshNonce, type View } from "./lib/stores";

  const tabs: { id: View; label: string }[] = [
    { id: "vector", label: "Vector field" },
    { id: "sankey", label: "Sankey" },
    { id: "manifold", label: "Manifold" },
    { id: "architecture", label: "Architecture" },
    { id: "geometry", label: "Geometry" },
  ];

  const settingsLabels: Record<View, string> = {
    vector: "Vector field",
    sankey: "Sankey",
    manifold: "Manifold",
    architecture: "Architecture",
    geometry: "Geometry",
  };

  // The two explorer tabs own their controls (per-view stores; see specs/002 plan §3b) —
  // the shared model/prompt/temperature sidebar semantics don't apply to them.
  const sharedControlsViews: View[] = ["vector", "sankey", "manifold"];

  // Static boot (feature 003): before the 001 views fire their first fetch, point the
  // shared stores at preset 1 / the recorded preset model so the default state is an
  // exact precomputed hit (instant render — no miss-note flash). Backend builds skip
  // this entirely (STATIC_MODE is compile-time false).
  let staticBooting = $state(STATIC_MODE);
  onMount(() => {
    if (STATIC_MODE) void applyStaticDefaults().finally(() => (staticBooting = false));
  });
</script>

<div class="app">
  <header class="masthead">
    <div class="brand">
      <span class="logo"></span>
      <div>
        <h1>llm-geometry</h1>
        <p>Explore the geometry of a transformer's embedding space.</p>
      </div>
    </div>
    {#if STATIC_MODE}<StaticBadge />{/if}
  </header>

  <main class="layout">
    <aside class="controls panel" data-testid="controls">
      <div class="controls-head">
        <h2>Controls</h2>
        {#if !STATIC_MODE}
          <!-- Static builds serve precomputed artifacts: a force-recompute can never
               succeed there, so the button hides instead of lying (red-team static F1). -->
          <button class="recompute" data-testid="recompute" title="Force re-compute the current view" onclick={() => refreshNonce.update((n) => n + 1)}>↻ Recompute</button>
        {/if}
      </div>
      {#if sharedControlsViews.includes($view)}
        {#if STATIC_MODE && ($view === "vector" || $view === "sankey" || $view === "manifold")}
          <StaticPresetPicker view={$view} />
        {/if}
        <ModelSelector />
        <PromptPrefix />
        <Temperature />
      {/if}
      <!-- View-specific controls: the readout layer only shapes the vector field; the response
           animates the vector field + manifold and highlights a path on the Sankey; the particle
           swarm (count + sequence length) and RBF width are each their own view's. -->
      <div class="divider"></div>
      <span class="group-label">{settingsLabels[$view]} settings</span>
      {#if $view === "vector"}<LayerSlider />{/if}
      {#if sharedControlsViews.includes($view)}<ResponseAnimator />{/if}
      {#if $view === "sankey"}<SwarmControls />{/if}
      {#if $view === "manifold"}<ManifoldControls />{/if}
      {#if sharedControlsViews.includes($view)}
        <p class="hint">Cached results return instantly; the first computation streams progress. ▶ Play animates token-by-token. Each figure can be exported as a vector PDF/SVG, a high-res PNG, or an animated GIF/MP4.</p>
      {:else}
        <p class="hint">This view's controls live inside the panel to the right.</p>
      {/if}
    </aside>

    <div class="main">
      <nav class="tabs" data-testid="view-tabs">
        {#each tabs as t (t.id)}
          <button
            class:active={$view === t.id}
            onclick={() => view.set(t.id)}
            data-testid={`tab-${t.id}`}
          >{t.label}</button>
        {/each}
      </nav>

      {#if staticBooting}
        <!-- one short beat while index.json + preset states load (static builds only) -->
        <div class="panel static-boot" data-testid="static-boot">loading the precomputed demo state…</div>
      {:else if $view === "vector"}
        <VectorField />
      {:else if $view === "sankey"}
        <Sankey />
      {:else if $view === "manifold"}
        <Manifold />
      {:else if $view === "architecture"}
        <ArchitectureExplorer />
      {:else if $view === "geometry"}
        <GeometryLab />
      {/if}
    </div>
  </main>

  <Tooltip />
</div>

<style>
  .app { max-width: 1180px; margin: 0 auto; padding: 1.6rem; }
  .masthead { display: flex; align-items: center; justify-content: space-between; gap: 0.9rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
  .static-boot { padding: 2.2rem 1.6rem; color: var(--text-dim); font-size: 0.84rem; font-family: var(--mono); }
  .brand { display: flex; align-items: center; gap: 0.9rem; }
  .logo { width: 38px; height: 38px; border-radius: 11px; background: var(--accent-grad); box-shadow: 0 0 22px rgba(110,168,254,0.5); }
  h1 { margin: 0; font-size: 1.4rem; letter-spacing: -0.01em; }
  .brand p { margin: 0.1rem 0 0; color: var(--text-dim); font-size: 0.86rem; }
  .layout { display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 1.2rem; align-items: start; }
  @media (max-width: 900px) { .layout { grid-template-columns: minmax(0, 1fr); } }
  .controls { padding: 1.2rem 1.3rem; display: flex; flex-direction: column; gap: 1.05rem; }
  .controls-head { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
  .controls h2 { margin: 0; font-size: 1rem; letter-spacing: 0.01em; }
  .recompute {
    background: var(--bg-elev-2); color: var(--text-dim); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.3rem 0.6rem; font-size: 0.74rem; font-family: var(--mono); font-weight: 500; cursor: pointer;
  }
  .recompute:hover { color: var(--text); border-color: var(--accent); filter: none; }
  .divider { height: 1px; background: var(--border); margin: 0.1rem -0.2rem; }
  .group-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  .hint { margin: 0; color: var(--text-dim); font-size: 0.78rem; line-height: 1.45; }
  .main { display: flex; flex-direction: column; gap: 0.9rem; min-width: 0; }
  /* flex-wrap + max-width keep all five pills reachable on narrow phones (three tab
     red-teams measured 95px+ of horizontal page overflow at 390-420px). */
  .tabs { display: inline-flex; flex-wrap: wrap; gap: 0.3rem; padding: 0.3rem; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 999px; align-self: flex-start; max-width: 100%; }
  .tabs button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.84rem;
    font-weight: 500;
    transition: color 0.15s ease, background 0.15s ease;
  }
  .tabs button:hover:not(.active) { color: var(--text); }
  .tabs button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
    box-shadow: 0 2px 10px rgba(110,168,254,0.25);
  }
</style>
