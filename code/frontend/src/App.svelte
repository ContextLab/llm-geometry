<script lang="ts">
  import ModelSelector from "./controls/ModelSelector.svelte";
  import PromptPrefix from "./controls/PromptPrefix.svelte";
  import ResponseAnimator from "./controls/ResponseAnimator.svelte";
  import Temperature from "./controls/Temperature.svelte";
  import LayerSlider from "./controls/LayerSlider.svelte";
  import SwarmControls from "./controls/SwarmControls.svelte";
  import VectorField from "./viz/VectorField.svelte";
  import Sankey from "./viz/Sankey.svelte";
  import Manifold from "./viz/Manifold.svelte";
  import Tooltip from "./lib/Tooltip.svelte";
  import { view, type View } from "./lib/stores";

  const tabs: { id: View; label: string }[] = [
    { id: "vector", label: "Vector field" },
    { id: "sankey", label: "Sankey" },
    { id: "manifold", label: "Manifold" },
  ];
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
  </header>

  <main class="layout">
    <aside class="controls panel" data-testid="controls">
      <h2>Controls</h2>
      <ModelSelector />
      <PromptPrefix />
      <Temperature />
      <!-- View-specific controls: layers only shape the vector field's representation layer;
           the response trajectory animates the vector field + manifold; the particle swarm
           (count + sequence length) is the Sankey only. -->
      <div class="divider"></div>
      <span class="group-label">{$view === "sankey" ? "Sankey" : $view === "manifold" ? "Manifold" : "Vector field"} settings</span>
      {#if $view === "vector"}<LayerSlider />{/if}
      {#if $view !== "sankey"}<ResponseAnimator />{/if}
      {#if $view === "sankey"}<SwarmControls />{/if}
      <p class="hint">Cached results return instantly; the first computation streams progress. ▶ Play animates token-by-token. Each figure can be exported as a vector PDF/SVG or an animated GIF.</p>
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

      {#if $view === "vector"}
        <VectorField />
      {:else if $view === "sankey"}
        <Sankey />
      {:else if $view === "manifold"}
        <Manifold />
      {/if}
    </div>
  </main>

  <Tooltip />
</div>

<style>
  .app { max-width: 1180px; margin: 0 auto; padding: 1.6rem; }
  .masthead { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.4rem; }
  .brand { display: flex; align-items: center; gap: 0.9rem; }
  .logo { width: 38px; height: 38px; border-radius: 11px; background: var(--accent-grad); box-shadow: 0 0 22px rgba(110,168,254,0.5); }
  h1 { margin: 0; font-size: 1.4rem; letter-spacing: -0.01em; }
  .brand p { margin: 0.1rem 0 0; color: var(--text-dim); font-size: 0.86rem; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 1.2rem; align-items: start; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  .controls { padding: 1.2rem 1.3rem; display: flex; flex-direction: column; gap: 1.05rem; }
  .controls h2 { margin: 0 0 0.1rem; font-size: 1rem; letter-spacing: 0.01em; }
  .divider { height: 1px; background: var(--border); margin: 0.1rem -0.2rem; }
  .group-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  .hint { margin: 0; color: var(--text-dim); font-size: 0.78rem; line-height: 1.45; }
  .main { display: flex; flex-direction: column; gap: 0.9rem; }
  .tabs { display: inline-flex; gap: 0.3rem; padding: 0.3rem; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 999px; align-self: flex-start; }
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
