<script lang="ts">
  import ModelSelector from "./controls/ModelSelector.svelte";
  import PromptPrefix from "./controls/PromptPrefix.svelte";
  import Temperature from "./controls/Temperature.svelte";
  import LayerSlider from "./controls/LayerSlider.svelte";
  import Preview from "./preview/Preview.svelte";
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
    <span class="tag">core machinery</span>
  </header>

  <main class="layout">
    <aside class="controls panel" data-testid="controls">
      <h2>Controls</h2>
      <ModelSelector />
      <PromptPrefix />
      <Temperature />
      <LayerSlider />
      <p class="hint">
        Cached results return instantly; the first computation streams progress.
      </p>
    </aside>

    <Preview />
  </main>
</div>

<style>
  .app { max-width: 1180px; margin: 0 auto; padding: 1.6rem; }
  .masthead { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.4rem; }
  .brand { display: flex; align-items: center; gap: 0.9rem; }
  .logo {
    width: 38px;
    height: 38px;
    border-radius: 11px;
    background: var(--accent-grad);
    box-shadow: 0 0 22px rgba(110, 168, 254, 0.5);
  }
  h1 { margin: 0; font-size: 1.4rem; letter-spacing: -0.01em; }
  .brand p { margin: 0.1rem 0 0; color: var(--text-dim); font-size: 0.86rem; }
  .tag {
    font-size: 0.72rem;
    color: var(--text-dim);
    border: 1px solid var(--border);
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
  }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 1.2rem; align-items: start; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  .controls { padding: 1.2rem 1.3rem; display: flex; flex-direction: column; gap: 1.1rem; }
  .controls h2 { margin: 0; font-size: 1rem; }
  .hint { margin: 0; color: var(--text-dim); font-size: 0.78rem; line-height: 1.4; }
</style>
