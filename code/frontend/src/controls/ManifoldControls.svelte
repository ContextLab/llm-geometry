<script lang="ts">
  import { rbfWidth, showSurface } from "../lib/stores";
</script>

<div class="control">
  <span class="label">Bump width <b data-testid="rbfwidth-value">{$rbfWidth.toFixed(2)}</b></span>
  <input
    type="range" min="0.05" max="0.6" step="0.01" bind:value={$rbfWidth}
    data-testid="rbfwidth-input"
  />
  <span class="hint">RBF cap width on the sphere: smaller = tighter, more localized domes toward likely tokens; larger = broad, blended swells.</span>
  <label class="toggle">
    <input type="checkbox" bind:checked={$showSurface} data-testid="surface-toggle" />
    <span>Surface flow field <span class="opt">(from a likely token → its predicted next token)</span></span>
  </label>
</div>

<style>
  .control input[type="range"] { width: 100%; }
  .hint { font-size: 0.72rem; color: var(--text-dim); line-height: 1.35; }
  .toggle { display: flex; align-items: flex-start; gap: 0.5rem; font-size: 0.8rem; cursor: pointer; }
  .toggle input { margin-top: 0.15rem; }
  .opt { color: var(--text-dim); }
</style>
