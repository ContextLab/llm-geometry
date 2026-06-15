<script lang="ts">
  import { layerFrom, layerTo, numLayers } from "../lib/stores";

  function onFrom(v: number) {
    layerFrom.set(v);
    if (v > $layerTo) layerTo.set(v);
  }
  function onTo(v: number) {
    layerTo.set(v);
    if (v < $layerFrom) layerFrom.set(v);
  }
</script>

<div class="control">
  <span class="label">
    Layer{$layerTo !== $layerFrom ? "s" : ""}
    <b data-testid="layer-value">{$layerFrom}{#if $layerTo !== $layerFrom}–{$layerTo}{/if}</b> / {$numLayers}
  </span>
  <div class="row">
    <span class="mini">from</span>
    <input
      type="range" min="0" max={$numLayers} step="1" value={$layerFrom}
      oninput={(e) => onFrom(+e.currentTarget.value)}
      disabled={$numLayers === 0} data-testid="layer-from"
    />
  </div>
  <div class="row">
    <span class="mini">to</span>
    <input
      type="range" min="0" max={$numLayers} step="1" value={$layerTo}
      oninput={(e) => onTo(+e.currentTarget.value)}
      disabled={$numLayers === 0} data-testid="layer-to"
    />
  </div>
</div>

<style>
  .row { display: flex; align-items: center; gap: 0.5rem; }
  .mini { font-size: 0.72rem; color: var(--text-dim); width: 2.4rem; }
  .row input { flex: 1; }
</style>
