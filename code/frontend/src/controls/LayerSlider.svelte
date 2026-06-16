<script lang="ts">
  import { layerFrom, layerTo, numLayers } from "../lib/stores";

  // The vector field positions tokens by their hidden state at ONE layer (the readout
  // layer); the final layer is where next-token probabilities are read. Keep layerFrom in
  // sync so the cache key / API stay single-layer.
  function setLayer(v: number) {
    layerTo.set(v);
    layerFrom.set(v);
  }
</script>

<div class="control">
  <span class="label">Readout layer <b data-testid="layer-value">{$layerTo}</b> / {$numLayers}</span>
  <input
    type="range" min="0" max={$numLayers} step="1" value={$layerTo}
    oninput={(e) => setLayer(+e.currentTarget.value)}
    disabled={$numLayers === 0} data-testid="layer-to"
  />
  <span class="hint">Which layer's representation positions the tokens. The final layer ({$numLayers}) is where the model's next-token probabilities are computed.</span>
</div>

<style>
  .control input[type="range"] { width: 100%; }
  .hint { font-size: 0.72rem; color: var(--text-dim); line-height: 1.35; }
</style>
