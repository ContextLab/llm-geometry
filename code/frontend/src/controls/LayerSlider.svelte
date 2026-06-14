<script lang="ts">
  import { layer, numLayers } from "../lib/stores";
</script>

<label class="control">
  <span class="label">Layer <b data-testid="layer-value">{$layer}</b> / {$numLayers}</span>
  <input
    type="range"
    min="0"
    max={$numLayers}
    step="1"
    bind:value={$layer}
    disabled={$numLayers === 0}
    data-testid="layer-input"
  />
</label>
