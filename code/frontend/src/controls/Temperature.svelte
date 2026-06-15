<script lang="ts">
  import { temperature, clampTemperature } from "../lib/stores";
</script>

<label class="control">
  <span class="label">Temperature <b data-testid="temp-value">{$temperature.toFixed(2)}</b></span>
  <div class="row">
    <input type="range" min="0" max="2" step="0.05" bind:value={$temperature} data-testid="temp-input" />
    <input
      type="number" min="0" max="2" step="0.05" class="num" value={$temperature}
      oninput={(e) => temperature.set(clampTemperature(+e.currentTarget.value))}
      data-testid="temp-number"
    />
  </div>
</label>

<style>
  .row { display: flex; align-items: center; gap: 0.6rem; }
  .row input[type="range"] { flex: 1; }
  .num { width: 4.2rem; }
</style>
