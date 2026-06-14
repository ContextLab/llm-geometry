<script lang="ts">
  import { temperature } from "../lib/stores";
</script>

<label class="control">
  <span class="label">Temperature <b data-testid="temp-value">{$temperature.toFixed(2)}</b></span>
  <input type="range" min="0" max="2" step="0.05" bind:value={$temperature} data-testid="temp-input" />
</label>
