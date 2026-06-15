<script lang="ts">
  import { responseText } from "../lib/stores";
</script>

<label class="control">
  <span class="label">Response trajectory <span class="opt">(optional)</span></span>
  <textarea
    rows="2"
    bind:value={$responseText}
    placeholder="e.g. Paris, the capital…"
    data-testid="response-input"
  ></textarea>
</label>

<style>
  .opt { color: var(--text-dim); font-weight: 400; }
</style>
