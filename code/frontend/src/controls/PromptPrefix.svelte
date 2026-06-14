<script lang="ts">
  import { prefixText } from "../lib/stores";
</script>

<label class="control">
  <span class="label">Prompt / context prefix</span>
  <textarea rows="2" bind:value={$prefixText} placeholder="Type a prompt…" data-testid="prefix-input"></textarea>
</label>
