<script lang="ts">
  import { prefixText } from "../lib/stores";

  // Preset prompts (§1/§2: "select from a dropdown menu").
  const presets = [
    "The capital of France is",
    "Once upon a time,",
    "The meaning of life is",
    "Q: What is 2 + 2?\nA:",
    "Roses are red, violets are",
    "def fibonacci(n):",
    "Translate to French: good morning ->",
  ];
</script>

<label class="control">
  <span class="label">Prompt / context prefix</span>
  <select
    onchange={(e) => { if (e.currentTarget.value) prefixText.set(e.currentTarget.value); }}
    data-testid="prompt-presets"
  >
    <option value="">Presets…</option>
    {#each presets as p (p)}<option value={p}>{p.replace(/\n/g, " ⏎ ").slice(0, 34)}</option>{/each}
  </select>
  <textarea rows="2" bind:value={$prefixText} placeholder="Type a prompt…" data-testid="prefix-input"></textarea>
</label>
