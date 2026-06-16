<script lang="ts">
  import { prefixText } from "../lib/stores";

  // Preset prompts (§1/§2: "select from a dropdown menu"). Chosen to give confident,
  // varied next-token predictions that make the geometry legible.
  const presets = [
    "The capital of France is",
    "The opposite of hot is",
    "Roses are red, violets are",
    "Once upon a time, there was a",
    "2 + 2 =",
    "The three primary colours are",
    "She opened the door and saw",
    "def factorial(n):",
    "Translate to French: \"good morning\" →",
    "In conclusion, the most important thing is",
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
