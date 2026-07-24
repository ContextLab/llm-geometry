<script lang="ts">
  import type { GeoTokenizeResult } from "../../lib/dataClient";
  import { showTip, hideTip } from "../../lib/tooltip";

  // Tokenization strip: one chip per token, <unk> substitutions visibly marked
  // (spec acceptance 2.4), truncation indicator when the prompt exceeds the
  // 50-token context window.
  interface Props {
    result: GeoTokenizeResult | null;
  }
  let { result }: Props = $props();
</script>

{#if result && result.tokens.length > 0}
  <div class="strip" data-testid="geo-tokenize-strip">
    {#each result.tokens as t, i (i)}
      {#if t.unk}
        <span
          class="chip unk"
          role="note"
          onmouseenter={(e) => showTip(e, `"${t.text}" is not in the 1000-word vocabulary — the model sees <unk>`)}
          onmouseleave={hideTip}
        >{t.text}<small>&lt;unk&gt;</small></span>
      {:else}
        <span class="chip">{t.text}</span>
      {/if}
    {/each}
    {#if result.truncated}
      <span class="chip trunc" title="the prompt exceeds the 50-token context window — extra tokens are cut">⋯ truncated at 50</span>
    {/if}
    <span class="meta">{result.tokens.length} tokens{result.n_unk > 0 ? ` · ${result.n_unk} unknown` : ""}</span>
  </div>
{/if}

<style>
  .strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.28rem;
  }
  .chip {
    font-family: var(--mono);
    font-size: 0.72rem;
    padding: 0.14rem 0.42rem;
    border-radius: 7px;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    color: var(--text);
    white-space: pre;
  }
  .chip.unk {
    background: rgba(255, 122, 144, 0.10);
    border: 1px dashed var(--bad);
    color: var(--bad);
    display: inline-flex;
    align-items: baseline;
    gap: 0.3rem;
    cursor: help;
  }
  .chip.unk small {
    font-size: 0.6rem;
    opacity: 0.85;
    letter-spacing: 0.03em;
  }
  .chip.trunc {
    background: rgba(255, 180, 84, 0.12);
    border-color: rgba(255, 180, 84, 0.55);
    color: #ffb454;
  }
  .meta {
    margin-left: 0.3rem;
    font-size: 0.7rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
</style>
