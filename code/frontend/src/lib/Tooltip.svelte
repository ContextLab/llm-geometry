<script lang="ts">
  import { tooltip } from "./tooltip";
</script>

{#if $tooltip.show}
  <div
    class="tip"
    style="left: {$tooltip.x + 14}px; top: {$tooltip.y + 14}px"
    data-testid="hover-tooltip"
  >
    {$tooltip.text}
  </div>
{/if}

<style>
  .tip {
    position: fixed;
    z-index: 50;
    pointer-events: none;
    background: rgba(10, 14, 24, 0.96);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.55rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    max-width: 320px;
    box-shadow: var(--shadow);
    white-space: pre-line;
  }
</style>
