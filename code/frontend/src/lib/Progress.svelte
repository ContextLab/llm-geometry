<script lang="ts">
  // Determinate progress indicator with a smooth animated bar (FR-009, SC-004).
  let { progress = 0, message = "" }: { progress?: number; message?: string } = $props();
  const pct = $derived(Math.max(0, Math.min(1, progress)) * 100);
</script>

<div class="wrap" data-testid="progress">
  <div class="track" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
    <div class="bar" style="width: {pct}%"></div>
  </div>
  {#if message}<div class="msg">{message}</div>{/if}
</div>

<style>
  .wrap { display: flex; flex-direction: column; gap: 0.4rem; }
  .track {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: var(--bg-elev-2);
    overflow: hidden;
  }
  .bar {
    height: 100%;
    border-radius: 999px;
    background: var(--accent-grad);
    transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 12px rgba(110, 168, 254, 0.5);
  }
  .msg { font-size: 0.8rem; color: var(--text-dim); font-family: var(--mono); }
</style>
