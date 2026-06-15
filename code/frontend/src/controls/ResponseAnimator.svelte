<script lang="ts">
  import { onDestroy } from "svelte";
  import { get } from "svelte/store";
  import { responseText, responseStep, responseTokenCount, isPlaying, modelId } from "../lib/stores";
  import { client } from "../lib/dataClient";

  let timer: ReturnType<typeof setInterval> | undefined;

  // Keep the response token count in sync (drives the step slider + animation length).
  $effect(() => {
    const m = $modelId;
    const r = $responseText;
    if (!r.trim()) {
      responseTokenCount.set(0);
      responseStep.set(0);
      return;
    }
    let cancelled = false;
    client
      .tokenize(m, r)
      .then((res) => {
        if (cancelled) return;
        responseTokenCount.set(res.tokens.length);
        responseStep.update((s) => Math.min(s, res.tokens.length));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  });

  function stop() {
    isPlaying.set(false);
    if (timer) {
      clearInterval(timer);
      timer = undefined;
    }
  }

  function play() {
    const total = get(responseTokenCount);
    if (total === 0) return;
    isPlaying.set(true);
    timer = setInterval(() => {
      const next = get(responseStep) + 1;
      if (next > get(responseTokenCount)) {
        stop();
      } else {
        responseStep.set(next);
      }
    }, 1100);
  }

  function toggle() {
    if (get(isPlaying)) {
      stop();
    } else {
      if (get(responseStep) >= get(responseTokenCount)) responseStep.set(0);
      play();
    }
  }

  onDestroy(stop);
</script>

<label class="control">
  <span class="label">Response trajectory <span class="opt">(optional · ▶ animates the views)</span></span>
  <textarea rows="2" bind:value={$responseText} placeholder="e.g. Paris, the capital of France." data-testid="response-input"></textarea>
  {#if $responseTokenCount > 0}
    <div class="row">
      <button onclick={toggle} data-testid="play-button">{$isPlaying ? "⏸ Pause" : "▶ Play"}</button>
      <input type="range" min="0" max={$responseTokenCount} step="1" bind:value={$responseStep} data-testid="step-input" />
      <span class="mini" data-testid="step-label">{$responseStep}/{$responseTokenCount}</span>
    </div>
  {/if}
</label>

<style>
  .opt { color: var(--text-dim); font-weight: 400; }
  .row { display: flex; align-items: center; gap: 0.5rem; }
  .row input[type="range"] { flex: 1; }
  .mini { font-size: 0.72rem; color: var(--text-dim); font-variant-numeric: tabular-nums; }
  button { padding: 0.35rem 0.6rem; font-size: 0.8rem; white-space: nowrap; }
</style>
