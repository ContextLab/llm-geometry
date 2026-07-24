<script lang="ts">
  import { onDestroy } from "svelte";
  import { get } from "svelte/store";
  import { responseText, responseStep, responseTokenCount, isPlaying, modelId, view } from "../lib/stores";
  import { client } from "../lib/dataClient";

  // On the Sankey the response isn't stepped — it's highlighted over the swarm — so the
  // ▶ Play / step row is hidden and the label reflects that.
  const sankeyMode = $derived($view === "sankey");

  let timer: ReturnType<typeof setInterval> | undefined;

  // Preset response trajectories (pair these with a prompt to watch the context shift them).
  const presets = [
    "Paris, the capital of France.",
    "cold, of course.",
    "blue, and full of wonder.",
    "the answer is obviously four.",
    "money is the root of all evil.",
    "once told me that nothing lasts forever.",
  ];

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
        responseStep.set(res.tokens.length); // at rest show the full trajectory; ▶ Play resets to 0
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
    }, 1500); // give the per-step morph time to settle (slow, watchable playback)
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
  <span class="label">
    {#if sankeyMode}Response to highlight <span class="opt">(optional · marks this path on the swarm)</span>
    {:else}Response trajectory <span class="opt">(optional · ▶ animates the views)</span>{/if}
  </span>
  <select
    onchange={(e) => { if (e.currentTarget.value) responseText.set(e.currentTarget.value); e.currentTarget.selectedIndex = 0; }}
    data-testid="response-presets"
  >
    <option value="">Example responses…</option>
    {#each presets as p (p)}<option value={p}>{p.slice(0, 34)}</option>{/each}
  </select>
  <textarea rows="2" bind:value={$responseText} placeholder="e.g. Paris, the capital of France." data-testid="response-input"></textarea>
  {#if $responseTokenCount > 0 && !sankeyMode}
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
