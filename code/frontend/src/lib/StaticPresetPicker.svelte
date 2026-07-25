<script lang="ts">
  import { onMount } from "svelte";

  import type { StaticPresetSummary } from "./staticClient";
  import {
    applyPresetState,
    staticExtras,
    staticPresetModel,
    stateFieldEqual,
  } from "./staticUx";
  import {
    fanout,
    layerFrom,
    layerTo,
    modelId,
    nParticles,
    nSteps,
    prefixText,
    rbfWidth,
    responseText,
    temperature,
  } from "./stores";

  // Static-mode preset picker for the three 001 views (spec US-3): each option is
  // a control state the real backend replayed and recorded at build time.
  // Selecting one writes that state into the shared stores, so the view's normal
  // fetch path hits the precomputed artifact exactly. When the current controls
  // don't match any preset, the picker itself says so — the view's own note
  // explains what to do (FR-203: nothing silently no-ops).
  interface Props {
    view: "vector" | "sankey" | "manifold";
  }
  let { view }: Props = $props();

  let presets = $state<StaticPresetSummary[]>([]);
  let presetModel = $state<string | null>(null);
  let loadError = $state("");

  // Reload the list when the active 001 tab changes (the component is keyed on it).
  $effect(() => {
    const v = view;
    const sc = staticExtras();
    if (!sc) return;
    let cancelled = false;
    sc.staticPresets(v)
      .then((p) => {
        if (!cancelled) {
          presets = p;
          loadError = "";
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) loadError = e instanceof Error ? e.message : String(e);
      });
    return () => {
      cancelled = true;
    };
  });

  onMount(() => {
    void staticPresetModel().then((m) => (presetModel = m));
  });

  // The current control state, projected onto each view's preset-state keys.
  const current = $derived.by((): Record<string, unknown> => {
    switch (view) {
      case "vector":
        return {
          prefix_text: $prefixText,
          temperature: $temperature,
          layer_from: $layerFrom,
          layer_to: $layerTo,
          fanout: $fanout,
          response_text: $responseText,
        };
      case "sankey":
        return {
          prefix_text: $prefixText,
          temperature: $temperature,
          n_particles: $nParticles,
          n_steps: $nSteps,
          response_text: $responseText,
        };
      case "manifold":
        return {
          prefix_text: $prefixText,
          temperature: $temperature,
          width: $rbfWidth,
          response_text: $responseText,
        };
    }
  });

  // Which preset the controls currently sit on (0 = none — custom settings).
  const activeN = $derived.by(() => {
    if (presetModel !== null && $modelId !== presetModel) return 0;
    const cur = current;
    const hit = presets.find((p) =>
      Object.entries(p.state).every(([k, v]) => k in cur && stateFieldEqual(cur[k], v)),
    );
    return hit?.n ?? 0;
  });

  function apply(n: number): void {
    const p = presets.find((x) => x.n === n);
    if (!p) return;
    if (presetModel) modelId.set(presetModel); // presets were recorded against this model
    applyPresetState(p.state);
  }
</script>

<label class="control">
  <span class="label">
    Demo presets
    <span class="chip" title="Each preset was computed by the real backend at build time and is served verbatim — the static demo carries these exact states.">precomputed</span>
  </span>
  <select
    data-testid={`static-preset-${view}`}
    value={String(activeN)}
    onchange={(e) => apply(Number(e.currentTarget.value))}
  >
    {#if activeN === 0}
      <option value="0" disabled>— your own settings (not precomputed) —</option>
    {/if}
    {#each presets as p (p.n)}
      <option value={String(p.n)}>{p.label}</option>
    {/each}
  </select>
  {#if loadError}
    <span class="err">preset list unavailable: {loadError}</span>
  {:else}
    <span class="hint">
      real backend outputs, captured at build time — free-form edits beyond them need the full stack
    </span>
  {/if}
</label>

<style>
  .chip {
    font-size: 0.6rem;
    font-family: var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(110, 168, 254, 0.12);
    border: 1px solid rgba(110, 168, 254, 0.35);
    border-radius: 999px;
    padding: 0.08rem 0.45rem;
    margin-left: 0.35rem;
    vertical-align: middle;
  }
  .hint {
    font-size: 0.7rem;
    color: var(--text-dim);
    line-height: 1.4;
  }
  .err {
    font-size: 0.7rem;
    color: var(--bad);
    font-family: var(--mono);
  }
</style>
