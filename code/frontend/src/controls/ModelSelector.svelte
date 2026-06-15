<script lang="ts">
  import { onMount } from "svelte";
  import { modelId, numLayers, layerFrom, layerTo } from "../lib/stores";
  import { client, type ModelReference } from "../lib/dataClient";

  let models = $state<ModelReference[]>([]);
  let custom = $state("");
  let status = $state<"idle" | "resolving" | "ok" | "error">("idle");
  let message = $state("");

  onMount(async () => {
    try {
      models = (await client.listModels()).models;
    } catch (e) {
      // listing is best-effort; selection still works via custom id
    }
  });

  // Resolve the selected model whenever it changes -> capabilities drive the layer slider.
  $effect(() => {
    const id = $modelId;
    let cancelled = false;
    status = "resolving";
    message = "checking model…";
    client
      .resolveModel(id)
      .then((ref) => {
        if (cancelled) return;
        status = "ok";
        message = `${ref.display_name} · ${ref.capabilities.num_layers ?? "?"} layers`;
        const nl = ref.capabilities.num_layers ?? 0;
        numLayers.set(nl);
        layerFrom.update((l: number) => Math.min(l, nl));
        layerTo.update((l: number) => Math.min(l, nl));
      })
      .catch((e) => {
        if (cancelled) return;
        status = "error";
        message = e.message ?? "could not resolve model";
      });
    return () => {
      cancelled = true;
    };
  });

  function applyCustom() {
    const id = custom.trim();
    if (id) modelId.set(id);
  }
</script>

<div class="control">
  <span class="label">Model <span class="badge {status}" data-testid="model-status">{status}</span></span>
  <select bind:value={$modelId} data-testid="model-select">
    {#each models as m (m.model_id)}
      <option value={m.model_id}>{m.display_name ?? m.model_id}</option>
    {/each}
    {#if !models.some((m) => m.model_id === $modelId)}
      <option value={$modelId}>{$modelId}</option>
    {/if}
  </select>
  <div class="custom">
    <input
      type="text"
      placeholder="…or any open-weights HF id (e.g. distilgpt2)"
      bind:value={custom}
      onkeydown={(e) => e.key === "Enter" && applyCustom()}
      data-testid="model-custom"
    />
    <button onclick={applyCustom}>Load</button>
  </div>
  <div class="msg" data-testid="model-message">{message}</div>
</div>

<style>
  .custom { display: flex; gap: 0.5rem; }
  .custom input { flex: 1; }
  .msg { font-size: 0.78rem; color: var(--text-dim); font-family: var(--mono); min-height: 1em; }
</style>
