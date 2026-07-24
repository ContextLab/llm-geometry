<script lang="ts">
  import { onMount } from "svelte";
  import { modelId, numLayers, layerFrom, layerTo, modelError } from "../lib/stores";
  import { client, type ModelReference } from "../lib/dataClient";

  let models = $state<ModelReference[]>([]);
  let custom = $state("");
  let status = $state<"idle" | "resolving" | "ok" | "error">("idle");
  let message = $state("");
  let detail = $state(""); // raw resolver error (HF traceback etc.) — shown only via title

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
        detail = "";
        modelError.set(""); // the selected model is live again — views un-mark stale content
        const nl = ref.capabilities.num_layers ?? 0;
        numLayers.set(nl);
        // Default the readout layer to the FINAL layer (where next-token probabilities are
        // computed). Single layer → keep from/to in sync.
        layerFrom.set(nl);
        layerTo.set(nl);
      })
      .catch((e) => {
        if (cancelled) return;
        status = "error";
        // Concise plain-language first line; the raw detail (often a multi-page HF
        // traceback) stays behind the title tooltip instead of flooding the sidebar.
        detail = e?.message ?? "could not resolve model";
        message = e?.type === "NetworkError"
          ? "Could not reach the backend server"
          : "Model not found or could not be loaded";
        modelError.set(detail); // views dim their previous-model content as stale
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
  <!-- One-way (value + onchange), NOT bind:value: the store is the single source of truth,
       so the DOM can never write a spurious option back while the list loads async. -->
  <select value={$modelId} onchange={(e) => modelId.set(e.currentTarget.value)} data-testid="model-select">
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
  <div class="msg" class:err={status === "error"} title={detail || undefined} data-testid="model-message">{message}</div>
</div>

<style>
  .custom { display: flex; gap: 0.5rem; }
  .custom input { flex: 1; }
  .msg { font-size: 0.78rem; color: var(--text-dim); font-family: var(--mono); min-height: 1em; }
  .msg.err { color: var(--bad); cursor: help; }
</style>
