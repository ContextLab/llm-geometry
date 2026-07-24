<script lang="ts">
  import { onMount } from "svelte";
  import { get } from "svelte/store";
  import { archModelId } from "../../lib/explorerStores";
  import { client, type ModelReference } from "../../lib/dataClient";
  import { plainError } from "./archShared";

  // Candidate models are resolved (existence + capability check) BEFORE the store
  // updates, so a rejected pick leaves the active model untouched (FR-107). Size
  // gating (ModelTooLargeError) happens server-side at graph time — the parent
  // reverts the store and surfaces that message here via `externalError`.
  interface Props {
    externalError?: string;
  }
  let { externalError = "" }: Props = $props();

  let models = $state<ModelReference[]>([]);
  let custom = $state("");
  let status = $state<"idle" | "checking" | "ok" | "error">("idle");
  let message = $state("");

  onMount(async () => {
    try {
      models = (await client.listModels()).models;
    } catch {
      // listing is best-effort; the free-text HF-id path still works
    }
  });

  let checkSeq = 0;
  async function pick(id: string): Promise<void> {
    const target = id.trim();
    if (!target || target === get(archModelId)) return;
    const my = ++checkSeq;
    status = "checking";
    message = `checking ${target}…`;
    try {
      const ref = await client.resolveModel(target);
      if (my !== checkSeq) return;
      status = "ok";
      const layers = ref.capabilities.num_layers;
      message = `${ref.display_name ?? target}${layers ? ` · ${layers} layers` : ""}`;
      archModelId.set(target); // only a verified pick changes the model
    } catch (e) {
      if (my !== checkSeq) return;
      status = "error";
      message = plainError(e);
    }
  }

  // One-way select (store is the source of truth). On a failed pick the store never
  // changes, so snap the DOM select back to the still-active model.
  async function onSelectChange(e: Event & { currentTarget: HTMLSelectElement }): Promise<void> {
    const el = e.currentTarget;
    await pick(el.value);
    el.value = get(archModelId);
  }

  const badge = $derived(
    status === "checking" ? "resolving" : status === "error" ? "error" : "ok",
  );
</script>

<div class="control" data-testid="arch-model-picker">
  <span class="label">
    Model
    <span class="badge {badge}" data-testid="arch-model-status">{status === "idle" ? "ready" : status}</span>
  </span>
  <select value={$archModelId} onchange={onSelectChange} data-testid="arch-model-select">
    {#each models as m (m.model_id)}
      <option value={m.model_id}>{m.display_name ?? m.model_id}</option>
    {/each}
    {#if !models.some((m) => m.model_id === $archModelId)}
      <option value={$archModelId}>{$archModelId}</option>
    {/if}
  </select>
  <div class="custom">
    <input
      type="text"
      placeholder="…or any open-weights HF id"
      bind:value={custom}
      onkeydown={(e) => e.key === "Enter" && void pick(custom)}
      data-testid="arch-model-custom"
    />
    <button onclick={() => void pick(custom)}>Load</button>
  </div>
  {#if externalError || status === "error"}
    <div class="err" data-testid="arch-error">{externalError || message}</div>
  {:else}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .custom {
    display: flex;
    gap: 0.5rem;
  }
  .custom input {
    flex: 1;
    min-width: 0;
  }
  .msg {
    font-size: 0.76rem;
    color: var(--text-dim);
    font-family: var(--mono);
    min-height: 1em;
  }
  .err {
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.5rem 0.65rem;
    font-size: 0.78rem;
    line-height: 1.4;
  }
</style>
