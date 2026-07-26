<script lang="ts">
  import { onMount } from "svelte";
  import { get } from "svelte/store";
  import { archModelId } from "../../lib/explorerStores";
  import { client, type ModelReference } from "../../lib/dataClient";
  import { plainError } from "./archShared";
  import { STATIC_MODE } from "../../lib/staticUx";

  const ISSUE_URL = "https://github.com/ContextLab/llm-geometry/issues/4";

  // Curated menu ONLY (feature 004, FR-412). The old free-text "any open-weights HF
  // id" box promised something the deployed demo cannot keep: the static build needs a
  // community ONNX export, which most HF repos do not have. Growing the list — or
  // filtering the Hub for repos that genuinely load — is tracked in issue #4.
  //
  // Picks are still resolved (existence + capability) BEFORE the store updates, so a
  // rejected model leaves the active one untouched (FR-107). Size gating
  // (ModelTooLargeError) happens server-side at graph time — the parent reverts the
  // store and surfaces that message here via `externalError`.
  interface Props {
    externalError?: string;
  }
  let { externalError = "" }: Props = $props();

  let models = $state<ModelReference[]>([]);
  let status = $state<"idle" | "checking" | "ok" | "error">("idle");
  let message = $state("");

  // The dropdown must ALWAYS offer the app's default model (users can return to it
  // after switching away — F2) plus the currently active id, even when the curated
  // /api/models list contains neither. Deduped against the curated entries.
  // STATIC build: the list IS the complete catalog (only precomputed models can be
  // explored), so don't offer the backend default when this build doesn't carry it —
  // only the transiently-active id until the boot steer lands.
  const DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"; // archModelId's initial value (explorerStores.ts)
  const extraOptions = $derived(
    (STATIC_MODE ? [$archModelId] : [DEFAULT_MODEL_ID, $archModelId]).filter(
      (id, i, arr) => arr.indexOf(id) === i && !models.some((m) => m.model_id === id),
    ),
  );

  onMount(async () => {
    try {
      models = (await client.listModels()).models;
    } catch {
      // Listing failed: the dropdown still offers the default + active id via
      // extraOptions, and the graph loader surfaces the real error.
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

  // A non-empty externalError (a server-side rejection surfaced by the parent, e.g.
  // ModelTooLargeError at graph time) must override the picker's own stale "ok"
  // badge/message with error styling (F1) — until the user interacts again
  // (status flips to "checking" on the next pick).
  const badge = $derived(
    status === "checking" ? "resolving" : status === "error" || externalError ? "error" : "ok",
  );
  const badgeText = $derived(
    status === "checking"
      ? "checking"
      : status === "error" || externalError
        ? "error"
        : status === "idle"
          ? "ready"
          : status,
  );
</script>

<div class="control" data-testid="arch-model-picker">
  <span class="label">
    Model
    <span class="badge {badge}" data-testid="arch-model-status">{badgeText}</span>
  </span>
  <select value={$archModelId} onchange={onSelectChange} data-testid="arch-model-select">
    {#each models as m (m.model_id)}
      <option value={m.model_id}>{m.display_name ?? m.model_id}</option>
    {/each}
    {#each extraOptions as id (id)}
      <option value={id}>{id}</option>
    {/each}
  </select>
  <p class="curated" data-testid="arch-model-curated-note">
    A curated set of small open-weights models. {#if STATIC_MODE}They run in your browser
    via their community ONNX exports;{:else}Each is traced live by the backend;{/if}
    <a href={ISSUE_URL} target="_blank" rel="noopener noreferrer">issue #4</a> tracks
    widening the list.
  </p>
  {#if status === "checking"}
    <div class="msg">{message}</div>
  {:else if status === "error"}
    <div class="err" data-testid="arch-error">{message}</div>
  {:else if externalError}
    <div class="err" data-testid="arch-error">{externalError}</div>
  {:else}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .curated {
    margin: 0;
    font-size: 0.72rem;
    line-height: 1.45;
    color: var(--text-dim);
  }
  .curated a {
    color: var(--accent);
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
