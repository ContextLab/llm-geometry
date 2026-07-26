<script lang="ts">
  import {
    client,
    ApiError,
    type GeoMatrixName,
    type GeoPresetName,
    type GeoWeightsData,
  } from "../../lib/dataClient";
  import { geoModelNote, geoWeightsToken } from "../../lib/explorerStores";
  import MatrixHeatmap from "../../lib/MatrixHeatmap.svelte";
  import { STATIC_MODE } from "../../lib/staticUx";

  // Weight Lab (spec acceptance 2.3): inspect any of the tiny model's matrices, apply
  // presets or edit 3×3 cells directly; every edit round-trips through POST
  // /api/geo/weights and the returned content-hash weights_token (persisted in
  // sessionStorage via the geoWeightsToken store) so the vector field, trace and
  // attention all recompute against the edited model.
  interface Props {
    label: (id: number) => string; // token id -> text, for embedding row labels
    // The canonical learned checkpoint's content-hash id (from /api/geo/spec).
    // Weight tokens are content-hashes too, so a minted token EQUAL to this id means
    // the edit chain reproduced the learned weights bit-for-bit (e.g. preset
    // `learned`) — that state is shown and treated as "no edit" (G1).
    checkpointId?: string | null;
  }
  let { label, checkpointId = null }: Props = $props();

  const MATRICES: GeoMatrixName[] = ["embedding", "W_Q", "W_K", "W_V", "W_O"];
  const PRESETS: GeoPresetName[] = ["identity", "toeplitz_fuzzy", "random", "random_autocorr", "zero", "learned"];
  // STATIC build: numpy's seeded MT19937 stream isn't reproducible in float64 JS, so
  // the TS engine ships REAL backend-computed fixture matrices for exactly these seeds
  // (src/lib/geoEngine/presetFixtures.json `seeds`; requesting any other seed raises
  // InvalidWeightEditError). The free seed integer becomes this honest dropdown.
  const FIXTURE_SEEDS = [0, 1, 2, 7];

  let matrix = $state<GeoMatrixName>("W_V");
  let layer = $state(0);
  let preset = $state<GeoPresetName>("identity");
  let seed = $state(0); // used only for the seeded presets in the static build
  let data = $state<GeoWeightsData | null>(null);
  let loading = $state(false);
  let posting = $state(false);
  let error = $state("");
  let runId = 0;

  const isEmbedding = $derived(matrix === "embedding");
  // Content-based "edited" state: a non-null token whose hash matches the learned
  // checkpoint is NOT an edit. Decision (G1): we normalize such tokens to null (the
  // canonical learned state) rather than keep them — sessionStorage, the badge, and
  // every geo fetch then agree on one representation of "learned".
  const editedActive = $derived($geoWeightsToken !== null && $geoWeightsToken !== checkpointId);
  $effect(() => {
    // Self-heal tokens persisted before this rule (or minted below on a slow path).
    if (checkpointId && $geoWeightsToken === checkpointId) {
      geoModelNote.set(null);
      geoWeightsToken.set(null);
    }
  });
  // Embedding (1003×3) renders transposed as a horizontally scrollable ribbon:
  // columns = tokens, rows = x/y/z. 12px cells keep the canvas under browser limits.
  const displayGrid = $derived.by(() => {
    if (!data) return [] as number[][];
    if (!isEmbedding) return data.values;
    const [x, y, z] = [0, 1, 2].map((d) => data!.values.map((row) => row[d]));
    return [x, y, z];
  });
  const dimLabels = ["x", "y", "z"];
  const tokenLabels = $derived(
    isEmbedding && data ? data.values.map((_, i) => `"${label(i)}" (token ${i})`) : undefined,
  );

  $effect(() => {
    // reload the displayed matrix whenever the picker or the active token changes
    void loadWeights(matrix, isEmbedding ? undefined : layer, $geoWeightsToken ?? undefined);
  });

  async function loadWeights(m: GeoMatrixName, l: number | undefined, token: string | undefined) {
    const my = ++runId;
    loading = true;
    error = "";
    try {
      const d = await client.getGeoWeights({ matrix: m, layer: l, weights_token: token });
      if (my !== runId) return;
      data = d;
    } catch (e) {
      if (my !== runId) return;
      error = friendly(e);
    } finally {
      if (my === runId) loading = false;
    }
  }

  async function post(edits: { layer: number; matrix: GeoMatrixName; preset?: GeoPresetName; values?: number[][]; seed?: number }[]) {
    posting = true;
    error = "";
    try {
      const res = await client.postGeoWeights({ base: $geoWeightsToken ?? "learned", edits });
      // Minted token == learned checkpoint hash ⇒ bit-identical weights ⇒ no-edit:
      // store the canonical null instead (G1). Otherwise persist it; every geo fetch
      // now uses it.
      const backToCanonical = res.weights_token === checkpointId;
      geoModelNote.set(backToCanonical ? null : "hand-edited weights");
      geoWeightsToken.set(backToCanonical ? null : res.weights_token);
    } catch (e) {
      error = friendly(e);
    } finally {
      posting = false;
    }
  }

  const seededPreset = $derived(preset === "random" || preset === "random_autocorr");

  function applyPreset() {
    // The backend's default seed is 0; only the static build exposes a seed picker
    // (limited to the shipped fixture seeds), so only it ever sends a non-default one.
    const edit: { layer: number; matrix: GeoMatrixName; preset: GeoPresetName; seed?: number } = {
      layer: isEmbedding ? 0 : layer,
      matrix,
      preset,
    };
    if (STATIC_MODE && seededPreset) edit.seed = seed;
    void post([edit]);
  }

  function onCellEdit(row: number, col: number, value: number) {
    if (!data || isEmbedding) return;
    if (!Number.isFinite(value)) {
      error = "That cell needs a finite number — try something like 0.5.";
      return;
    }
    const values = data.values.map((r, ri) => r.map((v, ci) => (ri === row && ci === col ? value : v)));
    void post([{ layer, matrix, values }]);
  }

  function resetToLearned() {
    geoModelNote.set(null);
    geoWeightsToken.set(null); // canonical learned checkpoint everywhere
    error = "";
  }

  function friendly(e: unknown): string {
    if (e instanceof ApiError) {
      if (e.type === "InvalidWeightEditError") return `The edit was rejected: ${e.message}`;
      if (e.type === "NetworkError") return "Could not reach the server — is the backend running?";
      return e.message;
    }
    return String(e);
  }
</script>

<div class="panel-body" data-testid="geo-weight-panel">
  <div class="head">
    <h3>Weight lab</h3>
    {#if editedActive}
      <span class="badge edited" title={`weights_token ${$geoWeightsToken}`}>{$geoModelNote ?? "a model you created"} active</span>
      <button class="ghost" data-testid="geo-reset" onclick={resetToLearned}>reset to learned</button>
    {:else}
      <span class="badge learned">shipped checkpoint</span>
    {/if}
  </div>

  <div class="picker">
    <label class="field">
      <span>matrix</span>
      <select bind:value={matrix} data-testid="geo-matrix">
        {#each MATRICES as m (m)}<option value={m}>{m}</option>{/each}
      </select>
    </label>
    {#if !isEmbedding}
      <label class="field">
        <span>layer</span>
        <select bind:value={layer}>
          {#each [0, 1, 2, 3] as l (l)}<option value={l}>{l}</option>{/each}
        </select>
      </label>
    {/if}
    <label class="field">
      <span>preset</span>
      <select bind:value={preset} data-testid="geo-preset">
        {#each PRESETS as p (p)}<option value={p}>{p}</option>{/each}
      </select>
    </label>
    {#if STATIC_MODE && seededPreset}
      <label class="field">
        <span>seed</span>
        <select
          bind:value={seed}
          data-testid="geo-seed"
          title="the static demo ships real backend-computed matrices for these numpy seeds — other seeds need the full stack"
        >
          {#each FIXTURE_SEEDS as s (s)}<option value={s}>{s}</option>{/each}
        </select>
      </label>
    {/if}
    <button data-testid="geo-apply" onclick={applyPreset} disabled={posting}>
      {posting ? "applying…" : "Apply"}
    </button>
  </div>

  {#if error}
    <div class="error" data-testid="geo-error">{error}</div>
  {/if}

  {#if data}
    <div class="matrix-meta">
      <span class="mono">{matrix}{isEmbedding ? "" : ` · layer ${layer}`} · {data.shape.join("×")}</span>
      <span class="badge src">{data.source}</span>
      {#if !isEmbedding}<span class="hint">click a cell to edit its value</span>
      {:else}<span class="hint">cells are read-only (1003 rows) — use a preset above to change it (rows x/y/z)</span>{/if}
    </div>
    {#if isEmbedding}
      <div class="ribbon">
        <MatrixHeatmap values={displayGrid} rowLabels={dimLabels} colLabels={tokenLabels} maxCanvasPx={12040} />
      </div>
    {:else}
      <MatrixHeatmap
        values={displayGrid}
        rowLabels={["row 0", "row 1", "row 2"]}
        colLabels={["col 0", "col 1", "col 2"]}
        editable
        {onCellEdit}
        maxCanvasPx={132}
      />
    {/if}
  {:else if loading}
    <p class="hint">loading matrix…</p>
  {/if}
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  h3 {
    margin: 0;
    font-size: 0.92rem;
  }
  .badge {
    font-size: 0.68rem;
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
    font-family: var(--mono);
  }
  .badge.edited {
    color: #ffb454;
    background: rgba(255, 180, 84, 0.14);
    border: 1px solid rgba(255, 180, 84, 0.4);
  }
  .badge.learned {
    color: var(--good);
    background: rgba(91, 224, 176, 0.1);
  }
  .badge.src {
    color: var(--accent);
    background: rgba(110, 168, 254, 0.12);
  }
  .ghost {
    background: var(--bg-elev-2);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.22rem 0.55rem;
    font-size: 0.72rem;
    font-family: var(--mono);
  }
  .ghost:hover {
    color: var(--text);
    border-color: var(--accent);
    filter: none;
  }
  .picker {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.72rem;
    color: var(--text-dim);
  }
  .field select {
    padding: 0.35rem 0.5rem;
    font-size: 0.8rem;
    width: auto;
  }
  .picker button {
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
  }
  .matrix-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .mono {
    font-family: var(--mono);
    font-size: 0.74rem;
    color: var(--text);
  }
  .hint {
    margin: 0;
    font-size: 0.72rem;
    color: var(--text-dim);
  }
  .ribbon {
    max-width: 100%;
    overflow-x: auto;
    padding-bottom: 0.3rem;
  }
  .error {
    background: rgba(255, 122, 144, 0.12);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    font-size: 0.78rem;
  }
</style>
