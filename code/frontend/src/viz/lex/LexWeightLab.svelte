<script lang="ts">
  /**
   * US-6 — modify the model, and watch the geometry and the text answer.
   *
   * This follows the Geometry Lab's Weight Lab exactly, because the rule it encodes is
   * the same one: an edit NEVER overwrites the model you trained. Every edit clones the
   * active weights, changes one tensor, and mints a new **content-hash** weight set; the
   * trained values stay untouched underneath, and `trained` (the preset, or the button in
   * the header) always takes you back.
   *
   * That hash also means "back" is decided by CONTENT, not by history: an edit chain that
   * happens to reproduce the trained weights bit-for-bit mints the trained model's own
   * token, and this panel then reports "trained", because it IS the trained model. Halve
   * a matrix and double it again and the badge goes out on its own.
   *
   * The spectrum, the token cloud and the sampler all read the tab's ACTIVE model, so
   * nothing here is a preview: applying an edit re-runs the eigendecomposition and the
   * next sample against the edited weights.
   *
   * Per-cell editing is offered where a cell is a thing you can reason about. The
   * vocabulary-rowed matrices — the embedding, and the readout when untied — are one row
   * per token and are read-only per cell; presets act on them whole. The panel says so
   * with the real row count rather than a vague "too big".
   */
  import {
    LexModel,
    LexVocab,
    cloneWeights,
    initWeights,
    lexEngineShapes,
    lexWeightsTokenOf,
    weightNames,
    type WeightSet,
  } from "../../lib/lexEngine";
  import Explain from "../../lib/Explain.svelte";
  import MatrixHeatmap from "../../lib/MatrixHeatmap.svelte";

  interface Props {
    /** The model an edit departs from and returns to: the trained one, or the random
     *  init when nothing has been trained yet. */
    base: LexModel | null;
    /** How the header should name that model ("trained" / "random init"). */
    baseLabel: string;
    /** The active vocabulary, for labelling embedding rows. */
    vocab: LexVocab | null;
    /** The current edit, if any — owned by the tab so every panel sees the same model. */
    edited: { model: LexModel; token: string; note: string } | null;
    onEdited: (model: LexModel, token: string, note: string) => void;
    onRestore: () => void;
  }
  let { base, baseLabel, vocab, edited, onEdited, onRestore }: Props = $props();

  /** Cells drawn per axis. Beyond this the panel shows the top-left window and says so —
   *  a 512×128 MLP matrix is not a thing to render 65,536 rectangles of. */
  const DISPLAY_MAX = 48;

  type Preset = "trained" | "zero" | "identity" | "randomize" | "halve" | "double";
  const PRESETS: { id: Preset; label: string; title: string }[] = [
    { id: "trained", label: "trained", title: "Restore this one matrix to the model you trained." },
    { id: "zero", label: "zero", title: "Set every element to 0 — the matrix stops contributing." },
    { id: "identity", label: "identity", title: "The identity matrix. Square matrices only." },
    { id: "randomize", label: "randomize", title: "Re-draw from this tensor's own initializer at the seed below." },
    { id: "halve", label: "×0.5", title: "Halve every element." },
    { id: "double", label: "×2", title: "Double every element." },
  ];

  let name = $state("embed");
  let preset = $state<Preset>("halve");
  let error = $state("");

  /** The ceiling the seed box declares, and the one `applyPreset` actually enforces. */
  const MAX_SEED = 9999;

  /**
   * The seed box's text, not its number.
   *
   * `bind:value` on `<input type="number">` hands back `null` for an empty box and a
   * `number` for everything the element will parse — so clearing the field re-drew the
   * tensor at seed 0 while the note under it read "at seed null", and `2.5` or `1e3` were
   * applied verbatim to an RNG that expects an integer in 0..9999. `randomize` is
   * reproducible ONLY by its seed: a seed that is not the one on screen makes the note a
   * false record of what the weights are, which is the one thing this panel exists to say.
   */
  let seedText = $state("0");

  /** The seed to draw with, or a sentence saying why nothing was drawn. Never a third thing. */
  function parseSeed(raw: string): { seed: number } | { error: string } {
    const text = raw.trim();
    // Digits only, as in the Vacancy panel's seed box: `Number()` reads "0x10", "1e3",
    // " 12 " and "3.0" and returns a number that is not the text that was typed.
    if (!/^\d+$/.test(text) || Number(text) > MAX_SEED) {
      return {
        error:
          `A seed is a whole number from 0 to ${MAX_SEED}. ${JSON.stringify(raw)} is not ` +
          "one, so nothing was re-drawn — `randomize` is reproducible only by its seed, " +
          "and a seed that is not the one you typed would make the note below wrong.",
      };
    }
    return { seed: Number(text) };
  }

  const cfg = $derived(base?.cfg ?? null);
  const names = $derived(cfg ? weightNames(cfg) : []);
  const shapes = $derived<Record<string, number[]>>(cfg ? lexEngineShapes(cfg) : {});
  /** The weights everything on the page is currently running: the edit if there is one. */
  const active = $derived<WeightSet | null>(edited?.model.weights ?? base?.weights ?? null);
  /** Content hash of the trained model — the canonical "no edit" state. */
  const baseToken = $derived(cfg && base ? lexWeightsTokenOf(cfg, base.weights) : "");

  // A shape change retires the selection: `layers.3.fc1_w` does not exist at n_layers = 2.
  $effect(() => {
    if (names.length > 0 && !names.includes(name)) {
      name = "embed";
      error = "";
    }
  });

  const shape = $derived<number[]>(shapes[name] ?? []);
  const rows = $derived(shape.length === 2 ? shape[0] : 1);
  const cols = $derived(shape.length === 2 ? shape[1] : (shape[0] ?? 0));
  const isSquare = $derived(shape.length === 2 && shape[0] === shape[1]);
  /** One row per token: the embedding, and the readout when the model is untied. */
  const vocabRowed = $derived(name === "embed" || name === "head_w");
  const cellsEditable = $derived(!vocabRowed);
  const truncated = $derived(rows > DISPLAY_MAX || cols > DISPLAY_MAX);
  const shownRows = $derived(Math.min(rows, DISPLAY_MAX));
  const shownCols = $derived(Math.min(cols, DISPLAY_MAX));

  const grid = $derived.by<number[][]>(() => {
    const values = active?.[name];
    if (!values || cols === 0) return [];
    const out: number[][] = [];
    for (let r = 0; r < shownRows; r++) {
      const row = new Array<number>(shownCols);
      for (let c = 0; c < shownCols; c++) row[c] = values[r * cols + c];
      out.push(row);
    }
    return out;
  });
  const rowLabels = $derived.by<string[]>(() => {
    if (vocabRowed && vocab) {
      return Array.from({ length: shownRows }, (_, r) => `"${vocab.itos[r] ?? "?"}" (row ${r})`);
    }
    return Array.from({ length: shownRows }, (_, r) => (shape.length === 2 ? `row ${r}` : "value"));
  });
  const colLabels = $derived(Array.from({ length: shownCols }, (_, c) => `col ${c}`));

  const presetAvailable = (p: Preset): boolean => (p === "identity" ? isSquare : true);
  /**
   * What the selected preset does, as TEXT. These descriptions used to exist only as
   * `title` attributes on `<option>` elements, which most assistive technology does not
   * expose at all and no keyboard user can reach — so the one place that said what
   * `randomize` re-draws from was invisible to some of the people reading it.
   */
  const presetNote = $derived(PRESETS.find((p) => p.id === preset)?.title ?? "");
  $effect(() => {
    if (!presetAvailable(preset)) preset = "halve";
  });

  /** Apply `next` as the active weight set, or fall back to the trained model when the
   *  content hash says the two are the same model. */
  function commit(next: WeightSet, note: string): void {
    if (!cfg) return;
    const token = lexWeightsTokenOf(cfg, next);
    if (token === baseToken) {
      onRestore(); // bit-identical to the trained weights: that IS the trained model
      return;
    }
    onEdited(new LexModel(cfg, next), token, note);
  }

  function applyPreset(): void {
    error = "";
    if (!cfg || !active || !base) return;
    const size = active[name].length;
    const next = cloneWeights(active);
    const out = new Float32Array(size);
    let note = "";
    if (preset === "trained") {
      out.set(base.weights[name]);
      note = `${name} back to its ${baseLabel} values`;
    } else if (preset === "zero") {
      note = `${name} zeroed`;
    } else if (preset === "identity") {
      for (let i = 0; i < rows; i++) out[i * cols + i] = 1;
      note = `${name} set to the identity`;
    } else if (preset === "randomize") {
      const parsed = parseSeed(seedText);
      if (!("seed" in parsed)) {
        error = parsed.error;
        return;
      }
      // The tensor's OWN initializer — N(0, 0.02²) for every matrix except the packed
      // QKV projection, which keeps PyTorch's xavier-uniform default.
      out.set(initWeights(cfg, parsed.seed)[name]);
      note = `${name} re-drawn from its initializer at seed ${parsed.seed}`;
    } else {
      const k = preset === "halve" ? 0.5 : 2;
      const src = active[name];
      for (let i = 0; i < size; i++) out[i] = src[i] * k;
      note = `${name} scaled by ${k}`;
    }
    next[name] = out;
    commit(next, note);
  }

  function onCellEdit(row: number, col: number, value: number): void {
    error = "";
    if (!cfg || !active) return;
    if (!cellsEditable) return;
    if (!Number.isFinite(value)) {
      error = "That cell needs a finite number — try something like 0.5.";
      return;
    }
    const next = cloneWeights(active);
    const arr = Float32Array.from(active[name]);
    arr[row * cols + col] = value;
    next[name] = arr;
    commit(next, `${name}[${row}, ${col}] = ${value}`);
  }

  function restore(): void {
    error = "";
    onRestore();
  }
</script>

<div class="panel-body" data-testid="lex-weight-panel">
  <div class="head">
    <h3>Weight lab</h3>
    {#if edited}
      <span class="badge edited" title={`weights_token ${edited.token}`} data-testid="lex-weight-badge">
        edited weights active
      </span>
      <button class="ghost" data-testid="lex-weight-restore" onclick={restore}>
        back to {baseLabel}
      </button>
    {:else}
      <span class="badge base" data-testid="lex-weight-badge">{baseLabel} weights</span>
    {/if}
  </div>

  {#if !cfg || !active}
    <p class="hint">The model appears once a budget resolves — pick one above.</p>
  {:else}
    <p class="panel-note">
      Pick a tensor and apply a preset, or click a cell to type a value. Editing does
      <b>not</b> overwrite the model you trained: each edit mints a new
      content-hash-addressed weight set, and <code>trained</code> — or the button above —
      restores it. The spectrum, the token cloud and the sampler all re-run against
      whatever is active, so an edit is never a preview.
    </p>

    <div class="picker">
      <label class="field">
        <span>tensor</span>
        <select bind:value={name} data-testid="lex-weight-matrix">
          {#each names as n (n)}
            <option value={n}>{n} · {(shapes[n] ?? []).join("×")}</option>
          {/each}
        </select>
      </label>
      <label class="field">
        <span>preset</span>
        <select bind:value={preset} data-testid="lex-weight-preset">
          {#each PRESETS as p (p.id)}
            <option value={p.id} disabled={!presetAvailable(p.id)} title={p.title}>
              {p.label}{presetAvailable(p.id) ? "" : " (square only)"}
            </option>
          {/each}
        </select>
      </label>
      {#if preset === "randomize"}
        <label class="field">
          <span>seed</span>
          <input
            type="number"
            min="0"
            max={MAX_SEED}
            step="1"
            value={seedText}
            oninput={(e) => (seedText = e.currentTarget.value)}
            data-testid="lex-weight-seed"
          />
        </label>
      {/if}
      <button data-testid="lex-weight-apply" onclick={applyPreset}>Apply</button>
    </div>

    <p class="hint small" data-testid="lex-weight-preset-note">
      <b>{preset}</b> — {presetNote}
    </p>

    {#if error}
      <div class="err" data-testid="lex-weight-error">{error}</div>
    {/if}

    <div class="matrix-meta">
      <span class="mono">{name} · {shape.join("×")}</span>
      {#if edited}
        <span class="badge tok mono" title="content hash of the whole active weight set">
          {edited.token.slice(0, 12)}…
        </span>
      {/if}
      {#if cellsEditable}
        <span class="hint">click a cell to edit its value</span>
      {:else}
        <span class="hint" data-testid="lex-weight-readonly">
          cells are read-only — one row per token, {rows.toLocaleString()} of them — so change
          this matrix with a preset
        </span>
      {/if}
    </div>

    {#if truncated}
      <p class="hint small" data-testid="lex-weight-truncated">
        showing the first {shownRows}×{shownCols} of {rows}×{cols}; a preset acts on the
        whole tensor
      </p>
    {/if}

    <div class="plot">
      <MatrixHeatmap
        values={grid}
        {rowLabels}
        {colLabels}
        editable={cellsEditable}
        {onCellEdit}
        maxCanvasPx={520}
      />
    </div>

    {#if edited}
      <p class="active-edit" data-testid="lex-weight-note">active edit: {edited.note}</p>
    {/if}
  {/if}

  <Explain
    title="What an edit actually does"
    hint="a new weight set, addressed by its own hash — never an overwrite"
    testid="lex-explain-weights"
  >
    <p>
      Applying an edit clones the active weight set, replaces one tensor, and hashes the
      result: sha256 over every tensor's name, its shape and its float32 bytes, in a fixed
      order, truncated to 32 hex characters. That hash <i>is</i> the identity of the weight
      set — the same construction the Geometry Lab uses, and the same one a saved model
      file declares. Two routes to the same numbers are the same model, and this panel
      says so: an edit whose hash equals the {baseLabel} model's is reported as the
      {baseLabel} model, not as an edit.
    </p>
    <p>
      <code>randomize</code> re-draws the selected tensor from the initializer it was born
      with — <code>N(0, 0.02²)</code> for every matrix except the packed QKV projection,
      which keeps PyTorch's xavier-uniform default. It is not noise added on top; it is
      that tensor, re-initialized, with the rest of the model left alone.
    </p>
    <p>
      The interesting edits are the ones that separate the two halves of the geometry.
      Zeroing an attention output projection leaves the embedding's spectrum untouched
      while the text collapses; halving the embedding rescales every singular value by the
      same factor and so leaves <b>effective rank unchanged</b> — a useful reminder that
      effective rank is an entropy of the normalized spectrum, not a size.
    </p>
  </Explain>
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    min-width: 0;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.95rem;
  }
  .panel-note {
    margin: 0.1rem 0 0;
    font-size: 0.74rem;
    line-height: 1.55;
    color: var(--text-dim);
  }
  .panel-note b {
    color: var(--text);
  }
  .panel-note code,
  .matrix-meta .mono {
    font-family: var(--mono);
  }
  .panel-note code {
    color: var(--accent);
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
  .badge.base {
    color: var(--good);
    background: rgba(91, 224, 176, 0.1);
  }
  .badge.tok {
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
    min-width: 0;
  }
  .field select,
  .field input {
    padding: 0.35rem 0.5rem;
    font-size: 0.8rem;
    width: auto;
    max-width: 100%;
  }
  .field input {
    width: 5.5rem;
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
    font-size: 0.74rem;
    color: var(--text);
  }
  .hint {
    margin: 0;
    font-size: 0.72rem;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .hint.small {
    font-size: 0.68rem;
  }
  /* Wide tensors scroll inside their own box — the page never scrolls sideways. */
  .plot {
    max-width: 100%;
    overflow-x: auto;
    padding-bottom: 0.3rem;
  }
  .active-edit {
    margin: 0;
    font-size: 0.72rem;
    font-family: var(--mono);
    color: var(--text-dim);
    overflow-wrap: anywhere;
  }
  .err {
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
    line-height: 1.45;
  }
</style>
