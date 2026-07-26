<script lang="ts">
  import { onDestroy } from "svelte";

  import {
    client,
    ApiError,
    debounced,
    type CorpusStatsResult,
    type GeoTrainScratchResult,
  } from "../../lib/dataClient";
  import { corpusStats } from "../../lib/geoEngine/scratch";
  import { geoModelNote, geoWeightsToken } from "../../lib/explorerStores";
  import Progress from "../../lib/Progress.svelte";

  // Train a BRAND NEW model on your own corpus (feature 004, FR-420), and save/load
  // models as files (FR-422).
  //
  // This is deliberately separate from the fine-tune panel: fine-tuning keeps the
  // shipped ~1000-word vocabulary, so text about anything Alice in Wonderland does not
  // cover mostly becomes <unk>. Training from scratch rebuilds the vocabulary from YOUR
  // text, which is the only way the model can actually learn a different domain — and
  // the reason a saved model has to carry its vocabulary with it.
  type Source = "paste" | "file" | "hf";

  let source = $state<Source>("paste");
  let text = $state("");
  let file = $state<File | null>(null);
  let hfDataset = $state("");
  let epochs = $state(12);
  let busy = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let result = $state<GeoTrainScratchResult | null>(null);
  let stats = $state<CorpusStatsResult | null>(null);
  let unsubscribe: (() => void) | null = null;

  // Save / load
  let ioError = $state("");
  let ioNote = $state("");
  let fileInput: HTMLInputElement | undefined = $state();

  onDestroy(() => {
    unsubscribe?.();
    statsDeb.cancel();
  });

  // Live corpus stats, so the vocabulary requirement is visible BEFORE a long run
  // that would be refused.
  //
  // Computed LOCALLY, not via GET /api/geo/corpus_stats: a real corpus is hundreds of
  // kilobytes and would not survive a query string (measured — the request simply
  // fails and the counter silently vanished). The rule is pure text processing and
  // geoScratch.test.ts proves the TS implementation reproduces the Python tokenizer's
  // vocabulary exactly, so there is nothing to gain from the round-trip.
  const statsDeb = debounced((t: string) => {
    stats = t.trim() ? corpusStats(t) : null;
  }, 250);

  $effect(() => {
    if (source === "paste") statsDeb(text);
    else stats = null;
  });

  const enoughText = $derived(
    stats === null || stats.n_distinct >= stats.vocab_words_required,
  );

  function onFileChange(e: Event): void {
    const input = e.currentTarget as HTMLInputElement;
    file = input.files?.[0] ?? null;
  }

  async function submitOnce(): Promise<GeoTrainScratchResult> {
    if (source === "file") {
      if (!file) throw new ApiError("InvalidParamError", "Choose a .txt or .md file first.");
      if (!/\.(txt|md)$/i.test(file.name)) {
        throw new ApiError("InvalidParamError", `Only .txt/.md files are accepted, got '${file.name}'.`);
      }
      return client.geoTrainScratch({ text: await file.text(), epochs });
    }
    if (source === "hf") {
      if (!hfDataset.trim()) {
        throw new ApiError(
          "InvalidParamError",
          "Enter a Hugging Face dataset id (like `roneneldan/TinyStories`).",
        );
      }
      return client.geoTrainScratch({ hf_dataset: hfDataset.trim(), epochs });
    }
    if (!text.trim()) throw new ApiError("InvalidParamError", "Paste some text to train on first.");
    return client.geoTrainScratch({ text, epochs });
  }

  async function run(): Promise<void> {
    unsubscribe?.();
    unsubscribe = null;
    busy = true;
    error = "";
    result = null;
    progress = 0;
    progressMsg = "submitting…";
    try {
      const first = await submitOnce();
      if (first.ready) {
        finish(first);
        return;
      }
      if (!first.job_id) {
        throw new ApiError("ComputeError", "Training was accepted but no job id came back.");
      }
      unsubscribe = client.subscribeProgress(first.job_id, {
        onProgress: (p, m) => {
          progress = p;
          progressMsg = m;
        },
        onDone: (data) => {
          const d = data as unknown as GeoTrainScratchResult | undefined;
          if (d && typeof d.weights_token === "string") finish({ ...d, ready: true });
          else fail(new ApiError("ComputeError", "Training finished but reported no model."));
        },
        onError: (type, message) => fail(new ApiError(type, message)),
      });
    } catch (e) {
      fail(e);
    }
  }

  function finish(r: GeoTrainScratchResult): void {
    busy = false;
    result = r;
    // The new model becomes active, so the sphere re-renders under its own geometry
    // AND its own vocabulary.
    if (r.weights_token) {
      geoModelNote.set(
        source === "hf"
          ? `trained from scratch on ${hfDataset.trim()}`
          : "trained from scratch on your text",
      );
      geoWeightsToken.set(r.weights_token);
    }
  }

  function fail(e: unknown): void {
    busy = false;
    if (e instanceof ApiError) {
      error =
        e.type === "NetworkError"
          ? "Could not reach the server — is the backend running?"
          : e.message;
    } else {
      error = String(e);
    }
  }

  // --- save / load ------------------------------------------------------------------

  async function saveModel(): Promise<void> {
    ioError = "";
    ioNote = "";
    try {
      const bundle = await client.geoExportModel($geoWeightsToken ?? "learned");
      const blob = new Blob([JSON.stringify(bundle)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `geotransformer-${bundle.weights_token.slice(0, 8)}.llmgeo.json`;
      a.click();
      URL.revokeObjectURL(url);
      ioNote = `saved ${a.download}`;
    } catch (e) {
      ioError = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function loadModel(e: Event): Promise<void> {
    ioError = "";
    ioNote = "";
    const input = e.currentTarget as HTMLInputElement;
    const f = input.files?.[0];
    input.value = ""; // allow re-picking the same file
    if (!f) return;
    try {
      const parsed = JSON.parse(await f.text());
      const loaded = await client.geoImportModel(parsed);
      geoModelNote.set(`loaded from ${f.name}`);
      geoWeightsToken.set(loaded.weights_token);
      ioNote = `loaded ${f.name} · ${loaded.vocab_size}-token vocabulary`;
    } catch (err) {
      if (err instanceof SyntaxError) ioError = `${f.name} is not valid JSON.`;
      else ioError = err instanceof ApiError ? err.message : String(err);
    }
  }
</script>

<div class="panel-body" data-testid="geo-train">
  <div class="head">
    <h3>Train a new model</h3>
    <span class="hint">a fresh vocabulary + fresh weights — not a fine-tune of the shipped one</span>
  </div>

  <div class="tabs" role="tablist">
    {#each [["paste", "paste text"], ["file", "upload .txt/.md"], ["hf", "HF dataset"]] as [id, lbl] (id)}
      <button
        role="tab"
        aria-selected={source === id}
        class:active={source === id}
        data-testid={`geo-train-src-${id}`}
        onclick={() => (source = id as Source)}
      >{lbl}</button>
    {/each}
  </div>

  {#if source === "paste"}
    <textarea
      data-testid="geo-train-text"
      bind:value={text}
      placeholder="Paste a book, a few chapters, a long article — the model learns its vocabulary from exactly this text…"
    ></textarea>
    {#if stats}
      <p class="stats" class:short={!enoughText} data-testid="geo-train-stats">
        {stats.n_tokens.toLocaleString()} tokens · {stats.n_distinct.toLocaleString()} distinct
        words
        {#if !enoughText}
          — needs {stats.vocab_words_required.toLocaleString()} distinct to fill the
          vocabulary, so this is {(stats.vocab_words_required - stats.n_distinct).toLocaleString()}
          short
        {:else}
          — enough to fill the {stats.vocab_words_required.toLocaleString()}-word vocabulary
        {/if}
      </p>
    {/if}
  {:else if source === "file"}
    <input type="file" accept=".txt,.md,text/plain,text/markdown" onchange={onFileChange} data-testid="geo-train-file" />
  {:else}
    <input
      type="text"
      bind:value={hfDataset}
      placeholder="roneneldan/TinyStories"
      data-testid="geo-train-hf"
    />
    <p class="hint small">
      Read live from the Hugging Face dataset viewer — real rows, no download.
    </p>
  {/if}

  <label class="epochs">
    <span>epochs <b>{epochs}</b></span>
    <input type="range" min="1" max="30" step="1" bind:value={epochs} data-testid="geo-train-epochs" />
  </label>

  <button
    class="go"
    data-testid="geo-train-run"
    disabled={busy || (source === "paste" && !enoughText)}
    onclick={() => void run()}
  >{busy ? "training…" : "Train from scratch"}</button>

  {#if busy}
    <Progress {progress} message={progressMsg} />
  {/if}

  {#if error}
    <div class="err" data-testid="geo-train-error">{error}</div>
  {/if}

  {#if result?.weights_token}
    <div class="ok" data-testid="geo-train-result">
      trained a new model · final loss {result.final_loss?.toFixed(2)} ·
      {result.n_tokens?.toLocaleString()} tokens · {result.epochs} epochs — it is now the
      active model, with its own vocabulary
    </div>
  {/if}

  <div class="divider"></div>
  <div class="io">
    <span class="hint">Models live in this browser session — save one to keep it.</span>
    <div class="io-row">
      <button data-testid="geo-save-model" onclick={() => void saveModel()}>↓ Save model</button>
      <button data-testid="geo-load-model" onclick={() => fileInput?.click()}>↑ Load model</button>
      <input
        bind:this={fileInput}
        type="file"
        accept=".json,application/json"
        class="hidden-file"
        data-testid="geo-load-model-input"
        onchange={(e) => void loadModel(e)}
      />
    </div>
    {#if ioError}<div class="err" data-testid="geo-io-error">{ioError}</div>{/if}
    {#if ioNote}<div class="ok" data-testid="geo-io-note">{ioNote}</div>{/if}
  </div>
</div>

<style>
  .panel-body { display: flex; flex-direction: column; gap: 0.55rem; }
  .head { display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap; }
  h3 { margin: 0; font-size: 0.95rem; }
  .hint { font-size: 0.72rem; color: var(--text-dim); line-height: 1.4; }
  .hint.small { margin: 0; }
  .tabs { display: inline-flex; gap: 0.25rem; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 999px; padding: 0.2rem; align-self: flex-start; }
  .tabs button { background: transparent; color: var(--text-dim); border: none; border-radius: 999px; padding: 0.22rem 0.7rem; font-size: 0.74rem; }
  .tabs button.active { background: var(--accent-grad); color: #0b0e14; font-weight: 600; }
  textarea { min-height: 84px; resize: vertical; font-size: 0.78rem; }
  .stats { margin: 0; font-size: 0.72rem; font-family: var(--mono); color: var(--good, #5be0b0); }
  .stats.short { color: #ffb454; }
  .epochs { display: flex; flex-direction: column; gap: 0.22rem; font-size: 0.74rem; color: var(--text-dim); }
  .epochs b { color: var(--text); font-variant-numeric: tabular-nums; }
  .go { align-self: flex-start; }
  .go:disabled { opacity: 0.45; cursor: default; }
  .err {
    background: rgba(255, 122, 144, 0.1); color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3); border-radius: 9px;
    padding: 0.45rem 0.6rem; font-size: 0.76rem; line-height: 1.45;
  }
  .ok {
    background: rgba(91, 224, 176, 0.08); color: #5be0b0;
    border: 1px solid rgba(91, 224, 176, 0.28); border-radius: 9px;
    padding: 0.45rem 0.6rem; font-size: 0.76rem; line-height: 1.45;
  }
  .divider { height: 1px; background: var(--border); margin: 0.15rem 0; }
  .io { display: flex; flex-direction: column; gap: 0.4rem; }
  .io-row { display: flex; gap: 0.4rem; flex-wrap: wrap; }
  .io-row button {
    background: var(--bg-elev); color: var(--text-dim); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.28rem 0.7rem; font-size: 0.75rem; font-family: var(--mono);
  }
  .io-row button:hover { color: var(--text); border-color: var(--accent); }
  .hidden-file { display: none; }
</style>
