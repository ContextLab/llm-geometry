<script lang="ts">
  import { onDestroy } from "svelte";

  import { client, ApiError, type GeoFinetuneResult } from "../../lib/dataClient";
  import { geoModelNote, geoWeightsToken } from "../../lib/explorerStores";
  import Progress from "../../lib/Progress.svelte";

  // Fine-tune the tiny model on the student's own data (FR-106): pasted text, an
  // uploaded .txt/.md, or an HF dataset id. Real SGD runs as a job (SSE phase
  // "finetune"); the finished checkpoint is a NEW content-hash weights_token (the
  // canonical `learned` checkpoint is never mutated) which becomes the active token,
  // so the sphere immediately re-renders under the fine-tuned weights.
  type Source = "paste" | "file" | "hf";

  let source = $state<Source>("paste");
  let text = $state("");
  let file = $state<File | null>(null);
  let hfDataset = $state("");
  let steps = $state(100);
  let busy = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let result = $state<{ before: number; after: number } | null>(null);
  /** How much of the fine-tuning stream the ACTIVE model's vocabulary knew. A loss drop
   * is only "on your text" to the extent your text was in the vocabulary, so this is
   * shown beside the loss rather than left for the user to assume. */
  let unk = $state<{ nTokens: number; nUnk: number; rate: number } | null>(null);
  /** What the finished run was actually trained on — "your text" was printed even for
   * a Hugging Face dataset, which is not the user's text at all. */
  let sourceLabel = $state("your text");
  let unsubscribe: (() => void) | null = null;

  onDestroy(() => unsubscribe?.());

  function onFileChange(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    file = input.files?.[0] ?? null;
  }

  function submitOnce(): Promise<GeoFinetuneResult> {
    const base = $geoWeightsToken ?? "learned";
    if (source === "file") {
      if (!file) throw new ApiError("InvalidParamError", "Choose a .txt or .md file first.");
      return client.geoFinetuneFile(file, file.name, { steps, base });
    }
    if (source === "hf") {
      if (!hfDataset.trim()) throw new ApiError("InvalidParamError", "Enter a Hugging Face dataset id (like `roneneldan/TinyStories`).");
      return client.geoFinetune({ hf_dataset: hfDataset.trim(), steps, base });
    }
    if (!text.trim()) throw new ApiError("InvalidParamError", "Paste some text to fine-tune on first.");
    return client.geoFinetune({ text, steps, base });
  }

  async function run() {
    unsubscribe?.();
    unsubscribe = null;
    busy = true;
    error = "";
    result = null;
    unk = null;
    progress = 0;
    progressMsg = "submitting…";
    try {
      const first = await submitOnce();
      if (first.ready) {
        finish(first);
        return;
      }
      if (!first.job_id) throw new ApiError("ComputeError", "The server accepted the job but returned no job id.");
      unsubscribe = client.subscribeProgress(first.job_id, {
        onProgress: (p, m) => {
          progress = p;
          progressMsg = m;
        },
        onDone: (data) => {
          // The done event carries the minted token + losses directly — never
          // re-submit (a mid-run weight edit would change `base`, miss the cache,
          // and report a false failure for a SUCCESSFUL fine-tune).
          const d = data as unknown as GeoFinetuneResult | undefined;
          if (d && typeof d.weights_token === "string") {
            finish({ ...d, ready: true });
          } else {
            fail(
              new ApiError(
                "ComputeError",
                "Fine-tune finished but the completion event carried no result.",
              ),
            );
          }
        },
        onError: (_type, message) => fail(new ApiError(_type, message)),
      });
    } catch (e) {
      fail(e);
    }
  }

  function finish(r: GeoFinetuneResult) {
    busy = false;
    if (r.weights_token) {
      sourceLabel = source === "hf" ? hfDataset.trim() : source === "file" ? (file?.name ?? "your file") : "your text";
      geoModelNote.set(`fine-tuned on ${sourceLabel}`);
      geoWeightsToken.set(r.weights_token); // the geometry re-fetches under the new weights
    }
    if (r.loss_before != null && r.loss_after != null) {
      result = { before: r.loss_before, after: r.loss_after };
    }
    unk =
      r.n_tokens != null && r.n_unk != null && r.unk_rate != null
        ? { nTokens: r.n_tokens, nUnk: r.n_unk, rate: r.unk_rate }
        : null;
  }

  function fail(e: unknown) {
    busy = false;
    if (e instanceof ApiError) {
      if (e.type === "NetworkError") error = "Could not reach the server — is the backend running?";
      else if (e.type === "InvalidParamError" && /dataset/i.test(e.message)) error = `That dataset didn't work: ${e.message}`;
      else error = e.message;
    } else {
      error = String(e);
    }
  }
</script>

<div class="panel-body" data-testid="geo-finetune">
  <div class="head">
    <h3>Fine-tune on your text</h3>
    <span class="hint">a new checkpoint is minted — the learned one is never touched</span>
  </div>
  <p class="panel-note">
    Real gradient steps (SGD, lr 1e-2) on the <b>currently active</b> weights — so this adapts the
    model you are looking at rather than starting one. Watch the embedding drift on the sphere as
    the loss falls. Your text is tokenized with the <b>active model's own</b> vocabulary, and the
    new checkpoint keeps that vocabulary, so fine-tuning a model you trained from scratch really
    does use its words. Words the model has never seen become <code>&lt;unk&gt;</code>; the share
    of them is reported with the loss, and text that is almost entirely unknown is refused rather
    than turned into a loss drop that says nothing about it. To build a vocabulary from your own
    text, use <b>Train a new model</b>.
  </p>

  <div class="tabs" role="tablist">
    <!-- All three sources work in BOTH builds (feature 004): the Hub's dataset viewer
         is a public CORS-enabled service, so the static build reads real rows instead
         of refusing. This tab used to be disabled here with a "needs the Python
         backend" note that is no longer true. -->
    {#each [["paste", "paste text"], ["file", "upload .txt/.md"], ["hf", "HF dataset"]] as [id, lbl] (id)}
      <button
        class:active={source === id}
        role="tab"
        aria-selected={source === id}
        data-testid={id === "hf" ? "geo-finetune-hf-tab" : undefined}
        onclick={() => (source = id as Source)}
      >{lbl}</button>
    {/each}
  </div>

  {#if source === "paste"}
    <textarea rows="4" placeholder="Paste a paragraph or two — the model will take real gradient steps on it…" bind:value={text}></textarea>
  {:else if source === "file"}
    <input class="file" type="file" accept=".txt,.md" onchange={onFileChange} />
    {#if file}<span class="hint mono">{file.name} · {(file.size / 1024).toFixed(1)} kB</span>{/if}
  {:else}
    <input type="text" placeholder="dataset id, e.g. roneneldan/TinyStories" bind:value={hfDataset} />
  {/if}

  <label class="steps">
    <span>steps <b>{steps}</b></span>
    <input type="range" min="10" max="500" step="10" bind:value={steps} />
  </label>

  <div class="actions">
    <button onclick={run} disabled={busy}>{busy ? "fine-tuning…" : "Fine-tune"}</button>
    {#if result}
      <span class="loss" data-testid="geo-finetune-loss">
        loss {result.before.toFixed(2)} <span class="arrow">→</span> {result.after.toFixed(2)} on {sourceLabel}
      </span>
    {/if}
  </div>

  {#if unk}
    <!-- The loss above is only "on your text" to the extent your text was in the
         model's vocabulary. Stated always, and flagged once it is large enough to
         change how the number should be read. -->
    <div class="unk" class:high={unk.rate > 0.25} data-testid="geo-finetune-unk">
      {#if unk.rate > 0.25}
        <b>{unk.nUnk.toLocaleString()} of {unk.nTokens.toLocaleString()} tokens
          ({Math.round(unk.rate * 100)}%)</b>
        were outside this model's vocabulary and trained as <code>&lt;unk&gt;</code> — the loss is
        partly about the unknown-word token, not only your words.
      {:else}
        {unk.nUnk.toLocaleString()} of {unk.nTokens.toLocaleString()} tokens
        ({Math.round(unk.rate * 100)}%) were outside this model's vocabulary
        (<code>&lt;unk&gt;</code>).
      {/if}
    </div>
  {/if}

  {#if busy && progressMsg}
    <Progress {progress} message={progressMsg} />
  {/if}
  {#if error}
    <div class="error" data-testid="geo-error">{error}</div>
  {/if}
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .panel-note {
    margin: 0.1rem 0 0;
    font-size: 0.74rem;
    line-height: 1.55;
    color: var(--text-dim);
  }
  .panel-note b {
    color: var(--text);
    font-weight: 600;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.92rem;
  }
  .hint {
    font-size: 0.72rem;
    color: var(--text-dim);
  }
  .mono {
    font-family: var(--mono);
  }
  .tabs {
    display: inline-flex;
    gap: 0.25rem;
    padding: 0.2rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    align-self: flex-start;
  }
  .tabs button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font-size: 0.74rem;
    font-weight: 500;
  }
  .tabs button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
  }
  .tabs button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    text-decoration: line-through;
    text-decoration-thickness: 1px;
  }
  textarea {
    font-size: 0.84rem;
  }
  .file {
    color: var(--text-dim);
    font-size: 0.8rem;
  }
  .steps {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.76rem;
    color: var(--text-dim);
  }
  .steps b {
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex-wrap: wrap;
  }
  .actions button {
    padding: 0.45rem 1rem;
    font-size: 0.84rem;
  }
  .loss {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--good);
  }
  .loss .arrow {
    color: var(--accent);
  }
  .unk {
    font-size: 0.72rem;
    line-height: 1.5;
    color: var(--text-dim);
  }
  .unk.high {
    color: var(--warn, var(--bad));
    background: rgba(255, 190, 92, 0.1);
    border: 1px solid rgba(255, 190, 92, 0.3);
    border-radius: 10px;
    padding: 0.4rem 0.55rem;
  }
  .unk code {
    font-family: var(--mono);
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
