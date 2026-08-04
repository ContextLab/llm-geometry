<script lang="ts">
  /**
   * US-2 (train from scratch, live), US-3 (compare budgets by retraining at matched |V|)
   * and US-5 (train on your own text).
   *
   * BOTH modes go to `lexEngine/trainWorker.ts` (FR-617), so the loss curve, the samples
   * and the geometry keep animating while the model really learns, and Stop means stop.
   * The worker owns the whole pipeline — resolve the budget, encode the corpus, train —
   * and hands back the weights, the config AND the vocabulary words, so a model can never
   * be paired with a vocabulary it was not trained on (feature 004's issue #6).
   *
   * FINE-TUNING (FR-619) is the same request with two fields added: `initialWeights`, the
   * active model's own weights, and `vocabWords`, its own budget. The explicit word list
   * WINS over the budget controls, which is precisely issue #6's fix — continuing training
   * on new text must not silently rebuild the vocabulary underneath the weights, because
   * every embedding row would then mean a different word.
   *
   * The recipe is `specs/006-lexicon-lab-tiny/architecture.md`, shared with the PyTorch
   * backend: AdamW with weight decay on 2-D weight matrices only, a one-cycle schedule
   * with `lr` as the PEAK, gradient clipping at global norm 1.0.
   */
  import { onDestroy } from "svelte";

  import {
    DEFAULT_BATCH,
    DEFAULT_LR,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WEIGHT_DECAY,
    GRAD_CLIP_NORM,
    LexModel,
    LexVocab,
    MAX_STEPS,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    VAL_FRACTION,
    type BudgetSource,
    type LexConfig,
    type LexTrainRequest,
    type LexTrainResponse,
    type TrainPoint,
  } from "../../lib/lexEngine";
  import { fetchDatasetText } from "../../lib/staticClient/hfDatasets";
  import Explain from "../../lib/Explain.svelte";
  import Progress from "../../lib/Progress.svelte";

  interface Props {
    cfg: LexConfig;
    budgetSource: BudgetSource;
    budgetName: string;
    corpusText: string;
    corpusLabel: string;
    /**
     * An EXPLICIT word list for a from-scratch run, or null to let the worker resolve the
     * budget from `corpusText` as it always has.
     *
     * Feature 007 needs this: under the vacancy transform's mapped condition the budget is
     * not a function of the text at all — it is the ORIGINAL budget pushed through the same
     * transform, in the same order, so every word keeps the embedding row its pre-image had.
     * Letting the worker rebuild a Dolch list from a vacated corpus would hand the model a
     * vocabulary of English words the text no longer contains, and the invariance theorem
     * this tab demonstrates would be false for a reason nothing on screen explained.
     */
    vocabWords: readonly string[] | null;
    trainedModel: LexModel | null;
    trainedVocab: LexVocab | null;
    trainedNote: string;
    onTrained: (model: LexModel, vocab: LexVocab, note: string) => void;
    onAdoptCorpus: (text: string, label: string) => void;
  }
  let {
    cfg,
    budgetSource,
    budgetName,
    corpusText,
    corpusLabel,
    vocabWords,
    trainedModel,
    trainedVocab,
    trainedNote,
    onTrained,
    onAdoptCorpus,
  }: Props = $props();

  type Mode = "scratch" | "finetune";
  type Source = "shipped" | "paste" | "file" | "hf";

  let mode = $state<Mode>("scratch");
  let source = $state<Source>("shipped");
  let pasted = $state("");
  let file = $state<File | null>(null);
  let hfDataset = $state("");

  let steps = $state<number>(DEFAULT_STEPS);
  let lr = $state<number>(DEFAULT_LR);
  let batchSize = $state<number>(DEFAULT_BATCH);
  let sampleEvery = $state<number>(DEFAULT_SAMPLE_EVERY);
  let seed = $state<number>(DEFAULT_SEED);

  let busy = $state(false);
  let error = $state("");
  let loading = $state("");
  let fraction = $state(0);
  let point = $state<TrainPoint | null>(null);
  /** The engine's step count, which it validates and truncates — not the slider's. */
  let totalSteps = $state(0);
  let elapsedMs = $state(0);
  let history = $state<TrainPoint[]>([]);
  let samples = $state<{ step: number; text: string }[]>([]);
  let done = $state<{
    initialTrainLoss: number;
    finalTrainLoss: number;
    valLoss: number;
    nTokens: number;
    elapsedMs: number;
  } | null>(null);

  let worker: Worker | null = null;
  let startedAt = 0;

  onDestroy(() => {
    worker?.terminate();
    worker = null;
  });

  /** See BudgetPanel: arrows must move an ARIA radiogroup's selection and focus. */
  function segKey(e: KeyboardEvent, apply: (value: string) => void): void {
    const dir =
      e.key === "ArrowRight" || e.key === "ArrowDown"
        ? 1
        : e.key === "ArrowLeft" || e.key === "ArrowUp"
          ? -1
          : 0;
    if (dir === 0) return;
    e.preventDefault();
    const group = e.currentTarget as HTMLElement;
    const radios = Array.from(
      group.querySelectorAll<HTMLButtonElement>('[role="radio"]:not([disabled])'),
    );
    if (radios.length === 0) return;
    const cur = radios.findIndex((r) => r.getAttribute("aria-checked") === "true");
    const next = radios[((cur < 0 ? 0 : cur) + dir + radios.length) % radios.length];
    next.focus();
    apply(next.dataset.value ?? "");
  }

  /**
   * A uniform model over the embedding rows scores exactly `ln(rows)` nats. That is the
   * line below which the model has learned something rather than nothing, and it is
   * derived from the vocabulary on screen rather than quoted.
   */
  const uniformLoss = $derived(cfg.vocabRows > 1 ? Math.log(cfg.vocabRows) : null);
  const canFinetune = $derived(trainedModel !== null && trainedVocab !== null);
  const ready = $derived(cfg.vocabRows > 4 && corpusText.length > 0);

  const MODES = [
    {
      id: "scratch",
      label: "from scratch",
      title: "Fresh random weights at the current shape, trained in a worker on the active corpus.",
    },
    {
      id: "finetune",
      label: "fine-tune",
      title: "Continue training the model you already have, in a worker, keeping its vocabulary.",
    },
  ];
  const SOURCES = [
    { id: "shipped", label: "shipped corpus" },
    { id: "paste", label: "paste text" },
    { id: "file", label: "upload .txt/.md" },
    { id: "hf", label: "HF dataset" },
  ];

  function onFileChange(e: Event): void {
    const input = e.currentTarget as HTMLInputElement;
    file = input.files?.[0] ?? null;
  }

  /**
   * Resolve whichever source is selected into text and hand it to the tab, so the BUDGET
   * is rebuilt from it (US-5). The coverage counters move before a single step runs —
   * which is the whole point of measuring a budget against the corpus it will be used on.
   */
  async function adoptSelectedCorpus(): Promise<void> {
    error = "";
    loading = "";
    try {
      if (source === "paste") {
        if (!pasted.trim()) throw new Error("Paste some text first.");
        onAdoptCorpus(pasted, "your pasted text");
      } else if (source === "file") {
        if (!file) throw new Error("Choose a .txt or .md file first.");
        if (!/\.(txt|md)$/i.test(file.name)) {
          throw new Error(`Only .txt/.md files are accepted, got '${file.name}'.`);
        }
        onAdoptCorpus(await file.text(), file.name);
      } else if (source === "hf") {
        if (!hfDataset.trim()) {
          throw new Error("Enter a HuggingFace dataset id, like roneneldan/TinyStories.");
        }
        loading = `reading ${hfDataset.trim()} from the HuggingFace dataset viewer…`;
        const res = await fetchDatasetText(hfDataset.trim(), {
          onProgress: (_f, m) => (loading = m),
        });
        onAdoptCorpus(res.text, `${res.dataset} · ${res.config}/${res.split} · ${res.rows} rows`);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = "";
    }
  }

  function resetRun(): void {
    error = "";
    done = null;
    history = [];
    samples = [];
    point = null;
    fraction = 0;
    totalSteps = steps;
    elapsedMs = 0;
    startedAt = Date.now();
    busy = true;
  }

  // ---- the worker, for both modes -----------------------------------------------------

  /**
   * Drive one training run in a worker. `noteFor` turns the finished payload into the
   * provenance line the tab displays — the only thing the two modes disagree about.
   */
  function runInWorker(req: LexTrainRequest, noteFor: (msg: { budgetSource: string; budgetName: string; vocabWords: string[] }) => string): void {
    resetRun();
    worker?.terminate();
    const w = new Worker(new URL("../../lib/lexEngine/trainWorker.ts", import.meta.url), {
      type: "module",
    });
    worker = w;
    w.onmessage = (event: MessageEvent<LexTrainResponse>) => {
      const msg = event.data;
      elapsedMs = Date.now() - startedAt;
      if (msg.type === "progress") {
        fraction = msg.fraction;
        point = msg.point;
        totalSteps = msg.totalSteps;
        history = [...history, msg.point];
      } else if (msg.type === "sample") {
        samples = [{ step: msg.step, text: msg.text }, ...samples].slice(0, 6);
      } else if (msg.type === "error") {
        busy = false;
        error = msg.message;
        w.terminate();
        if (worker === w) worker = null;
      } else {
        busy = false;
        fraction = 1;
        history = msg.history;
        done = {
          initialTrainLoss: msg.initialTrainLoss,
          finalTrainLoss: msg.finalTrainLoss,
          valLoss: msg.valLoss,
          nTokens: msg.nTokens,
          elapsedMs,
        };
        // The final sample lands on the last step, which the periodic sampler has ALREADY
        // emitted whenever `steps % sampleEvery === 0` — true of the defaults (400 / 50).
        // Keying the list by step then produced two entries with the same key and Svelte
        // threw `each_key_duplicate` on the live site. The final sample supersedes the
        // periodic one for that step rather than joining it.
        samples = [
          { step: steps, text: msg.sample.text },
          ...samples.filter((s) => s.step !== steps),
        ].slice(0, 6);
        // The vocabulary comes back FROM the worker — the one it actually encoded and
        // trained against — so the model and its word list are written together.
        onTrained(
          new LexModel(msg.config, msg.weights),
          new LexVocab(msg.vocabWords, msg.budgetSource as BudgetSource, msg.budgetName),
          noteFor(msg),
        );
        w.terminate();
        if (worker === w) worker = null;
      }
    };
    w.onerror = (e) => {
      busy = false;
      error = e.message || "the training worker failed to start";
      w.terminate();
      if (worker === w) worker = null;
    };
    w.postMessage(req);
  }

  const modelDims = $derived({
    dModel: cfg.dModel,
    nLayers: cfg.nLayers,
    nHeads: cfg.nHeads,
    ctx: cfg.ctx,
    tied: cfg.tied,
    dropout: cfg.dropout,
  });

  function runScratch(): void {
    runInWorker(
      {
        text: corpusText,
        budgetSource,
        budget: budgetName,
        // Only when the tab hands one down: otherwise the worker resolves the budget from
        // the text, which is what every pre-007 run did and must keep doing.
        vocabWords: vocabWords ? [...vocabWords] : undefined,
        model: modelDims,
        steps,
        lr,
        batchSize,
        seed,
        sampleEvery,
      },
      (msg) => `trained from scratch · ${msg.budgetSource} ${msg.budgetName} · ${steps} steps on ${corpusLabel}`,
    );
  }

  /**
   * FR-619. The request carries the ACTIVE model's weights and the ACTIVE model's word
   * list; `trainWorker` prefers an explicit `vocabWords` over the budget controls, so a
   * fine-tune keeps its own vocabulary even when the new text would have produced a
   * different one. That is feature 004's issue #6, and it is fixed here by construction
   * rather than by remembering to pass the right thing: the model's dimensions come off
   * `model.cfg`, not off the sliders.
   */
  function runFinetune(): void {
    const model = trainedModel;
    const vocab = trainedVocab;
    if (!model || !vocab) return;
    const { vocabRows: _rows, ...dims } = model.cfg;
    runInWorker(
      {
        text: corpusText,
        budgetSource: vocab.source,
        budget: vocab.budgetName,
        vocabWords: [...vocab.words],
        initialWeights: model.weights,
        model: dims,
        steps,
        lr,
        batchSize,
        seed,
        sampleEvery,
      },
      (msg) =>
        `fine-tuned ${steps} more steps on ${corpusLabel}, keeping its ` +
        `${msg.vocabWords.length}-word ${msg.budgetSource} vocabulary`,
    );
  }

  function run(): void {
    if (mode === "finetune") runFinetune();
    else runScratch();
  }

  function stop(): void {
    worker?.terminate();
    worker = null;
    busy = false;
  }

  // ---- the loss curve ----------------------------------------------------------------
  // A plain SVG polyline: the y-axis is nats, the dashed rule is ln(rows), and the curve
  // is the actual per-step training loss, not a smoothing of it.
  const W = 320;
  const H = 96;
  const curve = $derived.by(() => {
    if (history.length < 2) return null;
    const losses = history.map((h) => h.loss);
    const hi = Math.max(...losses, uniformLoss ?? 0);
    const lo = Math.min(...losses, 0);
    const span = hi - lo || 1;
    const maxStep = Math.max(totalSteps, history[history.length - 1].step, 1);
    const pts = history
      .map(
        (h) =>
          `${((h.step / maxStep) * W).toFixed(1)},${(H - ((h.loss - lo) / span) * H).toFixed(1)}`,
      )
      .join(" ");
    const uniformY =
      uniformLoss != null && uniformLoss <= hi ? H - ((uniformLoss - lo) / span) * H : null;
    return { pts, uniformY };
  });

  const fmtSecs = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
</script>

<div class="panel-body" data-testid="lex-train">
  <div class="head">
    <h3>Train</h3>
    <span class="hint">in this browser, on a model really learning</span>
  </div>

  <div class="ctl">
    <span class="ctl-label" id="lex-train-mode-label">what to train</span>
    <div
      class="seg"
      role="radiogroup"
      tabindex="-1"
      aria-labelledby="lex-train-mode-label"
      data-testid="lex-train-mode"
      onkeydown={(e) => segKey(e, (v) => (mode = v as Mode))}
    >
      {#each MODES as m (m.id)}
        <button
          role="radio"
          aria-checked={mode === m.id}
          tabindex={mode === m.id ? 0 : -1}
          data-value={m.id}
          class:active={mode === m.id}
          disabled={m.id === "finetune" && !canFinetune}
          title={m.id === "finetune" && !canFinetune
            ? "Train a model from scratch first — there is nothing to fine-tune yet."
            : m.title}
          onclick={() => (mode = m.id as Mode)}
        >{m.label}</button>
      {/each}
    </div>
    {#if mode === "finetune" && canFinetune && trainedVocab}
      <p class="active-corpus" data-testid="lex-finetune-vocab">
        keeps the active model's vocabulary: <b>{trainedVocab.budgetSize} words</b>
        ({trainedVocab.source} {trainedVocab.budgetName}, {trainedVocab.rows} embedding
        rows). Words the new text uses but this budget lacks arrive as
        <code>&lt;unk&gt;</code> — fine-tuning cannot add a word to a budget.
      </p>
    {/if}
  </div>

  <div class="ctl">
    <span class="ctl-label" id="lex-train-src-label">corpus</span>
    <div
      class="seg wrap"
      role="radiogroup"
      tabindex="-1"
      aria-labelledby="lex-train-src-label"
      data-testid="lex-train-source"
      onkeydown={(e) => segKey(e, (v) => (source = v as Source))}
    >
      {#each SOURCES as s (s.id)}
        <button
          role="radio"
          aria-checked={source === s.id}
          tabindex={source === s.id ? 0 : -1}
          data-value={s.id}
          class:active={source === s.id}
          onclick={() => (source = s.id as Source)}
        >{s.label}</button>
      {/each}
    </div>
    <p class="active-corpus" data-testid="lex-train-active-corpus">
      active corpus: <b>{corpusLabel || "—"}</b>
      {#if source !== "shipped"}
        <span class="why">
          — your text replaces it for the budget <i>and</i> the training data, so the
          coverage counters rebuild before you train
        </span>
      {/if}
    </p>
  </div>

  {#if source === "paste"}
    <textarea
      data-testid="lex-train-text"
      bind:value={pasted}
      placeholder="Paste a poem, a chapter, a corpus — the budget and the model are both rebuilt from exactly this text…"
    ></textarea>
  {:else if source === "file"}
    <input
      type="file"
      accept=".txt,.md,text/plain,text/markdown"
      data-testid="lex-train-file"
      onchange={onFileChange}
    />
  {:else if source === "hf"}
    <input
      type="text"
      bind:value={hfDataset}
      placeholder="roneneldan/TinyStories"
      data-testid="lex-train-hf"
    />
    <p class="hint small">Read live from the HuggingFace dataset viewer — real rows, no download.</p>
  {/if}

  {#if source !== "shipped"}
    <button
      class="secondary"
      data-testid="lex-train-adopt"
      disabled={busy}
      onclick={() => void adoptSelectedCorpus()}
    >Use this text</button>
    {#if loading}<p class="loading">{loading}</p>{/if}
  {/if}

  <div class="hypers">
    <label class="slider">
      <span class="ctl-label">steps <b>{steps}</b></span>
      <input type="range" min="20" max={MAX_STEPS} step="20" data-testid="lex-steps" bind:value={steps} />
    </label>
    <label class="slider">
      <span class="ctl-label">peak lr <b>{lr.toExponential(1)}</b></span>
      <input
        type="range"
        min="-4"
        max="-2"
        step="0.1"
        data-testid="lex-lr"
        value={Math.log10(lr)}
        oninput={(e) => (lr = Number((10 ** Number(e.currentTarget.value)).toPrecision(3)))}
      />
    </label>
    <label class="slider">
      <span class="ctl-label">batch <b>{batchSize}</b></span>
      <input type="range" min="4" max="64" step="4" data-testid="lex-batch" bind:value={batchSize} />
    </label>
    <label class="slider">
      <span class="ctl-label">sample every <b>{sampleEvery}</b></span>
      <input type="range" min="10" max="200" step="10" data-testid="lex-sample-every" bind:value={sampleEvery} />
    </label>
    <label class="slider">
      <span class="ctl-label">seed <b>{seed}</b></span>
      <input type="range" min="0" max="99" step="1" data-testid="lex-seed" bind:value={seed} />
    </label>
  </div>

  <div class="actions">
    <button class="go" data-testid="lex-train-run" disabled={busy || !ready} onclick={run}>
      {busy ? "training…" : mode === "finetune" ? "Fine-tune" : "Train from scratch"}
    </button>
    {#if busy}
      <button class="secondary" data-testid="lex-train-stop" onclick={stop}>Stop</button>
    {/if}
  </div>

  {#if busy || history.length > 0}
    <div class="live" data-testid="lex-train-live">
      <Progress
        progress={fraction}
        message={`step ${point?.step ?? 0}/${totalSteps} · loss ${point?.loss.toFixed(3) ?? "—"} · lr ${point?.lr.toExponential(2) ?? "—"} · ${fmtSecs(elapsedMs)}`}
      />
      <div class="chart-wrap">
        <svg
          class="chart"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`training loss, ${point?.loss.toFixed(3) ?? "no"} nats at step ${point?.step ?? 0} of ${totalSteps}`}
          data-testid="lex-loss-chart"
        >
          {#if curve}
            {#if curve.uniformY != null}
              <line x1="0" x2={W} y1={curve.uniformY} y2={curve.uniformY} class="uniform" stroke-dasharray="4 4" />
            {/if}
            <polyline points={curve.pts} class="loss-line" />
          {/if}
        </svg>
      </div>
      <p class="chart-key">
        training loss in nats
        {#if uniformLoss != null}
          · the dashed line is <b>ln {cfg.vocabRows} = {uniformLoss.toFixed(2)}</b>, what a
          uniform model over the embedding rows scores — below it the model has learned
          something
        {/if}
      </p>
    </div>
  {/if}

  {#if samples.length > 0}
    <div class="samples" data-testid="lex-train-samples">
      <span class="ctl-label">samples as it learns</span>
      {#each samples as s (s.step)}
        <p class="sample"><span class="at">step {s.step}</span>{s.text}</p>
      {/each}
    </div>
  {/if}

  {#if error}
    <div class="err" data-testid="lex-train-error">{error}</div>
  {/if}

  {#if done}
    <div class="ok" data-testid="lex-train-done">
      loss {done.initialTrainLoss.toFixed(3)} → {done.finalTrainLoss.toFixed(3)} ·
      held-out {done.valLoss.toFixed(3)} · {done.nTokens.toLocaleString()} tokens (the
      budget's words plus one <code>&lt;eos&gt;</code> per line) ·
      {fmtSecs(done.elapsedMs)} — this model now drives the spectrum, the cloud and the
      sampler
    </div>
  {/if}

  {#if trainedNote}
    <p class="active-model" data-testid="lex-active-model">active model: {trainedNote}</p>
  {/if}

  <Explain
    title="The training recipe, exactly"
    hint="AdamW on matrices only, one-cycle with lr as the peak, clip at 1.0"
    testid="lex-explain-training"
  >
    <p>
      Loss is the mean cross-entropy of <code>logits[:, :-1]</code> against
      <code>x[:, 1:]</code>, ignoring <code>&lt;pad&gt;</code>. Batches draw start offsets
      uniformly <b>with replacement</b> from a contiguous
      {((1 - VAL_FRACTION) * 100).toFixed(0)}/{(VAL_FRACTION * 100).toFixed(0)}
      train/validation split of the token stream; each window is <code>ctx+1</code> tokens
      so the last one has a target.
    </p>
    <p>
      <b>AdamW</b> with weight decay <code>{DEFAULT_WEIGHT_DECAY}</code> applied to
      <b>2-D weight matrices only</b> — not the embedding, not the positions, not any bias,
      not a LayerNorm gain. This is a deliberate departure from the source model, which
      decays every parameter; the standard convention is followed here and named rather
      than inherited quietly.
    </p>
    <p>
      The <b>one-cycle</b> schedule takes your <code>lr</code> as its <i>peak</i>: it starts
      at <code>lr / {ONECYCLE_DIV_FACTOR}</code>, rises over the first
      <code>{(ONECYCLE_PCT_START * 100).toFixed(0)}%</code> of the run, then anneals
      cosine-wise to
      <code>lr / {ONECYCLE_DIV_FACTOR} / {ONECYCLE_FINAL_DIV_FACTOR.toExponential(0)}</code>.
      Gradients are clipped at global L2 norm <code>{GRAD_CLIP_NORM}</code> before every
      step.
    </p>
    <p>
      The number worth watching is not the final loss but the <b>drop</b>: the panel reports
      the training loss measured <i>before</i> the first step alongside the last one, so a
      run that went nowhere looks like a run that went nowhere.
    </p>
    <p>
      The browser and the Python backend run <b>the same recipe</b>, and a golden test holds
      their forward pass, loss and spectrum statistics to ≤1e-5 on fixed weights. Whole-run
      equality is <b>not</b> claimed: platform BLAS and RNG streams diverge, so the same
      seed gives the same shape of curve, not the same curve.
    </p>
    <p>
      The <b>data</b> is part of that recipe, and is built the same way in both: each line
      with at least one word contributes its ids followed by <code>&lt;eos&gt;</code>, and
      blank lines contribute nothing. That is what lets the model learn where a line
      <i>ends</i>, which is why a sample comes back as verse rather than one long run of
      words. On the shipped corpus this is 19,071 tokens, 3,071 of them
      <code>&lt;eos&gt;</code> — a number both runtimes reproduce exactly, and a test
      pins it.
    </p>
    <p>
      <b>Fine-tuning</b> keeps the model's vocabulary, and it runs in the same worker a
      from-scratch run does — the request simply also carries the active model's weights
      and its word list. The explicit word list <i>wins</i> over the budget controls, which
      is the point: continuing training on text a budget cannot express does not teach it
      new words, it teaches it to predict <code>&lt;unk&gt;</code>, and rebuilding the
      budget underneath the weights would silently change what every embedding row means.
      To learn a different lexicon you must train from scratch at a budget rebuilt from
      that text — which is what "use this text" does.
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
    align-items: baseline;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.95rem;
  }
  .hint {
    font-size: 0.72rem;
    color: var(--text-dim);
    line-height: 1.4;
  }
  .hint.small {
    margin: 0;
  }
  .ctl {
    display: flex;
    flex-direction: column;
    gap: 0.28rem;
  }
  .ctl-label {
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .ctl-label b {
    color: var(--text);
    font-variant-numeric: tabular-nums;
    text-transform: none;
  }
  .seg {
    display: inline-flex;
    gap: 0.2rem;
    padding: 0.2rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 999px;
    align-self: flex-start;
    max-width: 100%;
  }
  .seg.wrap {
    flex-wrap: wrap;
    border-radius: 14px;
  }
  .seg button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.24rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    white-space: nowrap;
  }
  .seg button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
  }
  .seg button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .seg button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .active-corpus {
    margin: 0.1rem 0 0;
    font-size: 0.72rem;
    color: var(--text-dim);
    line-height: 1.5;
  }
  .active-corpus b {
    color: var(--text);
    font-family: var(--mono);
  }
  .why {
    color: #ffb454;
  }
  textarea {
    min-height: 84px;
    resize: vertical;
    font-size: 0.78rem;
  }
  .loading {
    margin: 0;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent);
  }
  .hypers {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr));
    gap: 0.4rem 0.8rem;
  }
  .slider {
    display: flex;
    flex-direction: column;
    gap: 0.22rem;
    min-width: 0;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .go:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .secondary {
    align-self: flex-start;
    background: var(--bg-elev);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-size: 0.75rem;
    font-family: var(--mono);
    font-weight: 500;
  }
  .secondary:hover {
    color: var(--text);
    border-color: var(--accent);
  }
  .live {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .chart-wrap {
    min-width: 0;
  }
  .chart {
    width: 100%;
    height: 96px;
    display: block;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .loss-line {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.6;
    vector-effect: non-scaling-stroke;
  }
  .uniform {
    stroke: var(--text-dim);
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
    opacity: 0.7;
  }
  .chart-key {
    margin: 0;
    font-size: 0.68rem;
    color: var(--text-dim);
    line-height: 1.5;
  }
  .chart-key b {
    color: var(--text);
    font-family: var(--mono);
  }
  .samples {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .sample {
    margin: 0;
    font-family: var(--mono);
    font-size: 0.74rem;
    line-height: 1.5;
    color: var(--text);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.4rem 0.55rem;
    overflow-wrap: anywhere;
  }
  .sample .at {
    color: var(--accent);
    margin-right: 0.5rem;
    font-size: 0.9em;
  }
  .active-model {
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
  .ok {
    background: rgba(91, 224, 176, 0.08);
    color: var(--good);
    border: 1px solid rgba(91, 224, 176, 0.28);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
    line-height: 1.45;
  }
</style>
