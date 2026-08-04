<script lang="ts">
  /**
   * The model's shape, and the parameter count that follows from it.
   *
   * The count is the closed form verified against the source implementation on 7
   * configurations (`config.py::param_count`, mirrored by `lexEngine.paramCount`), not an
   * estimate and not a framework's allocation total. It updates as you move any control,
   * because seeing `|V|·d` dominate at a small `d` is the point of putting the budget and
   * the dimensions on the same screen.
   *
   * `n_heads` is filtered to the divisors of `d_model` (FR-611) rather than accepted and
   * rejected later: every one of D_MODEL_CHOICES is divisible by every one of
   * N_HEAD_CHOICES today, so the filter is a guard against a future choice list, and it
   * disables rather than hides so the constraint stays visible.
   */
  import {
    CTX_CHOICES,
    DEFAULT_CTX,
    DEFAULT_N_LAYERS,
    DEFAULT_TIED,
    D_MODEL_CHOICES,
    MLP_RATIO,
    N_HEAD_CHOICES,
    N_LAYER_CHOICES,
    SPECIAL_TOKENS,
    dolchSizes,
    paramCount,
  } from "../../lib/lexEngine";
  import Explain from "../../lib/Explain.svelte";

  interface Props {
    dModel: number;
    nLayers: number;
    nHeads: number;
    ctx: number;
    tied: boolean;
    dropout: number;
    nParams: number;
    vocabRows: number;
    budgetSize: number;
    onDModel: (v: number) => void;
    onNLayers: (v: number) => void;
    onNHeads: (v: number) => void;
    onCtx: (v: number) => void;
    onTied: (v: boolean) => void;
    onDropout: (v: number) => void;
  }
  let {
    dModel,
    nLayers,
    nHeads,
    ctx,
    tied,
    dropout,
    nParams,
    vocabRows,
    budgetSize,
    onDModel,
    onNLayers,
    onNHeads,
    onCtx,
    onTied,
    onDropout,
  }: Props = $props();

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

  const headChoices = $derived(N_HEAD_CHOICES.filter((h: number) => dModel % h === 0));

  // The three terms of `N = (2 if untied else 1)·V·d + ctx·d + L·(12d² + 13d) + 2d`,
  // shown separately so the budget's share of the model is legible rather than implied.
  const embedParams = $derived((tied ? 1 : 2) * vocabRows * dModel);
  const posParams = $derived(ctx * dModel);
  const blockParams = $derived(nLayers * (12 * dModel * dModel + 13 * dModel));
  const finalNormParams = $derived(2 * dModel);
  const embedShare = $derived(nParams > 0 ? embedParams / nParams : 0);

  const fmt = (n: number) => n.toLocaleString();
  const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

  // ---- the crossover the explainer describes, MEASURED rather than asserted -----------
  //
  // The old sentence read "at d = 16 with the full budget the embedding is the majority of
  // the model". That is true at one layer and false at the tab's default two — the
  // embedding is 42% there — so a reader who followed it literally (move d_model, touch
  // nothing else) was told the opposite of what the bars in front of them showed. These
  // shares are now computed with the same closed form as the live count, at the endpoints
  // named in the prose, so the sentence cannot drift from the arithmetic again.
  const FULL_BUDGET_ROWS =
    Math.max(...Object.values(dolchSizes())) + SPECIAL_TOKENS.length;
  const SMALLEST_D = D_MODEL_CHOICES[0];
  const LARGEST_D = D_MODEL_CHOICES[D_MODEL_CHOICES.length - 1];
  const MOST_LAYERS = N_LAYER_CHOICES[N_LAYER_CHOICES.length - 1];
  /** Embedding share at the full Dolch budget, `d = SMALLEST_D`, `L` layers, defaults otherwise. */
  function fullBudgetEmbedShare(layers: number): number {
    const total = paramCount(FULL_BUDGET_ROWS, SMALLEST_D, layers, DEFAULT_CTX, DEFAULT_TIED);
    return ((DEFAULT_TIED ? 1 : 2) * FULL_BUDGET_ROWS * SMALLEST_D) / total;
  }
  const embedShareOneLayer = fullBudgetEmbedShare(1);
  const embedShareDefaultLayers = fullBudgetEmbedShare(DEFAULT_N_LAYERS);
  const blockShareLargest =
    (MOST_LAYERS * (12 * LARGEST_D * LARGEST_D + 13 * LARGEST_D)) /
    paramCount(FULL_BUDGET_ROWS, LARGEST_D, MOST_LAYERS, DEFAULT_CTX, DEFAULT_TIED);
</script>

<div class="panel-body" data-testid="lex-model">
  <div class="head">
    <h3>Model</h3>
    <span class="hint">pre-norm decoder-only — the source model's exact shape</span>
  </div>

  <div class="dims">
    <div class="ctl">
      <span class="ctl-label" id="lex-dmodel-label">d_model</span>
      <div
        class="seg"
        role="radiogroup"
        tabindex="-1"
        aria-labelledby="lex-dmodel-label"
        data-testid="lex-dmodel"
        onkeydown={(e) => segKey(e, (v) => onDModel(Number(v)))}
      >
        {#each D_MODEL_CHOICES as d (d)}
          <button
            role="radio"
            aria-checked={dModel === d}
            tabindex={dModel === d ? 0 : -1}
            data-value={String(d)}
            class:active={dModel === d}
            onclick={() => onDModel(d)}
          >{d}</button>
        {/each}
      </div>
    </div>

    <div class="ctl">
      <span class="ctl-label" id="lex-layers-label">layers</span>
      <div
        class="seg"
        role="radiogroup"
        tabindex="-1"
        aria-labelledby="lex-layers-label"
        data-testid="lex-layers"
        onkeydown={(e) => segKey(e, (v) => onNLayers(Number(v)))}
      >
        {#each N_LAYER_CHOICES as l (l)}
          <button
            role="radio"
            aria-checked={nLayers === l}
            tabindex={nLayers === l ? 0 : -1}
            data-value={String(l)}
            class:active={nLayers === l}
            onclick={() => onNLayers(l)}
          >{l}</button>
        {/each}
      </div>
    </div>

    <div class="ctl">
      <span class="ctl-label" id="lex-heads-label">heads</span>
      <div
        class="seg"
        role="radiogroup"
        tabindex="-1"
        aria-labelledby="lex-heads-label"
        data-testid="lex-heads"
        onkeydown={(e) => segKey(e, (v) => onNHeads(Number(v)))}
      >
        {#each N_HEAD_CHOICES as h (h)}
          <button
            role="radio"
            aria-checked={nHeads === h}
            tabindex={nHeads === h ? 0 : -1}
            data-value={String(h)}
            class:active={nHeads === h}
            disabled={!headChoices.includes(h)}
            title={headChoices.includes(h)
              ? `${h} head${h === 1 ? "" : "s"} of width ${dModel / h}`
              : `d_model must divide evenly into the heads, and ${h} does not divide ${dModel}`}
            onclick={() => onNHeads(h)}
          >{h}</button>
        {/each}
      </div>
    </div>

    <div class="ctl">
      <span class="ctl-label" id="lex-ctx-label">context</span>
      <div
        class="seg"
        role="radiogroup"
        tabindex="-1"
        aria-labelledby="lex-ctx-label"
        data-testid="lex-ctx"
        onkeydown={(e) => segKey(e, (v) => onCtx(Number(v)))}
      >
        {#each CTX_CHOICES as c (c)}
          <button
            role="radio"
            aria-checked={ctx === c}
            tabindex={ctx === c ? 0 : -1}
            data-value={String(c)}
            class:active={ctx === c}
            onclick={() => onCtx(c)}
          >{c}</button>
        {/each}
      </div>
    </div>
  </div>

  <div class="row">
    <label class="check">
      <input
        type="checkbox"
        data-testid="lex-tied"
        checked={tied}
        onchange={(e) => onTied(e.currentTarget.checked)}
      />
      <span>tie the readout to the embedding</span>
    </label>
    <label class="slider">
      <span class="ctl-label">dropout <b>{dropout.toFixed(2)}</b></span>
      <input
        type="range"
        min="0"
        max="0.5"
        step="0.05"
        data-testid="lex-dropout"
        value={dropout}
        oninput={(e) => onDropout(Number(e.currentTarget.value))}
      />
    </label>
  </div>

  <div class="params" data-testid="lex-params">
    <div class="total">
      <span class="k">parameters</span>
      <span class="v">{fmt(nParams)}</span>
    </div>
    <div class="breakdown">
      <div class="term">
        <span class="tk">embedding{tied ? "" : " + readout"}</span>
        <span class="tv">{fmt(embedParams)}</span>
        <span class="tb"><span class="fill" style={`width:${(embedShare * 100).toFixed(1)}%`}></span></span>
        <span class="td">
          {tied ? "1" : "2"} × {fmt(vocabRows)} rows × {dModel} —
          <b>{(embedShare * 100).toFixed(0)}%</b> of the model
        </span>
      </div>
      <div class="term">
        <span class="tk">{nLayers} block{nLayers === 1 ? "" : "s"}</span>
        <span class="tv">{fmt(blockParams)}</span>
        <span class="tb"><span class="fill dim" style={`width:${nParams ? ((blockParams / nParams) * 100).toFixed(1) : 0}%`}></span></span>
        <span class="td">{nLayers} × (12·{dModel}² + 13·{dModel})</span>
      </div>
      <div class="term">
        <span class="tk">positions + final norm</span>
        <span class="tv">{fmt(posParams + finalNormParams)}</span>
        <span class="tb"><span class="fill dim" style={`width:${nParams ? (((posParams + finalNormParams) / nParams) * 100).toFixed(1) : 0}%`}></span></span>
        <span class="td">{ctx} × {dModel} learned absolute positions, + 2·{dModel}</span>
      </div>
    </div>
  </div>

  <Explain
    title="Where the parameters go"
    hint="how much of the model the vocabulary is depends on d AND on L"
    testid="lex-explain-params"
  >
    <p>
      The count is exact:
    </p>
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div class="eq" role="group" aria-label="parameter count" tabindex="0">
      N = (2 if untied else 1)·V·d + ctx·d + L·(12d² + 13d) + 2d
    </div>
    <p>
      Per block the <code>12d² + 13d</code> is: the packed QKV projection
      <code>3d² + 3d</code>, the attention output projection <code>d² + d</code>, the MLP
      up-projection <code>{MLP_RATIO}d² + {MLP_RATIO}d</code>, the MLP down-projection
      <code>{MLP_RATIO}d² + d</code>, and two LayerNorms at <code>4d</code>.
    </p>
    <p>
      <b>Tying</b> makes the readout <i>be</i> the embedding matrix rather than a second
      one — it halves the vocabulary's cost, and it means the model has exactly one
      embedding geometry to look at. Untied, the spectrum panel shows two spectra, because
      there genuinely are two matrices and they need not agree.
    </p>
    <p>
      The crossover depends on <code>L</code> as much as on <code>d</code>, so it is worth
      stating with both fixed. At <code>d = {SMALLEST_D}</code> with the full
      {FULL_BUDGET_ROWS - SPECIAL_TOKENS.length}-word budget, tied, at
      <code>ctx = {DEFAULT_CTX}</code>: the embedding is
      <b>{pct(embedShareOneLayer)}</b> of the model at <b>one</b> layer — a majority — but
      only <b>{pct(embedShareDefaultLayers)}</b> at the default
      <code>L = {DEFAULT_N_LAYERS}</code>, where the blocks have already overtaken it. At
      <code>d = {LARGEST_D}</code> with {MOST_LAYERS} layers the blocks hold
      <b>{pct(blockShareLargest)}</b>. The bar next to each term is that share at
      <i>your</i> shape, so the crossover is visible rather than arithmetical.
    </p>
    <p>
      <b>Dropout</b> defaults to 0 so that re-running the same configuration gives the
      same model. The source hard-codes 0.1 and does not expose it; this exposes it and
      tells you what changed. Its placement — embedding sum, attention weights, after the
      second MLP linear, and <i>not</i> on the attention residual branch — is the source's
      and is kept.
    </p>
    <p class="tiny">
      Current shape: <code>|V| = {fmt(budgetSize)}</code> words in
      <code>{fmt(vocabRows)}</code> rows, <code>{nHeads}</code>
      head{nHeads === 1 ? "" : "s"} of width <code>{dModel / nHeads}</code>.
    </p>
  </Explain>
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
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
  .dims {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 9.5rem), 1fr));
    gap: 0.5rem 0.8rem;
  }
  .ctl {
    display: flex;
    flex-direction: column;
    gap: 0.28rem;
    min-width: 0;
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
  }
  .seg {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 0.2rem;
    padding: 0.2rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 999px;
    align-self: flex-start;
    max-width: 100%;
  }
  .seg button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.24rem 0.62rem;
    font-size: 0.75rem;
    font-family: var(--mono);
    font-weight: 500;
  }
  .seg button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
  }
  .seg button:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
  .seg button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .row {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .check {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: var(--text);
  }
  .check input {
    accent-color: var(--accent);
  }
  .slider {
    display: flex;
    flex-direction: column;
    gap: 0.22rem;
    min-width: 150px;
    flex: 1;
  }
  .params {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.7rem;
  }
  .total {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.6rem;
  }
  .total .k {
    font-size: 0.66rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .total .v {
    font-family: var(--mono);
    font-size: 1.25rem;
    font-variant-numeric: tabular-nums;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .breakdown {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .term {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.1rem 0.6rem;
    align-items: baseline;
  }
  .term .tk {
    font-size: 0.72rem;
    color: var(--text);
  }
  .term .tv {
    font-family: var(--mono);
    font-size: 0.74rem;
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }
  .term .tb {
    grid-column: 1 / -1;
    height: 4px;
    border-radius: 999px;
    background: var(--bg-elev-2);
    overflow: hidden;
  }
  .term .fill {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: var(--accent-grad);
    transition: width 0.25s ease;
  }
  .term .fill.dim {
    background: var(--border);
  }
  .term .td {
    grid-column: 1 / -1;
    font-size: 0.67rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .term .td b {
    color: var(--text);
  }
  .panel-body :global(.eq b),
  .panel-body :global(.tiny) {
    font-size: 0.76rem;
  }
</style>
