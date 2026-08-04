<script lang="ts">
  /**
   * US-1 — move the budget, and see what it costs BEFORE training anything.
   *
   * The three counters FR-606 requires are visible from a cold load: token coverage,
   * `<unk>` rate, and how many corpus lines are wholly inside the budget. They are the
   * measurable form of "what this vocabulary cannot say", and they are the reason the
   * budget is a real experimental control rather than a setting.
   *
   * Every size shown is MEASURED from the word lists at runtime (`dolchSizes()`), never
   * quoted — the published Dolch counts disagree with each other, and the largest list
   * here is one word shorter than the usual citation because `Santa Claus` cannot be a
   * word-level token.
   */
  import {
    DOLCH_ORDER,
    SPECIAL_TOKENS,
    dolchSizes,
    type Coverage,
    type DolchBudgetName,
    type LexVocab,
  } from "../../lib/lexEngine";
  import Explain from "../../lib/Explain.svelte";

  interface Props {
    source: string;
    budget: string;
    vocab: LexVocab | null;
    coverage: Coverage | null;
    corpusLabel: string;
    onSource: (s: string) => void;
    onBudget: (b: string) => void;
  }
  let { source, budget, vocab, coverage, corpusLabel, onSource, onBudget }: Props =
    $props();

  const sizes = dolchSizes();

  /** Human labels for the graded lists. The NUMBER always comes from `sizes`. */
  const BUDGET_LABELS: Record<DolchBudgetName, string> = {
    pre_primer: "pre-primer",
    primer: "primer",
    first: "first grade",
    service: "service words",
    full: "full list",
  };

  const SOURCES = [
    {
      id: "dolch",
      label: "Dolch (1936)",
      title: "The real graded sight-word lists published by Edward William Dolch in 1936 — a vocabulary somebody prescribed.",
    },
    {
      id: "frequency",
      label: "corpus top-N",
      title: "The N most frequent word types of the corpus you are training on, ties broken alphabetically — a vocabulary the corpus describes.",
    },
  ];

  /**
   * ARIA radiogroup keyboard behaviour: arrows move the selection and the focus together.
   * Written out per panel on purpose — this feature slice owns no shared control module,
   * and a segmented control that only responds to the mouse is not finished.
   */
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

  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
</script>

<div class="panel-body" data-testid="lex-budget">
  <div class="head">
    <h3>Vocabulary budget</h3>
    <span class="hint">the independent variable — everything else responds to it</span>
  </div>

  <div class="ctl">
    <span class="ctl-label" id="lex-budget-source-label">budget source</span>
    <div
      class="seg"
      role="radiogroup"
      tabindex="-1"
      aria-labelledby="lex-budget-source-label"
      data-testid="lex-budget-source"
      onkeydown={(e) => segKey(e, onSource)}
    >
      {#each SOURCES as s (s.id)}
        <button
          role="radio"
          aria-checked={source === s.id}
          tabindex={source === s.id ? 0 : -1}
          data-value={s.id}
          class:active={source === s.id}
          title={s.title}
          onclick={() => onSource(s.id)}
        >{s.label}</button>
      {/each}
    </div>
  </div>

  <div class="ctl">
    <span class="ctl-label" id="lex-budget-size-label">
      budget size
      {#if source === "frequency"}<span class="matched">· matched to the Dolch sizes</span>{/if}
    </span>
    <div
      class="seg wrap"
      role="radiogroup"
      tabindex="-1"
      aria-labelledby="lex-budget-size-label"
      data-testid="lex-budget-size"
      onkeydown={(e) => segKey(e, onBudget)}
    >
      {#each DOLCH_ORDER as name (name)}
        <button
          role="radio"
          aria-checked={budget === name}
          aria-label={`${BUDGET_LABELS[name]}, ${sizes[name]} words`}
          tabindex={budget === name ? 0 : -1}
          data-value={name}
          class:active={budget === name}
          onclick={() => onBudget(name)}
        >
          {BUDGET_LABELS[name]}
          <b>{sizes[name]}</b>
        </button>
      {/each}
    </div>
  </div>

  {#if vocab}
    <p class="rows" data-testid="lex-budget-rows">
      <b>|V| = {vocab.budgetSize.toLocaleString()}</b> words ·
      <b>{vocab.rows.toLocaleString()}</b> embedding rows
      <span class="why" title={`the ${SPECIAL_TOKENS.length} reserved rows: ${SPECIAL_TOKENS.join(" ")}`}
        >(+{SPECIAL_TOKENS.length} specials: {SPECIAL_TOKENS.join(" ")})</span
      >
    </p>
  {/if}

  <div class="counters" data-testid="lex-coverage">
    {#if coverage}
      <div class="counter">
        <span class="k">token coverage</span>
        <span class="v good" data-testid="lex-coverage-tokens">{pct(coverage.token_coverage)}</span>
        <span class="d">
          {coverage.in_budget_tokens.toLocaleString()} of
          {coverage.total_tokens.toLocaleString()} corpus tokens are in budget
        </span>
      </div>
      <div class="counter">
        <span class="k">&lt;unk&gt; rate</span>
        <span class="v warn" data-testid="lex-coverage-unk">{pct(coverage.unk_rate)}</span>
        <span class="d">
          the share of the corpus this budget cannot express — those tokens train as
          <code>&lt;unk&gt;</code>
        </span>
      </div>
      <div class="counter">
        <span class="k">whole lines in budget</span>
        <span class="v" data-testid="lex-coverage-lines">
          {coverage.whole_lines_in_budget.toLocaleString()}
          <span class="of">/ {coverage.total_lines.toLocaleString()}</span>
        </span>
        <span class="d">
          lines with every word inside the budget — the ones the model could reproduce
          exactly
        </span>
      </div>
      <div class="counter">
        <span class="k">out-of-budget types</span>
        <span class="v" data-testid="lex-coverage-oov">{coverage.oov_types.toLocaleString()}</span>
        <span class="d">
          distinct words in {corpusLabel || "the corpus"} with no row of their own, out of
          {coverage.distinct_types.toLocaleString()}
        </span>
      </div>
    {:else}
      <p class="waiting">measuring the budget against the corpus…</p>
    {/if}
  </div>

  <Explain
    title="Why coverage before training"
    hint="the budget's cost is measurable without running anything"
    testid="lex-explain-coverage"
  >
    <p>
      A budget's price is paid before the first gradient step. Coverage is a property of
      the word list and the corpus alone, so it is shown the moment you move the control —
      you can decide what the model will be unable to say without waiting for it to fail
      to say it.
    </p>
    <p>
      The two sources are offered at <b>matched</b> sizes so a comparison is honest: the
      same <code>|V|</code>, the same model shape, the same corpus, different words. The
      Dolch lists <b>nest</b> — a larger budget only ever adds words — so growing
      <code>|V|</code> is not confounded by words leaving.
    </p>
    <p>
      Out-of-budget tokens map to <code>&lt;unk&gt;</code> in both inputs and targets, so
      the model spends real probability mass on a token it is then forbidden to emit. That
      is a cost of the budget, not a bug, and the <code>&lt;unk&gt;</code> rate is how big
      it is.
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
  .matched {
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
    color: var(--accent);
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
    padding: 0.26rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    white-space: nowrap;
  }
  .seg button b {
    font-family: var(--mono);
    font-size: 0.9em;
    opacity: 0.75;
    margin-left: 0.25rem;
    font-variant-numeric: tabular-nums;
  }
  .seg button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
  }
  .seg button.active b {
    opacity: 0.85;
  }
  .seg button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .rows {
    margin: 0;
    font-size: 0.76rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .rows b {
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .why {
    color: var(--text-dim);
    opacity: 0.8;
  }
  .counters {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 11rem), 1fr));
    gap: 0.5rem;
  }
  .counter {
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.6rem;
    min-width: 0;
  }
  .counter .k {
    font-size: 0.66rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .counter .v {
    font-family: var(--mono);
    font-size: 1.1rem;
    color: var(--text);
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
  }
  .counter .v.good {
    color: var(--good);
  }
  .counter .v.warn {
    color: #ffb454;
  }
  .counter .of {
    font-size: 0.72rem;
    color: var(--text-dim);
  }
  .counter .d {
    font-size: 0.68rem;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .counter .d code {
    font-family: var(--mono);
    color: var(--accent);
  }
  .waiting {
    margin: 0;
    font-size: 0.74rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
</style>
