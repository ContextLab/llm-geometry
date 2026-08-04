<script lang="ts">
  /**
   * Generation, with the in-budget guarantee stated where you can check it (FR-605).
   *
   * The logits of `GENERATION_BANNED_IDS` are set to −∞ before sampling, and the model's
   * vocabulary IS the budget — so every word it emits is a budget word by construction.
   * There is no trie, no post-filter, and no filtering step that could be wrong: the
   * source project's word-boundary trie emitted run-together tokens like `" hameat"`,
   * which is precisely the class of bug a word-level budget cannot have.
   *
   * `<eos>` is deliberately NOT banned — it is how the model ends a line.
   *
   * The model and the vocabulary always arrive as a pair, and `generate` refuses a
   * mismatched one outright — a model can only be sampled with the vocabulary its ids
   * mean something in.
   */
  import {
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    GENERATION_BANNED_IDS,
    LexModel,
    LexVocab,
    MAX_NEW_TOKENS,
    SPECIAL_TOKENS,
    generate,
    tokenize,
  } from "../../lib/lexEngine";
  import Explain from "../../lib/Explain.svelte";
  import type { Provenance } from "./provenance";

  interface Props {
    model: LexModel | null;
    vocab: LexVocab | null;
    /** What these weights are: untrained, trained, or either of those hand-edited. */
    provenance: Provenance;
  }
  let { model, vocab, provenance }: Props = $props();

  /** One phrase per state, so the header never names weights that are not running. */
  const sourceHint = $derived(
    {
      trained: "from the model you trained",
      untrained: "from the untrained model — noise, in budget, which is the point",
      "edited-trained": "from hand-edited weights, not from the model you trained",
      "edited-untrained":
        "from hand-edited weights over an untrained model — still noise, still in budget",
    }[provenance],
  );

  let prompt = $state("");
  let temperature = $state<number>(DEFAULT_TEMPERATURE);
  let maxNewTokens = $state<number>(DEFAULT_MAX_NEW_TOKENS);
  let seed = $state(0);
  let output = $state<{ text: string; words: string[] } | null>(null);
  let error = $state("");
  let busy = $state(false);

  const bannedNames = GENERATION_BANNED_IDS.map((id) => SPECIAL_TOKENS[id]).join(", ");

  /**
   * A prompt is tokenized against the budget like any other text, so out-of-budget words
   * in it become `<unk>`. Saying so beats letting somebody wonder why `sarsaparilla` had
   * no effect.
   */
  const promptOov = $derived.by(() => {
    const v = vocab;
    if (!v || !prompt.trim()) return [];
    // The engine's own tokenizer and membership test, so this agrees with what the model
    // will actually be handed rather than approximating it with a second regex.
    return [...new Set(tokenize(prompt).filter((t) => !v.has(t)))];
  });

  function run(): void {
    if (!model || !vocab) return;
    error = "";
    busy = true;
    try {
      const r = generate(model, vocab, { prompt, temperature, maxNewTokens, seed });
      output = { text: r.text, words: r.words };
    } catch (e) {
      // Drop the previous sample too: the engine now REFUSES to generate from weights
      // that have left float32 range (an edit of `1e40`, or a `×2` preset applied often
      // enough), and leaving the last good text on screen would put an "in budget by
      // construction" caption beside a model that can no longer produce anything.
      output = null;
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<div class="panel-body" data-testid="lex-sample">
  <div class="head">
    <h3>Generate</h3>
    <span class="hint">{sourceHint}</span>
  </div>

  <div class="controls">
    <label class="prompt">
      <span class="ctl-label">prompt</span>
      <input
        type="text"
        data-testid="lex-prompt"
        bind:value={prompt}
        placeholder="leave empty to start from &lt;bos&gt;, or type a few budget words"
      />
    </label>
    <label class="slider">
      <span class="ctl-label">temperature <b>{temperature.toFixed(2)}</b></span>
      <input
        type="range"
        min="0"
        max="2"
        step="0.05"
        data-testid="lex-temperature"
        bind:value={temperature}
      />
    </label>
    <label class="slider">
      <span class="ctl-label">max tokens <b>{maxNewTokens}</b></span>
      <input
        type="range"
        min="5"
        max={MAX_NEW_TOKENS}
        step="5"
        data-testid="lex-max-tokens"
        bind:value={maxNewTokens}
      />
    </label>
    <label class="slider narrow">
      <span class="ctl-label">seed <b>{seed}</b></span>
      <input type="range" min="0" max="99" step="1" data-testid="lex-gen-seed" bind:value={seed} />
    </label>
  </div>

  <div class="actions">
    <button data-testid="lex-generate" disabled={busy || !model || !vocab} onclick={run}>
      {busy ? "generating…" : "Generate"}
    </button>
    <span class="badge guarantee" data-testid="lex-inbudget-badge" title={`The logits of ${bannedNames} are set to −∞ before sampling, and the model's whole vocabulary is the budget. Every emitted word is therefore a budget word — by construction, not by filtering.`}>
      in budget by construction
    </span>
    {#if temperature === 0}
      <span class="badge greedy" title="At temperature 0 the sampler takes the argmax, so the same prompt and model always give the same text.">greedy · deterministic</span>
    {/if}
  </div>

  {#if promptOov.length > 0}
    <p class="oov" data-testid="lex-prompt-oov">
      not in this budget, so the model reads
      {promptOov.length === 1 ? "it" : "them"} as <code>&lt;unk&gt;</code>:
      <b>{promptOov.slice(0, 12).join(", ")}</b>{promptOov.length > 12
        ? `, and ${promptOov.length - 12} more`
        : ""}
    </p>
  {/if}

  {#if error}
    <div class="err" data-testid="lex-generate-error">{error}</div>
  {/if}

  {#if output}
    <div class="output" data-testid="lex-output">
      <p class="text">{output.text}</p>
      <p class="meta">
        {output.words.length} word{output.words.length === 1 ? "" : "s"} · every one drawn
        from the {vocab?.budgetSize ?? 0}-word budget · <code>{bannedNames}</code> were masked;
        <code>&lt;eos&gt;</code> was not, because that is how a line ends — it appears as the
        line breaks above rather than being counted here
      </p>
    </div>
  {/if}

  <Explain
    title="Why nothing out of budget can appear"
    hint="the vocabulary IS the budget — no filter to get wrong"
    testid="lex-explain-generation"
  >
    <p>
      At <code>temperature = 0</code> the sampler is greedy; above 0 it samples from
      <code>softmax(logits[-1] / T)</code> after setting the logits of
      <code>{bannedNames}</code> to −∞. Those are the only ids ever masked.
    </p>
    <p>
      The guarantee is <b>structural</b>, not enforced after the fact. This model's output
      layer has one row per budget word plus the {SPECIAL_TOKENS.length} reserved rows — it
      has no way to name a word outside the budget, so there is nothing to filter and no
      filter to have a bug in. A model that shares a large tokenizer and then restricts
      generation has to get a filter right; this one does not.
    </p>
    <p>
      The one thing removed after the fact is <code>&lt;eos&gt;</code>, and only from the
      word <i>count</i>: it is a line ending, so it is drawn as the line breaks in the text
      above. The masked ids are not removed — they are never chosen. If one ever were, or
      if the model's weights left <code>float32</code> range (a cell edit of
      <code>1e40</code>, or a <code>×2</code> preset applied enough times, makes every
      logit <code>NaN</code>), generation <b>refuses with an error</b> instead of printing
      something and calling it in budget. That is the same refusal the Python backend makes.
    </p>
    <p>
      Which is also the honest limitation: the budget's cost has already been paid, in the
      <code>&lt;unk&gt;</code> rate the budget panel reports. The model does not work around
      a word it lacks; it simply cannot refer to it.
    </p>
    <p>
      Before training, generation still runs and still respects the budget — it produces
      budget words in a random order. If that surprises you, it is worth sitting with:
      "in budget" and "meaningful" are different properties, and only one of them is
      guaranteed here.
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
  .controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
    gap: 0.45rem 0.8rem;
    align-items: end;
  }
  .prompt {
    display: flex;
    flex-direction: column;
    gap: 0.24rem;
    grid-column: 1 / -1;
    min-width: 0;
  }
  .prompt input {
    font-family: var(--mono);
    font-size: 0.84rem;
  }
  .slider {
    display: flex;
    flex-direction: column;
    gap: 0.22rem;
    min-width: 0;
  }
  .slider.narrow {
    max-width: 12rem;
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
  .actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .actions button:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .badge {
    font-family: var(--mono);
    font-size: 0.7rem;
    border-radius: 999px;
    padding: 0.18rem 0.6rem;
  }
  .badge.guarantee {
    color: var(--good);
    background: rgba(91, 224, 176, 0.12);
    border: 1px solid rgba(91, 224, 176, 0.35);
  }
  .badge.greedy {
    color: var(--text-dim);
    background: var(--bg-elev);
    border: 1px solid var(--border);
  }
  .oov {
    margin: 0;
    font-size: 0.72rem;
    line-height: 1.5;
    color: #ffb454;
  }
  .oov b {
    font-family: var(--mono);
  }
  .oov code {
    font-family: var(--mono);
  }
  .output {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.7rem;
  }
  .output .text {
    margin: 0;
    font-family: var(--mono);
    font-size: 0.84rem;
    line-height: 1.7;
    color: var(--text);
    overflow-wrap: anywhere;
  }
  .output .meta {
    margin: 0;
    font-size: 0.68rem;
    color: var(--text-dim);
    line-height: 1.5;
  }
  .output .meta code {
    font-family: var(--mono);
    color: var(--accent);
  }
  .err {
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
  }
</style>
