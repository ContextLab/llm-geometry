<script lang="ts">
  import { get } from "svelte/store";
  import { archModelId } from "../../lib/explorerStores";
  import { client, type ArchVacancyScoreResult } from "../../lib/dataClient";
  import { plainError } from "./archShared";
  import {
    errorBarTerms,
    formCostsLessThanContent,
    formSharePercent,
    nats,
    num,
    quantizationNote,
    secondaryLine,
    upperBoundLabel,
    verdictKind,
  } from "./vacancyVerdict";
  import Explain from "../../lib/Explain.svelte";
  import StaticNotice from "../../lib/StaticNotice.svelte";
  import { STATIC_MODE } from "../../lib/staticUx";
  import { view } from "../../lib/stores";

  /**
   * What a word's FORM is worth to a model that HAS one for it (feature 007, ui.md §2).
   *
   * The Lexicon Lab's arm of this instrument measures a from-scratch word-level model, for
   * which the answer is an exact 0 — the vacancy transform is a pure relabelling of its
   * vocabulary, so its loss is bit-identical. That null is only interpretable next to a
   * model that does have lexical entries, which is what this panel runs: three real
   * forward passes per passage on a real pretrained transformer.
   *
   * THE DECOMPOSITION IS THE WHOLE POINT (contract §8.3). Vacating content words changes
   * three things at once: the forms become unknown, they fragment into more subword
   * tokens, and the passage stops meaning anything. A caveat cannot separate those; a
   * control can. The `swap` variant replaces each vacated WORD with another type of the
   * passage's own domain, frequency-rank-matched inside the same suffix class. Nothing is
   * assembled, so the image IS a domain type: a real word, of the same inflection as the
   * word it replaces — equally nonsensical, ordinarily tokenized — so:
   *
   *     nll(swap)  − nll(english)  =  the cost of WRONG CONTENT
   *     nll(nonce) − nll(swap)     =  the cost of UNKNOWN FORM
   *
   * `nll(nonce) − nll(english)` is their SUM. It is rendered, small and labelled, and
   * never as the headline: presenting it as "what location was worth" would credit the
   * cost of nonsense to the cost of an unknown word.
   *
   * NOTHING IN THIS FILE IS A CONSTANT COPIED FROM A DOCUMENT. Every number rendered
   * below comes out of the response of the run the reader just triggered; every caveat
   * string (`confound`, the unknown-form note, the refusals) is served by the stack that
   * computed the numbers, so a stack that may not report something says so in its own
   * words rather than having this component guess on its behalf.
   *
   * The mint strategy is deliberately NOT a control here, unlike in the Lexicon Lab's
   * panel: the decomposition needs both strategies at once, so every run computes all
   * three variants. Choosing one would make the two differences uncomputable.
   */

  let passage = $state("");
  let useDefaults = $state(true);
  let p = $state(1);
  let seed = $state(0);
  let busy = $state(false);
  let error = $state("");
  let result = $state<ArchVacancyScoreResult | null>(null);
  let forModel = "";
  let elapsed = $state(0);
  let timer: ReturnType<typeof setInterval> | null = null;

  // A result belongs to the model it was measured on. Dropping it on a model change is
  // not tidiness: leaving it on screen would attribute one model's numbers to another.
  $effect(() => {
    const m = $archModelId;
    if (result && forModel !== m) {
      result = null;
      error = "";
    }
  });

  const int = (v: number): string => v.toLocaleString();

  const headline = $derived((result?.differences ?? []).filter((d) => d.headline));
  const secondary = $derived((result?.differences ?? []).filter((d) => !d.headline));
  const absoluteRefusal = $derived(result?.variants.find((v) => v.refused)?.refused ?? null);
  const wrongContent = $derived(result?.differences.find((d) => d.id === "wrong_content") ?? null);
  const unknownForm = $derived(result?.differences.find((d) => d.id === "unknown_form") ?? null);
  const total = $derived(result?.differences.find((d) => d.id === "total") ?? null);

  /**
   * Which closing sentence this run licenses — decided in `vacancyVerdict.ts`, against
   * real payloads, under test. Rendering is this file's job; deciding what may be said
   * about a number is the part that was wrong on the live site, so it lives where it can
   * be asserted with the numbers that were actually observed.
   */
  const verdict = $derived(verdictKind(unknownForm));
  /** `null` where "N% of the total damage" would not be a share of anything. */
  const formShare = $derived(formSharePercent(unknownForm, total));
  /** Whether the closing ordering claim is one THIS run supports. */
  const formCostsLess = $derived(formCostsLessThanContent(unknownForm, wrongContent));
  /**
   * What the quantization term beside a number actually is. The term used to call itself
   * "measured"; it is a bound retained from a measurement whose texts the swap rewrite
   * replaced, and the reader gets told that where the number is, not only in the Info tab.
   */
  const qNote = $derived(
    quantizationNote((result?.differences ?? []).find((d) => d.quantizationUncertaintyNats)),
  );

  let seq = 0;
  async function score(): Promise<void> {
    const my = ++seq;
    const m = get(archModelId);
    busy = true;
    error = "";
    elapsed = 0;
    const started = Date.now();
    timer = setInterval(() => (elapsed = (Date.now() - started) / 1000), 100);
    try {
      const body = useDefaults || !passage.trim() ? {} : { passage };
      const r = await client.archVacancyScore({
        model_id: m,
        p,
        seed,
        ...body,
      });
      if (my !== seq) return;
      forModel = m;
      result = r;
      // Show the reader the text the number came from, so "editable" means editing THIS
      // passage rather than typing into an empty box and hoping it matches.
      if (!passage.trim() && r.passages_used?.length) passage = r.passages_used[0];
    } catch (e) {
      if (my !== seq) return;
      result = null;
      error = plainError(e);
    } finally {
      if (my === seq) busy = false;
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }
  }
</script>

<section class="vac" data-testid="arch-vacancy">
  <div class="head">
    <h3>What is a word's form worth?</h3>
    <span class="sub">
      three real forward passes per passage · full vacancy of open-class stems · the closed-class
      scaffolding is character-identical in all three, and is what gets scored
    </span>
  </div>

  <p class="lede">
    Vacate the content words of a passage and the scaffolding around them — <i>the</i>, <i>and</i>,
    <i>did</i> — is still there, byte for byte. Does a model that knows English still predict it?
    Reading only those tokens, this measures the mean negative log-likelihood in nats under three
    conditions and reports the two differences that can be told apart: the cost of <b>wrong
    content</b>, and the cost of <b>unknown form</b>. Their sum is not the headline, because it
    conflates them.
  </p>

  <div class="controls">
    <label class="scope">
      <input
        type="checkbox"
        data-testid="arch-vac-defaults"
        checked={useDefaults}
        onchange={(e) => (useDefaults = e.currentTarget.checked)}
      />
      <span>
        score the shipped corpus excerpts (pooled) — uncheck to score just the passage below
      </span>
    </label>
    <label class="num">
      <span>p</span>
      <input
        type="number"
        min="0"
        max="1"
        step="0.05"
        data-testid="arch-vac-p"
        value={p}
        onchange={(e) => (p = Number(e.currentTarget.value))}
      />
    </label>
    <label class="num">
      <span>seed</span>
      <input
        type="number"
        step="1"
        data-testid="arch-vac-seed"
        value={seed}
        onchange={(e) => (seed = Math.trunc(Number(e.currentTarget.value)))}
      />
    </label>
    <button class="run" data-testid="arch-vac-run" disabled={busy} onclick={score}>
      {busy ? "scoring…" : "Score"}
    </button>
  </div>

  <textarea
    class="passage"
    rows="4"
    spellcheck="false"
    data-testid="arch-vac-passage"
    placeholder="Leave empty to use the shipped corpus excerpts, or paste a passage of your own."
    value={passage}
    oninput={(e) => (passage = e.currentTarget.value)}
  ></textarea>
  <p class="hint">
    Nothing runs while you type: this is real inference, so it waits for the button.
  </p>

  <!--
    The `p` control used to be a bare number input at 0.05 steps with nothing saying what
    `p` was — and 19 of its 21 values are configurations in which the swap map is provably
    non-injective, which the Lexicon Lab refuses by name (contract §5.2a) while this panel
    reported a decomposition through them anyway. The engine refuses them here too now;
    this says why before the reader spends a run finding out, exactly as the Lexicon Lab's
    mint note does.
  -->
  <p class="pnote" data-testid="arch-vac-pnote">
    <b><code>p</code> is the vacancy rate</b> — the fraction of eligible open-class stems replaced.
    The decomposition needs the <code>swap</code> control, and a swap map is injective only at
    <code>p = 0</code> and <code>p = 1</code>: its replacements are drawn <i>from</i> the passage's
    own types, so in between a vacated type can land on one that was not vacated. That is a
    theorem (contract §5.2a), not a rough edge, so intermediate values are <b>refused</b> rather
    than scored. <code>p = 1</code> is full vacancy and the configuration the reference numbers
    were measured at; <code>p = 0</code> is the null control, where all three variants are one
    string and every difference is exactly 0 <i>by construction</i>.
  </p>

  {#if busy}
    <div class="running" data-testid="arch-vac-busy">
      <div class="bar"><div class="fill"></div></div>
      <span
        >running real forward passes{STATIC_MODE
          ? " in this browser"
          : ""} — {elapsed.toFixed(1)} s</span
      >
    </div>
  {/if}

  {#if error}
    <div class="err" data-testid="arch-vac-error">
      <span>{error}</span>
      {#if error.includes("§5.2a")}
        <!-- The Lexicon Lab offers the two defined ends rather than moving the control
             silently; so does this. Nothing is substituted for the refused run. -->
        <div class="actions">
          <button data-testid="arch-vac-set-p1" onclick={() => ((p = 1), void score())}>
            Set p = 1 — full vacancy
          </button>
          <button data-testid="arch-vac-set-p0" onclick={() => ((p = 0), void score())}>
            Set p = 0 — the null control
          </button>
        </div>
      {/if}
    </div>
  {/if}

  {#if result}
    <div class="grid">
      <div class="pair" data-testid="arch-vac-headline">
        {#each headline as d (d.id)}
          <div class="delta" class:refused={Boolean(d.refused)}>
            <span class="dlabel">{d.label}</span>
            <span class="dexpr">{d.expr}</span>
            {#if d.refused}
              <StaticNotice message={d.refused.message} testid={`arch-vac-refused-${d.id}`} />
            {:else}
              <span class="dvalue" data-testid={`arch-vac-${d.id}`}>
                {nats(d.nats)}<span class="unit">nats</span>
              </span>
              <span class="derr" data-testid={`arch-vac-${d.id}-err`}>
                {errorBarTerms(d).join(" · ")}
              </span>
            {/if}
            {#if d.upperBound && !d.refused}
              <span class="bound" data-testid={`arch-vac-${d.id}-bound`}>
                {upperBoundLabel(d)}
              </span>
            {/if}
          </div>
        {/each}

        <div class="delta tiny" data-testid="arch-vac-tiny-arm">
          <span class="dlabel">the same measurement, tiny model</span>
          <span class="dexpr">{result.tiny_arm.label}</span>
          <span class="dvalue exact">
            {result.tiny_arm.delta_nats.toFixed(0)}<span class="unit">nats, exactly</span>
          </span>
          <span class="derr">an identity, not a rounding</span>
        </div>
      </div>

      {#if qNote}
        <p class="qnote" data-testid="arch-vac-quantization-note">{qNote}</p>
      {/if}

      <p class="twobytwo" data-testid="arch-vac-verdict">
        That juxtaposition is the whole result: a word's form is worth <b>exactly nothing</b> to a
        model trained from scratch with no lexical entries, and
        {#if verdict === "refused" || !unknownForm}
          a quantity this build may not report to one that has them (see the refusal above).
        {:else if verdict === "identity"}
          <b>exactly nothing</b> to one that has them either — but by construction, not by
          measurement: {unknownForm.identityNote}
        {:else if verdict === "unresolved"}
          <b>{nats(unknownForm.nats)} ± {num(unknownForm.se)} nats</b> to one that has them — an
          effect this sample does not resolve, because the standard error is more than half of it.
          Score more text (the pooled corpus excerpts above are the measured configuration) before
          reading anything into the sign.
        {:else if verdict === "backwards"}
          <b>{nats(unknownForm.nats)} nats</b> to one that has them — which is
          <b>negative</b>, and a cost cannot be. It says the nonce variant was <i>easier</i> to
          predict than the swap variant, so on this sample the contrast ran backwards and no
          conclusion is drawn from it. That happens on short passages, where the swap variant's
          own replacements can be rarer than the nonce forms that replace them; it is not the
          measured configuration. (Intermediate <code>p</code> used to be the other way in, and
          is now refused outright, so it cannot be the cause of this one.) Score the pooled corpus
          excerpts at <code>p = 1</code> — the run the reference numbers come from — before
          reading a direction into this.
        {:else}
          <b>{nats(unknownForm.nats)} nats</b> to one that has them — against
          <b>{nats(wrongContent?.nats)} nats</b>
          for simply saying the wrong thing{#if formShare}, i.e. {formShare} of the total damage{/if}.
          {#if formCostsLess}
            Even where a location exists, losing it costs far less than losing the content.
          {:else}
            On this sample the form cost at least as much as the content did — the reverse of the
            ordering the pooled corpus run finds, so read it as this sample's result rather than
            as the finding.
          {/if}
        {/if}
      </p>

      <table class="rows" data-testid="arch-vac-table">
        <thead>
          <tr>
            <th>variant</th>
            <th>nllPreserved</th>
            <th>nllAll</th>
            <th>bitsPerChar</th>
            <th>nTokens</th>
            <th>nPreservedTokens</th>
          </tr>
        </thead>
        <tbody>
          {#each result.variants as v (v.id)}
            <tr data-testid={`arch-vac-row-${v.id}`}>
              <th class="vname">{v.id}</th>
              <td class="mono">{nats(v.pooled.nllPreserved)}</td>
              <td class="mono">{nats(v.pooled.nllAll)}</td>
              <td class="mono">{num(v.pooled.bitsPerChar)}</td>
              <td class="mono">{int(v.pooled.nTokens)}</td>
              <td class="mono">{int(v.pooled.nPreservedTokens)}</td>
            </tr>
          {/each}
        </tbody>
      </table>

      {#if absoluteRefusal}
        <StaticNotice message={absoluteRefusal.message} testid="arch-vac-refused-absolute" />
      {/if}
      {#if result.passagesRefused}
        <StaticNotice
          message={result.passagesRefused.message}
          testid="arch-vac-refused-passages"
        />
      {/if}

      <!--
        A secondary row is still a number with an error bar, and it gets EVERY term of it.
        This row rendered `nll(nonce) − nll(english) = 0.879 ± 0.074` on the live static
        site while ignoring the `quantizationUncertaintyNats: 0.2` the same response
        carried — and the interval it stated, [0.805, 0.953], excludes the float32 truth
        0.9892. FR-720a: a stated ± that was never measured is a fabricated error bar and
        is worse than no number; so is a measured one that was dropped. The two headline
        cards above render both terms, and so does this.
      -->
      {#each secondary as d (d.id)}
        <p class="secondary" data-testid={`arch-vac-${d.id}`}>
          <span class="sname">{d.label}</span>
          <span class="mono" data-testid={`arch-vac-${d.id}-err`}>{secondaryLine(d)}</span>
          <span class="snote">{d.note}</span>
        </p>
      {/each}

      <ul class="honesty" data-testid="arch-vac-honesty">
        <li>
          <b>The remainder is not pure location.</b>
          {headline.find((d) => d.id === "unknown_form")?.note ?? ""}
        </li>
        <li><b>The confound.</b> {result.confound}</li>
        <li>
          <b>Alignment.</b> Tokens are attributed to words by {result.alignment.mechanism} ({result
            .alignment.unit}). {result.alignment.note}
        </li>
        <li>
          <b>What ran.</b>
          {result.model_id} on the <code>{result.stack}</code> stack at
          <code>{result.dtype}</code>{#if result.device}
            / <code>{result.device}</code>{/if}, p = {result.p}, seed = {result.seed}.
        </li>
      </ul>

      <div class="previews">
        {#each result.variants as v (v.id)}
          <figure data-testid={`arch-vac-preview-${v.id}`}>
            <figcaption>{v.id}</figcaption>
            <pre>{v.preview}</pre>
          </figure>
        {/each}
      </div>
    </div>
  {/if}

  <Explain
    title="How this number is produced"
    hint="one forward pass per variant, scored on the scaffolding only"
    testid="arch-vac-explain"
  >
    <p>
      Each variant is tokenized as written — no special tokens, no chat template — and scored in a
      single teacher-forced forward pass. Position <i>t</i> contributes
      <code>−log p(x_t | x_&lt;t)</code>; position 0 has no prediction and is excluded rather than
      counted as zero. <code>nllPreserved</code> averages only over tokens belonging to
      <b>preserved</b> words: words that are character-identical in all three variants. Those words
      tokenize identically everywhere, so the preserved token lists correspond one-for-one across
      variants, and each difference is a <b>paired</b> mean — the same function word compared with
      itself in the other condition — which is what makes a standard error on an effect of a
      tenth of a nat worth printing.
    </p>
    <p>
      Tokens are attributed to words in <b>UTF-8 byte</b> coordinates, never characters. The
      tokenizer's byte-level pieces are decoded back to bytes and the concatenation is checked
      against the passage byte for byte; a mismatch raises rather than mis-attributing. This is
      not fussiness: the browser tokenizer exposes no offsets at all, decoding tokens one at a
      time destroys multi-byte characters, and Python's character indices and JavaScript's UTF-16
      indices disagree about the same string. Bytes are the only unit both stacks can mean the
      same thing by. The passage is NFC-normalized once up front, because one curated tokenizer
      normalizes internally and the others do not.
    </p>
    <p>
      <code>bitsPerChar</code> is <code>nllAll · nTokens / (ln 2 · nChars)</code>, so it is
      comparable across variants whose tokenizations differ in length — which is exactly what
      happens when nonce forms fragment.
    </p>
    <p>
      For the transform itself — nesting, stability, the invariance theorem and what the swap
      control does and does not isolate — see the
      <button class="linklike" onclick={() => view.set("info")}>Info tab</button>.
    </p>
  </Explain>
</section>

<style>
  .vac {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    border-top: 1px solid var(--border);
    padding-top: 0.9rem;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.95rem;
  }
  .sub {
    font-size: 0.74rem;
    color: var(--text-dim);
  }
  .lede {
    margin: 0;
    font-size: 0.82rem;
    line-height: 1.6;
    color: var(--text-dim);
    max-width: 78ch;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    flex-wrap: wrap;
  }
  .scope {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.76rem;
    color: var(--text-dim);
  }
  .num {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.76rem;
    color: var(--text-dim);
  }
  .num input {
    width: 4.6rem;
    font-family: var(--mono);
    font-size: 0.76rem;
    padding: 0.2rem 0.35rem;
    border-radius: 7px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
  }
  .run {
    border-radius: 8px;
    border: 1px solid var(--accent);
    background: rgba(110, 168, 254, 0.12);
    color: var(--accent);
    font-size: 0.78rem;
    padding: 0.3rem 0.95rem;
  }
  .run:disabled {
    opacity: 0.55;
  }
  .passage {
    width: 100%;
    resize: vertical;
    font-family: var(--mono);
    font-size: 0.72rem;
    line-height: 1.55;
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
  }
  .hint {
    margin: 0;
    font-size: 0.7rem;
    color: var(--text-dim);
  }
  .running {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.74rem;
    color: var(--text-dim);
  }
  .bar {
    flex: 0 0 8rem;
    height: 4px;
    border-radius: 999px;
    background: rgba(110, 168, 254, 0.15);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    width: 40%;
    border-radius: 999px;
    background: var(--accent);
    animation: sweep 1.1s ease-in-out infinite;
  }
  @keyframes sweep {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(250%);
    }
  }
  .err {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    line-height: 1.5;
  }
  .err .actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .err .actions button {
    background: transparent;
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.5);
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.74rem;
  }
  .pnote {
    margin: 0;
    font-size: 0.72rem;
    line-height: 1.55;
    color: var(--text-dim);
    max-width: 84ch;
  }
  .pnote code {
    font-family: var(--mono);
  }
  .grid {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .pair {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
    gap: 0.7rem;
  }
  .delta {
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.7rem 0.85rem;
    background: linear-gradient(180deg, rgba(110, 168, 254, 0.06), transparent 70%);
  }
  .delta.tiny {
    background: linear-gradient(180deg, rgba(126, 231, 180, 0.07), transparent 70%);
  }
  .dlabel {
    font-size: 0.78rem;
    font-weight: 600;
  }
  .dexpr {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-dim);
  }
  .dvalue {
    font-family: var(--mono);
    font-size: 1.45rem;
    line-height: 1.2;
    margin-top: 0.15rem;
  }
  .dvalue.exact {
    color: var(--good, #7ee7b4);
  }
  .unit {
    font-size: 0.68rem;
    color: var(--text-dim);
    margin-left: 0.35rem;
  }
  .derr,
  .bound {
    font-size: 0.68rem;
    color: var(--text-dim);
  }
  .bound {
    font-style: italic;
  }
  .twobytwo {
    margin: 0;
    font-size: 0.8rem;
    line-height: 1.6;
    max-width: 78ch;
  }
  .qnote {
    margin: 0;
    font-size: 0.72rem;
    line-height: 1.55;
    max-width: 78ch;
    color: var(--text-dim);
    border-left: 2px solid var(--border);
    padding-left: 0.6rem;
  }
  .rows {
    border-collapse: collapse;
    font-size: 0.74rem;
    width: 100%;
  }
  .rows th,
  .rows td {
    border-bottom: 1px solid var(--border);
    padding: 0.28rem 0.5rem;
    text-align: right;
  }
  .rows thead th {
    color: var(--text-dim);
    font-weight: 500;
    font-size: 0.68rem;
  }
  .rows .vname {
    text-align: left;
    font-family: var(--mono);
  }
  .mono {
    font-family: var(--mono);
  }
  .secondary {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: baseline;
    font-size: 0.72rem;
    color: var(--text-dim);
  }
  .sname {
    font-weight: 600;
  }
  .snote {
    flex: 1 1 22rem;
  }
  .honesty {
    margin: 0;
    padding-left: 1.1rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    font-size: 0.74rem;
    line-height: 1.55;
    color: var(--text-dim);
    max-width: 84ch;
  }
  .previews {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
    gap: 0.6rem;
  }
  figure {
    margin: 0;
  }
  figcaption {
    font-size: 0.68rem;
    color: var(--text-dim);
    margin-bottom: 0.2rem;
    font-family: var(--mono);
  }
  pre {
    margin: 0;
    max-height: 9rem;
    overflow: auto;
    white-space: pre-wrap;
    font-size: 0.66rem;
    line-height: 1.5;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.6rem;
    background: var(--panel);
  }
  .linklike {
    background: none;
    border: none;
    padding: 0;
    color: var(--accent);
    font: inherit;
    cursor: pointer;
    text-decoration: underline;
  }
</style>
