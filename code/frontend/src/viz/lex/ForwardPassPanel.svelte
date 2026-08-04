<script lang="ts">
  /**
   * US-7 — step the forward pass.
   *
   * Everything on this panel comes from `traceForward`, which re-presents the SAME
   * `LexModel.forward` the rest of the tab runs. There is no illustrative pattern here and
   * no code path that could produce one: with no corpus there is no model and the panel
   * says so, and when the model is untrained its trace is labelled as an untrained
   * model's rather than dressed up as a finding.
   *
   * Two lessons are inherited from the Architecture tab's trace panel, which learned both
   * the hard way:
   *
   *   * the playhead must not overwrite a layer the user chose. "follow playhead" is an
   *     explicit mode that unticks itself the moment you pick a layer yourself;
   *   * a chart that clips must SAY it clips. The residual-norm scale is the largest
   *     non-outlier norm, and bars past it are drawn striped and counted — a red-team
   *     finding was that scaling to a single huge value flattened every other bar to the
   *     floor and the panel silently showed nothing.
   */
  import { onDestroy } from "svelte";

  import { LexModel, LexVocab } from "../../lib/lexEngine";
  import { traceForward, type LexTrace, type TraceStage } from "../../lib/lexEngine/trace";
  import Explain from "../../lib/Explain.svelte";
  import MatrixHeatmap from "../../lib/MatrixHeatmap.svelte";
  import { hideTip, showTip } from "../../lib/tooltip";
  import type { Provenance } from "./provenance";

  interface Props {
    model: LexModel | null;
    vocab: LexVocab | null;
    /** What these weights are: untrained, trained, or either of those hand-edited. */
    provenance: Provenance;
  }
  let { model, vocab, provenance }: Props = $props();

  /** One phrase per state: the trace is real in all four, of a different model in each. */
  const sourceHint = $derived(
    {
      trained: "the trained model's own activations, stage by stage",
      untrained: "the untrained model's own activations — real weights, real trace, random patterns",
      "edited-trained": "hand-edited weights' own activations — real trace, not the trained model's",
      "edited-untrained":
        "hand-edited weights over an untrained model — real trace, edited weights, random patterns",
    }[provenance],
  );

  /** In budget in every Dolch list and in any frequency budget of the shipped corpus. */
  const DEFAULT_PROMPT = "the little";

  let prompt = $state(DEFAULT_PROMPT);

  /**
   * Trace and error travel together in ONE derived. Svelte 5 forbids writing `$state`
   * from inside a `$derived`, and splitting them into a derived plus an effect would let
   * a stale error outlive the trace it described for a frame.
   */
  const traced = $derived.by<{ trace: LexTrace | null; error: string }>(() => {
    if (!model || !vocab) return { trace: null, error: "" };
    try {
      return { trace: traceForward(model, vocab, { prompt }), error: "" };
    } catch (e) {
      return { trace: null, error: e instanceof Error ? e.message : String(e) };
    }
  });
  const trace = $derived(traced.trace);
  const error = $derived(traced.error);

  // ---- playback ----------------------------------------------------------------------

  let idx = $state(0);
  let playing = $state(false);
  let speed = $state(1);
  let layerSel = $state(0);
  let headSel = $state(0);
  /** See the header comment: the playhead follows until the user picks a layer. */
  let followLayer = $state(true);
  let raf = 0;
  let lastTs = 0;
  let acc = 0;

  onDestroy(() => {
    hideTip();
    cancelAnimationFrame(raf);
  });

  const stages = $derived(trace?.stages ?? []);

  // A new trace (a new prompt, or a model that just finished training) rewinds.
  let lastTrace: LexTrace | null = null;
  $effect(() => {
    const t = trace;
    if (t !== lastTrace) {
      lastTrace = t;
      idx = 0;
      playing = false;
      acc = 0;
      lastTs = 0;
      layerSel = Math.min(layerSel, Math.max(0, (t?.nLayers ?? 1) - 1));
      headSel = Math.min(headSel, Math.max(0, (t?.nHeads ?? 1) - 1));
    }
  });

  /** One stage per second at 1×: a 2-layer trace is 6 stages, so ~6 s end to end. */
  const STAGES_PER_SEC = 1;
  $effect(() => {
    if (!playing) {
      cancelAnimationFrame(raf);
      lastTs = 0;
      return;
    }
    const tick = (ts: number) => {
      if (lastTs) {
        acc += ((ts - lastTs) / 1000) * STAGES_PER_SEC * speed;
        const n = Math.floor(acc);
        if (n > 0) {
          acc -= n;
          idx = Math.min(idx + n, stages.length - 1);
          if (idx >= stages.length - 1) {
            playing = false;
            return;
          }
        }
      }
      lastTs = ts;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  });

  const stage = $derived<TraceStage | null>(stages[Math.min(idx, stages.length - 1)] ?? null);

  /** Follow the playhead's layer — until the user says otherwise. */
  $effect(() => {
    const l = stage?.layer;
    if (followLayer && l !== null && l !== undefined) layerSel = l;
  });

  function pickLayer(v: number): void {
    followLayer = false;
    layerSel = v;
  }

  function togglePlay(): void {
    if (stages.length === 0) return;
    if (!playing && idx >= stages.length - 1) {
      idx = 0;
      acc = 0;
    }
    lastTs = 0;
    playing = !playing;
  }

  function stepBy(n: number): void {
    playing = false;
    idx = Math.max(0, Math.min(stages.length - 1, idx + n));
  }

  function scrub(v: number): void {
    playing = false;
    idx = v;
  }

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

  const SPEEDS = [0.5, 1, 2, 4];

  // ---- per-layer detail ---------------------------------------------------------------

  /** The attention stage of the SELECTED layer — not necessarily the playhead's. */
  const attnStage = $derived(
    stages.find((s) => s.kind === "attention" && s.layer === layerSel) ?? null,
  );
  const heads = $derived(attnStage?.attention?.length ?? 0);
  const headIdx = $derived(Math.min(headSel, Math.max(0, heads - 1)));
  const attn = $derived(attnStage?.attention?.[headIdx] ?? null);
  const tokenLabels = $derived((trace?.tokens ?? []).map((t) => t.word));

  /**
   * Bar scale: the largest NON-outlier norm (outliers being past 8× the median), so one
   * huge position cannot flatten the rest. Anything above it is drawn striped and counted.
   */
  const normScale = $derived.by(() => {
    const vals = (stage?.residualNorm ?? []).filter((v) => Number.isFinite(v) && v > 0);
    if (vals.length === 0) return 1;
    const sorted = [...vals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] || sorted[0];
    const inliers = vals.filter((v) => v <= median * 8);
    const m = Math.max(...(inliers.length > 0 ? inliers : vals));
    return m > 0 ? m : 1;
  });
  const offScale = $derived(
    (stage?.residualNorm ?? []).reduce((n, v) => (v > normScale ? n + 1 : n), 0),
  );
  const maxProb = $derived(Math.max(...(stage?.lens.probs ?? [1]), 1e-9));
</script>

<div class="panel-body" data-testid="lex-forward">
  <div class="head">
    <h3>Step the forward pass</h3>
    <span class="hint">{sourceHint}</span>
  </div>

  {#if !model || !vocab}
    <p class="empty" data-testid="lex-forward-empty">
      There is no model to trace yet — the corpus has not loaded, so no vocabulary and no
      weights exist. Nothing is drawn here in the meantime: an invented attention pattern
      would look exactly like a real one.
    </p>
  {:else}
    {#if provenance === "untrained"}
      <p class="untrained" data-testid="lex-forward-untrained">
        <b>Nothing has been trained yet.</b> What follows is a real trace of the
        <b>random-init</b> model at this shape — the attention maps and the readout below are
        genuinely what these weights compute, and they are genuinely meaningless. Train
        above, and step this again.
      </p>
    {:else if provenance === "edited-untrained"}
      <p class="untrained" data-testid="lex-forward-untrained">
        <b>Nothing has been trained yet, and the weights have been hand-edited.</b> What
        follows is a real trace of the <b>random-init model with your edit applied</b> — the
        attention maps and the readout below are genuinely what those weights compute, and
        they are genuinely meaningless. Train above, or restore in the Weight Lab, and step
        this again.
      </p>
    {:else if provenance === "edited-trained"}
      <p class="untrained" data-testid="lex-forward-edited">
        <b>These weights have been hand-edited.</b> The trace below is real, but it is the
        trace of the edited weights — not of the model you trained. The Weight Lab's restore
        button brings that model back.
      </p>
    {/if}

    <label class="prompt">
      <span class="ctl-label">prompt</span>
      <input
        type="text"
        data-testid="lex-forward-prompt"
        bind:value={prompt}
        placeholder="a few budget words — &lt;bos&gt; is prepended for you"
      />
    </label>

    {#if error}
      <div class="err" data-testid="lex-forward-error">{error}</div>
    {/if}

    {#if trace}
      <!-- the sequence the model is really given, <unk> included and visible -->
      <div class="strip" data-testid="lex-forward-strip">
        {#if trace.truncated}
          <span class="chip trunc" title={`the prompt was longer than ctx, so its first ${trace.droppedTokens} tokens were dropped`}>
            {trace.droppedTokens} earlier token{trace.droppedTokens === 1 ? "" : "s"} dropped ⋯
          </span>
        {/if}
        {#each trace.tokens as t (t.position)}
          <span
            class="chip tok"
            class:oov={!t.inBudget && !t.special}
            class:special={t.special}
            role="note"
            aria-label={`position ${t.position}: ${t.word}${t.inBudget ? "" : " — not in budget, read as <unk>"} (id ${t.id})`}
            onmousemove={(e) =>
              showTip(e, `position ${t.position} · id ${t.id}` + (t.inBudget ? "" : " · <unk>"))}
            onmouseleave={hideTip}
          >
            <span class="tword">{t.word}</span>
            <span class="tid">{t.inBudget ? t.id : "unk"}</span>
          </span>
        {/each}
      </div>

      {#if trace.unkCount > 0}
        <p class="oov-note" data-testid="lex-forward-oov">
          <b>{trace.unkCount}</b> prompt token{trace.unkCount === 1 ? "" : "s"} outside this
          budget entered the model as <code>&lt;unk&gt;</code>:
          <b>{trace.unkWords.slice(0, 12).join(", ")}</b>{trace.unkWords.length > 12
            ? `, and ${trace.unkWords.length - 12} more`
            : ""}. The model has no row for
          {trace.unkWords.length === 1 ? "it" : "them"}, so the pass below never sees the
          word — only the hole where it was.
        </p>
      {/if}

      <!-- transport -->
      <div class="player">
        <button
          class="tbtn"
          data-testid="lex-forward-prev"
          onclick={() => stepBy(-1)}
          disabled={idx === 0}
          aria-label="previous stage">◀</button
        >
        <button
          class="tbtn play"
          data-testid="lex-forward-play"
          onclick={togglePlay}
          aria-label={playing ? "pause the forward pass" : "play the forward pass"}
        >{playing ? "❚❚" : "▶"}</button>
        <button
          class="tbtn"
          data-testid="lex-forward-next"
          onclick={() => stepBy(1)}
          disabled={idx >= stages.length - 1}
          aria-label="next stage">▶</button
        >
        <input
          class="scrub"
          type="range"
          min="0"
          max={Math.max(0, stages.length - 1)}
          value={idx}
          aria-label="scrub through the forward pass"
          oninput={(e) => scrub(Number(e.currentTarget.value))}
        />
        <div
          class="seg"
          role="radiogroup"
          tabindex="-1"
          aria-label="playback speed"
          data-testid="lex-forward-speed"
          onkeydown={(e) => segKey(e, (v) => (speed = Number(v)))}
        >
          {#each SPEEDS as s (s)}
            <button
              role="radio"
              aria-checked={speed === s}
              tabindex={speed === s ? 0 : -1}
              data-value={String(s)}
              class:active={speed === s}
              onclick={() => (speed = s)}
            >{s}×</button>
          {/each}
        </div>
      </div>

      {#if stage}
        <div class="playhead" data-testid="lex-forward-playhead">
          <span class="ph-step">stage {idx + 1}/{stages.length}</span>
          <span class="ph-label">{stage.label}</span>
          <code class="ph-detail">{stage.detail}</code>
        </div>

        <div class="detail-grid">
          <div class="cell">
            <div class="cell-head">
              <span class="cell-label">
                attention · layer {layerSel} · all {heads} head{heads === 1 ? "" : "s"} · rows attend to columns
              </span>
              <label class="follow" title="while the pass plays, show whichever layer the current stage belongs to">
                <input
                  type="checkbox"
                  data-testid="lex-forward-follow"
                  checked={followLayer}
                  onchange={(e) => (followLayer = e.currentTarget.checked)}
                />
                <span class="dlabel">follow playhead</span>
              </label>
            </div>
            {#if trace.nLayers > 1}
              <div
                class="seg wrap"
                role="radiogroup"
                tabindex="-1"
                aria-label="attention layer"
                data-testid="lex-forward-layer"
                onkeydown={(e) => segKey(e, (v) => pickLayer(Number(v)))}
              >
                {#each Array.from({ length: trace.nLayers }, (_, l) => l) as l (l)}
                  <button
                    role="radio"
                    aria-checked={layerSel === l}
                    tabindex={layerSel === l ? 0 : -1}
                    data-value={String(l)}
                    class:active={layerSel === l}
                    onclick={() => pickLayer(l)}
                  >layer {l}</button>
                {/each}
              </div>
            {/if}
            {#if attnStage?.attention}
              <div class="heads" data-testid="lex-forward-heads">
                {#each attnStage.attention as m, h (h)}
                  <button
                    class="headtile"
                    class:sel={h === headIdx}
                    data-testid={`lex-forward-head-${h}`}
                    aria-label={`attention head ${h} of layer ${layerSel}`}
                    aria-pressed={h === headIdx}
                    onclick={() => (headSel = h)}
                  >
                    <MatrixHeatmap values={m} maxCanvasPx={64} />
                    <span class="headnum">{h}</span>
                  </button>
                {/each}
              </div>
              <span class="cell-label">head {headIdx} · each row sums to 1 over the past</span>
              {#if attn}
                <div class="hmwrap">
                  <MatrixHeatmap
                    values={attn}
                    rowLabels={tokenLabels}
                    colLabels={tokenLabels}
                    maxCanvasPx={252}
                  />
                </div>
              {/if}
            {/if}
          </div>

          <div class="cell">
            <span class="cell-label">
              ‖residual stream‖ per position · {stage.label}
              {#if offScale > 0}
                <span
                  class="offscale-note"
                  data-testid="lex-forward-offscale"
                  title="The scale is the largest non-outlier norm, so one big position cannot flatten the rest. Bars past it are striped and clipped rather than silently squashing everything else."
                >· {offScale} off-scale ↑</span>
              {/if}
            </span>
            <div class="bars">
              {#each stage.residualNorm as v, i (i)}
                <div
                  class="bar"
                  class:over={v > normScale}
                  role="img"
                  aria-label={`${tokenLabels[i] ?? `#${i}`}: norm ${v.toFixed(2)}${v > normScale ? " (off scale)" : ""}`}
                  style:height={`${Math.max(4, Math.min(100, (v / normScale) * 100))}%`}
                  onmousemove={(e) =>
                    showTip(
                      e,
                      `${tokenLabels[i] ?? `#${i}`} · ‖h‖ = ${v.toFixed(3)}` +
                        (v > normScale ? " · off scale (clipped)" : ""),
                    )}
                  onmouseleave={hideTip}
                ></div>
              {/each}
            </div>

            <span class="cell-label" data-testid="lex-forward-lens-label">
              next-token readout here
              {#if stage.lens.exact}
                <span class="tag exact">exact</span>
              {:else}
                <span class="tag approx">logit lens · approximate</span>
              {/if}
            </span>
            <div class="logits" data-testid="lex-forward-lens">
              {#each stage.lens.ids as id, i (id)}
                <div class="lrow">
                  <code class="ltok">{stage.lens.words[i]}</code>
                  <div class="lbarwrap">
                    <div class="lbar" style:width={`${(stage.lens.probs[i] / maxProb) * 100}%`}></div>
                  </div>
                  <span class="lpct">{(stage.lens.probs[i] * 100).toFixed(1)}%</span>
                </div>
              {/each}
            </div>
            <p class="lens-note">
              {#if stage.lens.exact}
                This model applies its final LayerNorm to exactly this state, so here the
                readout <b>is</b> the model's distribution — a plain softmax over the real
                logits, with no temperature and none of generation's
                <code>&lt;unk&gt;</code>/<code>&lt;bos&gt;</code>/<code>&lt;pad&gt;</code>
                masking, which belongs to the sampler rather than to the model.
              {:else}
                An <b>approximation</b>, not a prediction. This state has not been through
                the remaining {stages.length - 1 - idx} stage{stages.length - 1 - idx === 1
                  ? ""
                  : "s"} of the pass, and the final LayerNorm it is read through was fitted
                to the statistics of the <i>last</i> layer's residual stream, not to this
                one's. Read it as "what this state most resembles", never as what the model
                would predict.
              {/if}
            </p>
          </div>
        </div>
      {/if}
    {/if}

    <Explain
      title="What stepping this actually shows"
      hint="the same forward pass the rest of the tab runs, paused"
      testid="lex-explain-forward"
    >
      <p>
        The stages are <code>embed</code>, then for each of the {trace?.nLayers ?? "L"}
        layers its <b>attention</b> residual and its <b>MLP</b> residual, then the final
        LayerNorm and readout. They are not a re-implementation: the panel calls the same
        <code>forward</code> that training and generation call, in eval mode with dropout
        off, and reads the intermediates it already keeps. The final logits shown here and
        the logits the model generates from are the same array, which a unit test pins to
        1e-9 against an independently-run forward pass.
      </p>
      <p>
        <b>Attention</b> is <code>softmax(q kᵀ / √dh)</code> under a causal mask, so each
        row sums to 1 and everything strictly above the diagonal is exactly 0 — a position
        can attend to itself and its past, never to its future. That triangle is the mask,
        visible.
      </p>
      <p>
        <b>The readout at an intermediate stage is a logit lens</b>, and it is an
        approximation for two reasons worth separating. The state has not passed through
        the remaining layers; and this model ends with a <b>final LayerNorm</b> whose gain
        and bias were learned against the last layer's residual statistics, so applying it
        to an earlier layer's state re-scales something it was never fitted to. Only the
        last layer's output — which <i>is</i> the final LayerNorm's input — reads out
        exactly. The panel marks each stage <span class="tag exact">exact</span> or
        <span class="tag approx">approximate</span> rather than letting you guess.
      </p>
      <p>
        Out-of-budget prompt words are shown as what they become: a
        <code>&lt;unk&gt;</code> row. Nothing is dropped quietly — the model genuinely has
        no row for a word outside the budget, and the pass you are stepping through never
        sees it.
      </p>
    </Explain>
  {/if}
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
  .empty {
    margin: 0;
    color: var(--text-dim);
    font-size: 0.8rem;
    line-height: 1.55;
  }
  .untrained {
    margin: 0;
    font-size: 0.74rem;
    line-height: 1.55;
    color: #ffb454;
    background: rgba(255, 180, 84, 0.1);
    border: 1px solid rgba(255, 180, 84, 0.3);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
  }
  .untrained b {
    color: var(--text);
  }
  .prompt {
    display: flex;
    flex-direction: column;
    gap: 0.24rem;
    min-width: 0;
  }
  .prompt input {
    font-family: var(--mono);
    font-size: 0.84rem;
  }
  .ctl-label {
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .err {
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
  }
  .strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.28rem;
    align-items: center;
  }
  .chip {
    font-family: var(--mono);
    font-size: 0.7rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.14rem 0.42rem;
    background: var(--bg-elev-2);
    color: var(--text);
  }
  .chip.tok {
    display: inline-flex;
    align-items: baseline;
    gap: 0.3rem;
    cursor: default;
  }
  .chip.special {
    color: var(--accent-2);
    border-color: rgba(183, 148, 246, 0.35);
    background: rgba(183, 148, 246, 0.12);
  }
  /* Out of budget is never silent: the chip changes colour and the id column says unk. */
  .chip.oov {
    color: #ffb454;
    border-color: rgba(255, 180, 84, 0.45);
    background: rgba(255, 180, 84, 0.12);
  }
  .chip.trunc {
    color: #ffb454;
    border-color: rgba(255, 180, 84, 0.35);
    background: rgba(255, 180, 84, 0.12);
  }
  .tword {
    white-space: pre;
  }
  .tid {
    font-size: 0.6rem;
    color: var(--text-dim);
  }
  .oov-note {
    margin: 0;
    font-size: 0.72rem;
    line-height: 1.55;
    color: #ffb454;
  }
  .oov-note code {
    font-family: var(--mono);
  }
  .player {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .tbtn {
    width: 36px;
    height: 30px;
    padding: 0;
    font-size: 0.72rem;
    flex-shrink: 0;
  }
  .tbtn.play {
    width: 42px;
  }
  .tbtn:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .scrub {
    flex: 1;
    min-width: 6rem;
  }
  .seg {
    display: flex;
    gap: 0.25rem;
    flex-shrink: 0;
  }
  .seg.wrap {
    flex-wrap: wrap;
  }
  .seg button {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.22rem 0.5rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-dim);
    cursor: pointer;
  }
  .seg button:hover {
    border-color: var(--accent);
  }
  .seg button.active {
    color: var(--text);
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }
  .playhead {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    flex-wrap: wrap;
    font-size: 0.74rem;
    min-height: 1.2em;
  }
  .ph-step {
    font-family: var(--mono);
    color: var(--accent);
  }
  .ph-label {
    font-weight: 600;
  }
  .ph-detail {
    font-family: var(--mono);
    font-size: 0.66rem;
    color: var(--text-dim);
    overflow-wrap: anywhere;
  }
  .detail-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(15rem, 21rem);
    gap: 1.1rem;
    align-items: start;
  }
  @media (max-width: 900px) {
    .detail-grid {
      grid-template-columns: 1fr;
    }
  }
  .cell {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    min-width: 0;
  }
  .cell-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .cell-label {
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .follow {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    cursor: pointer;
  }
  .follow input {
    width: auto;
    margin: 0;
    accent-color: var(--accent);
  }
  .dlabel {
    font-size: 0.7rem;
    color: var(--text-dim);
  }
  /* Fixed-width tiles that wrap, NOT `1fr` columns: this model has 1, 2 or 4 heads, and
     stretching two of them across the full width made each tile a wide empty box with a
     64px map adrift inside it. Every head is on screen either way — nothing scrolls. */
  .heads {
    display: grid;
    grid-template-columns: repeat(auto-fit, 70px);
    justify-content: start;
    gap: 0.3rem;
  }
  .headtile {
    position: relative;
    display: flex;
    justify-content: center;
    padding: 2px;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    line-height: 0;
    cursor: pointer;
  }
  .headtile:hover {
    border-color: var(--accent);
  }
  .headtile.sel {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }
  .headnum {
    position: absolute;
    bottom: 2px;
    right: 3px;
    font-family: var(--mono);
    font-size: 0.58rem;
    line-height: 1;
    color: var(--text);
    background: rgba(11, 14, 20, 0.72);
    border-radius: 3px;
    padding: 0 2px;
  }
  /* 390px: a long prompt's heatmap scrolls inside itself, never sideways off the page. */
  .hmwrap {
    max-width: 100%;
    overflow-x: auto;
  }
  .bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 68px;
    background: rgba(0, 0, 0, 0.22);
    border-radius: 8px;
    padding: 4px;
  }
  .bar {
    flex: 1;
    min-width: 2px;
    border-radius: 2px 2px 0 0;
    background: linear-gradient(180deg, var(--accent) 0%, rgba(110, 168, 254, 0.35) 100%);
    transition: height 0.25s ease;
  }
  .bar:hover {
    background: var(--accent-2);
  }
  /* Striped = "taller than shown", so a clipped bar cannot read as a tie with the top. */
  .bar.over {
    background: repeating-linear-gradient(
      -45deg,
      var(--accent-2) 0 3px,
      rgba(183, 148, 246, 0.35) 3px 6px
    );
  }
  .offscale-note {
    color: #ffb454;
    text-transform: none;
    letter-spacing: 0;
    cursor: help;
  }
  .tag {
    font-family: var(--mono);
    font-size: 0.62rem;
    text-transform: none;
    letter-spacing: 0;
    border-radius: 999px;
    padding: 0.08rem 0.42rem;
    /* A pill that breaks across two lines reads as a broken border rather than a badge:
       at 390px it drops onto its own line whole instead. */
    white-space: nowrap;
  }
  .tag.exact {
    color: var(--good);
    background: rgba(91, 224, 176, 0.12);
    border: 1px solid rgba(91, 224, 176, 0.35);
  }
  .tag.approx {
    color: #ffb454;
    background: rgba(255, 180, 84, 0.12);
    border: 1px solid rgba(255, 180, 84, 0.35);
  }
  .logits {
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
  }
  .lrow {
    display: grid;
    grid-template-columns: minmax(48px, auto) minmax(0, 1fr) 46px;
    align-items: center;
    gap: 0.5rem;
  }
  .ltok {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text);
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .lbarwrap {
    height: 9px;
    border-radius: 999px;
    background: var(--bg-elev-2);
    overflow: hidden;
  }
  .lbar {
    height: 100%;
    border-radius: 999px;
    background: var(--accent-grad);
    transition: width 0.3s ease;
  }
  .lpct {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    text-align: right;
  }
  .lens-note {
    margin: 0;
    font-size: 0.7rem;
    line-height: 1.55;
    color: var(--text-dim);
  }
  .lens-note b {
    color: var(--text);
  }
  .lens-note code {
    font-family: var(--mono);
    color: var(--accent);
  }
</style>
