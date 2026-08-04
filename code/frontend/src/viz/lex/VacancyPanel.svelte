<script lang="ts">
  /**
   * The vacancy transform, live on the corpus this tab trains on (feature 007, ui.md §1).
   *
   * What it is for: a word can carry meaning two ways. Its FORM can be known — you have
   * seen `crow` before and something about the string is already yours — or its meaning
   * can be entirely FIELD, fixed by the company it keeps. This panel manufactures the
   * second condition on demand: closed-class scaffolding, inflection, punctuation and
   * line structure survive byte for byte, while a controlled fraction `p` of open-class
   * STEMS is replaced by phonotactically legal nonce forms carrying the stem's syllable
   * count and stress.
   *
   * Everything below is computed here, in this browser, by `lib/lexEngine/vacancy.ts` —
   * the TypeScript half of `specs/007-vacancy-transform-field/architecture.md`, which the
   * Python backend implements too. No number in the prose is retyped: every one of them
   * is read out of `vacancyStats` or out of a source constant, so changing the constant
   * changes the sentence.
   *
   * THREE THINGS THIS PANEL HAS TO MAKE VISIBLE, and how:
   *
   *   * NESTING and STABILITY (FR-711) — the ribbon, not the prose. `u(stem)` is a hash of
   *     `(seed, stem)` alone, so `{vacated at p} ⊆ {vacated at p'}` for `p < p'`; and the
   *     map is built ONCE over the whole type set in canonical order, so a stem's nonce is
   *     the same string at every `p` where it is vacated. The ribbon shows both by showing
   *     the same eligible stems at five values of `p` side by side.
   *   * THE INVARIANCE THEOREM (FR-714) — as a live computation in two tiers. The instant
   *     tier compares the two token id streams element for element on every control
   *     change; the on-demand tier really trains twice and subtracts the loss curves. In
   *     a condition that breaks the theorem both tiers show the real, broken result. There
   *     is no hard-coded tick anywhere in this file.
   *   * THAT THE NULL IS THE FINDING (§1.6) — an exact zero drawn as a flat line looks
   *     like a broken chart. It is stated as the result it is, in words, next to the
   *     measurement.
   *
   * PROSODY HONESTY (FR-712 / SC-708): the stress table is rule-seeded and unverified, so
   * no prosody statistic is ever rendered without the three-way stress split beside it and
   * the sentence saying what fraction of this corpus's tokens the hand table actually
   * covers — itself a measured number, `stressFromTable*`.
   *
   * The type counts shown are `corpusTypes*`, never `domainTypes*` (contract §10): the
   * domain adds budget words that never appear in the text, and counting words the reader
   * cannot see inflates the vacancy rate they are being shown.
   */
  import { onDestroy } from "svelte";

  import {
    LexVocab,
    UNK_ID,
    WORD_RE,
    hasWord,
    splitLines,
    tokenStream,
    tokenize,
    trainInWorker,
  } from "../../lib/lexEngine";
  import {
    FUNCTION_WORDS,
    STRESS_TABLE,
    SUFFIXES,
    effectiveKeepSet,
    isEligible,
    stemAndSuffix,
    vacancyStats,
    vacancyU,
    type VacancyMap,
    type VacancyParams,
  } from "../../lib/lexEngine/vacancy";
  import Explain from "../../lib/Explain.svelte";
  import Progress from "../../lib/Progress.svelte";
  import { view } from "../../lib/stores";

  interface Props {
    /** The untransformed corpus — the `p = 0` reference for every comparison here. */
    corpusText: string;
    /** `vacateText(corpusText, map, params)`, computed by the tab (it also trains on it). */
    vacatedText: string;
    /** The `p`-independent nonce assignment. Null only before the corpus has loaded. */
    map: VacancyMap | null;
    params: VacancyParams;
    /** `V` — the budget resolved against the untransformed corpus. */
    baseVocab: LexVocab | null;
    /** `V_p` — mapped in the mapped condition, rebuilt from the vacated corpus otherwise. */
    vocab: LexVocab | null;
    /** "consistent" | "inconsistent" | "reveal" — `params` encodes it, this names it. */
    condition: string;
    /** Kept even while the condition is not `reveal`, so the input does not lose its value. */
    revealAfter: number;
    /** "nonce" | "swap" (contract §8.3). */
    mint: string;
    onP: (v: number) => void;
    onSeed: (v: number) => void;
    onCondition: (v: string) => void;
    onRevealAfter: (v: number) => void;
    onProsody: (v: boolean) => void;
    onMint: (v: string) => void;
  }
  let {
    corpusText,
    vacatedText,
    map,
    params,
    baseVocab,
    vocab,
    condition,
    revealAfter,
    mint,
    onP,
    onSeed,
    onCondition,
    onRevealAfter,
    onProsody,
    onMint,
  }: Props = $props();

  const CONDITIONS = [
    {
      id: "consistent",
      label: "consistent",
      title:
        "One nonce per source type, corpus-wide. The mapped condition — the invariance theorem holds here.",
    },
    {
      id: "inconsistent",
      label: "inconsistent",
      title:
        "A fresh nonce for every OCCURRENCE. Same vacancy rate, no learnable identity — the source's control.",
    },
    {
      id: "reveal",
      label: "partial reveal",
      title:
        "The first N occurrences of a vacated stem keep their English form, so the type is split in two.",
    },
  ];
  const MINTS = [
    {
      id: "nonce",
      label: "nonce",
      title: "Replace the stem with a phonotactically legal invented form.",
      enabled: true,
    },
    {
      id: "swap",
      label: "swap",
      title:
        "Contract §8.3: draw a real, frequency-rank-matched English word instead. Both engines implement it; this control is not wired to them yet.",
      enabled: false,
    },
  ];

  /** See BudgetPanel: arrows must move an ARIA radiogroup's selection and focus together. */
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

  // ---- statistics (contract §10) -------------------------------------------------------

  const keep = $derived(effectiveKeepSet(params.keep));
  const stats = $derived.by(() =>
    map && corpusText.length > 0 ? vacancyStats(corpusText, vacatedText, map, params) : null,
  );
  /** The mapped condition — the only one in which §7.3's theorem is claimed to hold. */
  const mapped = $derived(params.consistent && params.revealAfter === 0);

  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
  const n = (x: number) => x.toLocaleString();

  // ---- the corpus view (ui.md §1.2 — the doc's Figure 5, live) -------------------------

  const WINDOW_LINES = 40;

  /**
   * The shipped corpus opens with 618 token-producing lines of front matter — a title page,
   * a list of rhymes, and an index of first lines — before `LITTLE BO-PEEP` starts the verse
   * at line 619. Opening the reader on a table of contents makes the transform look like it
   * is rewriting an index, which is the least interesting thing it does.
   *
   * Pinned rather than detected, deliberately. Every general rule tried here picks the wrong
   * boundary: the index of first lines has verse-length lines (median 6 tokens), so a
   * line-length heuristic lands on line 320, still inside the front matter. Separating an
   * index from a stanza needs to know the book. The corpus is committed and digest-verified,
   * so a constant is the honest way to say that, and `vacancy.spec.ts` asserts the default
   * page really opens on Bo-Peep.
   *
   * It applies ONLY to the shipped corpus. Pasted text and HuggingFace datasets have no front
   * matter, so they open at line 1.
   */
  const SHIPPED_BODY_LINE = 619;
  const SHIPPED_LINE_COUNT = 3071;

  let windowIndex = $state(0);
  /** Once the reader pages, their choice wins over `defaultWindow`. */
  let userPaged = $state(false);

  /**
   * Page by `delta`, reading the CURRENT window before taking ownership of it.
   *
   * The order matters and is not obvious: setting `userPaged` first makes `win` fall straight
   * back to `windowIndex`, which is still 0 while the default is in force, so the first click
   * would jump to line 1 instead of stepping. Capture, then assign.
   */
  function page(delta: number): void {
    const from = win;
    windowIndex = from + delta;
    userPaged = true;
  }

  type Seg = { text: string; cls: "gap" | "kept" | "open" | "minted" };

  /** Raw `WORD_RE` matches with their offsets, case preserved. */
  function wordsOf(line: string): { text: string; at: number }[] {
    const re = new RegExp(WORD_RE.source, "g");
    const out: { text: string; at: number }[] = [];
    for (let m = re.exec(line); m !== null; m = re.exec(line)) out.push({ text: m[0], at: m.index });
    return out;
  }

  /**
   * One line, split into coloured runs. The classification comes from the REAL map — the
   * eligibility test of §2.2 and the transform's own output — never from a hand-annotated
   * list (FR-711). A word is `minted` iff the transform actually changed it, which is what
   * makes `revealAfter` and the inconsistent control show up here honestly rather than
   * being predicted from `u` alone.
   */
  function segmentsOf(orig: string, vac: string): Seg[] {
    const before = wordsOf(orig);
    const after = wordsOf(vac);
    const segs: Seg[] = [];
    let cursor = 0;
    for (let i = 0; i < before.length; i++) {
      const w = before[i];
      if (w.at > cursor) segs.push({ text: orig.slice(cursor, w.at), cls: "gap" });
      const out = after[i]?.text ?? w.text;
      const [stem] = stemAndSuffix(w.text.toLowerCase());
      const cls: Seg["cls"] =
        out.toLowerCase() !== w.text.toLowerCase()
          ? "minted"
          : isEligible(stem, keep)
            ? "open"
            : "kept";
      segs.push({ text: out, cls });
      cursor = w.at + w.text.length;
    }
    if (cursor < orig.length) segs.push({ text: orig.slice(cursor), cls: "gap" });
    return segs;
  }

  const origLines = $derived(splitLines(corpusText));
  const vacLines = $derived(splitLines(vacatedText));
  /** Indices of the token-producing lines — the ones `tokenStream` turns into training data. */
  const wordLines = $derived.by(() => {
    const out: number[] = [];
    for (let i = 0; i < origLines.length; i++) if (hasWord(origLines[i])) out.push(i);
    return out;
  });
  const nWindows = $derived(Math.max(1, Math.ceil(wordLines.length / WINDOW_LINES)));
  /**
   * Where the view opens. The shipped corpus is recognised by its line count — it is
   * digest-verified upstream, so this cannot be some other book of the same length — and
   * opens on the verse; anything else opens at line 1.
   */
  const defaultWindow = $derived(
    wordLines.length === SHIPPED_LINE_COUNT ? Math.floor(SHIPPED_BODY_LINE / WINDOW_LINES) : 0,
  );
  /** Clamped rather than reset by an effect: a shorter corpus must not strand the view. */
  const win = $derived(
    Math.min(Math.max(0, userPaged ? windowIndex : defaultWindow), nWindows - 1),
  );
  const shown = $derived.by(() => {
    const start = win * WINDOW_LINES;
    return wordLines.slice(start, start + WINDOW_LINES).map((i) => ({
      i,
      segs: segmentsOf(origLines[i], vacLines[i] ?? origLines[i]),
    }));
  });

  // ---- the nesting ribbon (ui.md §1.3) -------------------------------------------------

  const P_CELLS = [0, 0.25, 0.5, 0.75, 1] as const;
  const RIBBON_ROWS = 8;

  /** Eligible stems the reader can actually find in the text above. */
  const corpusStems = $derived.by(() => {
    const out = new Set<string>();
    for (const t of new Set(tokenize(corpusText))) {
      const [stem] = stemAndSuffix(t);
      if (isEligible(stem, keep)) out.add(stem);
    }
    return out;
  });

  /**
   * ~8 stems spanning the `u` range, each shown at five values of `p`. Chosen by rank in
   * `u` rather than by hand, so the row set is a function of the corpus and the seed.
   */
  const ribbon = $derived.by(() => {
    if (!map) return [];
    const rows = [...map.mapping.keys()]
      .filter((s) => corpusStems.has(s))
      .map((stem) => ({ stem, u: vacancyU(stem, params.seed), nonce: map.mapping.get(stem) ?? stem }))
      .sort((a, b) => a.u - b.u || (a.stem < b.stem ? -1 : 1));
    if (rows.length === 0) return [];
    const picked: typeof rows = [];
    for (let k = 0; k < RIBBON_ROWS; k++) {
      const idx = Math.round((k * (rows.length - 1)) / (RIBBON_ROWS - 1));
      const row = rows[Math.min(idx, rows.length - 1)];
      if (!picked.some((r) => r.stem === row.stem)) picked.push(row);
    }
    return picked.map((r) => ({
      ...r,
      cells: P_CELLS.map((pc) => ({ p: pc, vacated: r.u < pc, form: r.u < pc ? r.nonce : r.stem })),
    }));
  });

  // ---- tier 1: the instant invariance check (ui.md §1.5) -------------------------------

  /**
   * §7.3, computed rather than asserted, on every control change: encode the original
   * corpus under `V` and the vacated corpus under `V_p`, and compare the id streams
   * element for element. `tokenStream` is the SAME function the trainer feeds on, so this
   * is the object the theorem is about and not a proxy for it.
   */
  const streams = $derived.by(() => {
    if (!baseVocab || !vocab || corpusText.length === 0) return null;
    const a = tokenStream(corpusText, baseVocab);
    const b = tokenStream(vacatedText, vocab);
    const common = Math.min(a.length, b.length);
    let differ = Math.abs(a.length - b.length);
    for (let i = 0; i < common; i++) if (a[i] !== b[i]) differ++;
    const unk = (xs: number[]) =>
      xs.length === 0 ? 0 : xs.reduce((c, x) => c + (x === UNK_ID ? 1 : 0), 0) / xs.length;
    return {
      compared: Math.max(a.length, b.length),
      differ,
      identical: differ === 0,
      unkBefore: unk(a),
      unkAfter: unk(b),
    };
  });

  // ---- tier 2: the on-demand training demonstration (ui.md §1.5) -----------------------

  /** Small on purpose: the point is the comparison, not the final loss. Two runs, not three. */
  const DEMO_STEPS = 40;
  const DEMO_SEED = 0;
  const DEMO_DIMS = { dModel: 16, nLayers: 1, nHeads: 2, ctx: 32, dropout: 0 };

  let demoBusy = $state(false);
  let demoFraction = $state(0);
  let demoMessage = $state("");
  let demoError = $state("");
  let demo = $state<{
    atP: number;
    conditionLabel: string;
    a: number[];
    b: number[];
    maxDelta: number;
    lengthsAgree: boolean;
    finalA: number;
    finalB: number;
    valA: number;
    valB: number;
    expectedZero: boolean;
    elapsedMs: number;
  } | null>(null);
  let demoAbort: AbortController | null = null;

  onDestroy(() => demoAbort?.abort());

  async function trainOnce(
    text: string,
    v: LexVocab,
    label: string,
    base: number,
    signal: AbortSignal,
  ): Promise<{ history: number[]; finalLoss: number; valLoss: number }> {
    const res = await trainInWorker(
      {
        ...DEMO_DIMS,
        vocab: { words: v.words, source: v.source, budgetName: v.budgetName },
        text,
        steps: DEMO_STEPS,
        seed: DEMO_SEED,
        sampleEvery: DEMO_STEPS,
        signal,
      },
      (prog) => {
        demoFraction = base + (prog.step / Math.max(1, prog.totalSteps)) * 0.5;
        demoMessage = `${label} · step ${prog.step}/${prog.totalSteps} · loss ${prog.loss.toFixed(3)}`;
      },
    );
    return {
      history: res.model.history.map((h) => h.loss),
      finalLoss: res.finalLoss,
      valLoss: res.valLoss,
    };
  }

  /**
   * Two real training runs — `p = 0` under `V`, and the current `p` under `V_p` — with the
   * same seed, the same dimensions and the same step count, then `max |Δloss|` between the
   * two curves. Under the mapped condition the theorem says this is EXACTLY zero, and the
   * UI reports whatever it actually is: a non-zero here is a bug, not a rounding artifact.
   */
  async function runDemo(): Promise<void> {
    if (!baseVocab || !vocab) return;
    demoAbort?.abort();
    const ctrl = new AbortController();
    demoAbort = ctrl;
    demoBusy = true;
    demoError = "";
    demo = null;
    demoFraction = 0;
    demoMessage = "";
    const started = Date.now();
    try {
      const runA = await trainOnce(corpusText, baseVocab, "p = 0", 0, ctrl.signal);
      const runB = await trainOnce(
        vacatedText,
        vocab,
        `p = ${params.p.toFixed(2)}`,
        0.5,
        ctrl.signal,
      );
      const common = Math.min(runA.history.length, runB.history.length);
      let maxDelta = 0;
      for (let i = 0; i < common; i++) {
        const d = Math.abs(runA.history[i] - runB.history[i]);
        if (d > maxDelta) maxDelta = d;
      }
      demo = {
        atP: params.p,
        conditionLabel: condition,
        a: runA.history,
        b: runB.history,
        maxDelta,
        lengthsAgree: runA.history.length === runB.history.length,
        finalA: runA.finalLoss,
        finalB: runB.finalLoss,
        valA: runA.valLoss,
        valB: runB.valLoss,
        expectedZero: mapped,
        elapsedMs: Date.now() - started,
      };
      demoFraction = 1;
      demoMessage = "";
    } catch (e) {
      demoError = e instanceof Error ? e.message : String(e);
    } finally {
      demoBusy = false;
      if (demoAbort === ctrl) demoAbort = null;
    }
  }

  function stopDemo(): void {
    demoAbort?.abort();
    demoAbort = null;
    demoBusy = false;
  }

  // The two loss curves, drawn on one pair of axes so an exact overlap is what an exact
  // zero LOOKS like: the dashed curve sits on the solid one for its whole length.
  const CW = 320;
  const CH = 96;
  const demoCurves = $derived.by(() => {
    if (!demo || demo.a.length < 2 || demo.b.length < 2) return null;
    const all = [...demo.a, ...demo.b];
    const hi = Math.max(...all);
    const lo = Math.min(...all);
    const span = hi - lo || 1;
    const path = (xs: number[]) =>
      xs
        .map(
          (y, i) =>
            `${((i / Math.max(1, xs.length - 1)) * CW).toFixed(1)},${(CH - ((y - lo) / span) * CH).toFixed(1)}`,
        )
        .join(" ");
    return { a: path(demo.a), b: path(demo.b), hi, lo };
  });
</script>

<div class="panel-body" data-testid="lex-vacancy">
  <div class="head">
    <h3>Vacancy — field without location</h3>
    <span class="hint">rewrite the corpus so a word's form tells you nothing</span>
  </div>

  <p class="lede">
    A word can mean something to you two ways: because you know the <b>form</b> — you have met
    <code>crow</code> before, and the string itself already carries a prior — or because the
    <b>field</b> around it fixes what it must be. This control separates them. Raising
    <code>p</code> replaces that fraction of eligible open-class <b>stems</b> with invented
    forms, while closed-class scaffolding, inflectional suffixes, punctuation and line breaks
    survive byte for byte. Everything below is measured on the corpus this tab actually trains
    on, in this browser, by the same transform the Python backend runs.
  </p>

  <!-- ---- controls (ui.md §1.1) --------------------------------------------------- -->
  <div class="controls">
    <label class="slider wide">
      <span class="ctl-label">vacancy rate <b>p = {params.p.toFixed(2)}</b></span>
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        class="p-slider"
        data-testid="lex-vacancy-p"
        style={`--fill: ${(params.p * 100).toFixed(1)}%`}
        value={params.p}
        oninput={(e) => onP(Number(e.currentTarget.value))}
      />
    </label>

    <label class="slider">
      <span class="ctl-label">seed</span>
      <input
        type="number"
        min="0"
        max="9999"
        step="1"
        class="num"
        data-testid="lex-vacancy-seed"
        value={params.seed}
        oninput={(e) => {
          const v = Math.trunc(Number(e.currentTarget.value));
          if (Number.isFinite(v)) onSeed(Math.max(0, v));
        }}
      />
    </label>

    <div class="ctl">
      <span class="ctl-label" id="lex-vacancy-condition-label">condition</span>
      <div
        class="seg wrap"
        role="radiogroup"
        tabindex="-1"
        aria-labelledby="lex-vacancy-condition-label"
        data-testid="lex-vacancy-condition"
        onkeydown={(e) => segKey(e, onCondition)}
      >
        {#each CONDITIONS as c (c.id)}
          <button
            role="radio"
            aria-checked={condition === c.id}
            tabindex={condition === c.id ? 0 : -1}
            data-value={c.id}
            class:active={condition === c.id}
            title={c.title}
            onclick={() => onCondition(c.id)}>{c.label}</button
          >
        {/each}
      </div>
    </div>

    {#if condition === "reveal"}
      <label class="slider">
        <span class="ctl-label">reveal first <b>{revealAfter}</b></span>
        <input
          type="number"
          min="1"
          max="99"
          step="1"
          class="num"
          data-testid="lex-vacancy-reveal"
          value={revealAfter}
          oninput={(e) => {
            const v = Math.trunc(Number(e.currentTarget.value));
            if (Number.isFinite(v)) onRevealAfter(Math.max(1, v));
          }}
        />
      </label>
    {/if}

    <label class="check">
      <input
        type="checkbox"
        data-testid="lex-vacancy-prosody"
        checked={params.matchProsody}
        onchange={(e) => onProsody(e.currentTarget.checked)}
      />
      <span>match prosody</span>
    </label>

    <div class="ctl">
      <span class="ctl-label" id="lex-vacancy-mint-label">mint</span>
      <div
        class="seg"
        role="radiogroup"
        tabindex="-1"
        aria-labelledby="lex-vacancy-mint-label"
        data-testid="lex-vacancy-mint"
        onkeydown={(e) => segKey(e, onMint)}
      >
        {#each MINTS as m (m.id)}
          <button
            role="radio"
            aria-checked={mint === m.id}
            tabindex={mint === m.id ? 0 : -1}
            data-value={m.id}
            class:active={mint === m.id}
            disabled={!m.enabled}
            title={m.title}
            onclick={() => m.enabled && onMint(m.id)}>{m.label}</button
          >
        {/each}
      </div>
    </div>
  </div>

  <p class="note" data-testid="lex-vacancy-mint-note">
    <b>swap</b> — draw a real, frequency-rank-matched English word instead of an invented one
    (contract §8.3) — is the control that separates <i>wrong content</i> from
    <i>unknown form</i> for the pretrained arm. <b>Both engines now implement it</b>, and they
    agree word for word; this panel is not wired to them yet, so it is offered
    <b>disabled</b> rather than silently falling back to <code>nonce</code>. One thing it will
    have to say when it is: a swap map is injective only at <b>p = 0 or p = 1</b>. Its
    replacements are real corpus words, so at an intermediate <i>p</i> a vacated word can land
    on one that has <i>not</i> moved — and contract §5.2a proves no stable swap avoids that.
    The engines refuse the mapped vocabulary there rather than duplicate a row.
  </p>

  <!-- ---- the corpus, live (ui.md §1.2) -------------------------------------------- -->
  <div class="ctl">
    <div class="corpus-head">
      <span class="ctl-label">the corpus, transformed</span>
      <div class="pager">
        <button
          class="page"
          data-testid="lex-vacancy-prev"
          disabled={win === 0}
          onclick={() => page(-1)}>◀</button
        >
        <span class="range" data-testid="lex-vacancy-window">
          token-producing lines {n(win * WINDOW_LINES + 1)}–{n(
            Math.min((win + 1) * WINDOW_LINES, wordLines.length),
          )} of {n(wordLines.length)}
        </span>
        <button
          class="page"
          data-testid="lex-vacancy-next"
          disabled={win >= nWindows - 1}
          onclick={() => page(1)}>▶</button
        >
      </div>
    </div>
    <!-- A scroll container must be focusable or keyboard-only users cannot reach the
         overflow (WCAG 2.1.1). The linter only knows the element is non-interactive. -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div class="corpus" data-testid="lex-vacancy-corpus" role="group" aria-label="the corpus, transformed" tabindex="0">
      {#each shown as line (line.i)}
        <div class="line">
          {#each line.segs as seg, k (k)}<span class={seg.cls}>{seg.text}</span>{/each}
        </div>
      {/each}
    </div>
    <p class="legend" data-testid="lex-vacancy-legend">
      <span class="swatch kept">closed class, preserved</span>
      <span class="swatch open">open class, not yet vacated</span>
      <span class="swatch minted">minted</span>
      <span class="legend-why">
        — the classes come from the real map: a word is <b>minted</b> when the transform
        actually changed it, <b>open</b> when its stem passes the eligibility test of
        contract §2.2 but <code>u(stem) ≥ p</code>, and <b>preserved</b> when the stem is in
        the closed class or fails eligibility (too short, or not ASCII letters — which is why
        <code>good-bye</code> never moves). Nothing here is annotated by hand. The corpus opens
        with its own table of contents; page forward for verse.
      </span>
    </p>
  </div>

  <!-- ---- the nesting ribbon (ui.md §1.3) ------------------------------------------ -->
  <div class="ctl">
    <span class="ctl-label">nesting &amp; stability</span>
    <div class="ribbon-wrap">
      <table class="ribbon" data-testid="lex-vacancy-ribbon">
        <thead>
          <tr>
            <th scope="col" class="stem-h">stem</th>
            <th scope="col" class="u-h">u</th>
            {#each P_CELLS as pc (pc)}
              <th scope="col">p = {pc.toFixed(2)}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each ribbon as row (row.stem)}
            <tr class:live={row.u < params.p}>
              <th scope="row" class="stem">{row.stem}</th>
              <td class="u">{row.u.toFixed(3)}</td>
              {#each row.cells as cell (cell.p)}
                <td class={cell.vacated ? "cell minted" : "cell open"}>{cell.form}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <p class="caption" data-testid="lex-vacancy-ribbon-caption">
      Read each row left to right. <b>Nesting</b>: once a cell turns minted it never reverts —
      a stem is vacated iff <code>u(stem) &lt; p</code>, and <code>u</code> is a hash of
      <code>(seed, stem)</code> alone, so the vacated sets are nested as <code>p</code> grows.
      <b>Stability</b>: the minted string is <i>the same string</i> in every later cell — the
      nonce map is built once over the whole type set in canonical order, so it does not depend
      on <code>p</code>, on document order, or on which other words exist. The source project's
      minter built its map lazily while rewriting and had neither property; contract §5.2 is the
      correction. Rows are the eligible stems of this corpus at eight evenly spaced ranks of
      <code>u</code>; a highlighted row is one that is vacated at the current
      <code>p = {params.p.toFixed(2)}</code>.
      {#if !mapped}
        <b class="warn">
          The corpus above is in the <i>{condition === "reveal" ? "partial reveal" : condition}</i>
          condition, which deliberately does not use this map for every occurrence — that control
          exists precisely to destroy the identity the ribbon is showing.
        </b>
      {/if}
    </p>
  </div>

  <!-- ---- statistics (ui.md §1.4, contract §10) ------------------------------------ -->
  {#if stats}
    <div class="counters" data-testid="lex-vacancy-stats">
      <div class="counter">
        <span class="k">types vacated</span>
        <span class="v" data-testid="lex-vacancy-types">
          {n(stats.corpusTypesVacated)}<span class="of">/ {n(stats.corpusTypesEligible)}</span>
        </span>
        <span class="d">
          distinct corpus types whose form changed, out of the eligible ones —
          <b>corpus scope</b>, not the map's domain, which adds
          {n(stats.domainTypesEligible - stats.corpusTypesEligible)} budget words that never appear
          in the text
        </span>
      </div>
      <div class="counter">
        <span class="k">tokens vacated</span>
        <span class="v" data-testid="lex-vacancy-tokens">
          {n(stats.tokensVacated)}<span class="of">/ {n(stats.tokensTotal)}</span>
        </span>
        <span class="d">{pct(stats.tokensVacated / Math.max(1, stats.tokensTotal))} of the corpus's word occurrences</span>
      </div>
      <div class="counter">
        <span class="k">stems vacated</span>
        <span class="v">{n(stats.stemsVacated)}<span class="of">/ {n(stats.stemsTotal)}</span></span>
        <span class="d">
          the map's size is every eligible stem; a stem vacates when <code>u &lt; p</code>
        </span>
      </div>
      <div class="counter">
        <span class="k">map</span>
        <span class="v" class:good={stats.bijective} data-testid="lex-vacancy-bijective">
          {stats.bijective ? "injective ✓" : "NOT injective"}
        </span>
        <span class="d">
          verified over assembled surface forms at every <code>p</code>, not assumed ·
          <b>{n(stats.remintRounds)}</b> re-mint {stats.remintRounds === 1 ? "round" : "rounds"} ·
          image {n(stats.imageSize)}
        </span>
      </div>
    </div>

    <div class="prosody" data-testid="lex-vacancy-prosody-stats">
      <div class="prosody-nums">
        <span class="ctl-label">prosody, before → after</span>
        <p class="metric">
          mean syllables
          <b>{stats.meanSyllablesBefore.toFixed(3)} → {stats.meanSyllablesAfter.toFixed(3)}</b>
        </p>
        <p class="metric">
          mean anapest
          <b>{stats.meanAnapestBefore.toFixed(3)} → {stats.meanAnapestAfter.toFixed(3)}</b>
        </p>
      </div>
      <div class="prosody-split">
        <span class="ctl-label">where each token's stress came from</span>
        <table class="split" data-testid="lex-vacancy-stress-split">
          <thead>
            <tr>
              <th scope="col">source</th>
              <th scope="col">before</th>
              <th scope="col">after</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">hand table</th>
              <td>{pct(stats.stressFromTableBefore)}</td>
              <td>{pct(stats.stressFromTableAfter)}</td>
            </tr>
            <tr>
              <th scope="row">minted</th>
              <td>{pct(stats.stressFromMintedBefore)}</td>
              <td>{pct(stats.stressFromMintedAfter)}</td>
            </tr>
            <tr>
              <th scope="row">spelling rule</th>
              <td>{pct(stats.stressFromRuleBefore)}</td>
              <td>{pct(stats.stressFromRuleAfter)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="honesty" data-testid="lex-vacancy-prosody-honesty">
        Read those two numbers only together with this table. The stress table is
        <b>{STRESS_TABLE.size} hand-set entries</b>, seeded by rule and never checked by a
        human, and it covers <b>{pct(stats.stressFromTableBefore)}</b> of this corpus's tokens
        before the transform and <b>{pct(stats.stressFromTableAfter)}</b> after. Everything else
        is either a form <i>we</i> minted — whose pattern we chose, and whose syllable
        <i>count</i> is checked while its pattern is only asserted — or the spelling heuristic
        of contract §6.2, which is a guess. So the prosody statistics are
        <b>indicative, not exact</b>.
      </p>
    </div>
  {/if}

  <!-- ---- tier 1: the instant invariance check (ui.md §1.5) ------------------------ -->
  {#if streams}
    <div
      class="verdict"
      class:ok={streams.identical}
      class:broken={!streams.identical}
      data-testid="lex-vacancy-invariance"
    >
      {#if streams.identical}
        <b data-testid="lex-vacancy-invariance-verdict">token id streams identical</b>
        · {n(streams.compared)} ids compared, element for element, just now. The vacated corpus
        under <code>V_p</code> encodes to exactly the token stream the original corpus encodes to
        under <code>V</code>, so the trainer cannot tell them apart.
        <code>&lt;unk&gt;</code> rate {pct(streams.unkBefore)} → {pct(streams.unkAfter)}.
      {:else}
        <b data-testid="lex-vacancy-invariance-verdict">token id streams differ</b>
        · {n(streams.differ)} of {n(streams.compared)} positions.
        <code>&lt;unk&gt;</code> rate {pct(streams.unkBefore)} → {pct(streams.unkAfter)}. This is
        the real result, not a warning: {condition === "inconsistent"
          ? "the inconsistent control mints a fresh form per occurrence, so the vacated corpus has types the budget cannot express and the theorem's premise is gone"
          : condition === "reveal"
            ? `partial reveal splits every vacated stem into two types — the first ${revealAfter} occurrences keep their English form — so the vocabulary is rebuilt from the vacated corpus and the ids move`
            : "the vocabulary was rebuilt rather than mapped, so ids no longer correspond"}.
      {/if}
    </div>
  {/if}

  <!-- ---- framing: the null is the finding (ui.md §1.6) ---------------------------- -->
  <div class="framing" data-testid="lex-vacancy-framing">
    <p>
      <b>The headline result here is an exact zero, and that is the finding — not a broken
      chart.</b>
      For a word-level model trained from scratch, the mapped condition is a
      <i>pure relabelling</i> of the vocabulary: the map is injective, the budget's words are
      pushed through the same transform in the same order, so every word keeps the embedding row
      its pre-image had. The model is provably blind to <code>p</code>, to
      <code>seed</code> and to <code>match prosody</code>. Read plainly: for this model class,
      <b>all of a word's meaning is field and none of it is form</b>.
    </p>
    <p>
      The number that is <b>not</b> zero belongs to a model that has forms it already knows — a
      real pretrained transformer, scored on a passage and its vacated twin. That measurement is
      this feature's pretrained arm and it lives in the
      <button class="linklike" onclick={() => view.set("architecture")}>
        Architecture Explorer</button
      >. The two numbers are the point of the pair: the same transform, worth exactly nothing to
      one model and something measurable to the other.
    </p>
  </div>

  <!-- ---- tier 2: train twice and subtract (ui.md §1.5) --------------------------- -->
  <div class="ctl">
    <span class="ctl-label">the invariance theorem, trained</span>
    <div class="actions">
      <button
        class="go"
        data-testid="lex-vacancy-demo-run"
        disabled={demoBusy || !baseVocab || !vocab}
        onclick={() => void runDemo()}
      >
        {demoBusy ? "training…" : `Train at p = 0 and p = ${params.p.toFixed(2)}`}
      </button>
      {#if demoBusy}
        <button class="secondary" data-testid="lex-vacancy-demo-stop" onclick={stopDemo}>Stop</button>
      {/if}
      <span class="hint">
        two real runs · {DEMO_STEPS} steps · d={DEMO_DIMS.dModel}, {DEMO_DIMS.nLayers} layer,
        ctx {DEMO_DIMS.ctx} · same seed {DEMO_SEED}, same hyperparameters
      </span>
    </div>

    {#if demoBusy}
      <Progress progress={demoFraction} message={demoMessage} />
    {/if}

    {#if demoError}
      <div class="err" data-testid="lex-vacancy-demo-error">{demoError}</div>
    {/if}

    {#if demo}
      <div class="demo" data-testid="lex-vacancy-demo-result">
        <div class="chart-wrap">
          <svg
            class="chart"
            viewBox={`0 0 ${CW} ${CH}`}
            preserveAspectRatio="none"
            role="img"
            aria-label={`two training loss curves, p = 0 and p = ${demo.atP.toFixed(2)}, maximum absolute difference ${demo.maxDelta}`}
          >
            {#if demoCurves}
              <polyline points={demoCurves.a} class="curve-a" />
              <polyline points={demoCurves.b} class="curve-b" stroke-dasharray="5 5" />
            {/if}
          </svg>
        </div>
        <p class="chart-key">
          <span class="key-a">p = 0</span>
          <span class="key-b">p = {demo.atP.toFixed(2)} ({demo.conditionLabel})</span>
          — training loss in nats, {n(demo.a.length)} steps each, in
          {(demo.elapsedMs / 1000).toFixed(1)}s
        </p>
        <p
          class="delta"
          class:ok={demo.expectedZero && demo.maxDelta === 0 && demo.lengthsAgree}
          class:broken={demo.expectedZero && !(demo.maxDelta === 0 && demo.lengthsAgree)}
          data-testid="lex-vacancy-demo-delta"
        >
          {#if !demo.lengthsAgree}
            The two runs produced different numbers of loss points
            ({n(demo.a.length)} vs {n(demo.b.length)}) — the curves are not comparable.
          {:else if demo.maxDelta === 0}
            <b>max |Δloss| = 0</b> — exactly zero, over every one of the {n(demo.a.length)} steps.
            {#if demo.expectedZero}
              Not "≈ 0", not "within tolerance": the two runs are the same computation on the same
              token ids, so they are bit-identical. Final loss
              {demo.finalA.toFixed(6)} in both, held-out {demo.valA.toFixed(6)} in both.
            {:else}
              This condition is not expected to be invariant, so a zero here means the controls
              happen to leave the token stream unchanged — check <code>p</code>.
            {/if}
          {:else}
            <b>max |Δloss| = {demo.maxDelta.toExponential(3)}</b> ·
            final {demo.finalA.toFixed(4)} vs {demo.finalB.toFixed(4)} · held-out
            {demo.valA.toFixed(4)} vs {demo.valB.toFixed(4)}.
            {#if demo.expectedZero}
              <b>This should have been exactly 0.</b> The mapped condition is a pure relabelling,
              so any difference at all is a defect in the transform, the vocabulary mapping or the
              trainer — report it rather than rounding it away.
            {:else}
              That is the expected direction: this condition breaks type identity, so the model
              really is seeing a different corpus.
            {/if}
          {/if}
        </p>
      </div>
    {/if}
  </div>

  <p class="note">
    The corpus every panel below now uses is the <b>vacated</b> one, so the budget's coverage
    counters, the training run, the samples and the embedding geometry all respond to
    <code>p</code> together. In the mapped condition the budget is pushed through the same
    transform in the same order and keeps its size; in the control conditions it is rebuilt from
    the vacated text instead, and the collapse in coverage <i>is</i> the measurement.
  </p>

  <Explain
    title="What is actually preserved, exactly"
    hint="closed class, inflection, punctuation, line structure — byte for byte"
    testid="lex-explain-vacancy"
  >
    <p>
      A word is split into <code>stem + suffix</code> by a spelling heuristic (contract §3):
      the first match among the {SUFFIXES.length} suffixes
      <code>{SUFFIXES.join(" · ")}</code> wins, tried in that order, and only when at least
      three characters would remain. A stem may be vacated when all three of §2.2 hold: it is
      not one of the
      <b>{FUNCTION_WORDS.size} closed-class words</b>, it is ASCII letters only, and it is
      longer than two characters. So <code>don't</code> splits to the stem <code>do</code> and
      never moves; <code>good-bye</code> fails the ASCII test and never moves;
      <code>dog's</code> becomes <code>&lt;nonce&gt;'s</code>.
    </p>
    <p>
      Every output is itself a single complete tokenizer match — checked at run time, not
      assumed — so <code>tokenize(vacated)</code> has the same length and ordering as
      <code>tokenize(original)</code>, and because line breaks are untouched the
      <code>&lt;eos&gt;</code>-per-line rule fires in exactly the same places. That is what the
      invariance theorem rests on, and it is why the transform's idea of a word is the
      tokenizer's own regular expression rather than a second one written beside it.
    </p>
    <p>
      The heuristic is not a morphological analyser and is wrong outside its exception list —
      <code>ladder</code> splits to <code>ladd + er</code>. That is tolerable (the nonce still
      carries a consistent identity and an inflected-looking surface) but it is a known
      artifact, and it is stated here rather than quietly absorbed.
    </p>
  </Explain>

  <Explain
    title="Why the theorem holds, and what it does not prove"
    hint="a pure relabelling is invisible; that is a fact about this model class"
    testid="lex-explain-vacancy-theorem"
  >
    <p>
      With <code>consistent = true</code> and <code>revealAfter = 0</code>, for every
      <code>p</code>, <code>seed</code>, budget and setting of <code>match prosody</code>, the
      token id stream of the vacated corpus under the mapped vocabulary equals the token id
      stream of the original under the original vocabulary, element for element. Three facts
      make it true: the transform is a bijection on word occurrences preserving order and line
      structure; the type map is injective on the union of the corpus's types and the budget's
      words, verified over assembled surface forms at map-build time and therefore at every
      <code>p</code> at once; and the budget is pushed through the same transform in the same
      order, so out-of-budget types still land out of budget and <code>&lt;unk&gt;</code>
      appears in exactly the same places.
    </p>
    <p>
      Injectivity is <b>verified, not assumed</b>. Two weaker checks were tried first and both
      were wrong: checking bare nonces misses a collision that arrives through the suffix, and
      checking the image size at <code>p = 1</code> only misses it too, because at full vacancy
      every eligible type has moved and no minted form can meet a surviving English word. The
      collision exists only at intermediate <code>p</code>, which is exactly where a reader
      sweeping this slider spends their time.
    </p>
    <p>
      What it does <b>not</b> prove: that form is worthless in general. It proves that a
      word-level model whose entire lexicon is a table of embedding rows has no channel through
      which a form could matter — it never sees the characters. A model with subword tokens has
      that channel, which is why the pretrained arm exists and why its answer is not zero.
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
  .lede {
    margin: 0;
    font-size: 0.78rem;
    line-height: 1.6;
    color: var(--text-dim);
    max-width: 62rem;
  }
  .lede b {
    color: var(--text);
  }
  .lede code,
  .note code,
  .caption code,
  .legend code,
  .verdict code,
  .framing code,
  .delta code,
  .counter .d code {
    font-family: var(--mono);
    font-size: 0.94em;
    color: var(--accent);
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 0.5rem 1rem;
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
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    text-transform: none;
    letter-spacing: 0;
  }
  .slider {
    display: flex;
    flex-direction: column;
    gap: 0.22rem;
    min-width: 6rem;
  }
  .slider.wide {
    flex: 1 1 14rem;
    min-width: min(100%, 12rem);
  }
  /* The headline knob has to be legible: the shell's default track is `--bg-elev-2`, which
     is exactly this card's background, so on this tab it disappears. Fill the travelled
     part with the accent so `p` is readable from the control itself and not only its
     label. */
  .slider input.p-slider {
    height: 8px;
    background: linear-gradient(
      90deg,
      var(--accent) 0 var(--fill),
      var(--bg-elev) var(--fill) 100%
    );
    border: 1px solid var(--border);
  }
  .num {
    width: 5.5rem;
    background: var(--bg-elev);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.24rem 0.45rem;
    font-family: var(--mono);
    font-size: 0.78rem;
  }
  .num:focus {
    border-color: var(--accent);
    outline: none;
  }
  .check {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--text-dim);
    padding-bottom: 0.2rem;
  }
  .check input {
    accent-color: var(--accent);
    width: 15px;
    height: 15px;
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
  .note {
    margin: 0;
    font-size: 0.72rem;
    line-height: 1.55;
    color: var(--text-dim);
    max-width: 62rem;
  }
  .note b {
    color: var(--text);
  }

  /* ---- the corpus view ---------------------------------------------------------- */
  .corpus-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .pager {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .pager .range {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }
  .page {
    background: var(--bg-elev);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.1rem 0.45rem;
    font-size: 0.7rem;
    line-height: 1.4;
  }
  .page:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--accent);
  }
  .page:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .corpus {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.75rem;
    font-family: var(--mono);
    font-size: 0.74rem;
    line-height: 1.75;
    max-height: 22rem;
    overflow: auto;
    min-width: 0;
  }
  .corpus:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .line {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    min-height: 1.2em;
  }
  .line .gap {
    color: var(--text-dim);
  }
  /* The three-way colour coding of ui.md §1.2, stated verbatim by the legend below. */
  .line .kept,
  .swatch.kept {
    color: var(--text);
  }
  .line .open,
  .swatch.open {
    color: var(--text-dim);
    opacity: 0.72;
  }
  .line .minted,
  .swatch.minted {
    color: var(--accent);
    font-weight: 600;
  }
  .legend {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.3rem 0.8rem;
    font-size: 0.7rem;
    line-height: 1.55;
    color: var(--text-dim);
  }
  .swatch {
    font-family: var(--mono);
    font-size: 0.68rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
    white-space: nowrap;
  }
  .swatch.minted {
    border-color: rgba(110, 168, 254, 0.45);
  }
  .legend-why {
    flex: 1 1 20rem;
    min-width: 0;
  }
  .legend-why b {
    color: var(--text);
  }

  /* ---- the nesting ribbon -------------------------------------------------------- */
  .ribbon-wrap {
    overflow-x: auto;
    min-width: 0;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--bg-elev);
  }
  .ribbon {
    border-collapse: collapse;
    width: 100%;
    font-family: var(--mono);
    font-size: 0.72rem;
  }
  .ribbon th,
  .ribbon td {
    padding: 0.28rem 0.55rem;
    text-align: left;
    white-space: nowrap;
    border-bottom: 1px solid var(--border);
  }
  .ribbon thead th {
    font-size: 0.64rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .ribbon tbody tr:last-child th,
  .ribbon tbody tr:last-child td {
    border-bottom: none;
  }
  .ribbon tbody tr.live {
    background: rgba(110, 168, 254, 0.07);
  }
  .ribbon .stem {
    color: var(--text);
    font-weight: 600;
  }
  .ribbon .u {
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }
  .ribbon .cell.open {
    color: var(--text-dim);
    opacity: 0.72;
  }
  .ribbon .cell.minted {
    color: var(--accent);
    font-weight: 600;
  }
  .caption {
    margin: 0;
    font-size: 0.7rem;
    line-height: 1.6;
    color: var(--text-dim);
    max-width: 62rem;
  }
  .caption b {
    color: var(--text);
  }
  .caption b.warn {
    color: #ffb454;
  }

  /* ---- statistics ---------------------------------------------------------------- */
  .counters {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr));
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
  .counter .of {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-left: 0.3rem;
  }
  .counter .d {
    font-size: 0.66rem;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .counter .d b {
    color: var(--text);
  }

  .prosody {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr));
    gap: 0.5rem 0.9rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.7rem;
  }
  .prosody-nums,
  .prosody-split {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    min-width: 0;
  }
  .metric {
    margin: 0;
    font-size: 0.74rem;
    color: var(--text-dim);
  }
  .metric b {
    color: var(--text);
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
  }
  .split {
    border-collapse: collapse;
    font-size: 0.72rem;
  }
  .split th,
  .split td {
    padding: 0.12rem 0.6rem 0.12rem 0;
    text-align: left;
    white-space: nowrap;
  }
  .split thead th {
    font-size: 0.62rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-dim);
  }
  .split tbody th {
    font-weight: 500;
    color: var(--text-dim);
  }
  .split tbody td {
    font-family: var(--mono);
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .honesty {
    grid-column: 1 / -1;
    margin: 0;
    font-size: 0.68rem;
    line-height: 1.6;
    color: var(--text-dim);
  }
  .honesty b {
    color: var(--text);
  }

  /* ---- invariance ---------------------------------------------------------------- */
  .verdict {
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    font-size: 0.74rem;
    line-height: 1.6;
  }
  .verdict.ok {
    background: rgba(91, 224, 176, 0.08);
    border: 1px solid rgba(91, 224, 176, 0.28);
    color: var(--good);
  }
  .verdict.broken {
    background: rgba(255, 180, 84, 0.09);
    border: 1px solid rgba(255, 180, 84, 0.3);
    color: #ffb454;
  }
  .verdict b {
    font-family: var(--mono);
  }
  .framing {
    background: linear-gradient(135deg, rgba(110, 168, 254, 0.09), rgba(183, 148, 246, 0.07));
    border: 1px solid rgba(110, 168, 254, 0.28);
    border-radius: 10px;
    padding: 0.6rem 0.75rem;
    font-size: 0.75rem;
    line-height: 1.65;
    color: var(--text-dim);
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
  .framing p {
    margin: 0;
    max-width: 62rem;
  }
  .framing b {
    color: var(--text);
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .go:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .secondary {
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
  .demo {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
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
  .curve-a {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2.2;
    vector-effect: non-scaling-stroke;
  }
  .curve-b {
    fill: none;
    stroke: var(--accent-2);
    stroke-width: 1.4;
    vector-effect: non-scaling-stroke;
  }
  .chart-key {
    margin: 0;
    font-size: 0.68rem;
    color: var(--text-dim);
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    align-items: baseline;
  }
  .key-a,
  .key-b {
    font-family: var(--mono);
  }
  .key-a::before,
  .key-b::before {
    content: "";
    display: inline-block;
    width: 14px;
    height: 0;
    margin-right: 0.3rem;
    vertical-align: middle;
  }
  .key-a::before {
    border-top: 2.5px solid var(--accent);
  }
  .key-b::before {
    border-top: 2px dashed var(--accent-2);
  }
  .delta {
    margin: 0;
    font-size: 0.74rem;
    line-height: 1.6;
    color: var(--text-dim);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
  }
  .delta b {
    color: var(--text);
    font-family: var(--mono);
  }
  .delta.ok {
    background: rgba(91, 224, 176, 0.08);
    border-color: rgba(91, 224, 176, 0.28);
    color: var(--good);
  }
  .delta.ok b {
    color: var(--good);
  }
  .delta.broken {
    background: rgba(255, 122, 144, 0.1);
    border-color: rgba(255, 122, 144, 0.3);
    color: var(--bad);
  }
  .delta.broken b {
    color: var(--bad);
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
  /* Reads as a link, stays a button: switching tabs is an action, not navigation. */
  .linklike {
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: var(--accent);
    text-decoration: underline;
    cursor: pointer;
  }
</style>
