<script lang="ts">
  /**
   * Lexicon Lab (feature 006) — a tiny word-level transformer whose VOCABULARY BUDGET is
   * the control you move. Pick a budget, pick the model's dimensions, train it in the
   * browser on real public-domain nursery rhymes, and watch the loss, the text it
   * generates, and the geometry of its embedding matrix respond together.
   *
   * Everything numeric on this page comes from `src/lib/lexEngine` — the TypeScript
   * mirror of `code/backend/src/llm_geometry/lex/`, golden-tested against PyTorch. No
   * constant is retyped into the prose here; where a sentence quotes a number it reads it
   * from the engine (or from the corpus asset described below), so a changed constant
   * changes the sentence.
   *
   * ---------------------------------------------------------------------------------
   * ONE INTEGRATION DEPENDENCY, owned outside this file slice.
   *
   * The engine builds budgets and trains from raw corpus TEXT, and neither it nor
   * `dataClient` ships the shipped corpus to the browser yet. This tab therefore reads
   * it from the project's established build-time asset path:
   *
   *     ${BASE_URL}static-data/lex/corpus.json
   *     { "title", "year", "gutenberg_id", "sha256", "bytes",
   *       "body_sha256", "body_bytes", "text" }
   *
   * exported by the real backend from `lex/config.py` + `lex/corpus.py::load_corpus_text`
   * (body only — the Gutenberg header and licence footer trimmed), exactly as
   * `static-data/geo/*` already is. Until that export exists the tab says so loudly and
   * refuses to run: an invented corpus would make every number below a fiction.
   *
   * `sha256`/`bytes` describe the COMMITTED file (Gutenberg header and licence footer
   * included); `body_sha256`/`body_bytes` describe the trimmed body — the string this tab
   * actually tokenizes. `loadCorpus` RECOMPUTES that body digest here and refuses the
   * asset on a mismatch or an absence, so the digest on screen is one this browser
   * checked rather than one the export asserted.
   * ---------------------------------------------------------------------------------
   */
  import { onMount } from "svelte";

  import {
    DEFAULT_BUDGET,
    DEFAULT_BUDGET_SOURCE,
    DEFAULT_CTX,
    DEFAULT_DROPOUT,
    DEFAULT_D_MODEL,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_SEED,
    DEFAULT_TIED,
    LAYER_NORM_EPS,
    LexModel,
    LexVocab,
    MLP_RATIO,
    buildVocab,
    isDolchBudgetName,
    paramCount,
    randomBaselineSpectrum,
    spectrum,
    type BudgetSource,
    type Coverage,
    type LexConfig,
    type SpectrumResult,
  } from "../../lib/lexEngine";
  import { sha256Hex, utf8Bytes } from "../../lib/geoEngine/hash";
  import Explain from "../../lib/Explain.svelte";
  import { view } from "../../lib/stores";
  import { baseLabelOf, provenanceOf } from "./provenance";
  import BudgetPanel from "./BudgetPanel.svelte";
  import ForwardPassPanel from "./ForwardPassPanel.svelte";
  import LexWeightLab from "./LexWeightLab.svelte";
  import ModelFile from "./ModelFile.svelte";
  import ModelPanel from "./ModelPanel.svelte";
  import SamplePanel from "./SamplePanel.svelte";
  import SpectrumPanel from "./SpectrumPanel.svelte";
  import TokenCloud from "./TokenCloud.svelte";
  import TrainPanel from "./TrainPanel.svelte";

  /** The build-time corpus export described in the header comment. */
  interface CorpusAsset {
    title: string;
    year: number;
    gutenberg_id: number;
    /** Digest and size of the committed file, header and licence footer included. */
    sha256: string;
    bytes: number;
    /** Digest and size of the trimmed body below — the text every number is measured on. */
    body_sha256: string;
    body_bytes: number;
    text: string;
  }
  /** The corpus every budget and every training run is measured against. */
  interface ActiveCorpus {
    text: string;
    label: string;
    /** Provenance is only claimable for the committed Project Gutenberg text. */
    shipped: CorpusAsset | null;
  }

  // ---- state ------------------------------------------------------------------------

  let corpus = $state<ActiveCorpus | null>(null);
  let corpusError = $state("");

  let budgetSource = $state<BudgetSource>(DEFAULT_BUDGET_SOURCE);
  let budgetName = $state<string>(DEFAULT_BUDGET);

  let dModel = $state<number>(DEFAULT_D_MODEL);
  let nLayers = $state<number>(DEFAULT_N_LAYERS);
  let nHeads = $state<number>(DEFAULT_N_HEADS);
  let ctx = $state<number>(DEFAULT_CTX);
  let tied = $state<boolean>(DEFAULT_TIED);
  let dropout = $state<number>(DEFAULT_DROPOUT);

  /**
   * What a training run produced, or null while nothing has been trained at the CURRENT
   * shape. Its vocabulary travels with it (FR-619): fine-tuning and generation both use
   * THIS vocabulary, never the one the controls happen to show. Any change of budget or
   * dimension retires it — those weights have the wrong shape, and describing their
   * geometry would describe a model that is no longer on screen.
   */
  let trained = $state<{ model: LexModel; vocab: LexVocab; note: string } | null>(null);

  /**
   * A hand-edited weight set (US-6), or null. It never REPLACES `trained` — it sits in
   * front of it, so `trained` is always the way back, and every panel below reads the
   * active model rather than deciding for itself which weights are current.
   */
  let edited = $state<{ model: LexModel; token: string; note: string } | null>(null);

  onMount(() => {
    void loadCorpus();
  });

  async function loadCorpus(): Promise<void> {
    corpusError = "";
    const url = `${import.meta.env.BASE_URL}static-data/lex/corpus.json`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      const asset = (await res.json()) as CorpusAsset;
      if (typeof asset?.text !== "string" || asset.text.length === 0) {
        throw new Error(`${url} carries no corpus text`);
      }
      // The digest is CHECKED here, in this browser, against the body that was just
      // loaded — not quoted from the export and displayed as if something had verified
      // it. A missing digest is treated as a wrong one: an export that declares nothing
      // cannot be checked, and an unverifiable corpus silently rewrites every coverage
      // counter, every loss and every spectrum below. (`sha256Hex` is the same pure-JS
      // implementation the model bundles are hashed with, so no secure context and no
      // async ceremony is needed.)
      if (typeof asset.body_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(asset.body_sha256)) {
        throw new Error(
          `${url} declares no usable body_sha256, so the text it carries cannot be verified`,
        );
      }
      const actualDigest = sha256Hex(utf8Bytes(asset.text));
      if (actualDigest !== asset.body_sha256) {
        throw new Error(
          `${url} carries text hashing to ${actualDigest.slice(0, 16)}… but declares ` +
            `${asset.body_sha256.slice(0, 16)}…`,
        );
      }
      corpus = {
        text: asset.text,
        label: `${asset.title} (${asset.year})`,
        shipped: asset,
      };
    } catch (e) {
      corpus = null;
      corpusError = e instanceof Error ? e.message : String(e);
    }
  }

  // ---- derived: vocabulary, coverage, parameter count --------------------------------

  const vocab = $derived.by<LexVocab | null>(() =>
    corpus ? buildVocab(budgetSource, budgetName, corpus.text) : null,
  );
  const coverage = $derived.by<Coverage | null>(() =>
    vocab && corpus ? vocab.coverage(corpus.text) : null,
  );
  const nParams = $derived(vocab ? paramCount(vocab.rows, dModel, nLayers, ctx, tied) : 0);

  const cfg = $derived<LexConfig>({
    vocabRows: vocab?.rows ?? 0,
    dModel,
    nLayers,
    nHeads,
    ctx,
    tied,
    dropout,
  });

  /** Every choice a set of weights depends on. A change here retires the trained model. */
  const shapeKey = $derived(
    `${budgetSource}/${budgetName}/${vocab?.rows ?? 0}/${dModel}/${nLayers}/${nHeads}/${ctx}/${tied}`,
  );
  let lastShapeKey = "";
  $effect(() => {
    const key = shapeKey;
    if (key !== lastShapeKey) {
      lastShapeKey = key;
      trained = null;
      edited = null; // those weights have the wrong shape now too
    }
  });

  /**
   * An untrained model at exactly this shape. Not a placeholder: it is what the sampler
   * generates from before you train (noise, in budget — which is the point), and it is
   * the model the spectrum panel's baseline describes.
   */
  const freshModel = $derived.by<LexModel | null>(() =>
    vocab && vocab.rows > 4 ? LexModel.fresh(cfg, DEFAULT_SEED) : null,
  );

  /** What an edit departs from and returns to: the trained model, or the random init. */
  const baseModel = $derived(trained?.model ?? freshModel);
  const activeModel = $derived(edited?.model ?? baseModel);
  const activeVocab = $derived(trained?.vocab ?? vocab);
  /**
   * What the weights every panel below is describing actually ARE. `trained !== null` is
   * not that question: an edit sits in front of the base model, so with nothing trained
   * one preset click leaves the page running weights that are neither the trained model
   * nor the random initialization. Panels take this, never a boolean.
   */
  const provenance = $derived(provenanceOf(trained !== null, edited !== null));
  const activeNote = $derived(edited?.note ?? trained?.note ?? "");

  const embedSpectrum = $derived.by<SpectrumResult | null>(() => {
    if (!activeModel) return null;
    const c = activeModel.cfg;
    return spectrum(activeModel.weights.embed, c.vocabRows, c.dModel);
  });
  const readoutSpectrum = $derived.by<SpectrumResult | null>(() => {
    if (!activeModel || activeModel.cfg.tied) return null;
    const c = activeModel.cfg;
    return spectrum(activeModel.headWeight, c.vocabRows, c.dModel);
  });
  /**
   * FR-622: a random matrix at the SAME shape, drawn from the embedding's own N(0, 0.02²)
   * initializer. Effective rank climbs with |V| for random matrices too, so the trained
   * number is meaningless without this one beside it.
   */
  const baselineSpectrum = $derived.by<SpectrumResult | null>(() => {
    if (!activeModel) return null;
    const c = activeModel.cfg;
    return randomBaselineSpectrum(c.vocabRows, c.dModel, DEFAULT_SEED);
  });

  // ---- events from the panels --------------------------------------------------------

  function adoptCorpus(text: string, label: string): void {
    corpus = { text, label, shipped: null };
  }

  function onTrained(model: LexModel, modelVocab: LexVocab, note: string): void {
    // Written together, and read together: a model and the vocabulary its ids mean
    // something in are one object here, never two loosely-associated pieces of state.
    trained = { model, vocab: modelVocab, note };
    edited = null; // an edit of the PREVIOUS weights says nothing about these
    lastShapeKey = shapeKey; // this IS the current shape — do not retire what just landed
  }

  /**
   * A model read back from a `.llmlex.json` file (US-8). Its weights and its vocabulary
   * arrive together and verified, so the controls are moved to match the file rather than
   * the file being reinterpreted through whatever the controls happened to say — and then
   * `lastShapeKey` is re-baselined, because this IS the shape now.
   */
  function onLoadedModel(model: LexModel, modelVocab: LexVocab, note: string): void {
    dModel = model.cfg.dModel;
    nLayers = model.cfg.nLayers;
    nHeads = model.cfg.nHeads;
    ctx = model.cfg.ctx;
    tied = model.cfg.tied;
    dropout = model.cfg.dropout;
    budgetSource = modelVocab.source;
    // Only a budget name this build can resolve against the corpus may drive the
    // controls; the model itself keeps its own vocabulary either way.
    if (modelVocab.source === "frequency" || isDolchBudgetName(modelVocab.budgetName)) {
      budgetName = modelVocab.budgetName;
    }
    trained = { model, vocab: modelVocab, note };
    edited = null;
    lastShapeKey = shapeKey;
  }
</script>

<section class="viz panel" data-testid="lex-view" data-ready={corpus ? 1 : 0}>
  <header>
    <h2>Lexicon Lab — what a bounded vocabulary can say</h2>
    <p class="sub">
      A word-level decoder-only transformer, trained <b>here, in this tab</b>, on real
      public-domain nursery rhymes. The control that matters is the <b>vocabulary budget</b>:
      the model's entire lexicon is the budget you pick, so out-of-budget words become
      <code>&lt;unk&gt;</code> in training and are masked at generation — its output is in
      budget by construction. Move the budget and three things answer together: the
      coverage counters, the loss and the text, and the geometry of the embedding matrix.
    </p>
    <p class="chips" data-testid="lex-corpus-chips">
      {#if corpus?.shipped}
        <span class="chip">corpus · {corpus.shipped.title} ({corpus.shipped.year})</span>
        <span class="chip">PG #{corpus.shipped.gutenberg_id}</span>
        <span class="chip mono verified" data-testid="lex-corpus-digest">
          body sha256 {corpus.shipped.body_sha256.slice(0, 12)}… ✓ checked here
        </span>
        <span class="chip mono">{corpus.shipped.body_bytes.toLocaleString()} bytes loaded</span>
      {:else if corpus}
        <span class="chip active">corpus · {corpus.label}</span>
        <button class="linklike" onclick={() => void loadCorpus()}>back to the shipped corpus</button>
      {:else}
        <span class="chip">loading the corpus…</span>
      {/if}
    </p>
    {#if corpus?.shipped}
      <!-- This used to live in a `title` attribute on a non-focusable chip, which is
           unreachable by keyboard and invisible to most assistive technology — and it
           claimed a verification nothing in the browser performed. Both halves are fixed
           here: the claim is visible text, and it is a claim about a check this page ran. -->
      <p class="provenance" data-testid="lex-corpus-provenance">
        Project Gutenberg ebook #{corpus.shipped.gutenberg_id} is committed to this
        repository whole — header and licence footer intact — and trimmed to its body only
        when the text is used. The
        <b>{corpus.shipped.body_bytes.toLocaleString()} bytes</b> loaded here were hashed
        <b>in this browser</b>, just now, and match the <code>body_sha256</code> the build
        recorded; a mismatch is fatal and the tab refuses to run rather than measure
        budgets against text it cannot identify. The committed file itself is
        {corpus.shipped.bytes.toLocaleString()} bytes and hashes to
        <span class="mono">{corpus.shipped.sha256.slice(0, 12)}…</span>, which is the digest
        the backend checks before it serves anything.
      </p>
    {/if}
  </header>

  <div class="explainers">
    <Explain
      title="What am I looking at?"
      hint="a budget, a model that fits in it, and the geometry that results"
      testid="lex-explain-what"
    >
      <p>
        Every language model has a vocabulary, and every vocabulary is a budget somebody
        chose. Here the budget is the dial. Two kinds are offered at <b>matched sizes</b>: the
        <b>Dolch</b> lists — a vocabulary somebody <i>prescribed</i>, published by Edward
        William Dolch in 1936 for teaching reading — and <b>frequency</b>, the top-N most
        frequent word types of the corpus itself, a vocabulary the corpus <i>describes</i>.
        Same number of words, different words.
      </p>
      <p>
        Everything here is computed from a model that really ran: no stored curve, no
        schematic, no number quoted from a paper. Before you train, the model is random
        initialization and its geometry is a random matrix's — which is exactly why the
        spectrum panel draws a random-init baseline next to the trained result instead of
        the trained result alone.
      </p>
    </Explain>

    <Explain
      title="The model, exactly"
      hint="pre-norm decoder-only, packed QKV with bias, exact-erf GELU, no attention-branch dropout"
      testid="lex-explain-model"
    >
      <p>
        With <code>V</code> embedding rows, <code>d</code> = <code>d_model</code>,
        <code>H</code> heads and <code>dh = d/H</code>; LayerNorm is affine with
        <code>eps = {LAYER_NORM_EPS}</code> over the last axis:
      </p>
      <!-- A scroll container must be focusable or keyboard-only users cannot reach the
           overflow (WCAG 2.1.1). The linter only knows the element is non-interactive. -->
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div class="eq" role="group" aria-label="forward pass" tabindex="0">
        h = embed[x] + pos[:T]<br />
        a = layernorm(h) &nbsp;·&nbsp; q, k, v = split(a W_qkvᵀ + b_qkv, 3)<br />
        A = softmax(q kᵀ / √dh + causal_mask)<br />
        h ← h + (A v) W_projᵀ + b_proj <span class="note">(no dropout on this branch)</span><br />
        m = layernorm(h) &nbsp;·&nbsp; h ← h + gelu(m W₁ᵀ + b₁) W₂ᵀ + b₂<br />
        logits = layernorm(h) · head_wᵀ <span class="note">(no bias)</span>
      </div>
      <p>
        The MLP is <code>d → {MLP_RATIO}d → d</code> and <code>gelu</code> is the
        <b>exact</b> form <code>0.5·x·(1 + erf(x/√2))</code>, not the tanh approximation.
        Dropout sits on the embedding sum, on the <b>attention weights</b>, and after the
        second MLP linear — an unusual placement, inherited faithfully from the source
        model this tab derives from. Its default here is <b>{DEFAULT_DROPOUT}</b>, not the
        source's hard-coded 0.1: a demo people re-run should be deterministic by default.
      </p>
      <p>
        The initialization is mixed, and deliberately so. Every matrix is
        <code>N(0, 0.02²)</code> except the packed QKV projection, which keeps PyTorch's
        xavier-uniform default because the source's initializer only matches
        <code>nn.Linear</code> and <code>nn.Embedding</code>. Reproducing that keeps us
        honest about what the source model actually is rather than what it meant to be.
      </p>
    </Explain>

    <Explain
      title="Reading the geometry honestly"
      hint="why the ceiling and the random-init baseline are never optional"
      testid="lex-explain-geometry"
    >
      <p>
        The spectrum comes from the <b>column-mean-centred</b> embedding
        <code>Ac = A − mean(A, axis=0)</code>, via the symmetric eigendecomposition of the
        <code>d×d</code> Gram matrix <code>G = Acᵀ Ac</code>: the eigenvalues <b>are</b>
        <code>σᵢ²</code>, so no SVD runs and the whole thing costs milliseconds in a
        browser.
      </p>
      <p>
        Centring is why the largest attainable rank is <code>min(V−1, d)</code>. That bound
        is <b>mechanical</b> — a random matrix at the same shape gains rank as the
        vocabulary grows too. So effective rank is never displayed on its own here: the
        panel draws the ceiling and an untrained model's spectrum beside it, and if the
        trained curve sits on the random one, that is the finding.
      </p>
      <p>
        The token cloud is a <b>PCA projection</b>, with its explained variance printed. It
        is not the model's representation — unlike the
        <button class="linklike" onclick={() => view.set("geometry")}>Geometry Lab</button>'s
        sphere, where <code>d_model = 3</code> and the picture <i>is</i> the space.
      </p>
    </Explain>

    <Explain
      title="Where this comes from, and what was left out"
      hint="a derivation, not a port — with the omissions named"
      testid="lex-explain-provenance"
    >
      <p>
        This tab imports ideas and corrected code from a source project, never its claims.
        That project ships no trained checkpoint, no corpus and no run manifest, and most
        of its figures are schematics. Every number here is generated live by a model that
        actually trained.
      </p>
      <ul>
        <li>
          The Dolch transcription is <b>corrected</b>: first grade has <code>going</code>,
          not <code>giving</code>, and <code>Santa Claus</code> — which a word-level
          tokenizer can never match as one token — is dropped, so the largest list is
          stated at the size it actually has rather than the often-cited 315.
        </li>
        <li>
          Weight decay applies to <b>2-D weight matrices only</b> — not embeddings, not
          positions, not biases, not LayerNorm gains. The source decays everything; this
          follows the standard convention and says so rather than inheriting it silently.
        </li>
        <li>
          A tied model reports <b>one</b> spectrum, labelled tied. The source logs the
          embedding and the readout separately even when they are the same array, which is
          one matrix counted twice.
        </li>
        <li>
          <b>Not shipped:</b> nonce-word minting (it needs a parameter-matched control that
          does not exist), and the meter/rhyme "fingerprint" (the source's meter score does
          not measure meter — under its scheme every word's stress pattern begins the same
          way, so the score converges to the template's own density whatever you feed it).
        </li>
        <li>
          Browser and Python run the <b>same recipe</b>, held to ≤1e-5 on the forward pass,
          the loss and every spectrum statistic by a golden test. Whole-run training
          equality is <b>not</b> claimed: platform BLAS and RNG streams diverge.
        </li>
      </ul>
      <p>
        Notation and this deployment's what-is-real table are in the
        <button class="linklike" onclick={() => view.set("info")}>Info tab</button>.
      </p>
    </Explain>
  </div>

  {#if corpusError}
    <div class="inline-error" data-testid="lex-corpus-error">
      <div>
        <b>The shipped corpus is not available, or did not verify — so nothing here can
        run.</b>
        {corpusError}. This tab reads
        <code>static-data/lex/corpus.json</code> — the build-time export of the real
        backend's verified corpus — and re-hashes the body it loads against the digest that
        export declares. It deliberately has no fallback: a substituted or invented corpus
        would silently change every coverage counter, every loss and every spectrum on this
        page.
      </div>
      <button class="retry" onclick={() => void loadCorpus()}>retry</button>
    </div>
  {/if}

  <div class="grid two">
    <div class="card">
      <BudgetPanel
        source={budgetSource}
        budget={budgetName}
        {vocab}
        {coverage}
        corpusLabel={corpus?.label ?? ""}
        onSource={(s) => (budgetSource = s as BudgetSource)}
        onBudget={(b) => (budgetName = b)}
      />
    </div>
    <div class="card">
      <ModelPanel
        {dModel}
        {nLayers}
        {nHeads}
        {ctx}
        {tied}
        {dropout}
        {nParams}
        vocabRows={vocab?.rows ?? 0}
        budgetSize={vocab?.budgetSize ?? 0}
        onDModel={(v) => (dModel = v)}
        onNLayers={(v) => (nLayers = v)}
        onNHeads={(v) => (nHeads = v)}
        onCtx={(v) => (ctx = v)}
        onTied={(v) => (tied = v)}
        onDropout={(v) => (dropout = v)}
      />
    </div>
  </div>

  <div class="card">
    <TrainPanel
      {cfg}
      {budgetSource}
      {budgetName}
      corpusText={corpus?.text ?? ""}
      corpusLabel={corpus?.label ?? ""}
      trainedModel={trained?.model ?? null}
      trainedVocab={trained?.vocab ?? null}
      trainedNote={trained?.note ?? ""}
      {onTrained}
      onAdoptCorpus={adoptCorpus}
    />
  </div>

  <div class="grid two">
    <div class="card">
      <SpectrumPanel
        embedding={embedSpectrum}
        readout={readoutSpectrum}
        baseline={baselineSpectrum}
        tied={activeModel?.cfg.tied ?? tied}
        {provenance}
        dModel={activeModel?.cfg.dModel ?? dModel}
        budgetSize={activeVocab?.budgetSize ?? 0}
        vocabRows={activeModel?.cfg.vocabRows ?? 0}
      />
    </div>
    <div class="card">
      <TokenCloud
        spectrumResult={embedSpectrum}
        words={activeVocab?.itos ?? []}
        {provenance}
      />
    </div>
  </div>

  <div class="card">
    <SamplePanel model={activeModel} vocab={activeVocab} {provenance} />
  </div>

  <div class="card">
    <ForwardPassPanel model={activeModel} vocab={activeVocab} {provenance} />
  </div>

  <div class="card">
    <LexWeightLab
      base={baseModel}
      baseLabel={baseLabelOf(provenance)}
      vocab={activeVocab}
      {edited}
      onEdited={(model, token, note) => (edited = { model, token, note })}
      onRestore={() => (edited = null)}
    />
  </div>

  <div class="card">
    <ModelFile
      model={activeModel}
      vocab={activeVocab}
      {provenance}
      note={activeNote}
      onLoaded={onLoadedModel}
    />
  </div>
</section>

<style>
  .viz {
    padding: 1.2rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    /* 390px: nothing in this tab may push the page sideways. Wide content (the spectrum
       chart, the token cloud, the equations) scrolls inside its own container instead. */
    overflow-x: hidden;
  }
  header h2 {
    margin: 0;
    font-size: 1.1rem;
  }
  .sub {
    margin: 0.2rem 0 0;
    color: var(--text-dim);
    font-size: 0.82rem;
    line-height: 1.6;
    max-width: 62rem;
  }
  .sub b {
    color: var(--text);
  }
  .sub code {
    font-family: var(--mono);
    font-size: 0.94em;
    color: var(--accent);
  }
  .chips {
    margin: 0.45rem 0 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .chip {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.12rem 0.55rem;
  }
  .chip.active {
    color: var(--text);
    border-color: var(--accent);
  }
  .chip.verified {
    color: var(--good);
    border-color: rgba(91, 224, 176, 0.4);
  }
  .provenance {
    margin: 0.4rem 0 0;
    font-size: 0.72rem;
    line-height: 1.6;
    color: var(--text-dim);
    max-width: 62rem;
  }
  .provenance b {
    color: var(--text);
  }
  .provenance code,
  .provenance .mono {
    font-family: var(--mono);
  }
  .explainers {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr));
    gap: 0.9rem;
    align-items: start;
  }
  .card {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    min-width: 0; /* grid children default to min-content and would refuse to shrink */
  }
  .inline-error {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    flex-wrap: wrap;
    background: rgba(255, 122, 144, 0.12);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.6rem 0.75rem;
    font-size: 0.78rem;
    line-height: 1.55;
  }
  .inline-error > div {
    flex: 1;
    min-width: 14rem;
  }
  .inline-error code {
    font-family: var(--mono);
  }
  .inline-error .retry {
    background: transparent;
    border: 1px solid rgba(255, 122, 144, 0.5);
    color: var(--bad);
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
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
  .chips .linklike {
    font-size: 0.68rem;
    font-family: var(--mono);
  }
</style>
