<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  import {
    client,
    debounced,
    ApiError,
    type GeoSpec,
    type GeoToken,
    type GeoTokenizeResult,
    type GeoTrace,
    type GeoVectorFieldData,
    type GeoVectorFieldParams,
  } from "../../lib/dataClient";
  import {
    geoPrompt,
    geoFieldMode,
    geoLayer,
    geoTemperature,
    geoTopM,
    geoAntisymmetrize,
    geoWeightsToken,
    geoModelNote,
    type GeoLayerSelection,
  } from "../../lib/explorerStores";
  import Progress from "../../lib/Progress.svelte";
  import ExportBar from "../../controls/ExportBar.svelte";
  import GeoScene from "./GeoScene.svelte";
  import TokenStrip from "./TokenStrip.svelte";
  import WeightLab from "./WeightLab.svelte";
  import FinetunePanel from "./FinetunePanel.svelte";
  import TrainPanel from "./TrainPanel.svelte";
  import AttentionView from "./AttentionView.svelte";
  import { SPECIALS, tokenText, verifyVocab } from "./vocab";

  // Geometry Lab (issue #1's "deeper" demo; spec User Story 2): a tiny transparent
  // GeoTransformer whose 3-D token embeddings live on a real sphere. First open
  // trains it once (job + epoch-labeled SSE progress); afterwards everything —
  // vector fields, weight edits, fine-tunes, attention — is live and real.
  type Phase = "boot" | "training" | "ready" | "error";

  let scene = $state<GeoScene | undefined>();
  let phase = $state<Phase>("boot");
  let bootError = $state("");
  let progress = $state(0);
  let progressMsg = $state("");
  let spec = $state<GeoSpec | null>(null);

  let field = $state<GeoVectorFieldData | null>(null);
  // Starts true: between "ready" and the first field landing the sphere would
  // otherwise sit unlabeled-empty for a beat (red-team static finding #4).
  let fieldLoading = $state(true);
  let fieldError = $state("");
  let trace = $state<GeoTrace | null>(null);
  let traceError = $state("");
  let tokenized = $state<GeoTokenizeResult | null>(null);

  let unsubTrain: (() => void) | null = null;
  let fieldCtl: AbortController | null = null;
  let traceCtl: AbortController | null = null;

  // ---- token labels: bundled deterministic vocab, verified live; plus everything the
  // API has told us (tokenize/trace) as an always-correct fallback.
  let vocabVerified = $state(false);
  /**
   * The ACTIVE model's own word list, when it has one (a scratch-trained or imported
   * model). Without it every hover label fell back to "token #N" while the header
   * still promised "hover a dot for its word" — and the words were right there in the
   * model bundle the whole time.
   */
  let modelWords = $state<string[] | null>(null);
  const learnedLabels = new Map<number, string>();
  function noteTokens(tokens: GeoToken[]) {
    for (const t of tokens) if (!t.unk) learnedLabels.set(t.id, t.text);
  }
  function label(id: number): string {
    if (id >= 0 && id < 3) return SPECIALS[id];
    if (modelWords) return modelWords[id - 3] ?? `token #${id}`;
    if (vocabVerified) return tokenText(id);
    return learnedLabels.get(id) ?? `token #${id}`;
  }

  onMount(() => {
    void boot();
  });
  onDestroy(() => {
    unsubTrain?.();
    fieldDeb.cancel();
    traceDeb.cancel();
    tokDeb.cancel();
    fieldCtl?.abort();
    traceCtl?.abort();
  });

  // ---- first-load gate: spec -> (train | attach) -> ready --------------------------

  async function boot() {
    phase = "boot";
    bootError = "";
    progress = 0;
    progressMsg = "checking the tiny model…";
    try {
      const s = await client.getGeoSpec();
      spec = s;
      const cp = s.checkpoint;
      if (cp.status === "ready") {
        readyUp();
      } else if (cp.status === "training" && cp.job_id) {
        attach(cp.job_id); // someone else already kicked training off — ride along
      } else {
        const t = await client.geoTrain(); // idempotent, single-flight
        if (t.ready) readyUp();
        else if (t.job_id) attach(t.job_id);
        else throw new ApiError("ComputeError", "training was accepted but no job id came back");
      }
    } catch (e) {
      phase = "error";
      bootError = friendly(e);
    }
  }

  function attach(jobId: string) {
    phase = "training";
    progressMsg = "training the tiny transformer…";
    unsubTrain?.();
    unsubTrain = client.subscribeProgress(jobId, {
      onProgress: (p, m) => {
        progress = p;
        progressMsg = m; // "epoch 7/30 · loss 4.12"
      },
      onDone: () => {
        void client.getGeoSpec().then((s) => (spec = s)).catch(() => {});
        readyUp();
      },
      onError: (_type, message) => {
        phase = "error";
        bootError = `Training hit a snag: ${message}`;
      },
    });
  }

  function readyUp() {
    phase = "ready";
    revalidateVocab();
  }

  /**
   * The bundled vocab table is the CANONICAL model's. A model trained from scratch has
   * its own words, so the probe legitimately fails for it and `label()` falls back to
   * the ids the API actually reported — which is correct rather than confidently wrong.
   * Re-run whenever the active model changes.
   */
  function revalidateVocab(): void {
    const token = $geoWeightsToken ?? undefined;
    vocabVerified = false;
    modelWords = null;
    learnedLabels.clear();
    void verifyVocab((text) => client.geoTokenize(text, token)).then((ok) => {
      if (($geoWeightsToken ?? undefined) === token) vocabVerified = ok;
    });
    if (!token) return;
    // The probe above fails for a model with its own vocabulary (correctly — the
    // bundled table is the canonical model's), so fetch the real word list.
    void client
      .geoExportModel(token)
      .then((bundle) => {
        if (($geoWeightsToken ?? undefined) !== token) return;
        const parsed = JSON.parse(bundle.vocab) as { words?: unknown };
        if (Array.isArray(parsed.words)) modelWords = parsed.words as string[];
      })
      .catch(() => {
        // Labels fall back to the ids the API reported — wrong-looking, never wrong.
      });
  }

  // ---- debounced, abortable data fetches (FR-108 cancel-and-restart) ---------------

  const fieldDeb = debounced((params: GeoVectorFieldParams) => {
    fieldCtl?.abort();
    const ctl = new AbortController();
    fieldCtl = ctl;
    fieldLoading = true;
    client
      .getGeoVectorField(params, ctl.signal)
      .then((d) => {
        if (ctl !== fieldCtl) return;
        field = d;
        fieldError = "";
        fieldLoading = false;
      })
      .catch((e) => {
        if (ctl !== fieldCtl || isAbort(e)) return;
        if (healEvictedToken(e)) return; // token cleared -> effect refires on learned
        fieldError = friendly(e);
        fieldLoading = false;
      });
  }, 400);

  const traceDeb = debounced((prompt: string, token: string | undefined) => {
    traceCtl?.abort();
    const ctl = new AbortController();
    traceCtl = ctl;
    client
      .getGeoTrace(prompt, token, ctl.signal)
      .then((d) => {
        if (ctl !== traceCtl) return;
        trace = d;
        traceError = "";
        noteTokens(d.tokens);
      })
      .catch((e) => {
        if (ctl !== traceCtl || isAbort(e)) return;
        if (healEvictedToken(e)) return;
        trace = null;
        traceError = friendly(e);
      });
  }, 400);

  const tokDeb = debounced((text: string) => {
    client
      .geoTokenize(text, $geoWeightsToken ?? undefined)
      .then((r) => {
        tokenized = r;
        noteTokens(r.tokens);
      })
      .catch(() => {}); // the strip is a preview — trace/field errors already surface
  }, 400);

  // Effective layer: force mode is per-layer by definition (the contract 400s on
  // "full"), so the UI never sends that combination.
  const effLayer = $derived<GeoLayerSelection>(
    $geoFieldMode === "force" && $geoLayer === "full" ? 0 : $geoLayer,
  );

  $effect(() => {
    if (phase !== "ready") return;
    const params: GeoVectorFieldParams = {
      mode: $geoFieldMode,
      layer: effLayer,
      prompt: $geoPrompt,
      weights_token: $geoWeightsToken ?? undefined,
      temperature: $geoFieldMode === "next_next" ? $geoTemperature : undefined,
      top_m: $geoFieldMode === "next_next" ? $geoTopM : undefined,
      antisymmetrize: $geoFieldMode === "force" ? $geoAntisymmetrize : undefined,
    };
    fieldDeb(params);
  });

  $effect(() => {
    if (phase !== "ready") return;
    // An empty prompt is a guaranteed 400 ("prompt is empty after tokenization"); the
    // UI already hid the message, but the request still went out and logged an error.
    if (!$geoPrompt.trim()) {
      traceDeb.cancel();
      traceCtl?.abort();
      trace = null;
      traceError = "";
      return;
    }
    traceDeb($geoPrompt, $geoWeightsToken ?? undefined);
  });

  let vocabForToken: string | null | undefined;
  $effect(() => {
    if (phase !== "ready") return;
    const token = $geoWeightsToken;
    if (token !== vocabForToken) {
      vocabForToken = token;
      revalidateVocab();
    }
  });

  $effect(() => {
    if (phase !== "ready") return;
    tokDeb($geoPrompt);
  });

  function setMode(mode: "next_next" | "force") {
    if (mode === "force" && $geoLayer === "full") geoLayer.set(0); // "full" is next_next-only
    geoFieldMode.set(mode);
  }

  const maxResidual = $derived.by(() => {
    const sf = field?.sequence_forces;
    if (!sf || sf.length === 0) return null;
    return Math.max(...sf.map((f) => f.normal_residual));
  });

  function isAbort(e: unknown): boolean {
    return e instanceof DOMException && e.name === "AbortError";
  }

  // A sessionStorage-restored weights_token can outlive its server-side artifact
  // (LRU eviction). Rather than wedging the view in a retry loop against a dead
  // token, drop it — the field/trace effects re-fire against the learned weights.
  function healEvictedToken(e: unknown): boolean {
    if (e instanceof ApiError && e.type === "NotFoundError" && $geoWeightsToken) {
      geoModelNote.set(null); // the note would otherwise describe a model that is gone
      geoWeightsToken.set(null);
      return true;
    }
    return false;
  }

  function friendly(e: unknown): string {
    if (e instanceof ApiError) {
      if (e.type === "NetworkError") return "Could not reach the server — is the backend running?";
      return e.message;
    }
    return String(e);
  }

  const LAYERS: GeoLayerSelection[] = ["full", 0, 1, 2, 3];
</script>

<section class="viz panel" data-testid="geo-view" data-ready={phase === "ready" ? 1 : 0}>
  <header>
    <div>
      <h2>Geometry Lab — a tiny transparent transformer</h2>
      <p class="sub">
        A GeoTransformer (<b>d_model = 3</b>, 4 layers, 1 head, 1000-word vocab) whose token
        embeddings genuinely live on this sphere — no dimensionality reduction. Hover a dot for
        its word; drag to rotate, scroll to zoom. Edit the weights, fine-tune it, or train a new
        one on your own text below. (W_V and the embedding move the field most; at
        temperature 0 the next-next field is an argmax, so W_Q/W_K often shift only a
        few arrows.)
      </p>
      {#if $geoWeightsToken}
        <!-- A different model is driving the sphere: describing the shipped checkpoint's
             corpus and loss here would be plainly false, so those chips are withheld. -->
        <p class="chips" data-testid="geo-active-model">
          <span class="chip active">active model: {$geoModelNote ?? "one you created this session"}</span>
          <span class="chip mono" title="content hash of the active weights">{$geoWeightsToken.slice(0, 8)}</span>
        </p>
      {:else if spec}
        <p class="chips" data-testid="geo-active-model">
          <span class="chip">shipped checkpoint · corpus {spec.model.corpus}</span>
          {#if spec.checkpoint.final_loss != null}<span class="chip">final loss {spec.checkpoint.final_loss.toFixed(2)}</span>{/if}
          {#if spec.checkpoint.coverage_uniformity != null}<span class="chip">coverage {spec.checkpoint.coverage_uniformity.toFixed(2)}</span>{/if}
          {#if spec.checkpoint.field_directional_entropy != null}<span class="chip">field entropy {spec.checkpoint.field_directional_entropy.toFixed(2)}</span>{/if}
        </p>
      {/if}
    </div>
  </header>

  {#if phase === "boot" || phase === "training"}
    <div class="gate" data-testid="geo-progress">
      <div class="gate-title">
        {phase === "boot" ? "Waking up the Geometry Lab…" : "Training the tiny transformer (once — it's cached forever after)"}
      </div>
      <Progress {progress} message={progressMsg} />
    </div>
  {:else if phase === "error"}
    <div class="gate error-gate" data-testid="geo-error">
      <div class="gate-title">The tiny model couldn't get ready</div>
      <p class="gate-msg">{bootError}</p>
      <button onclick={() => void boot()}>Try again</button>
    </div>
  {:else}
    <div class="controls-row">
      <div class="ctl">
        <span class="ctl-label">field</span>
        <div class="seg" data-testid="geo-mode" role="tablist">
          <button class:active={$geoFieldMode === "next_next"} onclick={() => setMode("next_next")} title="from each token: where would the model go next-next?">next-next</button>
          <button class:active={$geoFieldMode === "force"} onclick={() => setMode("force")} title="the paper's attention force field W_V·z (arXiv:2607.13295)">force</button>
        </div>
      </div>
      <div class="ctl">
        <span class="ctl-label">layer</span>
        <div class="seg" data-testid="geo-layer" role="tablist">
          {#each LAYERS as l (l)}
            <button
              class:active={effLayer === l}
              disabled={$geoFieldMode === "force" && l === "full"}
              title={$geoFieldMode === "force" && l === "full" ? "the force field is per-layer by definition" : undefined}
              onclick={() => geoLayer.set(l)}
            >{l === "full" ? "full" : l}</button>
          {/each}
        </div>
      </div>
      {#if $geoFieldMode === "next_next"}
        <label class="ctl slider">
          <span class="ctl-label">temperature <b>{$geoTemperature.toFixed(2)}</b></span>
          <input type="range" min="0" max="2" step="0.05" value={$geoTemperature} oninput={(e) => geoTemperature.set(Number(e.currentTarget.value))} />
        </label>
        <label class="ctl slider narrow" class:inert={$geoTemperature === 0}>
          <span class="ctl-label">
            arrows/point <b>{$geoTopM}</b>
            {#if $geoTemperature === 0}<span class="why" title="At temperature 0 the next-token distribution is one-hot, so there is only ever one arrow to draw. Raise the temperature to fan them out.">· needs T &gt; 0</span>{/if}
          </span>
          <input
            type="range"
            min="1"
            max="5"
            step="1"
            value={$geoTopM}
            disabled={$geoTemperature === 0}
            oninput={(e) => geoTopM.set(Number(e.currentTarget.value))}
          />
        </label>
      {:else}
        <label class="ctl check">
          <input type="checkbox" checked={$geoAntisymmetrize} onchange={(e) => geoAntisymmetrize.set(e.currentTarget.checked)} />
          <span>antisymmetrize (W_V−W_Vᵀ)/2</span>
        </label>
        <!-- Two DIFFERENT facts, so both are shown: whether the per-point field is
             tangent by construction, and how much radial pull was projected out of the
             aggregate forces before drawing them. Showing only the first would hide
             what the projection removed. -->
        {#if field?.tangent_exact}
          <span class="badge tangent" data-testid="geo-tangent-badge" title="(W_V−W_Vᵀ)/2 is antisymmetric, so ⟨Az,z⟩ = 0 — the per-token field is exactly tangent to the sphere at every point">per-token field: exactly tangent</span>
        {/if}
        {#if maxResidual != null}
          <span class="badge residual" data-testid="geo-residual-badge" title="Each aggregate force is drawn tangent at its own token, so its radial component is projected away first. This is the largest amount removed across the prompt — antisymmetrizing W_V does not reduce it, because each term is tangent at z_j rather than at the z_i where the sum is drawn.">radial pull projected out: {maxResidual.toFixed(3)} max</span>
        {/if}
      {/if}
      {#if fieldLoading}<span class="computing">computing field…</span>{/if}
    </div>

    <div class="prompt-row">
      <label class="prompt">
        <span class="ctl-label">prompt</span>
        <input
          type="text"
          value={$geoPrompt}
          data-testid="geo-prompt"
          placeholder="type a prompt (corpus words work best — others become <unk>)"
          oninput={(e) => geoPrompt.set(e.currentTarget.value)}
        />
      </label>
      <TokenStrip result={tokenized} />
      {#if traceError && $geoPrompt.trim().length > 0}
        <div class="inline-error" data-testid="geo-error">{traceError}</div>
      {/if}
    </div>

    {#if fieldError}
      <div class="inline-error" data-testid="geo-error">
        {fieldError}
        <button class="retry" onclick={() => fieldDeb({ mode: $geoFieldMode, layer: effLayer, prompt: $geoPrompt, weights_token: $geoWeightsToken ?? undefined })}>retry</button>
      </div>
    {/if}

    <GeoScene
      bind:this={scene}
      {field}
      traceEmbeddings={trace?.embeddings ?? null}
      traceTokens={trace?.tokens ?? null}
      {label}
    />
    <div class="caption-row">
      <p class="caption">
        {#if $geoFieldMode === "next_next"}
          each arrow: append that token to the prompt, then follow it to the model's <i>next</i> prediction — brighter = more probable
          {#if $geoTemperature > 0 && $geoTopM > 1}· {$geoTopM} weighted arrows per token at T={$geoTemperature.toFixed(2)}{/if}
        {:else}
          thin arrows: the per-token field {$geoAntisymmetrize ? '((W_V−W_Vᵀ)/2)·z' : 'W_V·z'} at layer {effLayer} · <span class="force-key">amber arrows</span>: the prompt's aggregate attention forces, drawn tangent to the sphere at each token{#if maxResidual != null}&nbsp;(up to {maxResidual.toFixed(3)} of radial pull projected away — see the badge above){/if} · <span class="path-key">green path</span>: the prompt's tokens across the sphere, hidden where it passes behind
        {/if}
      </p>
      <ExportBar name="geometry-sphere" webglCanvas={() => scene?.canvasEl()} />
    </div>

    <div class="grid">
      <div class="card"><WeightLab {label} checkpointId={spec?.checkpoint.checkpoint_id ?? null} /></div>
      <div class="card"><FinetunePanel /></div>
      <div class="card"><TrainPanel /></div>
    </div>
    <div class="card"><AttentionView {trace} /></div>
  {/if}
</section>

<style>
  .viz {
    padding: 1.2rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }
  header h2 {
    margin: 0;
    font-size: 1.1rem;
  }
  .sub {
    margin: 0.2rem 0 0;
    color: var(--text-dim);
    font-size: 0.82rem;
    max-width: 62rem;
  }
  .chips {
    margin: 0.45rem 0 0;
    display: flex;
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
  .gate {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    padding: 2.2rem 1.6rem;
    border: 1px dashed var(--border);
    border-radius: 12px;
    background: radial-gradient(circle at 50% 0%, rgba(110, 168, 254, 0.06), transparent 70%);
  }
  .gate-title {
    font-size: 0.95rem;
    font-weight: 600;
  }
  .error-gate {
    border-color: rgba(255, 122, 144, 0.4);
  }
  .gate-msg {
    margin: 0;
    color: var(--bad);
    font-size: 0.84rem;
  }
  .error-gate button {
    align-self: flex-start;
    padding: 0.45rem 1.1rem;
  }
  .controls-row {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    flex-wrap: wrap;
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
  }
  .seg {
    display: inline-flex;
    gap: 0.2rem;
    padding: 0.2rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
  }
  .seg button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.26rem 0.7rem;
    font-size: 0.76rem;
    font-weight: 500;
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
  .slider {
    min-width: 150px;
  }
  .slider.narrow {
    min-width: 110px;
  }
  .check {
    flex-direction: row;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: var(--text);
    padding-bottom: 0.35rem;
  }
  .check input {
    accent-color: var(--accent);
  }
  .badge {
    font-family: var(--mono);
    font-size: 0.7rem;
    border-radius: 999px;
    padding: 0.18rem 0.6rem;
    margin-bottom: 0.3rem;
  }
  .badge.tangent {
    color: var(--good);
    background: rgba(91, 224, 176, 0.12);
    border: 1px solid rgba(91, 224, 176, 0.35);
  }
  .badge.residual {
    color: #ffb454;
    background: rgba(255, 180, 84, 0.1);
    border: 1px solid rgba(255, 180, 84, 0.35);
  }
  .computing {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent);
    padding-bottom: 0.4rem;
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    50% {
      opacity: 0.45;
    }
  }
  .prompt-row {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
  .prompt {
    display: flex;
    flex-direction: column;
    gap: 0.28rem;
  }
  .prompt input {
    font-family: var(--mono);
    font-size: 0.86rem;
  }
  .inline-error {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    background: rgba(255, 122, 144, 0.12);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    font-size: 0.8rem;
  }
  .inline-error .retry {
    background: transparent;
    border: 1px solid rgba(255, 122, 144, 0.5);
    color: var(--bad);
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
  }
  .ctl.inert { opacity: 0.55; }
  .why { color: #ffb454; }
  .caption-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .caption-row .caption { flex: 1; min-width: 260px; }
  .caption {
    margin: 0;
    color: var(--text-dim);
    font-size: 0.75rem;
  }
  .force-key {
    color: #ffb454;
  }
  .path-key {
    color: var(--good);
  }
  .grid {
    display: grid;
    /* auto-fit: with three cards a fixed 2-column grid left the third one alone in a
       row beside an empty column. */
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr));
    gap: 0.9rem;
    align-items: start;
  }
  @media (max-width: 860px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
  .card {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
  }
</style>
