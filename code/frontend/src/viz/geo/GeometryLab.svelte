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
  import SegmentedControl from "../../lib/SegmentedControl.svelte";
  import ExportBar from "../../controls/ExportBar.svelte";
  import GeoScene from "./GeoScene.svelte";
  import TokenStrip from "./TokenStrip.svelte";
  import WeightLab from "./WeightLab.svelte";
  import FinetunePanel from "./FinetunePanel.svelte";
  import TrainPanel from "./TrainPanel.svelte";
  import AttentionView from "./AttentionView.svelte";
  import Explain from "../../lib/Explain.svelte";
  import { view } from "../../lib/stores";
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
        A GeoTransformer (<b>d_model = 3</b>, 4 layers, 1 head, 1000-word vocab), really trained on
        a real corpus. Its 1003 token embeddings are unit vectors, so this sphere <b>is</b> the
        model's embedding space at full rank — not a PCA, t-SNE, or UMAP projection of a bigger one.
        Each dot is a token; hover it for its word. Drag to rotate, scroll to zoom, and
        <b>⟳ spin</b> sets it turning on its own (grabbing it stops the spin). Below the sphere you can edit any weight matrix, fine-tune the model, or
        train an entirely new one on your own text — every view updates from the real model that
        results.
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
          <span class="chip" title="the checkpoint that ships with this build — trained by the real backend on the corpus named here">shipped checkpoint · corpus {spec.model.corpus}</span>
          {#if spec.checkpoint.final_loss != null}<span class="chip" title="next-token cross-entropy in nats at the end of training. A uniform model over 1003 tokens would score ln 1003 ≈ 6.91, so lower than that is real learning.">final loss {spec.checkpoint.final_loss.toFixed(2)}</span>{/if}
          {#if spec.checkpoint.coverage_uniformity != null}<span class="chip" title="how evenly the 1003 embeddings occupy the sphere: normalized entropy of their occupancy over 64 equal-area bins, 0 = one cluster, 1 = perfectly spread. The test suite requires ≥ 0.80. This measures SPREAD, not learning — an untrained model scores higher, not lower.">coverage {spec.checkpoint.coverage_uniformity.toFixed(2)}</span>{/if}
          {#if spec.checkpoint.field_directional_entropy != null}<span class="chip" title="how many distinct directions the next-next field points in: entropy of arrow directions over the same 64 bins, in nats, max ln 64 ≈ 4.16. The test suite requires ≥ 2.0. This guards against COLLAPSE, not against failing to learn — a model trained on structureless text scores higher than this one. Read the final loss for that.">field entropy {spec.checkpoint.field_directional_entropy.toFixed(2)}</span>{/if}
        </p>
      {/if}
    </div>
  </header>

  <!-- Outside the phase gate on purpose: the first open runs a real training job, and
       that wait is exactly when a newcomer most needs to know what they are waiting for.
       Nothing in these panels depends on `spec`, `field`, or `trace`. -->
  <div class="explainers">
    <Explain
      title="The model, exactly"
      hint="4 layers, 1 head, no layer norm, tied unembedding — the whole forward pass"
      testid="geo-explain-model"
    >
      <p>
        Decoder-only. Token embeddings <code>E</code> are 1003 unit vectors on <code>S²</code>;
        positions are learned absolute embeddings (50 more 3-vectors; trained and saved with the
        rest, though the weight editor does not expose them).
        With <code>z</code> the residual stream at each position:
      </p>
      <!-- A scroll container must be focusable or keyboard-only users cannot reach the
           overflow (WCAG 2.1.1). The linter only knows the element is non-interactive. -->
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div class="eq" role="group" aria-label="equation" tabindex="0">
        z_i⁽⁰⁾ = E[t_i] + p_i<br />
        q_i = W_Q z_i, &nbsp; k_i = W_K z_i, &nbsp; v_i = W_V z_i<br />
        A_ij = softmax_j ⟨k_j, q_i⟩ &nbsp; over j ≤ i<br />
        z_i ← z_i + W_O {"Σ_{j≤i}"} A_ij v_j<br />
        z_i ← z_i + W_outᵀ gelu(W_inᵀ z_i + b_in) + b_out<br />
        logits = E z <span class="note">(tied unembedding)</span>
      </div>
      <p>
        Two deliberate departures from the standard block. Attention scores are
        <b>unscaled</b> — <code>⟨k_j, q_i⟩</code> with no <code>1/√d</code> — so the trace and the
        force field are literally the same numbers. And there is <b>no layer norm</b>: at
        <code>d = 3</code> it would erase precisely the radial information this tab exists to
        show. The embedding is instead held on the sphere by renormalizing during training.
      </p>
      <p>
        Because the unembedding is tied, reading out at an intermediate layer <i>is</i> the
        <b>logit lens</b> — what the model would predict if it stopped after that layer. That is
        what the <b>layer</b> control selects in next-next mode.
      </p>
    </Explain>

    <Explain
      title="What the arrows mean"
      hint="the two field modes, and why the amber ones are projected"
      testid="geo-explain-fields"
    >
      <p>
        <b>next-next</b> asks, for every one of the 1003 tokens: <i>if the model went here, where
        would it go from here?</i> Each token is appended to your prompt, the model runs, and an
        arrow is drawn from that token's embedding toward what it would predict next — the argmax
        at temperature 0, or the top-m targets weighted by probability above 0. Brighter is more
        probable. At <code>T = 0</code> the distribution is one-hot, so there is exactly one arrow
        per point however “arrows/point” is set, which is why that slider disables itself.
      </p>
      <p>
        <b>force</b> draws the attention field of a single layer. Two different objects share the
        view:
      </p>
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div class="eq" role="group" aria-label="equation" tabindex="0">
        <b class="k-thin">thin arrows</b> — the per-point field x ↦ W_V x, over all 1003 embedding points<br />
        <b class="k-amber">amber arrows</b> — the prompt's aggregate forces F_i = {"Σ_{j≤i}"} A_ij v_j
      </div>
      <p>
        <code>F_i</code> is literally the model's <code>attention @ v</code> row. Ticking
        <b>antisymmetrize</b> substitutes <code>(W_V − W_Vᵀ)/2</code> in the per-point field only;
        for antisymmetric <code>A</code>, <code>⟨Ax, x⟩ = 0</code> identically, so that field is
        exactly tangent to the sphere everywhere — an algebraic fact, not a measurement.
      </p>
      <p>
        It does <b>not</b> make the amber arrows tangent. Each term <code>W_V z_j</code> is
        tangent at <code>z_j</code>, but the sum is drawn anchored at <code>z_i</code>, where it
        has no reason to be tangent. So each aggregate force is projected onto the tangent plane
        at the point it is drawn from, <code>F_i − ⟨F_i, ẑ_i⟩ẑ_i</code>, and the badge reports
        the largest radial component that projection removed. A picture that asserts a geometric
        property should show the number that could falsify it.
      </p>
      <p>
        The <span class="path-key">green path</span> traces your prompt's tokens across the
        sphere in order, hidden where it passes behind, ending on a white dot at the last token.
      </p>
      <p>
        <b>Arrow length is relative, not absolute.</b> Each render scales its arrows so the
        90th-percentile magnitude reaches a fixed on-screen length, clipping anything longer, and
        the two arrow classes are scaled independently. So multiplying <code>W_V</code> by a
        constant leaves the thin arrows pixel-identical, and lengths do not compare across renders
        or between the two classes. Trust directions, relative lengths within one field, and the
        colour ramp; for absolute magnitudes, read the badges.
      </p>
      <p>
        Both fields are <b>conditioned on your prompt</b> — retyping it redraws every arrow. And an
        arrow's tail is the token's embedding, not the residual-stream state the prediction came
        from: the tail is where the token lives, not where the model was.
      </p>
    </Explain>

    <Explain
      title="What can I change, and what happens"
      hint="which matrices move which picture — and what barely moves it"
      testid="geo-explain-controls"
    >
      <ul>
        <li>
          <b>W_V and the embedding</b> move the force field most — the thin arrows literally
          <i>are</i> <code>W_V x</code> over the embedding points.
        </li>
        <li>
          <b>W_Q and W_K</b> change only the attention matrix <code>A</code>. The attention map
          and the amber forces respond at once, but at temperature 0 the next-next field is an
          argmax, so small changes there often move only a few arrows.
        </li>
        <li>
          <b>W_O</b> controls how strongly attention output re-enters the residual stream — watch
          how far the green path departs from the sphere.
        </li>
        <li>
          <b>Presets</b> per matrix: <code>identity</code>, <code>toeplitz_fuzzy</code>,
          <code>random</code>, <code>random_autocorr</code>, <code>zero</code>, and
          <code>learned</code> (back to the trained value). Editing never touches the trained
          checkpoint — each edit mints a new weight set addressed by a hash of its own contents.
        </li>
        <li>
          <b>Fine-tune</b> continues training the current weights (≤ 500 steps, default 100, lr
          1e-2), tokenizing with the active model's own vocabulary. <b>Train from scratch</b>
          builds a genuinely new model: fresh weights <i>and</i> a fresh vocabulary rebuilt from
          your text, which is why a saved model file carries its vocabulary alongside its weights
          — and why every model you derive from it keeps that vocabulary too.
        </li>
      </ul>
      <p>
        Full notation, the training recipe, and what is real in this deployment are in the
        <button class="linklike" onclick={() => view.set("info")}>Info tab</button>.
      </p>
    </Explain>
  </div>

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
      <!-- Radio groups, not tablists: these choose one of N settings and own no panels,
           so `role="tablist"` around plain buttons left the current field and the current
           layer readable only as a background colour (red-team D F5; issue #7).
           SegmentedControl carries the roles, `aria-checked`, and arrow-key navigation. -->
      <div class="ctl">
        <span class="ctl-label">field</span>
        <SegmentedControl
          testid="geo-mode"
          label="field"
          value={$geoFieldMode}
          onSelect={(v) => setMode(v as "next_next" | "force")}
          options={[
            { value: "next_next", label: "next-next", title: "from each token: where would the model go next-next?" },
            { value: "force", label: "force", title: "the paper's attention force field W_V·z (arXiv:2607.13295)" },
          ]}
        />
      </div>
      <div class="ctl">
        <span class="ctl-label">layer</span>
        <SegmentedControl
          testid="geo-layer"
          label="layer"
          value={String(effLayer)}
          onSelect={(v) => {
            // Back through LAYERS rather than Number(v): the store's type is the literal
            // union "full" | 0 | 1 | 2 | 3, so a widened `number` must not reach it.
            const picked = LAYERS.find((l) => String(l) === v);
            if (picked !== undefined) geoLayer.set(picked);
          }}
          options={LAYERS.map((l) => ({
            value: String(l),
            label: l === "full" ? "full" : String(l),
            disabled: $geoFieldMode === "force" && l === "full",
            title: $geoFieldMode === "force" && l === "full" ? "the force field is per-layer by definition" : undefined,
          }))}
        />
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
          <input
            type="checkbox"
            data-testid="geo-antisymmetrize"
            checked={$geoAntisymmetrize}
            onchange={(e) => geoAntisymmetrize.set(e.currentTarget.checked)}
          />
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
  /* `.seg` moved to lib/SegmentedControl.svelte along with the markup it styles. */
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
  .explainers {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  /* A key that names a colour is printed in it (the scene's exact values). */
  .explainers :global(.eq b.k-amber) { color: #ffb454; }
  .explainers :global(.eq b.k-thin) { color: #b794f6; }
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
