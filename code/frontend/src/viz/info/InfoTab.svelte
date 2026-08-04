<script lang="ts">
  import { view } from "../../lib/stores";
  import { STATIC_MODE } from "../../lib/staticUx";

  // The reference tab: what each visualization is, what the mathematics behind it is,
  // what you can manipulate, and — as importantly — what is NOT claimed. Every number
  // and equation here is transcribed from the code it describes:
  //   geo/model.py, geo/fields.py, geo/config.py, geo/bundle.py, geo/scratch.py,
  //   arch/{graph,trace,generate}.py, config.py.
  // If you change one of those, change the matching sentence here; the e2e docs test
  // pins the values that are cheapest to let drift.

  const SECTIONS = [
    { id: "start", label: "Start here" },
    { id: "notation", label: "Notation" },
    { id: "arch", label: "Architecture Explorer" },
    { id: "geo", label: "Geometry Lab" },
    { id: "lex", label: "Lexicon Lab" },
    { id: "real", label: "What's real" },
    { id: "limits", label: "Known limits" },
    { id: "refs", label: "Source & references" },
  ];

  function jump(id: string): void {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
</script>

<section class="viz panel info" data-testid="info-view">
  <header>
    <h2>What you're looking at</h2>
    <p class="lede">
      Three views of a transformer at three magnifications. The <b>Architecture Explorer</b> takes a
      real open-weights model and unfolds one genuine forward pass into a clickable graph. The
      <b>Geometry Lab</b> takes a transformer small enough that its entire representation space is
      a sphere — <b>d_model = 3</b> — trains it for real, and lets you move its weights around and
      watch the geometry respond. The <b>Lexicon Lab</b> holds the model small and moves the other
      dial instead: it trains a word-level transformer in your browser under a
      <b>bounded vocabulary</b>, so you can watch what a budget of a few hundred words can and
      cannot say. Nothing on any of the three is a schematic or an illustration: every tensor shown
      came out of a model that actually ran.
    </p>
    <nav class="toc" aria-label="sections">
      {#each SECTIONS as s (s.id)}
        <button onclick={() => jump(s.id)}>{s.label}</button>
      {/each}
    </nav>
  </header>

  <!-- ------------------------------------------------------------------ start -->
  <h3 id="start">Which tab do I want?</h3>
  <div class="cards">
    <article class="card">
      <h4>Architecture Explorer</h4>
      <p>
        “What are the actual operations, and what do the tensors look like as they flow through?”
      </p>
      <p>
        A real model (Qwen2.5&#8209;0.5B&#8209;Instruct by default) is traced op by op. You get the
        computational graph, the attention pattern of every head at every layer, the residual-stream
        norm at every position, the next-token distribution, and the weights themselves.
      </p>
      <button class="go" onclick={() => view.set("architecture")}>Open the Architecture Explorer →</button>
    </article>
    <article class="card">
      <h4>Geometry Lab</h4>
      <p>“Where do tokens actually <i>live</i>, and what does attention do to them?”</p>
      <p>
        A 4-layer, 1-head transformer with a 3-dimensional residual stream, trained from scratch on a
        real corpus. Its 1003 token embeddings are unit vectors, so they are drawn on the sphere
        exactly where they live — no PCA, no t-SNE, no UMAP. Edit weights, fine-tune, or train a new
        model on your own text. (Against a full local stack, opening this tab the first time trains
        the checkpoint before anything appears; the deployed site ships it pre-trained.)
      </p>
      <button class="go" onclick={() => view.set("geometry")}>Open the Geometry Lab →</button>
    </article>
    <article class="card">
      <h4>Lexicon Lab</h4>
      <p>“What can a model with a few hundred words say — and what can it never say?”</p>
      <p>
        A word-level transformer trained from scratch <i>in your browser</i> on
        <i>The Real Mother Goose</i> (1916). You pick the <b>vocabulary budget</b> — a prescribed
        Dolch list, or the corpus's own most frequent words at the same size — and the coverage
        counters, the loss, the text it generates and the geometry of its embedding matrix all
        answer together. This tab runs entirely in the browser in both modes; it never calls the
        backend.
      </p>
      <button class="go" onclick={() => view.set("lexicon")}>Open the Lexicon Lab →</button>
    </article>
  </div>

  <!-- --------------------------------------------------------------- notation -->
  <h3 id="notation">Notation</h3>
  <p class="para">
    Column-vector convention throughout: activations are column vectors and matrices act on the
    left, so <code>W_Q z</code> means the matrix times the vector. (The source uses PyTorch's
    row-major <code>h @ W.T</code>, which is the same map.)
  </p>
  <!-- A scroll container must be focusable or keyboard-only users cannot reach the
       overflow (WCAG 2.1.1). The linter only knows the element is non-interactive. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="tblwrap" role="group" aria-label="notation" tabindex="0">
    <table class="tbl notation">
    <tbody>
      <tr><td><code>V</code></td><td>vocabulary size</td><td><code>T</code></td><td>number of tokens in the prompt</td></tr>
      <tr><td><code>d</code></td><td>residual-stream width (<code>d_model</code>)</td><td><code>L</code></td><td>number of layers</td></tr>
      <tr><td><code>z_i</code></td><td>residual stream at position <code>i</code></td><td><code>E</code></td><td>token-embedding matrix, <code>V × d</code></td></tr>
      <tr><td><code>A</code></td><td>attention matrix, row-stochastic and causal</td><td><code>S²</code></td><td>the unit sphere in ℝ³</td></tr>
    </tbody>
  </table>
  </div>

  <!-- ------------------------------------------------------------------- arch -->
  <h3 id="arch">The Architecture Explorer</h3>

  <h4 class="sub">The diagram</h4>
  <p class="para">
    The graph is not hand-drawn and it is not read off the model's config. The backend runs one real
    forward pass on a short fixed prompt (<i>“The quick brown fox jumps over the lazy dog.”</i>,
    capped at 12 tokens) with hooks on every tensor operation, and builds a node for
    <b>every step that transforms the hidden state</b> — including the parameterless ones that
    architecture diagrams usually omit: rotary position embedding, the attention softmax, residual
    adds, activation functions. Edges are real dataflow, recovered from tensor identity and storage
    aliasing, with execution order as a fallback where views and copies break identity.
  </p>
  <p class="para">
    Tied weights are detected by comparing storage pointers, so a model whose embedding and output
    projection are the same tensor shows that tensor <b>once</b>; the alias carries a
    <code>tied_to</code> badge rather than double-counting the parameters. Clicking a node opens the
    inspector: shape, parameter list, and a heat map of the matrix.
  </p>
  <p class="para">
    <b>The first map you see is an overview, and it is lossy.</b> A real weight tensor has far more
    than the 4096-cell budget, so the whole-matrix view is a strided mean — computed by the backend
    live, or precomputed and quantized to 8 bits per cell on this static site. Clicking into it
    fetches the <b>exact</b> sub-window at full float precision rather than magnifying pixels you
    already have. Both are real measurements of real weights; only the zoomed one is exact.
  </p>
  <p class="para">
    Models are gated at <b>1.5B parameters</b>, decided from hub metadata <i>before</i> any weights
    download — the safetensors header when the repo has one, otherwise an architectural estimate
    from the config with a 20% safety margin, because that estimate can undercount.
  </p>

  <h4 class="sub">Smaller things on screen, named</h4>
  <ul class="para">
    <li>
      The chip beside the model picker is its load state — <code>ready</code> while a model is
      usable, <code>resolving</code> during the config check, <code>error</code> with the reason.
    </li>
    <li>
      In the token strip, the small number under each token is its <b>token id</b> in that model's
      vocabulary, and <code>↵</code> stands in for a newline inside a token so the chip stays one
      line. A <code>chat template</code> chip appears when the model's template wrapped your prompt,
      and an <code>earlier tokens dropped ⋯</code> chip appears at the left end when the prompt
      exceeded 64 tokens.
    </li>
    <li>
      The <b>Export</b> control above the Geometry sphere writes what you are looking at to a PNG,
      including the WebGL canvas.
    </li>
    <li>
      In the weight lab, the <code>source</code> badge says where the displayed matrix came from —
      the trained checkpoint, or the preset/edit that replaced it.
    </li>
    <li>
      Hover targets (token probabilities, sphere labels, norm bars) are pointer-driven. On a
      touchscreen the same numbers are readable in the panels themselves — the top-10 list, the
      token strip, and the chips — rather than only in the tooltip.
    </li>
  </ul>

  <h4 class="sub">The processing breakdown</h4>
  <p class="para">
    Below the diagram, the same forward pass is replayed. The <b>▶ playhead</b> walks the traced ops
    in true execution order and highlights each one in the diagram as it goes; the layer detail can
    follow along or stay pinned where you put it. Per layer you get:
  </p>
  <ul class="para">
    <li>
      <b>Every attention head at once</b>, as a grid of <code>T × T</code> heat maps — rows attend to
      columns, lower-triangular because the mask is causal. Click one to enlarge it. Long sequences
      are downsampled to at most 64×64 per head, which the label states when it happens.
    </li>
    <li>
      <b>‖residual stream‖ per token</b> at that layer's output. Worth knowing before you read it:
      most decoder-only LMs — base models included — park a huge-norm <b>massive activation</b> on
      the first token, often an order of magnitude above everything else. It is the
      residual-stream face of the <b>attention sink</b>, the separate (and causally linked)
      phenomenon where a large share of every attention row lands on position 0, which you can see
      in the head grid beside it. Scaling the chart to that bar would flatten every other token to
      nothing, so the scale is the largest <i>non-outlier</i> norm and genuine outliers are drawn
      clipped with stripes, counted in the “off-scale” note.
    </li>
    <li><b>The next-token distribution</b>, top 10, from the real logits.</li>
  </ul>
  <p class="para">
    Prompts are rendered through the model's own chat template when it has one (a chip in the token
    strip tells you when that happened), and truncated to the last <b>64</b> tokens. Truncation drops
    the <i>earliest</i> tokens, so the elision marker sits at the left end where the loss actually
    occurred.
  </p>

  <h4 class="sub">Generation, and what its numbers mean</h4>
  <p class="para">
    Replies come from a hand-written decode loop over real logits, not from a canned string. At
    temperature 0 it is greedy argmax. Above 0 it samples the temperature softmax restricted to
    <b>top-k 50 ∩ top-p 0.9</b>, after a <b>repetition penalty of 1.1</b>:
  </p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="eq" role="group" aria-label="equation" tabindex="0">
    p(token) ∝ exp(logit / T) &nbsp; restricted to the 50 most probable tokens<br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    ∩ the shortest prefix whose mass reaches 0.9
  </div>
  <p class="para">
    Those constraints exist because unrestricted full-vocabulary sampling draws from the long tail
    on every step, which models this small do not survive — it was the single biggest cause of
    incoherent replies here. <b>Only the draw is filtered</b>: no reported number is computed from
    the truncated distribution. Precisely which distribution each number comes from:
  </p>
  <ul class="para">
    <li>
      the <b>percentage under a generated token</b> is that token's probability under the
      temperature-scaled softmax, unfiltered;
    </li>
    <li>
      the <b>top-5 alternatives on hover</b> come from the model's own softmax, with no temperature
      applied — otherwise a genuine 2% alternative reads as 0.0% at low temperature;
    </li>
    <li>
      at <b>temperature 0</b> nothing is sampled at all, so the chosen token's “100%” is the greedy
      decision, not a probability. Its alternatives are still the model's real distribution.
    </li>
  </ul>
  <p class="para">Generation is capped at 128 new tokens.</p>

  <!-- -------------------------------------------------------------------- geo -->
  <h3 id="geo">The Geometry Lab</h3>

  <h4 class="sub">Why three dimensions</h4>
  <p class="para">
    Every published picture of an embedding space is a projection: 4096 dimensions crushed into 2
    or 3 by PCA, t-SNE, or UMAP, and you are looking at the projection's artifacts as much as the
    model's structure. This model was instead <i>built</i> at <code>d_model = 3</code> and genuinely
    trained there. <b>No dimensionality reduction is applied</b>: the 1003 embedding rows are unit
    vectors in ℝ³ and are drawn where they are. The only projection left is your screen's, and you
    can rotate it away.
  </p>
  <p class="para">
    Be precise about what lives on the sphere, though: the <i>embedding rows</i> are pinned to
    <code>S²</code>. The residual stream is not — it moves through ℝ³, and that radial excursion is
    exactly what the green path shows and what removing layer norm preserves.
  </p>

  <h4 class="sub">The model, exactly</h4>
  <p class="para">
    Decoder-only, <b>4 layers, 1 head, no layer norm</b>, MLP hidden width 12, context window 50,
    vocabulary 1003 (the corpus's 1000 most frequent word and punctuation types, plus
    <code>&lt;unk&gt;=0</code>, <code>&lt;eos&gt;=1</code>, <code>&lt;pad&gt;=2</code>). Embedding
    rows are unit vectors on <code>S²</code>. Positions are learned absolute embeddings — just 50
    more 3-vectors. They are trained, saved, and loaded with everything else, but the weight editor
    exposes only <code>embedding</code>, <code>W_Q</code>, <code>W_K</code>, <code>W_V</code> and
    <code>W_O</code>, so there is no way to hand-edit a positional vector from the interface.
  </p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="eq" role="group" aria-label="equation" tabindex="0">
    z_i⁽⁰⁾ = E[t_i] + p_i<br />
    q_i = W_Q z_i, &nbsp; k_i = W_K z_i, &nbsp; v_i = W_V z_i<br />
    A_ij = softmax_j ⟨k_j, q_i⟩ &nbsp; over j ≤ i<br />
    z_i ← z_i + W_O {"Σ_{j≤i}"} A_ij v_j<br />
    z_i ← z_i + W_outᵀ gelu(W_inᵀ z_i + b_in) + b_out<br />
    logits = E z<span class="note">&nbsp;&nbsp;(tied unembedding)</span>
  </div>
  <p class="para">
    Two deliberate departures from the standard block, both of which you can see the consequences
    of:
  </p>
  <ul class="para">
    <li>
      <b>Attention scores are unscaled</b> — <code>⟨k_j, q_i⟩</code>, with no <code>1/√d</code>.
      This makes the trace and the force field literally the same numbers rather than the same
      numbers up to a constant.
    </li>
    <li>
      <b>There is no layer norm.</b> At <code>d = 3</code>, normalizing would erase exactly the
      radial information this tab exists to show — how far off the sphere the residual stream
      travels. Embeddings are held on <code>S²</code> by renormalizing during training instead, and
      the initialization is scaled so the norm-free stream stays stable.
    </li>
  </ul>
  <p class="para">
    Reading out at an intermediate layer is the <b>logit lens</b>: <code>E z⁽ˡ⁾</code> asks what the
    model would predict if it stopped thinking after layer <code>ℓ</code>. (The usual final layer
    norm has nothing to apply here — this model has none.) Tying is not what makes it a logit lens;
    what tying buys is that the readout direction for a token <i>is</i> that token's embedding
    point, which is the only reason the answer can be drawn as an arrow to another point on the
    same sphere. That is what the <b>layer</b> selector does in next-next mode.
  </p>

  <h4 class="sub">Field 1 — “next-next”</h4>
  <p class="para">
    For every one of the 1003 vocabulary tokens <code>v</code>, append <code>v</code> to your
    prompt, run the model, and read out at the selected layer. Draw an arrow from <code>E[v]</code>
    toward the embedding of what the model would say <i>next</i>:
  </p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="eq" role="group" aria-label="equation" tabindex="0">
    arrow(v) = E[argmax logits] − E[v] <span class="note">at temperature 0</span><br />
    arrow(v) = E[u] − E[v], weighted by p(u), for the top-m tokens u <span class="note">at T &gt; 0</span>
  </div>
  <p class="para">
    So the field answers: <i>if the model went here, where would it go from here?</i> At temperature
    0 the distribution is one-hot, so there is exactly one arrow per point no matter what
    “arrows/point” is set to — which is why that slider disables itself until you raise the
    temperature. Brighter arrows are more probable.
  </p>
  <p class="para">
    Two things to be clear about, because “vector field” invites a stronger reading than is
    warranted. First, this field is <b>conditioned on your prompt</b> — the construction is
    literally “append <code>v</code> <i>to this prompt</i>”, so editing the prompt redraws every
    arrow. It is a field over the vocabulary given a context, not an autonomous flow on
    <code>S²</code>. Second, an arrow's tail is the token's <b>embedding</b> <code>E[v]</code>, not
    the residual-stream state the prediction was actually made from: the tail is where the token
    <i>lives</i>, not where the model <i>was</i>.
  </p>

  <h4 class="sub">Field 2 — the attention force</h4>
  <p class="para">
    This is the construction from <b>“On Transformer Dynamics”</b> (Latifi Jebelli,
    <a href="https://arxiv.org/abs/2607.13295" target="_blank" rel="noopener">arXiv:2607.13295</a>),
    which reads attention as a two-body interaction law moving tokens as particles on a manifold.
    Two different things are drawn, and they are worth keeping apart:
  </p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="eq" role="group" aria-label="equation" tabindex="0">
    <b class="k-thin">thin arrows</b> — the per-point field &nbsp; x ↦ W_V x &nbsp; over all 1003 embedding points<br />
    <b class="k-amber">amber arrows</b> — the aggregate force at each prompt position, &nbsp; F_i = {"Σ_{j≤i}"} A_ij v_j
  </div>
  <p class="para">
    <code>F_i</code> is literally the model's <code>attention @ v</code> row — the same tensor the
    trace shows. Ticking <b>antisymmetrize</b> replaces <code>W_V</code> with
    <code>(W_V − W_Vᵀ)/2</code> in the per-point field only. For an antisymmetric
    <code>A</code>, <code>⟨Ax, x⟩ = 0</code> identically, so that field is <b>exactly tangent</b> to
    the sphere at every point — the badge saying so is a statement about the algebra, not a
    measurement.
  </p>

  <h4 class="sub">How to read arrow length (and how not to)</h4>
  <p class="para">
    Arrow <i>length</i> on screen is relative, not absolute. Each render scales its arrows so the
    90th-percentile magnitude reaches a fixed on-screen length, and anything longer is clipped to
    the maximum. The two arrow classes are scaled <b>independently</b>. Three consequences:
  </p>
  <ul class="para">
    <li>
      Scaling <code>W_V</code> by a positive constant leaves the thin-arrow picture
      <b>pixel-identical</b>. The field really is <code>W_V x</code>, but you are seeing its
      directions and its <i>relative</i> magnitudes, not its units.
    </li>
    <li>A thin arrow and an amber arrow of the same length are not the same magnitude.</li>
    <li>Lengths are not comparable between two renders — change anything and the scale moves.</li>
  </ul>
  <p class="para">
    Directions, relative lengths within one field, and the colour ramp are the trustworthy signals.
    For an absolute number, read the badges: they are in model units.
  </p>

  <h4 class="sub">Reading the sphere: a legend</h4>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="tblwrap" role="group" aria-label="sphere legend" tabindex="0">
    <table class="tbl">
    <thead>
      <tr><th>On screen</th><th>What it is</th></tr>
    </thead>
    <tbody>
      <tr><td><span class="sw tok"></span> blue dots</td><td>the 1003 token embeddings; hover for the word</td></tr>
      <tr><td><span class="sw dim"></span>→<span class="sw hi"></span> thin arrows</td><td>the field, dim to bright with the arrow's weight (probability in next-next, relative magnitude in force)</td></tr>
      <tr><td><span class="sw amber"></span> amber arrows</td><td>the prompt's aggregate attention forces, one per prompt position</td></tr>
      <tr><td><span class="sw path"></span> green path</td><td>your prompt's tokens in order, occluded where it passes behind the sphere</td></tr>
      <tr><td><span class="sw last"></span> white dot</td><td>the last token of the prompt — where the path ends</td></tr>
    </tbody>
  </table>
  </div>

  <h4 class="sub">The tangency subtlety worth reading</h4>
  <p class="para">
    Antisymmetrizing does <b>not</b> make the aggregate forces tangent. Each term
    <code>W_V z_j</code> is tangent at <code>z_j</code> — but the sum <code>F_i</code> is drawn
    anchored at <code>z_i</code>, a different point on the sphere, where it has no reason to be
    tangent at all. So the aggregate arrows are explicitly projected onto the tangent plane at the
    point they are drawn from:
  </p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="eq" role="group" aria-label="equation" tabindex="0">
    F_i<sup>∥</sup> = F_i − ⟨F_i, ẑ_i⟩ ẑ_i, &nbsp;&nbsp; ẑ_i = E[t_i] / ‖E[t_i]‖
  </div>
  <p class="para">
    Projection hides something, so the amount hidden is reported: the
    <b>“radial pull projected out”</b> badge is <code>max_i |⟨F_i, ẑ_i⟩|</code> across your prompt.
    An earlier version of this page projected at the layer's residual stream instead of at the
    embedding, which put the arrows up to <b>59° out of the plane</b> they were claimed to lie in.
    If a picture asserts a geometric property, the number that could falsify it belongs next to it.
  </p>

  <h4 class="sub">What you can change, and what it does</h4>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="tblwrap" role="group" aria-label="controls and their effects" tabindex="0">
    <table class="tbl">
    <thead>
      <tr><th>Control</th><th>What it changes</th><th>What to watch</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><b>prompt</b></td>
        <td>
          everything. Both fields are <i>conditioned on it</i>, and it also sets the green path, the
          amber aggregate forces, and the attention map
        </td>
        <td>
          all 1003 arrows redrawn when you retype it — this is not an autonomous field on
          <code>S²</code>
        </td>
      </tr>
      <tr>
        <td><b>W_V</b>, <b>embedding</b></td>
        <td>the force field directly — it <i>is</i> <code>W_V x</code> over the embedding points</td>
        <td>the largest visible effect of any edit</td>
      </tr>
      <tr>
        <td><b>W_Q</b>, <b>W_K</b></td>
        <td>only the attention matrix <code>A</code></td>
        <td>
          at temperature 0 the next-next field is an argmax, so small changes often move just a few
          arrows; the aggregate forces and the attention map respond immediately
        </td>
      </tr>
      <tr>
        <td><b>W_O</b></td>
        <td>how attention output re-enters the residual stream</td>
        <td>how far the green token path departs from the sphere</td>
      </tr>
      <tr>
        <td><b>layer</b></td>
        <td>which layer's readout (next-next) or <code>W_V</code> (force) is used</td>
        <td>
          <code>full</code> is next-next only — the force field is per-layer by definition, so that
          button is disabled in force mode
        </td>
      </tr>
      <tr>
        <td><b>temperature</b>, <b>arrows/point</b></td>
        <td>how many next-next targets per point, and their weights</td>
        <td>the field fanning out from argmax into a distribution as T rises</td>
      </tr>
    </tbody>
  </table>
  </div>
  <p class="para">
    The presets are <code>identity</code>, <code>toeplitz_fuzzy</code>, <code>random</code>,
    <code>random_autocorr</code>, <code>zero</code>, and <code>learned</code> (back to the trained
    value). All six apply to the 3×3 matrices; the embedding takes every one except
    <code>zero</code>, which is refused because a zero row cannot be a unit vector. Individual cells
    are editable for the 3×3 matrices only — the 1003-row embedding is changed through presets.
    Editing never mutates the trained checkpoint: each edit mints a <b>new</b> weight set addressed
    by a content hash of its own weights, and <code>learned</code> takes you back.
  </p>

  <h4 class="sub">Training, fine-tuning, and training from scratch</h4>
  <p class="para">
    The shipped checkpoint is trained on the text of <i>Alice's Adventures in Wonderland</i>
    (Project Gutenberg ebook #11, public domain, committed to the repository so no run depends on a
    download; the Gutenberg header and licence footer are stripped before training): 30 epochs of
    Adam at lr 2e-2, batch 64, windows every 10 tokens. The objective is
    next-token cross-entropy plus a spherical-uniformity term — a Gaussian repulsion between
    sampled embedding rows, weight 0.3 — which keeps the vocabulary spread over the sphere instead
    of collapsing into a cap. Rows are renormalized onto <code>S²</code> after every step.
  </p>
  <p class="para">
    Two non-degeneracy metrics are computed and stored with every checkpoint, and shown as chips in
    the Geometry Lab header: <b>coverage uniformity</b> ≥ 0.80 (how evenly the vocabulary occupies
    the sphere) and <b>field directional entropy</b> ≥ 2.0 nats (how many distinct directions the
    next-next field points in; the maximum is ln 64 ≈ 4.16). To be precise about what those
    thresholds are: they are asserted by the test suite, not enforced by the training code at
    runtime. A checkpoint that fell below them would still be saved — CI is what catches it.
  </p>
  <p class="para">
    Those chips under the Geometry Lab heading are exactly these numbers for whichever checkpoint is
    active, and each one explains itself on hover. The one to read first is <b>final loss</b>: it is
    next-token cross-entropy in nats, so the reference point is a uniform model over 1003 tokens at
    <code>ln 1003 ≈ 6.91</code>. The short hexadecimal chip on an edited or trained model is the
    first 8 characters of its weights' content hash.
  </p>
  <ul class="para">
    <li>
      <b>Fine-tune</b> continues training the currently active weights on new text — plain SGD,
      up to 500 steps, default 100, lr 1e-2, with the embedding renormalized onto the sphere after
      every step. It mints a new checkpoint; the trained one is never overwritten. One caveat worth
      stating plainly: fine-tuning always tokenizes with the <i>shipped</i> vocabulary, so
      fine-tuning a from-scratch model does not use that model's own word list
      (<a href="https://github.com/ContextLab/llm-geometry/issues/6" target="_blank" rel="noopener">issue #6</a>).
    </li>
    <li>
      <b>Train from scratch</b> is a genuinely new model: fresh random weights <i>and</i> a fresh
      vocabulary, rebuilt as the 1000 most frequent types in <b>your</b> text (ties broken
      alphabetically, so it is deterministic). The epochs slider runs 1–30 and starts at 12. Your
      text has to contain at least 1000 distinct types or the run is refused rather than quietly
      producing a vocabulary padded out with nothing — the panel counts them as you type.
    </li>
    <li>
      You can paste text, upload a file, or pull a real dataset from the HuggingFace Hub — streamed
      by the <code>datasets</code> library on the full stack, and read from the Hub's public,
      CORS-enabled dataset-viewer service when this runs in your browser. Real rows either way.
    </li>
  </ul>
  <p class="para">
    Because a from-scratch model builds its own vocabulary, <b>token id 17 means different words in
    different models</b>. That is why a saved model is a bundle rather than a weight file: format
    <code>llm-geometry/geo-model</code> v2 carries the weights, the vocabulary, a content hash of
    the weights, and a separate SHA-256 of the vocabulary. Both digests are mandatory on load. A
    file with real weights and a tampered word list would silently mislabel every point on the
    sphere, so it is refused instead.
  </p>

  <!-- ------------------------------------------------------------------- real -->
  <!-- -------------------------------------------------------------------------- lex -->
  <h3 id="lex">The Lexicon Lab</h3>

  <p class="para">
    The other two tabs vary the <i>model</i>. This one varies the <b>vocabulary</b>. You choose a
    word budget, choose the model's dimensions, and train it from scratch in your browser — then
    watch three things move together: the loss, the text it produces, and the geometry of its
    embedding matrix. The question it makes explorable is what a <i>bounded</i> vocabulary can
    learn and say, and what it cannot.
  </p>

  <h4 class="sub">Two kinds of budget, at the same size</h4>
  <p class="para">
    A budget can be <b>prescribed</b> or <b>described</b>, and the tab ships both so you can hold
    <code>|V|</code> fixed and swap which one you are using:
  </p>
  <ul class="para">
    <li>
      <b>Dolch</b> — the graded sight-word lists Edward William Dolch published in <b>1936</b>, a
      real pedagogical word list still in use. The five cumulative budgets are
      <b>40 / 92 / 133 / 220 / 314</b> words, and they <b>nest</b>: growing the budget only ever
      adds words, so a comparison across sizes is not confounded by words leaving.
    </li>
    <li>
      <b>Corpus frequency</b> — the most frequent <code>N</code> word types of whatever text you
      are training on, ties broken alphabetically so the budget is a deterministic function of the
      corpus.
    </li>
  </ul>
  <p class="para">
    On the shipped corpus the descriptive budget covers more of its own text than the prescribed one
    at every matched size — 70.7% against 60.8% at <code>|V| = 314</code>. That is not a defect in
    Dolch's list; it is what "descriptive" means. The interesting part is what each one <i>cannot</i>
    say.
  </p>

  <h4 class="sub">Why 314 and not 315</h4>
  <p class="para">
    The Dolch noun list contains <code>Santa Claus</code>, which has a space in it and can never be
    matched by a word-level tokenizer. The source project shipped it, so its “315-word” budget was
    silently 314 words wide. We drop it and report the number the code actually contains. The same
    pass fixed a transcription slip in the first-grade list, which had <code>giving</code> where the
    published list has <code>going</code> — a word that occurs 27 times in the shipped corpus, so
    the slip cost real coverage.
  </p>

  <h4 class="sub">What a budget cannot say</h4>
  <p class="para">
    Words outside the budget become <code>&lt;unk&gt;</code> when the model is trained, and
    <code>&lt;unk&gt;</code> is <b>banned at generation</b>. So the model learns from the whole
    corpus but can only ever speak in-budget — guaranteed by construction, because the vocabulary
    <i>is</i> the budget. There is no filter to leak through. The <code>&lt;unk&gt;</code> rate is
    displayed rather than hidden: at <code>|V| = 314</code> the Dolch budget cannot express
    <b>39.2%</b> of the corpus's tokens, and only <b>215 of 3,071</b> lines are wholly inside it.
  </p>

  <h4 class="sub">Reading the spectrum — and the trap in it</h4>
  <p class="para">
    The geometry panel shows the singular-value spectrum of the embedding matrix and its
    <b>effective rank</b>, <code>exp(−Σ pᵢ ln pᵢ)</code> with <code>pᵢ = σᵢ²/Σσⱼ²</code> — the
    exponentiated entropy of the normalized squared spectrum, computed on the column-mean-centred
    matrix.
  </p>
  <p class="para">
    <b>It is very easy to fool yourself with this number,</b> so the panel makes that hard. Effective
    rank is bounded above by <code>min(|V| − 1, d)</code>, which rises as you grow the budget
    <i>whatever the weights contain</i>. A rank curve that climbs with <code>|V|</code> and then
    flattens is therefore the <b>null result</b>, not evidence of learning. So the panel always draws
    two extra things beside your model: the <code>min(|V| − 1, d)</code> ceiling, and an
    <b>untrained random-init model of the same shape</b>. Any claim worth making is about the gap
    between the trained curve and those two — never about the trained curve alone.
  </p>
  <p class="para">
    The token cloud is a <b>PCA projection</b> onto the top three components, and is labelled as one
    wherever it appears, with its explained variance shown. This is the opposite of the Geometry
    Lab's sphere, where the coordinates are the representation itself. Do not read the two the same
    way.
  </p>

  <h4 class="sub">Where this came from, and what is not claimed</h4>
  <p class="para">
    This tab was built from a research bundle exploring what a closed word budget reveals about
    lexicalization. That bundle is explicitly <b>a proposal, not results</b>: by its own independent
    audit it is “a proposal plus partial instrumentation—not a result package,” shipping no trained
    checkpoint, no generated corpus and no run manifest, with 8 of its 9 figures schematic — one
    legend reads “schematic data,” another caption “I have drawn my bet, not a finding.”
  </p>
  <p class="para">
    So <b>nothing here reproduces its curves</b>. Every number in this tab is computed live from a
    model that actually trained in your browser. Three of its instruments were deliberately left out
    rather than shipped broken: its meter score, which does not measure meter (the line
    <i>“and I do not like green eggs and ham”</i> scores 0.333 against a nonsense corpus's 0.346);
    its constrained decoder, which can emit fused non-words and is unnecessary here anyway; and its
    nonce-word “vacancy” experiment, which needs a parameter-matched control that does not yet
    exist.
  </p>
  <p class="para">
    Finally, on the name: this is <b>not</b> a Dr. Seuss model. His work is under copyright and is
    not used, quoted, or trained on. The corpus is <i>The Real Mother Goose</i> (1916), and the
    budget is Dolch's 1936 word list. The source project's own paper puts it plainly — “the tiny
    model is therefore not a Seuss pastiche.”
  </p>

  <h3 id="real">What's real, and where it runs</h3>
  <p class="para">
    This page is deployed as a static site with no Python behind it, so it is worth being precise
    about what that changes. {#if STATIC_MODE}<b>You are currently on the static build</b>, so the
    right-hand column describes what you have.{:else}<b>You are currently running against the full
    stack</b>, so the middle column describes what you have.{/if} The Lexicon Lab is the one tab
    with nothing to choose between: it computes in the browser either way, so its rows span both
    columns.
  </p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="tblwrap" role="group" aria-label="capability comparison" tabindex="0">
    <table class="tbl">
    <thead>
      <tr><th>Capability</th><th>Full stack (local)</th><th>Static build (this site)</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Geometry Lab: fields and traces</td>
        <td>real, in PyTorch</td>
        <td>
          real, in a TypeScript port of the same model, golden-tested against the Python backend to
          ≤ 1e-5
        </td>
      </tr>
      <tr>
        <td>Geometry Lab: weight edits, fine-tune, scratch training</td>
        <td>real, in PyTorch</td>
        <td>
          real, in the same TypeScript port — but training is only held to the <i>recipe</i>, not to
          bit-equality with a Python run (see Known limits)
        </td>
      </tr>
      <tr>
        <td>Architecture: chat / generation</td>
        <td>real, in PyTorch</td>
        <td>real, in your browser via transformers.js + ONNX</td>
      </tr>
      <tr>
        <td>Architecture: op-by-op trace</td>
        <td>live, for any prompt</td>
        <td>
          precomputed by the real backend for a set of example prompts. Browser ONNX exports do not
          expose hidden states, so an arbitrary prompt cannot be traced — the tab says so rather
          than inventing tensors
        </td>
      </tr>
      <tr>
        <td>Lexicon Lab: budgets, training, generation, spectrum</td>
        <td colspan="2" class="span">
          <b>Real, in your browser, in both modes — this tab never calls the backend.</b> The
          vocabulary, the training loop (in a Web Worker), generation, the forward-pass trace and
          the eigendecomposition are all the same TypeScript engine whichever way you are running,
          so the two columns cannot differ. The backend's <code>/api/lex/*</code> routes exist for
          parity and for callers outside this tab; nothing here uses them
        </td>
      </tr>
      <tr>
        <td>Lexicon Lab: the corpus</td>
        <td colspan="2" class="span">
          In both modes the tab loads <code>static-data/lex/corpus.json</code> — the build-time
          export of the committed Project Gutenberg file, trimmed to its body by the real backend —
          and <b>re-hashes that body in your browser</b> against the digest the export declares. A
          mismatch, or a missing digest, is fatal: the tab refuses to run rather than measure
          budgets against text it cannot identify
        </td>
      </tr>
      <tr>
        <td>Architecture: weight matrices</td>
        <td>from the loaded model</td>
        <td>
          exact windows are HTTP range reads straight out of the safetensors file on HuggingFace's
          CDN. The whole-matrix overview is the backend's own response, quantized to 8 bits at
          build time — see the note on overviews above
        </td>
      </tr>
    </tbody>
  </table>
  </div>
  <p class="para">
    The rule the project holds itself to: anything shown is computed live, read from a real
    artifact, or precomputed by the real backend — <b>never fabricated and never silently
    degraded</b>. Where a capability is missing, the interface says which one and why.
  </p>

  <!-- ----------------------------------------------------------------- limits -->
  <h3 id="limits">Known limits</h3>
  <ul class="para">
    <li>
      <b>The model menu is curated, not open.</b> The static build's live path needs a community
      ONNX export, which most repositories do not have, so an “any HuggingFace id” box would promise
      what this deployment cannot keep. Widening it is
      <a href="https://github.com/ContextLab/llm-geometry/issues/4" target="_blank" rel="noopener">issue #4</a>.
    </li>
    <li>
      <b>ONNX mirrors are pinned to <code>main</code>, not to a commit.</b> The mirror is a separate
      repository from the model, so the model's commit sha does not address it —
      <a href="https://github.com/ContextLab/llm-geometry/issues/5" target="_blank" rel="noopener">issue #5</a>.
    </li>
    <li>
      <b>A browser training run and a Python training run are not bit-identical.</b> In the
      <b>Geometry Lab</b> the objective, optimizer, hyperparameters, clipping, sphere projection,
      and vocabulary construction are the same, and vocabulary building is exact — but the RNG
      streams are not portable, so two runs “from the same seed” are two independent runs of one
      recipe. The forward pass <i>is</i> held to ≤ 1e-5 against the Python reference.
    </li>
    <li>
      <b>The Geometry Lab's tokenizer is word-level</b>, lowercased, with punctuation as tokens.
      Words outside its 1000-type vocabulary become <code>&lt;unk&gt;</code>, which the token strip
      marks.
    </li>
    <li>
      <b>The Lexicon Lab has no server half to fall back on.</b> It trains in your browser in both
      modes, so what it can do is bounded by your machine and your patience rather than by the
      deployment — and the same caveat as above applies to its numbers: the forward pass, the loss
      and every spectrum statistic are pinned to ≤ 1e-5 against the Python implementation by a
      golden test, but <i>whole-run</i> training equality with a Python run is not claimed.
    </li>
    <li>
      <b>A model trained in the Lexicon Lab lives in that tab and nowhere else.</b> There is no
      account and no server-side checkpoint, so closing the page ends the model unless you save the
      <code>.llmlex.json</code> bundle — which carries the weights <i>and</i> the vocabulary,
      because a token id means nothing without the budget it was trained under.
    </li>
  </ul>

  <!-- ------------------------------------------------------------------- refs -->
  <h3 id="refs">Source &amp; references</h3>
  <ul class="para links">
    <li>
      <a href="https://github.com/ContextLab/llm-geometry" target="_blank" rel="noopener">
        github.com/ContextLab/llm-geometry</a> — the full stack, the tests, and the spec. The
      Architecture Explorer and the Geometry Lab run locally against real PyTorch with one command;
      the Lexicon Lab runs in your browser either way, and the backend's <code>/api/lex/*</code>
      routes are there for parity rather than for the tab.
    </li>
    <li>
      M. J. Latifi Jebelli, <i>On Transformer Dynamics</i>,
      <a href="https://arxiv.org/abs/2607.13295" target="_blank" rel="noopener">arXiv:2607.13295</a>
      — the attention-as-interaction-law framing behind the force field.
    </li>
    <li>
      T. Wang &amp; P. Isola, <i>Understanding Contrastive Representation Learning through Alignment
      and Uniformity on the Hypersphere</i>,
      <a href="https://arxiv.org/abs/2005.10242" target="_blank" rel="noopener">arXiv:2005.10242</a>
      — the uniformity potential used as the spherical-spread term in training.
    </li>
    <li>
      <a href="https://www.gutenberg.org/ebooks/11" target="_blank" rel="noopener">
        Project Gutenberg ebook #11</a> — <i>Alice's Adventures in Wonderland</i>, the training
      corpus for the <b>Geometry Lab</b>'s shipped checkpoint.
    </li>
    <li>
      <a href="https://www.gutenberg.org/ebooks/10607" target="_blank" rel="noopener">
        Project Gutenberg ebook #10607</a> — <i>The Real Mother Goose</i> (1916), illustrated by
      Blanche Fisher Wright: the <b>Lexicon Lab</b>'s corpus. It is committed to the repository
      whole, header and licence footer intact, and trimmed to its body only when it is used.
    </li>
    <li>
      E. W. Dolch, <i>A Basic Sight Vocabulary</i>, The Elementary School Journal 36(6):456–460,
      1936, <a href="https://doi.org/10.1086/457353" target="_blank" rel="noopener">
        doi:10.1086/457353</a> — the source of the Lexicon Lab's prescribed word budgets.
    </li>
  </ul>
</section>

<style>
  .info {
    padding: 1.2rem 1.4rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    max-width: 62rem;
  }
  h2 {
    margin: 0;
    font-size: 1.25rem;
    letter-spacing: -0.01em;
  }
  .lede {
    margin: 0.45rem 0 0;
    font-size: 0.92rem;
    line-height: 1.65;
    color: var(--text-dim);
  }
  .lede b {
    color: var(--text);
  }
  .toc {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.9rem;
  }
  .toc button {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    color: var(--text-dim);
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    font-size: 0.72rem;
  }
  .toc button:hover {
    color: var(--accent);
    border-color: var(--accent);
  }
  h3 {
    margin: 1.7rem 0 0.2rem;
    font-size: 1.02rem;
    scroll-margin-top: 1rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
  }
  h4.sub {
    margin: 1.1rem 0 0.1rem;
    font-size: 0.86rem;
    color: var(--accent);
    letter-spacing: 0.01em;
  }
  .para {
    margin: 0.45rem 0 0;
    font-size: 0.83rem;
    line-height: 1.68;
    color: var(--text-dim);
  }
  .para b {
    color: var(--text);
    font-weight: 600;
  }
  ul.para {
    padding-left: 1.15rem;
  }
  ul.para li {
    margin: 0.35rem 0;
  }
  ul.para li::marker {
    color: var(--accent);
  }
  code {
    font-family: var(--mono);
    font-size: 0.93em;
    color: var(--accent);
    background: var(--bg-elev-2);
    border-radius: 5px;
    padding: 0.05em 0.35em;
  }
  a {
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px solid rgba(110, 168, 254, 0.35);
  }
  a:hover {
    border-bottom-color: var(--accent);
  }
  .eq {
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--text);
    background: var(--bg-elev-2);
    border-left: 2px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.85rem;
    margin: 0.6rem 0 0.2rem;
    /* nowrap is what makes overflow-x mean anything: without it a long line reflows
       mid-formula on a phone and reads as two equations instead of scrolling. */
    white-space: nowrap;
    overflow-x: auto;
    line-height: 1.9;
  }
  .eq:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .eq .note {
    color: var(--text-dim);
  }
  .eq b {
    font-weight: 600;
    color: var(--text);
  }
  /* A key that names a colour must be printed in it — `.eq b`'s violet is the colour of
     the OTHER field's bright arrows, which made the amber label actively misleading. */
  .eq b.k-amber { color: #ffb454; }
  .eq b.k-thin { color: #b794f6; }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 20rem), 1fr));
    gap: 0.8rem;
    margin-top: 0.6rem;
  }
  .card {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .card h4 {
    margin: 0;
    font-size: 0.92rem;
  }
  .card p {
    margin: 0;
    font-size: 0.8rem;
    line-height: 1.6;
    color: var(--text-dim);
  }
  .card .go {
    align-self: flex-start;
    margin-top: 0.35rem;
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.75rem;
  }
  .card .go:hover {
    background: var(--accent-grad);
    color: #0b0e14;
    border-color: transparent;
    font-weight: 600;
  }
  .tblwrap {
    /* The scroll container, NOT the table: `display: block` on a <table> strips its
       implicit table role, so headers stop associating with cells for assistive tech.
       tabindex makes the overflow reachable without a mouse. */
    overflow-x: auto;
    margin-top: 0.6rem;
  }
  .tblwrap:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.79rem;
  }
  .tbl th {
    text-align: left;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent);
    padding: 0.35rem 0.6rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  .tbl td {
    padding: 0.42rem 0.6rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-dim);
    line-height: 1.55;
    vertical-align: top;
  }
  .tbl td b {
    color: var(--text);
  }
  /* A row whose two runtime columns say the same thing says it once, across both — the
     Lexicon Lab is byte-identical in either mode, and splitting that into two identical
     cells would invite a reader to hunt for a difference that does not exist. */
  .tbl td.span {
    background: rgba(91, 224, 176, 0.06);
  }
  .tbl.notation td:nth-child(odd) {
    white-space: nowrap;
    width: 1%;
  }
  .links li {
    margin: 0.45rem 0;
  }
  /* Legend swatches: the exact colours GeoScene paints with, so the key cannot drift
     into describing one colour while the scene draws another. */
  .sw {
    display: inline-block;
    width: 0.72rem;
    height: 0.72rem;
    border-radius: 3px;
    vertical-align: -1px;
    margin-right: 0.15rem;
  }
  .sw.tok { background: #6ea8fe; }
  .sw.dim { background: #2a3a6e; }
  .sw.hi { background: #b794f6; }
  .sw.amber { background: #ffb454; }
  .sw.path { background: var(--good); }
  .sw.last { background: #ffffff; }
</style>
