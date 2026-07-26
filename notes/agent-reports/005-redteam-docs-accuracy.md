# Red-team: accuracy audit of the new Info tab + Explain prose

Date: 2026-07-26. Scope: the uncommitted documentation added in the working tree —
`code/frontend/src/viz/info/InfoTab.svelte`, `code/frontend/src/lib/Explain.svelte`,
and the new `<Explain>` / `<p class="sub">` / `<p class="panel-note">` blocks in
`viz/arch/{ArchitectureExplorer,ArchTracePanel}.svelte` and
`viz/geo/{GeometryLab,WeightLab,FinetunePanel,TrainPanel}.svelte`.

Method: every checkable assertion was traced to the code that implements it. Findings
below are only the ones where the prose and the code disagree. A "verified correct"
appendix is at the end so the next reader does not re-check the same ground.

---

## CRITICAL

None. Every equation and every architectural number in the Geometry Lab prose checks
out against `geo/model.py::_run` — see the appendix. The tangency/projection story is
also correct, including the badge definition.

---

## HIGH

### H1. The positional embeddings are advertised as editable. They are not.

`code/frontend/src/viz/info/InfoTab.svelte:180-183`

> "Positions are learned absolute embeddings — just 50 more 3-vectors, which is why
> they are **directly editable**."

`code/frontend/src/viz/geo/GeometryLab.svelte:370-371`

> "positions are learned absolute embeddings (50 more 3-vectors, **so they are editable
> too**)."

Contradicting code — `pos_embedding` is not in the editable set anywhere in the stack:

- `code/backend/src/llm_geometry/geo/model.py:36`
  `EDITABLE_MATRICES = ("W_Q", "W_K", "W_V", "W_O", "embedding")`
- `code/backend/src/llm_geometry/geo/weights.py:64-67` — `preset_matrix` raises for
  anything else: `raise InvalidWeightEditError(f"Unknown matrix {matrix!r}; expected one of {EDITABLE_MATRICES}")`
- `code/backend/src/llm_geometry/geo/weights.py:160-163` — `build_weight_set` rejects
  the same way per edit.
- `code/frontend/src/viz/geo/WeightLab.svelte:28`
  `const MATRICES: GeoMatrixName[] = ["embedding", "W_Q", "W_K", "W_V", "W_O"];`
  — no `pos_embedding` option in the picker.

`pos_embedding` *is* in `model.weight_names()` (model.py:186) and therefore travels in a
saved bundle, which is presumably where the sentence came from — but "directly
editable", written next to a description of the editing UI, is false. The user cannot
change a positional vector by any route the app exposes.

Severity HIGH: false about a capability, in both the reference tab and the lab header.

### H2. Fine-tune does **not** keep the active model's vocabulary.

`code/frontend/src/viz/geo/FinetunePanel.svelte:123-128`

> "Real gradient steps (SGD, lr 1e-2) on the **currently active** weights, **keeping the
> existing vocabulary** — so this adapts the model you are looking at rather than
> starting one."

The "currently active weights" half is true (`FinetunePanel.svelte:38`
`const base = $geoWeightsToken ?? "learned";`). The vocabulary half is false: fine-tuning
always tokenizes with the *shipped* Alice-in-Wonderland vocabulary, whatever the base is.

- `code/backend/src/llm_geometry/geo/finetune.py:157`
  `ids = get_tokenizer().encode_stream(text)` — the module-level canonical tokenizer
  (`tokenizer.py:179-182`), never `tokenizer_for(base)`.
- `code/frontend/src/lib/staticClient/geo.ts:240`
  `const tokenIds = engine.tokenizer.encodeStream(text);` — same, and the file says so
  in its own docstring at `geo.ts:269`: *"Fine-tuning keeps the shipped vocabulary"*.
- Worse, the result **loses** the base's vocabulary:
  `finetune.py:207` `new_token = save_weight_set(new_ws, source="finetuned", store=store)`
  passes no `vocab_json`, so `tokenizer_for()` (`tokenizer.py:193-198`) falls back to the
  canonical tokenizer for the fine-tuned model too.

Concretely: train a model from scratch on your own text (which mints a *fresh* 1000-word
vocabulary — `scratch.py:125,145-146`), then fine-tune it. The fine-tuning text is
encoded with Alice's ids, the gradients land on the wrong rows, and every dot on the
sphere is relabelled with Alice's words. The panel note asserts the opposite.

Severity HIGH. (The underlying behaviour may itself be the bug; the doc is what makes it
a *claim*.)

### H3. Static build: the weight heat map you actually see is a precomputed uint8 tile, not a range read.

`code/frontend/src/viz/info/InfoTab.svelte:388-391` — the "Static build (this site)" cell
for "Architecture: weight matrices":

> "HTTP range reads straight out of the safetensors file on HuggingFace's CDN"

That is only the *zoomed* path. The default view — the whole matrix, which for any real
weight tensor is far over the 4096-cell budget — is served from a precomputed,
8-bit-quantized tile shipped with the site:

`code/frontend/src/lib/staticClient/arch.ts:195-219`
```ts
const cells = (r1 - r0) * (c1 - c0);
if (cells <= maxCells && cells <= EXACT_CELLS_HARD_CAP) {
  return this.exactWindow(m, tile, r0, r1, c0, c1);
}
// Over-budget window → the precomputed strided-mean overview of the FULL
// tensor (the backend's own full-window response, uint8-quantized at build time).
...
values: dequantizeTile(bytes, gr, gc, tile.vmin, tile.vmax),
```
`arch.ts:81` — *"Dequantize one uint8 overview tile: value = vmin + (u8/255)·(vmax−vmin)."*
`arch.ts:7-8` (the file's own header) — *"weights: EXACT windows via safetensors HTTP
Range reads at the pinned revision; **over-budget windows serve the precomputed uint8
overview tiles**"*.

The inspector requests exactly that over-budget window on every node click
(`ArchInspector.svelte:67-71`: no `r0/r1` and no `max_cells` for 2-D params ⇒ full
tensor ⇒ tile path). So on the deployed site the first heat map for, e.g., a 896×896
`q_proj.weight` is a 256-level requantization, not the file's float32 values.

This also undercuts the lede at `InfoTab.svelte:36-37`:

> "Nothing on either tab is a schematic or an illustration: **every tensor shown came out
> of a model that actually ran**."

— true in provenance, but the displayed numbers at that zoom level are a lossy
re-encoding, and nothing on the page says so. Given the project's stated rule ("never
fabricated and never silently degraded", `InfoTab.svelte:394-397`), the omission is
material.

Severity HIGH.

### H4. "Two gates have to pass before a checkpoint is accepted" — nothing enforces them.

`code/frontend/src/viz/info/InfoTab.svelte:325-327`

> "**Two gates have to pass before a checkpoint is accepted:** coverage uniformity ≥ 0.80
> and field directional entropy ≥ 2.0 nats (the maximum is ln 64 ≈ 4.16)."

The two numbers are right (`geo/config.py:75` `MIN_COVERAGE_UNIFORMITY = 0.80`,
`geo/config.py:79` `MIN_FIELD_DIRECTIONAL_ENTROPY = 2.0`, `SPHERE_BINS = 64`,
ln 64 = 4.1589). What is wrong is "have to pass before a checkpoint is accepted":
`train_canonical` computes the metrics and stores them, then accepts the checkpoint
unconditionally.

`code/backend/src/llm_geometry/geo/train.py:259-272`
```python
ws, final_loss = train_geo_model(seed=seed, progress_cb=progress_cb)
...
metrics = compute_gate_metrics(ws)
token = save_weight_set(ws, source="learned", store=store)
meta = {..., "coverage_uniformity": ..., "field_directional_entropy": ...}
store.put(key, spec, meta, {...})
return meta
```
There is no comparison against the thresholds, and `train.py` never imports them. Repo-wide,
`MIN_COVERAGE_UNIFORMITY` / `MIN_FIELD_DIRECTIONAL_ENTROPY` appear in exactly three
places: `geo/config.py`, `tests/integration/test_geo_fields.py:14-15,55-61`, and
`tests/contract/test_api_geo.py:21-22,149-150`. They are **test assertions**, not a
runtime gate — a degenerate checkpoint would be saved and served.

Severity HIGH (states a safety property the code does not have).

---

## MEDIUM

### M1. "golden-tested … to ≤ 1e-5" is asserted for fine-tune and scratch training, which are not.

`code/frontend/src/viz/info/InfoTab.svelte:368-372` — the row is
*"Geometry Lab: fields, traces, weight edits, **fine-tune, scratch training**"*, and the
static-build cell reads:

> "real, in a TypeScript port of the same model, **golden-tested against the Python
> backend to ≤ 1e-5**"

Traces and fields are held to 1e-5
(`tests/unit/geoEngine.test.ts:154` *"matches golden traces to <=1e-5"*, `:195`
*"matches golden fields (both modes) to <=1e-5"*). Fine-tune is not:
`tests/unit/geoEngineFinetune.test.ts:95-97` compares `loss_after` with
`).toBeLessThan(0.15);` — a **15 % relative** tolerance. No golden equality test for
from-scratch training against Python exists.

The page contradicts itself 45 lines later at `InfoTab.svelte:415-419`:

> "A browser training run and a Python training run are not bit-identical. … **The forward
> pass *is* held to ≤ 1e-5** against the Python reference."

The limits bullet is the accurate one; the table cell overclaims.

### M2. "This is what the model would sample from right now" — it isn't.

`code/frontend/src/viz/arch/ArchTracePanel.svelte:223-225`

> "**Next-token top-10** — the real next-token distribution after the full forward pass,
> not per-layer. **This is what the model would sample from right now.**"

The trace's distribution is a plain, unfiltered, temperature-1 softmax:
`arch/trace.py:88-90`
```python
logits = out.logits[0, -1, :].detach().float()
probs = torch.softmax(logits, dim=-1).cpu().numpy()
top = np.argsort(-probs)[:10]
```
What Chat actually samples from is a *different* distribution: temperature-scaled, with a
1.1 repetition penalty, restricted to top-k 50 ∩ top-p 0.9
(`arch/generate.py:44-67, 129-131`). The Info tab says so itself at lines 145-155. The
first clause of the sentence is fine; the second is false.

### M3. "Every probability displayed … is computed from the unfiltered distribution" — two different distributions are in play.

`code/frontend/src/viz/info/InfoTab.svelte:156-162` (and the shorter form at
`ArchitectureExplorer.svelte:354-355`)

> "**Only the draw is filtered.** Every probability displayed — the per-token percentage,
> the top-5 alternatives on hover — is computed from **the** unfiltered distribution, so
> no number on screen changes meaning because of the sampler."

Both numbers are unfiltered, but they come from different distributions:

- per-token percentage, T > 0: `generate.py:129`
  `probs = torch.softmax(logits / float(temperature), dim=-1)` — temperature-scaled.
- top-5 alternatives: `generate.py:140`
  `model_probs = torch.softmax(logits, dim=-1)` — **not** temperature-scaled.

At T = 0.5 a token can read "62 %" while the very same token in its own top-5 list reads
"31 %". Additionally at T = 0 the reported `prob` is a sampling artifact, not a
distribution value: `generate.py:121-124`
```python
next_id = int(torch.argmax(logits)); probs = torch.zeros_like(logits); probs[next_id] = 1.0
```
so the "100 %" shown in greedy mode *does* change meaning because of the sampler. The
singular "the unfiltered distribution" papers over both.

### M4. HF datasets are not pulled "through the dataset-viewer service" on the full stack.

`code/frontend/src/viz/info/InfoTab.svelte:342-345`

> "You can paste text, upload a file, or pull a real dataset from the HuggingFace Hub
> **through its public dataset-viewer service**."

True only in the static build (`lib/staticClient/hfDatasets.ts:19`
`const BASE = "https://datasets-server.huggingface.co";`). The full stack uses the
`datasets` library's streaming loader, not the viewer:
`geo/finetune.py:51-55`
```python
from datasets import load_dataset
...
stream = load_dataset(dataset_id, split=split, streaming=True)
```
The paragraph sits in the Geometry Lab section with no mode qualifier, and the
"What's real" table two sections later never mentions the split.

### M5. "retraced 400 ms after you stop typing" is false on the deployed (static) site.

`code/frontend/src/viz/arch/ArchitectureExplorer.svelte:346-349`

> "**Prompt** and **system prompt** — retraced 400 ms after you stop typing."

The 400 ms is right for the full stack (`ArchitectureExplorer.svelte:204`
`const fireTrace = debounced((m, p, sp) => runTrace(m, p, sp), 400);`), but in static mode
an arbitrary prompt is never traced — only the labeled example prompts are:
`lib/staticClient/arch.ts:4-6` — *"traces: precomputed for the labeled example prompts
only (ONNX exports expose no attentions/hidden states) → other prompts get
StaticModeError"*. Unlike the header `<p class="sub">` right above it (which *is* gated on
`{#if STATIC_MODE}`), this `<Explain>` block renders identically in both builds.

### M6. "click a cell to type a new value" is false for the embedding.

`code/frontend/src/viz/geo/WeightLab.svelte:161-166`

> "Pick a matrix, apply a preset, **or click a cell to type a new value.**"

`code/frontend/src/viz/geo/WeightLab.svelte:214-215`
```svelte
{#if !isEmbedding}<span class="hint">click a cell to edit its value</span>
{:else}<span class="hint">cells are read-only (1003 rows) — use a preset above to change it (rows x/y/z)</span>{/if}
```
`embedding` is the first entry in the picker (`WeightLab.svelte:28`). Mitigating: the
default selection is `W_V` (`WeightLab.svelte:36`) and the inline hint corrects the note
once you switch. Still an unconditional statement that is conditionally false.

---

## LOW

### L1. `zero` is not a preset "per matrix".

`code/frontend/src/viz/info/InfoTab.svelte:311-313`

> "Presets per matrix are `identity`, `toeplitz_fuzzy`, `random`, `random_autocorr`,
> `zero`, and `learned` (back to the trained value)."

`geo/weights.py:97-102` rejects `zero` for the embedding:
`raise InvalidWeightEditError("preset 'zero' is invalid for the embedding: zero rows cannot satisfy the unit-norm constraint")`.
The list is otherwise exactly `PRESETS` (`weights.py:40`).

### L2. "downsampled **server-side**" — there is no server in the static build.

`code/frontend/src/viz/info/InfoTab.svelte:107-109` and
`ArchitectureExplorer.svelte:320-323` both say the overview is "downsampled server-side to
at most 4096 cells". On the deployed site the downsampling happened at *build* time
(`staticClient/arch.ts:200-203`). Also, for 1-D params (biases/norms) the client asks for
`max_cells: 128`, not 4096 (`ArchInspector.svelte:66-71`) — "at most 4096" is still
literally true, just not what happens.

*(The companion claim — "clicking into it fetches the **exact** sub-window at full
resolution" — checks out: `ArchInspector.svelte:91-100 windowFor()` caps the zoom window
at ≤ 4096 cells, so `weights.py:88-91` returns `method: "exact"`, and the static path takes
`exactWindow()` at `arch.ts:196-197`.)*

### L3. "(10–60 s)" has no basis in the code.

`code/frontend/src/viz/arch/ArchitectureExplorer.svelte:342-344`

> "First load of a new one downloads and traces it (**10–60 s**), then it is cached."

No timeout, budget, or measurement in the repo corresponds to this range; it depends
entirely on network and CPU. Stated as fact. I could not verify it and it is not
falsifiable from the source.

### L4. "the full text of *Alice's Adventures in Wonderland*".

`code/frontend/src/viz/info/InfoTab.svelte:319-322`. `geo/corpus.py` strips the Project
Gutenberg header/footer at the `*** START/END OF THE PROJECT GUTENBERG` markers
(`geo/config.py:48-49`), so it is the book's body, not the full file. Nit.

### L5. Unverifiable browser claim in a code comment.

`code/frontend/src/lib/Explain.svelte:6-10`

> "is found by the browser's in-page search even while collapsed (**Chromium/Safari**
> `hidden=until-found` semantics for `<details>` content)"

I could not verify that WebKit/Safari implements find-in-page inside a closed
`<details>`; my understanding is that this is a Chromium behaviour. Not user-facing (it is
a source comment), so LOW — but it is stated as fact. Flagging rather than asserting the
opposite: **I don't know** Safari's current behaviour here.

---

## Appendix — assertions checked and found CORRECT

Recorded so this ground is not re-walked.

**The Geometry Lab equations** (`InfoTab.svelte:185-192`, `GeometryLab.svelte:374-381`)
against `geo/model.py::_run` (lines 97-133):

| Doc line | Code | Verdict |
|-|-|-|
| `z_i⁽⁰⁾ = E[t_i] + p_i` | `h = tok + self.pos_embedding[:T].unsqueeze(0)` (model.py:98) | ✓ |
| `q_i = W_Q z_i` etc. | `q = h @ layer.W_Q.T` (model.py:105-107) | ✓ (row-major transpose = the same map) |
| `A_ij = softmax_j ⟨k_j, q_i⟩ over j ≤ i` | `scores = q @ k.transpose(1,2) + causal`; `torch.softmax(scores, dim=-1)` with `causal = full((T,T), -inf).triu(1)` (model.py:99,108-109) | ✓ unscaled, causal, softmax over `j` |
| `z_i ← z_i + W_O Σ_{j≤i} A_ij v_j` | `attn_out = (attn @ v) @ layer.W_O.T; h = h + attn_out` (model.py:110-111) | ✓ |
| `z_i ← z_i + W_outᵀ gelu(W_inᵀ z_i + b_in) + b_out` | `gelu(h @ W_in + b_in) @ W_out + b_out` with `W_in (3,12)`, `W_out (12,3)` (model.py:46-48,112-113) | ✓ — `h @ W_in ≡ W_inᵀ z`, `g @ W_out ≡ W_outᵀ g`; the transposes in the doc are **right** |
| `logits = E z` | `readout: h @ self.embedding.T`, `E` is `V×d` (model.py:131-133) | ✓ |

**Numbers.** `d_model=3`, `4` layers, `1` head, `mlp_hidden=12`, `context 50`,
`vocab 1003 = 1000 + <unk>=0/<eos>=1/<pad>=2` (`geo/config.py:18-34`). Training: `30`
epochs, Adam `lr 2e-2`, batch `64`, stride `10`, repulsion weight `0.3`
(`geo/config.py:52-60`, `train.py:164,173,179-181`); rows renormalized every step ✓.
Fine-tune: SGD, `≤ 500` steps, default `100`, `lr 1e-2` (`geo/config.py:64-68`,
`finetune.py:167`, slider `min=10 max=500` default `100` at `FinetunePanel.svelte:19,157`).
Scratch epochs slider `1–30`, default `12` (`TrainPanel.svelte:29,256`) — matches the doc
exactly (note the *API* allows up to 60, `scratch.py:47`, but the doc describes the
slider, which is what it says). ≥ 1000 distinct types required (`scratch.py:109-115`) ✓.
`SPHERE_BINS = 64`, ln 64 ≈ 4.1589 ✓. Arch: `1.5B` (`config.py:56`), 20 % margin as a
*reduced ceiling* on the config estimate (`gate.py:89-91`), `64`-token left truncation
(`config.py:60`, `trace.py:56`), `64×64` attention downsample (`config.py:62`,
`trace.py:69-76`), `4096` cells (`config.py:64`), `128` max new tokens (`config.py:66`,
slider `max=128` at `ArchChat.svelte:131`), top-k `50` / top-p `0.9` / rep penalty `1.1`
(`config.py:74-76`), top-**10** next-token list (`trace.py:90`), graph prompt
*"The quick brown fox jumps over the lazy dog."* capped at 12 tokens
(`graph.py:32-33`), Qwen2.5-0.5B-Instruct default (`config.py:46`).

**Controls.** `⟳ spin` — default OFF and the button *starts* rotation, grabbing stops it
(`GeoScene.svelte:28-30,134-137,577-585`) — the header sentence is right. `arrows/point`
disabled at T = 0 (`GeometryLab.svelte:513`). `full` layer disabled in force mode
(`GeometryLab.svelte:490`, `fields.py:117-121`). `follow playhead` defaults on and unticks
on manual layer choice (`ArchTracePanel.svelte:63,120-122`). Fine-tune sends the active
token as `base` (`FinetunePanel.svelte:38`). Editing mints a new content-hash weight set
and never overwrites `learned` ✓.

**Tangency.** `antisymmetrize` applies to the per-point field only, aggregates always use
the real `W_V` (`fields.py:128,146-147`, `geoEngine/fields.ts:18,278-283`). Aggregates are
projected at the **token embedding** `ẑ_i = E[t_i]/‖E[t_i]‖`, exactly as written
(`fields.py:158-167`). The badge is `max_i |⟨F_i, ẑ_i⟩|`:
`GeometryLab.svelte:288` `return Math.max(...sf.map((f) => f.normal_residual));` over
`normal_residual = abs(radial)` (`fields.py:173`). The 59° figure matches the code comment
at `fields.py:150-153`.

**Bundles.** Format `llm-geometry/geo-model`, version `2`, carries weights + vocab +
`weights_token` (content hash of weights) + `vocab_sha256`; both digests mandatory and
fatal on mismatch (`bundle.py:30-31,86-90,134-168`) ✓.

**Graph.** Tied weights by `data_ptr` with a `tied_to` alias (`graph.py:10-11,64-88`);
edges from tensor identity + storage aliasing with an execution-order fallback
(`graph.py:6-8,97-141`) ✓. Left-side truncation marker renders before the first surviving
chip (`ArchTracePanel.svelte:260-272`) ✓. Residual-norm outlier rule (scale = largest
non-outlier, outliers = past 8× median, striped + counted) matches
`ArchTracePanel.svelte:155-170,385-405` ✓.

**Links.** github.com/ContextLab/llm-geometry issues **#4** ("Architecture Explorer:
expand model support beyond the curated list", OPEN) and **#5** ("Pin the community ONNX
mirrors to their own commit revisions", OPEN) exist and match their descriptions; the
`main`-pinning claim matches `staticClient/transformersRuntime.ts:17-18`. The arXiv id
`2607.13295` is the same one already cited in the frozen contract and in
`geo/{model,fields}.py` — not newly introduced here, and I did not independently verify it
resolves.

**TS/Python parity claim** (`InfoTab.svelte:415-419`): objective, optimizer,
hyperparameters, clipping, sphere projection and vocabulary construction really are
mirrored — `geoEngine/scratch.ts:39-46` (`TRAIN_LR = 2e-2 // Adam`, `REPULSION_WEIGHT =
0.3`, `REPULSION_SAMPLE = 256`, `REPULSION_T = 2.0`), Adam at `:201-290`, global-norm clip
at 1.0 at `:271-279`. Tokenizer top-1000-by-frequency with alphabetical tie-break
(`tokenizer.py:103-111`) ✓.
