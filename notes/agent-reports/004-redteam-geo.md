# Red-team review — Geometry Lab tab (live stack, 2026-07-26)

Target: `http://localhost:5173/` → **Geometry** tab, against the real backend on `:8000`
(`checkpoint_id be5359a1c66bda29c8c554269e589009`, `final_loss 4.88521`).
Method: Playwright driving a real Chromium against the real API; every visual claim
checked against a screenshot in `/tmp/redteam-geo/`, every numeric claim checked against
the API response that produced it. No source file was modified.

---

## CRITICAL

### C1. The amber force arrows are **not** tangent where they are drawn — and two on-screen strings say they are

**What I did.** Force mode, prompt `alice rabbit queen said the little door`, every layer
0–3. Read `GET /api/geo/vector_field?mode=force&layer=N&...` and `GET /api/geo/trace`,
and computed `cos(sequence_force[i].vec, trace.embeddings[i])` — i.e. the angle between
each drawn arrow and the surface normal **at the point the frontend anchors it to**.
Tangent ⇒ cos = 0.

**What I saw.**

| layer | max \|cos(force, anchor)\| | angle out of the tangent plane |
|-|-|-|
| 0 | 0.175 | 10° |
| 1 | 0.685 | 43° |
| 2 | **0.853** | **58°** |
| 3 | 0.440 | 26° |

Layer 2, position 0: `vec` magnitude 0.6374, cos −0.8528 ⇒ **0.544 of genuine radial
pull is present in the arrow as drawn**, while the badge for that view reads
`radial pull projected out: 0.786 max` — a number computed somewhere else entirely.

The exact user-visible strings that are false:

- Caption (force mode): *"amber arrows: the prompt's aggregate attention forces, **drawn
  tangent to the sphere at each token** (up to 0.144 of radial pull projected away — see
  the badge above)"*
- Badge tooltip: *"**Each aggregate force is drawn tangent at its own token**, so its
  radial component is projected away first. This is the largest amount removed across the
  prompt…"*
- Source comment: *"They stay tangent at their anchor (the backend projects them there),
  which keeps them outside the r=0.985 shell and therefore visible on the near side."*

**What I expected.** Either the arrow is tangent at the point it is anchored to, or the
UI does not claim it is.

**Cause.** Two different points are being used for "the token".
- Backend projects onto the tangent plane at the layer's **residual-stream input**:
  `code/backend/src/llm_geometry/geo/fields.py:146-171` — `z = tr["hidden_in"][0]`
  (= `tok + pos_embedding` at layer 0, the full residual stream at layers 1–3;
  `code/backend/src/llm_geometry/geo/model.py:98,104`). `normal_residual` is the radial
  part removed **there**.
- Frontend anchors the arrow at the raw **token embedding**:
  `code/backend/src/llm_geometry/geo/model.py:129` (`token_embeddings`) →
  `api/routes_geo.py:149` (`"embeddings"`) →
  `code/frontend/src/viz/geo/GeometryLab.svelte:403` (`traceEmbeddings={trace?.embeddings}`)
  → `code/frontend/src/viz/geo/GeoScene.svelte:339` (`const o = anchors[...]`).

Only the token embedding is on the unit sphere (verified: norms 0.999999–1.000001);
`hidden_in` is not, so the projection is w.r.t. a plane that has nothing to do with the
sphere at the anchor.

**Visible consequence.** `/tmp/redteam-geo/D-force-layer2-zoom.png`: for a 4-token prompt
only **one** amber arrow is visible, and it is a clipped stub at the silhouette — the
others have a large inward radial component and are swallowed by the r=0.985 shell.
`/tmp/redteam-geo/D-force-layer0-zoom.png`: none visible at that zoom.

**False strings to fix:** `GeometryLab.svelte:370` (badge `title`), `GeometryLab.svelte:413`
(caption), `GeoScene.svelte:163-166` (comment).

> Note: the *per-token* field badge, `per-token field: exactly tangent`, **is** true.
> Verified independently: with `antisymmetrize=true`, max |cos(arrow, radius)| over all
> 1003 arrows = 3.0e-6; with it off, mean |cos| = 0.408 and the badge correctly disappears.

---

## HIGH

### H2. A tampered model file with a fabricated vocabulary loads successfully — the integrity check is opt-in and never covers the vocabulary

**What I did.** Trained a model from scratch, saved it
(`geotransformer-6e1843fa.llmgeo.json`, 32 565 bytes), then loaded four mutations.

**What I saw.**

| file | result |
|-|-|
| not JSON | refused — *"not-json.llmgeo.json is not valid JSON."* ✅ |
| `{}` | refused — *"not a Geometry Lab model file (format=None, expected 'llm-geometry/geo-model')"* ✅ |
| one float32 flipped in `weights.embedding.data`, token left intact | refused — *"this model file is corrupt: its weights hash to d6ba8950… but it declares 6e1843fa…. **Loading it would pair the wrong vocabulary with these weights, so it is refused.**"* ✅ |
| every vocabulary word replaced with `zzzN`, weights untouched | **ACCEPTED** — green note *"loaded model-badvocab.llmgeo.json · 1003-token vocabulary"* ❌ |
| weights tampered **AND** vocabulary replaced with `FAKE0…FAKE1002` **AND** `weights_token` deleted | **ACCEPTED, HTTP 200**, `{"weights_token":"f662fcc0…","vocab_size":1003}` ❌ |

For the last case the model became active and every label was silently wrong —
`GET /api/geo/trace` returned `logits_topk.texts = ["FAKE1","FAKE0","FAKE2","FAKE3",…]`.

**What I expected.** The bundle's own docstring
(`code/backend/src/llm_geometry/geo/bundle.py:9-12`) and the on-screen refusal message
both promise exactly this is prevented: *"a silent mismatch would attach the wrong
vocabulary to a set of weights and every label in the UI would then be quietly wrong."*

**Cause.** `code/backend/src/llm_geometry/geo/bundle.py:117-124`:
```python
declared = bundle.get("weights_token")
actual = weights_token(ws)
if declared is not None and declared != actual:   # ← skipped entirely when omitted
```
and `weights_token()` (`geo/weights.py:193-199`) hashes only the weight arrays — `vocab`
is read afterwards (`bundle.py:126-131`) and stored unverified. So (a) omitting the field
skips validation, and (b) the vocabulary is never covered by any hash even when it is
present.

---

## MEDIUM

### M3. Clicking a weight cell and clicking away — without typing anything — silently rewrites the weight and flags the model "hand-edited"

**What I did.** Fresh page, badge `shipped checkpoint`. Clicked one W_V cell (editor
opened showing `0.00028039`), then clicked the prompt field to blur. Typed nothing.

**What I saw.** Badge → `hand-edited weights active | reset to learned`; header →
`active model: hand-edited weights | dad28815`; the sphere visibly changed
(`/tmp/redteam-geo/B1-blur.png` vs `B0-base.png`).

**What I expected.** Opening and abandoning an editor is a no-op.

**Cause.** `code/frontend/src/lib/MatrixHeatmap.svelte:216` `onblur={commitEdit}` commits
unconditionally, and the value it commits is the *display* string seeded at line 148 from
`formatValue()` (line 127-129, `toPrecision(6)`) — which is lossy for float32. So blur
writes a rounded weight. `commitEdit` (line 151-165) only cancels on an *empty* field.
This also races `reset to learned`: in my first pass the blur-commit fired after the
reset click and instantly re-flagged the model as edited.

### M4. Preset `learned` on the **embedding** never returns to the shipped checkpoint — the badge stays permanently wrong

**What I did.** `embedding → random`, then `embedding → learned`. Then compared the
minted token to `spec.checkpoint.checkpoint_id`, and the resulting matrices to the
canonical ones.

**What I saw.** Badge stuck at `hand-edited weights active`, header
`active model: hand-edited weights | 7864adf0` — forever. Control:
`W_V → random → learned` mints exactly `be5359a1c66bda29c8c554269e589009` and the badge
correctly returns to `shipped checkpoint`.
Diff after the embedding round-trip: **44 of 3009 entries differ, max |Δ| = 1.0e-6**;
`W_Q/W_K/W_V/W_O` are bit-identical. So the model *is* the learned checkpoint, and the UI
says it is not.

**Cause.** `code/backend/src/llm_geometry/geo/weights.py:174` — the `preset == "learned"`
fast path is explicitly `and matrix != "embedding"`, so the embedding falls through to
`preset_matrix("learned", ...)` and is then re-normalized at line 181-182
(`_unit_rows`) in float32, drifting ~1e-6. The frontend's "content hash equals the
checkpoint ⇒ not an edit" rule (`WeightLab.svelte:51,98-103`) then never fires.

### M5. The "arrows/point" slider silently no-ops at the default temperature

**What I did.** Temperature 0.00 (the load default), arrows/point dragged 1 → 5.

**What I saw.** Label reads `ARROWS/POINT 5`; the render is byte-identical to
arrows/point = 1 (`35418510:198132` both times). Nothing on screen explains why.
This is *documented backend behaviour* (`fields.py:81-84`, one-hot at T=0) and the
caption's optional clause is correctly suppressed — but a user moving a labelled slider
gets no feedback and no reason.

**Where:** `code/frontend/src/viz/geo/GeometryLab.svelte:353-356` (slider),
`:411` (the disclosure that only renders when `temperature > 0`).

### M6. The force caption names the wrong matrix in the default configuration

Caption: *"thin arrows: the per-token field **W_V·z** at layer 0"*. `antisymmetrize` is
ON by default, in which case the field actually plotted is `((W_V−W_Vᵀ)/2)·z`
(`fields.py:128`). Verified: with antisymmetrize on, arrows are tangent to 3e-6; with it
off, mean |cos| 0.408 — two visibly different fields, one caption.
**Where:** `GeometryLab.svelte:413`.

### M7. After training from scratch (or loading a model) every hover label becomes `token #N`, while the header still promises words

**What I did.** Trained on the full Alice corpus (34 672 tokens, 2 659 distinct, 3
epochs), then hovered the sphere.

**What I saw.** `"token #31" · token 31`, `"token #26" · token 26`, `"token #6" · token 6`
… (8/8 probes). Same after loading a saved model: `"token #67" · token 67`, etc.
Meanwhile the header still reads *"Hover a dot for its word"*, and the token strip and
the top-k list **do** show real words (`<unk>`, `"`, `,`, `the`, `.`, `and`) — so the
vocabulary is right there.

**What I expected.** Either real words, or the header not promising them.

**Note:** the whole vocabulary is already retrievable —
`GET /api/geo/model?weights_token=…` returns `vocab` containing the full `words` array
(the frontend already calls it via `client.geoExportModel`). The fallback in
`GeometryLab.svelte:69-73` only knows ids that happened to appear in a tokenize/trace
response.
**Where:** `GeometryLab.svelte:69-73` (`label()`), `:291-293` (the claim).

### M8. The sphere is cropped left and right at narrow viewports

**What I did.** 390 × 900. **What I saw.** `/tmp/redteam-geo/46-390-mid.png` — the sphere
is cut off at both canvas edges. No horizontal page overflow
(`scrollWidth == clientWidth == 390`, so the layout itself is fine).
**Cause.** `GeoScene.svelte:37` fixes `HEIGHT = 520` and `:215-223` updates
`camera.aspect` on resize but never dollies to fit; at aspect 0.65 with fov 42 the
horizontal half-extent at the sphere is 0.75 < 1. Recovering requires manually
scroll-zooming out.

---

## LOW

### L9. The fine-tune loss chip always says "on your text", even for a Hugging Face dataset
Fine-tuned on `roneneldan/TinyStories`; chip read `loss 5.18 → 4.76 **on your text**`
while the header correctly read `active model: fine-tuned on roneneldan/TinyStories`.
**Where:** `FinetunePanel.svelte:158`.

### L10. An empty prompt fires a request the app knows will fail
Clearing the prompt issues `GET /api/geo/trace?prompt=` → **400**
`{"type":"InvalidParamError","message":"prompt is empty after tokenization"}`. The UI
correctly hides the message (`GeometryLab.svelte:388` gates on a non-empty prompt) but
the request is still made and logged as a console error.
**Where:** `GeometryLab.svelte:227-230`.

### L11. "Edit the weights … and watch the field move" barely holds for W_Q / W_K
`W_Q → zero` changes **5 of 1003** arrows in the default view (max Δ 1.13); renders for
`W_Q/identity`, `W_Q/toeplitz_fuzzy` and `W_Q/random_autocorr` are pixel-identical, as
are three of the four `W_K` presets. Backend-verified — not a frontend no-op — but the
invitation over-promises for those matrices in the default `next_next / T=0 / full` view.

### L12. The embedding heatmap is labelled "read-only" yet presets overwrite it
`WeightLab.svelte:209`: *"read-only · scroll through all 1003 tokens"*. Cell editing is
indeed disabled, but `identity`, `toeplitz_fuzzy`, `random` and `random_autocorr` all
apply to the embedding and visibly change the sphere. (`zero` is correctly refused:
*"preset 'zero' is invalid for the embedding: zero rows cannot satisfy the unit-norm
constraint"*.)

### L13. Source comment states a falsehood (documentation debt, not user-visible)
`GeoScene.svelte:163-166` asserts the force arrows "stay tangent at their anchor … which
keeps them outside the r=0.985 shell". Disproved by C1.

---

## Verified working (I tried to break these and could not)

- **Auto-rotation is OFF on load.** Two canvas captures 2 s apart after settling: **0**
  differing pixels.
- **⟳ spin starts it** — 109 202 pixels change over 1.5 s; button text/`aria-pressed`
  flip correctly.
- **Dragging stops the spin** — `aria-pressed` → `false` and motion decays to 0 differing
  pixels within ~6 s (OrbitControls damping, not a stuck animation).
- **Scroll-zoom works.**
- **Hover tooltips are correct** — 12/12 probes returned sensible labels
  (`"the" · token 5`, `"her" · token 24`, `"even" · token 232`, `"sigh" · token 832`),
  cross-checked against `GET /api/geo/tokenize`. Prompt-path dots hover separately.
- **Occlusion is correct — this is *not* a bug.** Decisive test: 2-token prompt, force
  mode, orbited 360° in 30° steps counting path-green and force-amber pixels. Green went
  **336 → 0 → 0 → 0 → 114** and amber **266 → 777 → 0 → 40 → 283**; at 240°
  (`/tmp/redteam-geo/E-occl-08.png`) the sphere is completely clean. Nothing is drawn
  through the sphere.
- **`per-token field: exactly tangent` badge** — true (3.0e-6), appears only with
  antisymmetrize on, disappears when off.
- **`radial pull projected out: N max`** matches the API exactly (badge 0.144 vs API
  `0.14362`) and is correctly unchanged by the antisymmetrize toggle, exactly as its
  tooltip claims. (Its *meaning* is the C1 problem; its arithmetic is right.)
- **`full` is disabled in force mode** with the title *"the force field is per-layer by
  definition"*; the contract 400s on that combination, and the UI never sends it.
- **Every layer button changes the field** (0/1/2/3 all distinct renders).
  `layer 3` == `full` is correct, not a no-op — verified max |Δ| = 0 against the API,
  and documented in `fields.py`.
- **Weight lab:** 5 matrices × 6 presets = 29 valid applies, all landed; badge and header
  chip track the active model; `reset to learned` restores exactly (badge → `shipped
  checkpoint`, header → the shipped chips) when no cell editor is open; a typed cell edit
  (`2.5` + Enter) changes the sphere and flips the `source` chip to `edited`.
- **Header withholds the shipped-checkpoint chips** whenever another model is active —
  verified after training from scratch: no `shipped checkpoint`, no
  `gutenberg-11-alice-in-wonderland`, no `final loss 4.89`. It shows
  `active model: trained from scratch on your text` + the token prefix. This is right.
- **Fine-tune is real.** Pasted text: `loss 5.20 → 4.00`. HF tab is **enabled** and works:
  `roneneldan/TinyStories`, `loss 5.18 → 4.76`. Nonsense dataset id gives a plain-language
  error: *"HuggingFace dataset 'definitely/not-a-real-dataset-xyz123' (split 'train')
  could not be loaded: Dataset … doesn't exist on the Hub or cannot be accessed."*
- **Train-from-scratch gate.** 10 distinct words → *"12 tokens · 10 distinct words —
  needs 1,000 distinct to fill the vocabulary, so this is 990 short"*, button disabled.
  Full Alice corpus → *"34,672 tokens · 2,659 distinct words — enough to fill the
  1,000-word vocabulary"*, button enabled.
- **Training runs with real progress** (`epoch 2/3 · loss 5.26`) and reports
  *"trained a new model · final loss 5.14 · 34,672 tokens · 3 epochs"*.
- **Save produces a real file** — `geotransformer-6e1843fa.llmgeo.json`, 32 565 bytes,
  `weights_token` matching the active model, config + full vocab + base64 float32 weights.
- **Load restores in a fresh page** (sphere changes, header → `active model: loaded from
  model.llmgeo.json | 6e1843fa`).
- **Edge cases:** empty prompt → no strip, no visible error; out-of-vocab prompt → three
  dashed `<unk>` chips + *"3 tokens · 3 unknown"*; 70-word prompt → *"⋯ truncated at 50"*
  + *"50 tokens"*; 8 rapid mode toggles → no errors, consistent final state; tab-switch
  mid-compute and back → recovers to ready with no errors.
- **Attention panel is real** — all 4 rows sum to 1.000000 and are strictly causal.
- **Console:** zero errors and zero unhandled rejections across all runs, apart from the
  two expected-and-handled HTTP 400s noted in H2/L10 and benign
  `GPU stall due to ReadPixels` WebGL *warnings* caused by my own screenshotting.
- **390 px:** no horizontal page overflow; controls, weight lab and heatmaps all reflow.

## Not tested
- Train-from-scratch via the **HF dataset** tab (only the fine-tune HF path was exercised).
- Whether a saved model reloads after a **backend restart** — the artifact was still in
  the server's cache during the load test, so file-only restoration is unproven.
- The static (GitHub Pages) build; everything above is the live backend.
