# Red-team: Architecture tab (`/api/arch/*`, `src/viz/arch/**`)

Date: 2026-07-24 · Stance: adversarial, hunting incorrect DATA (not just crashes).
Method: real Chromium (Playwright, `--enable-unsafe-swiftshader`), screenshotted +
visually inspected every state in `/private/tmp/redteam-arch/`, cross-checked against
`curl` of the live API on :8000. Stack left running.

## Verdict
The Architecture tab is **solid and data-honest** on its core promise: the weight
heatmap, trace top-10, attention, playback, and generation all render REAL model
numbers that match the API to the digit. The bugs are in **model-lifecycle UX and
1-D/edge rendering**, not in the science. One HIGH (a false "success" state that hides
a rejection). No crashes; 0 console errors in normal use.

---

## Findings (severity-tagged, with repro + evidence)

### F1 [HIGH] Oversized/gated model shows a FALSE success state; the rejection is never shown
Repro: Architecture tab → type `Qwen/Qwen2.5-7B-Instruct` in the custom box → Enter.
- Picker badge stays green **"ok"**, message reads **"Qwen/Qwen2.5-7B-Instruct · 28 layers"**, yet the active model silently stays **gpt2** (meta/diagram/dropdown all gpt2). Screenshot `53-toolarge.png`, `55-toolarge-final.png`.
- Polled timeline (`out-lifecycle2.json`): `picker_err` is **null the entire time**; the `ModelTooLargeError` (422) fires in console but is **never surfaced to the user**. This violates FR-107 ("surface the plain-language error at the picker and keep the previous model").
- API confirms the gate works pre-download: `curl .../arch/graph?model_id=Qwen/Qwen2.5-7B-Instruct` → `422 ModelTooLargeError "~7,615,616,512 parameters, over the ceiling of 1,500,000,000"`.
- Root cause (code): `resolveModel` SUCCEEDS for 7B (real 28-layer model) → `ArchModelPicker` sets `status="ok"` + success message. Then `archModelId.set` → graph 422 → parent `ArchitectureExplorer.loadGraph` sets `pickerError` AND reverts `archModelId` to `lastGood`. The revert re-runs the `$effect`, reloads the **cached** gpt2 graph, whose success handler runs `pickerError = ""` — wiping the error within one tick. The picker's own `status/message` are never reset by the graph rejection. Net: user is told a 7B model loaded fine when it didn't.
- Contrast: an *invalid* id (`not-a-real/model-zzz999`) surfaces correctly (F-path via resolveModel failure, not cleared) — so only the size-gated / graph-time rejection is invisible.

### F2 [MEDIUM] Default model disappears from the dropdown after switching away
`archModelId` default `HuggingFaceTB/SmolLM2-135M-Instruct` is **not** in curated
`/api/models`; `ArchModelPicker` only appends it as an `<option>` while it is the
active model. Evidence (`out-lifecycle2.json`): options include SmolLM2-135M at start
(`start_smollm2_in_options: true`), but after selecting gpt2 it is **gone**
(`smollm2_still_in_options_after_gpt2: false`). A user who changes models cannot
return to the default via the dropdown — they must retype the full HF id. (Caused a
`selectOption` timeout in an early test run.) Back-via-custom-input is instant/cached
(`back_ms: 21`).

### F3 [MEDIUM] 1-D params (bias/norm) render as an unusable ~4px-wide, thousands-of-px-tall strip
Repro: Qwen2.5-0.5B → click `model.layers.0.self_attn.q_proj` → select the **bias
[896 × 1]** param. Screenshot `61-qwen-bias.png` shows a barely-visible vertical
hairline. Measured canvas: **width 4px, height 3584px** (`out-misc.json`
`bias_canvas_h:"3584px"`, `bias_canvas_w:"4px"`). The overview fetch (`zoom=null`)
requests the whole [896,1] tensor (896 ≤ 4096 cells → exact), and `MatrixHeatmap`
floors cell size to the 4px minimum → 896×4px. gpt2 `c_attn.bias` [2304×1] would be
9216px. Stats are correct (min -79 · max 47.75 · μ -0.152 · σ 7.853, matches API) but
the map itself is illegible and pushes the pan/stats controls thousands of px down.
`windowFor` caps the *zoom* window at 128 rows but the default *overview* does not.

### F4 [MEDIUM-LOW] Temperature 0 misrepresents alternative-token probabilities as 0.0%
Repro: temperature slider → 0 → Generate → hover first reply token. Tooltip:
`p = 100.0% · top-5: "The" 100.0% · "Paris" 0.0% · "As" 0.0% · "Bon" 0.0% · "I" 0.0%`
(`out-chat.json` `t0_tok_tooltip`, screenshot `41-gen-t0-hover.png`). The alternatives'
probabilities are all 0.0% because `arch/generate.py` builds a one-hot `probs` vector
at `temperature==0` and reports `probs[i]` for the top-k. Paris really carries ~2-4%
(confirmed at T=0.8: `"Paris" 2.2%`), so a student reading "Paris 0.0%" is misled. The
token *identities* (top-k over logits) are correct — only the displayed probabilities
are degenerate.

### F5 [LOW] Chat reply prints the raw special token `<|im_end|>`
`out-chat.json`: `t0_reply_1 = "The capital of France is Paris.<|im_end|>"` while
`.replymeta` says "finish: eos" and the API `text` field (skip_special_tokens) is
`"The capital of France is Paris."`. `ArchChat` renders per-token `t.text` spans, and
the eos token decodes (no-skip) to the literal marker. Looks like leaked template text.

### F6 [LOW] Weight-map hover labels are window-LOCAL, not global indices
Zoomed q_proj window "rows 197–260 · cols 312–375"; hovering the cell at global
(202,320) shows tooltip **"row 5 · col 8 · -0.328125"** (`out-xcheck.json`,
`21-qproj-hover.png`). Value is exactly right, but the row/col are window-relative
(`MatrixHeatmap` gets no `rowLabels/colLabels` from the inspector), so the user must
add the `viewmeta` offset to know which real row/col they're on.

### F7 [LOW/INFO] No reload persistence for Architecture state
After a page reload the app returns to the **Vector field** tab and arch
model/prompt reset to defaults (`out-misc.json` `reload_view:"other"`, screenshot
`64-after-reload.png`). `view` and the arch stores are plain in-memory writables (only
`geoWeightsToken` persists to sessionStorage). Likely by design; noted per checklist.

### F8 [LOW] Narrow viewport (390px) horizontal overflow
Screenshot `63-narrow.png`: the tab bar (Architecture pill clipped, Geometry off-screen)
and the arch controls (model select, prompt) overflow past the right edge — page scrolls
horizontally. Partly shared app chrome, but the Architecture controls are among the
clipped content.

### F9 [INFO] 500-word prompt left-truncates to 64 tokens with no indication
`curl` of a 500-"word" prompt returns exactly 64 tokens, all `" word"` — the chat
template's opening tokens were truncated away, yet `chat_template_used` stays `true`
and the UI still shows the "chat template" chip. The `/api/arch/trace` contract has no
`truncated` flag (unlike `/api/geo/tokenize`), so the UI cannot indicate left-truncation.

### F10 [INFO/expected] CJK/emoji tokens show replacement glyphs
Prompt `こんにちは 🌍 世界` → strip + top-k render `�`/◆ glyphs (`33-cjk.png`,
`out-trace.json` `cjk_topk`). Correct byte-level-BPE behavior (a token is a byte
fragment of a multi-byte char), just visually noisy — flagging so it isn't mistaken
for corruption.

---

## Verified CORRECT (attacked, held up)
- **Weight data**: hovered cell (local 5,8) = **-0.328125** matches API `window[5][8]` AND an independent single-cell fetch at global (202,320) = -0.328125. Exact-vs-downsampled label + stats match API. Pan clamps to r0=0 boundary (rows 0–63). Overview 576×576 downsamples to 64×64 "strided mean"; exact ≤4096-cell window is truly exact.
- **Trace top-10 UI == API** to the digit (The 83.1 / Paris 4.1 / As 1.8 / Bon 1.5 / I 1.4 / France 0.9 / B 0.6 / " 0.6 / In 0.4 / F 0.3). Token strip = 37 chips = API token count.
- **Attention row-stochastic**: all 9 heads' last-token rows sum to 1.0 (API, layer 0, non-downsampled).
- **Diagram**: 424 nodes SmolLM2; expand layer 15 / collapse layer 0 both work; wheel-zoom + drag-pan mutate viewBox; module=rect, functional=pill. Node/pill click opens the matching inspector; **inspector kind always matches node kind** (q_proj→linear+heatmap; softmax/rope→functional "no learned weights" note, no heatmap; embedding 49152×576; lm_head **tied → model.embed_tokens.weight** badge). ESC closes; focus+Enter (keyboard) opens.
- **Playback**: plays through **op 424/424**; scrub-backwards mid-play seeks + pauses; 0.5/1/2/4× speed; layer auto-follows playhead; typing a new prompt **cancels playback with no zombie** (resets to idle, new distribution loads).
- **Generation**: T=0 twice → **byte-identical** reply; Generate button disables while busy (`generating…`, no double reply); T=0.8 tooltip shows real top-5 probs (The 93.9 / Paris 2.2 …); max-tokens slider bounds **8–128**; temperature 0.00 reachable.
- **gpt2** renders with **softmax(12) + residual(24) pills and ZERO rope nodes** (`out-lifecycle.json` `gpt2_rope_nodes:0`), Conv1D `c_attn` [768×2304] exact, lm_head tied to `transformer.wte.weight`. Qwen meta 24L/896/14h·2KV/151936/494M and gpt2 12L/768/12h/50257/124M both match `/api/models`.
- **Model gating**: invalid id → typed `UnsupportedModelError` at picker, model unchanged; Qwen-7B → `422 ModelTooLargeError` from hub metadata BEFORE any download.
- **Edge prompts**: empty → cleared "Type a prompt" state; 1-char + CJK trace fine.
- **Robustness**: 10× tab cycling → **0 console errors**, model persists across tabs; 49k-row embedding overview downsamples & renders.

## Evidence
Screens/JSON in `/private/tmp/redteam-arch/` (01–64, `out-*.json`, `api-*.json`).
Throwaway Playwright drivers `rt*.mjs` there too.
