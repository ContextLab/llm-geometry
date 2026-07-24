# Batch-3 B3 — Architecture Explorer (feature 002, issue #1)

## Component structure (src/viz/arch/**, only files touched)
- `ArchitectureExplorer.svelte` — orchestrator + 3-zone layout: left rail (model/prompt/chat) | center diagram stage (inspector slides in over its right edge) | full-width "Processing breakdown" below. Owns graph loading (module-level per-model cache in `archShared.ts`), trace loading (400 ms debounce + AbortController cancel-and-restart, keyed so a landing trace can't re-trigger itself), ESC-closes-inspector, playback→diagram highlight wiring. Stacks at the app's 900 px breakpoint.
- `ArchModelPicker.svelte` — dropdown (listModels) + free-text HF id; resolves BEFORE committing so a rejected pick leaves the model unchanged; `externalError` slot for graph-time 422s.
- `ArchInspector.svelte` — drawer: kind/op/layer/param chips, per-kind explainer, param list with `tied →` badge; MatrixHeatmap overview (downsampled) with click-to-zoom → exact ≤4096-cell window (r1/c1 exclusive, verified), re-center click, pan arrows, reset, stats, abort-stale tile fetches. Functional ops get plain-language explainer instead of a heatmap.
- `ArchTracePanel.svelte` — tokenization strip (chips with ids, chat-template badge, `<unk>`-safe pre whitespace), rAF playback over `node_activations` execution order (play/pause/scrub/0.5–4× speed, live ‖out‖ readout), per-layer detail: per-head attention heatmap (token labels when exact), residual-norm bar chart, top-10 logits.
- `ArchChat.svelte` — temperature + max-tokens, Generate/Re-run, busy dots, reply with per-token underline colored by probability + hover tooltip (p + top-5 alternatives).
- All requested testids present: arch-view/-model-picker/-diagram/-prompt/-trace-strip/-play/-generate/-reply/-inspector/-error.

## Verified live (real stack, Chromium, screenshots /private/tmp/arch-*.png, visually inspected)
- Cold/warm graph load with phase copy + indeterminate bar + breathing mark (arch-10); SmolLM2 diagram + meta (30 layers · 576 · 9h/3KV · 134.5M).
- Auto-trace of default prompt: 37 chat-templated tokens, causal attention heatmaps, real top-10 ("The" 83.1%, "Paris" 4.1%) (arch-2).
- q_proj inspector: 576×576 downsampled → click → rows 279–342 exact window, hover shows exact cell values, pan/reset (arch-3/4). ESC closes.
- Playback: op 26/424 readout, layer auto-follow, diagram highlight (arch-5).
- Generate: real 64-token reply about Paris, token hover tooltip with top-5 alts, Re-run (arch-6).
- Prompt edit → debounced re-trace ("dragon" strip, top-10 flips to "Once" 21%) (arch-7/8).
- Bad HF id → designed picker error, model unchanged (arch-8). gpt2-xl → ModelTooLargeError before download, model reverted (arch-9).
- Second model Qwen2.5-0.5B-Instruct: 24 layers, lm_head `tied → model.embed_tokens.weight` badge (arch-14).

## Flagged bugs (NOT in my files)
1. `lib/PipelineDiagram.svelte`: pan's `setPointerCapture` on pointerdown retargets the derived click to the SVG, so node/group click handlers NEVER fire from a real pointer (keyboard/synthetic work). Worked around in ArchitectureExplorer (re-dispatch trusted clicks to `elementFromPoint`, drag-guarded); proper fix belongs in the shared component.
2. Backend: `/api/arch/graph?model_id=gpt2` → 500 ComputeError "missing functional steps (attention_softmax/residual_add)" — GPT-2-family fused ops break the 1-to-1 tracing invariant. UI shows the designed Retry state (arch-13), but gpt2 is in the model dropdown, so users can hit it.

## Checks
- `npm run check`: COMPLETED 501 FILES **0 ERRORS 0 WARNINGS**.
- `npm run test` (vitest): 31/31 passed.
- Dev stack was already running when I started (shared with sibling agent) — left it running; throwaway scripts/screenshots kept out of the repo (/private/tmp).
