# Batch-1 A3 — frontend scaffolding for feature 002 (issue #1)

Date: 2026-07-24 · Branch: 002-interactive-model-explorer · Frontend only; no git commands run.

## Files created / changed
- `code/frontend/src/lib/explorerStores.ts` (new) — per-view stores: arch (archModelId default SmolLM2-135M-Instruct, archPrompt, archSystemPrompt, archTemperature 0.8, archMaxNewTokens 64, archSelectedNode null) + geo (geoPrompt Alice-corpus default sentence, geoFieldMode "next_next", geoLayer "full", geoTemperature 0, geoTopM 1, geoAntisymmetrize false, geoWeightsToken). Exports `GeoFieldMode` (re-export from dataClient) + `GeoLayerSelection`.
- `code/frontend/src/lib/dataClient.ts` (extended, append-only; existing API untouched) — ~30 typed interfaces covering every shape in the frozen contract (GeoSpec/GeoTrace/GeoVectorFieldData/GeoWeights*/GeoFinetune*, ArchGraph/ArchWeightsData/ArchTrace/ArchGenerate*; tensors as number[][] / number[][][] per contract) + 13 client methods: getGeoSpec, geoTrain, geoTokenize, getGeoTrace, getGeoVectorField, getGeoWeights, postGeoWeights, geoFinetune, geoFinetuneFile (multipart), getArchGraph, getArchWeights, getArchTrace, archGenerate. Plus exported `debounced(fn, ms=400)` (trailing-edge, `.cancel()`).
- `code/frontend/src/lib/MatrixHeatmap.svelte` (new)
- `code/frontend/src/lib/PipelineDiagram.svelte` (new)
- `code/frontend/tests/unit/explorerClient.test.ts` (new, 12 tests), `tests/unit/explorerStores.test.ts` (new, 3 tests)

## Component API decisions
- **MatrixHeatmap** props `{values, rowLabels?, colLabels?, editable?, onCellEdit?, maxCanvasPx?=560}`. Accepts `number[][] | number[]` (1-D normalizes to a single column, matching the contract's C=1 biases). Diverging scale centered at 0 using app tokens: --bad → --bg-elev-2 → --accent. DPR-aware canvas; rect fills with hairline gap when cell ≥ 8 px; cell size clamped 4–36 px so ≤64×64 grids fit `maxCanvasPx`. Hover uses the shared `tooltip.ts` (`row · col · value`, labels substitute indices when given). Editable: click → absolute-positioned numeric input over the cell; Enter/blur commits via `onCellEdit(r,c,v)`, ESC closes; canvas is `tabindex=0` + role=grid, ESC on it also closes. `data-testid="matrix-heatmap"` / `"heatmap-cell-editor"`.
- **PipelineDiagram** props `{nodes, edges, selected?, onSelect?}` typed against `ArchNode`/`ArchEdge` from dataClient. Vertical flow grouped by `node.group` in first-appearance (trace) order. Param modules = rects (width ∝ log10 param count; kind→color: embed/lm_head purple, linear/mlp blue, norms dim); functional ops = smaller pills (1-to-1 invariant: parameterless steps stay visible). Layer groups collapsible; default-collapsed for layer ≥ 1 when >2 layer groups; user toggles kept as overrides, reset when a new graph arrives. Edges anchor bottom→top of nodes; edges into collapsed groups reroute to the placeholder and de-dupe; long skips (residuals) bow right as cubic arcs. Wheel zoom about cursor (manual non-passive listener) + pointer-capture drag pan, both on the SVG viewBox; view refits once per graph via `untrack`. `data-testid="diagram-node-<id>"`, `diagram-group-<group>`, `diagram-collapsed-<group>`; hover tooltip shows id/kind/param shapes/tied_to.

## sessionStorage + abort plumbing
- `geoWeightsToken` init ← `sessionStorage.getItem("llm-geometry:geo-weights-token")`; a module-level `subscribe` writes every change back (null ⇒ removeItem). try/catch guards make storage-less environments degrade to in-memory. Tokens are content hashes (stateless server-side), so a restored token stays valid after refresh/restart — spec acceptance 2.3.
- `getGeoTrace`, `getGeoVectorField`, `getArchTrace` take an optional `AbortSignal`, passed as `fetch(..., { signal })` through the shared `request()` helper; an abort rejects with a DOMException (`AbortError`), not an ApiError, so callers can distinguish cancellation from server errors. Pair with `debounced()` for FR-108 cancel-and-restart.

## Verification (tails)
`npm run check`:
```
1784907587412 COMPLETED 489 FILES 0 ERRORS 0 WARNINGS 0 FILES_WITH_PROBLEMS
```
`npm run test`:
```
 ✓ tests/unit/dataClient.test.ts (5 tests) 7ms
 ✓ tests/unit/explorerClient.test.ts (12 tests) 9ms
 ✓ tests/unit/explorerStores.test.ts (3 tests) 31ms
 Test Files  3 passed (3)
      Tests  20 passed (20)
```
All 5 pre-existing tests still pass. One svelte-check warning (non-$state `panning`) surfaced mid-work and was fixed. App.svelte, stores.ts, viz/, backend untouched.
