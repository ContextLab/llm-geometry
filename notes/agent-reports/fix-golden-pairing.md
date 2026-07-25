# Fix: golden-source/engine pairing for geoEngine tests (2026-07-25)

CI failure (run 30178823081): the suites merged two golden sets (macOS-trained
fixtures + Linux-trained static export) and evaluated the merge against ONE
engine. Training is platform-divergent at the bit level, so cross-source
evaluation legitimately fails in CI. Fix: pair each golden set with its OWN
checkpoint+vocab and run every golden-driven suite once per source.

## What changed

- `code/frontend/tests/unit/geoGoldenAssets.ts`
  - :100 `GoldenSource` interface; :113 `goldenSources()` — static-export
    source first (only when golden.json + checkpoint.json + vocab.json all
    exist and parse under STATIC_DIR, via existing `normalizeStaticGolden`),
    then the fixtures source (same three files; generate.py emits exactly
    those). Header comment (:1-21) documents WHY pairing matters.
  - Removed now-unreferenced `loadAsset` / `loadGoldens` / `loadMergedGolden`.
- `code/frontend/tests/unit/geoEngine.test.ts`
  - :105 module-level `for (const src of sources)` → `describe(geoEngine
    golden [src.name])` with `engine = GeoEngine.fromAssets(src.checkpoint,
    src.vocab)`; spec/tokenizer/trace/vector_field/weights golden tests run
    per source. Spec test compares each engine to ITS OWN golden
    checkpoint_id. `logitsForOrigin` now takes the engine (:87).
    Static tokenizer test `it.runIf` (static export ships no tokenize
    goldens; fixtures still strictly required non-empty).
  - :293 golden-independent behavior + perf, :404 persistence round-trip: run
    ONCE against the fixtures engine explicitly (platform-stable).
- `code/frontend/tests/unit/geoEngineFinetune.test.ts`
  - :51 same per-source loop for the golden loss-trajectory test; :120
    golden-independent finetune tests pinned to the fixtures engine.
- Tolerances/assertions unchanged (RTOL 1e-5 / ATOL 2e-6, exact token
  equality, finetune trajectory bands).

## Validation (all pass)

- Targeted (full static-data, BOTH sources exercised):
  `npx vitest run tests/unit/geoEngine.test.ts tests/unit/geoEngineFinetune.test.ts`
  → 2 files passed; **30 passed | 1 skipped (31)**.
  geoEngine.test.ts: 23 tests (7 per source ×2, static tokenizer skipped,
  +9 fixtures-only). geoEngineFinetune.test.ts: 8 tests (2 makeWindows,
  1 per source ×2, 4 fixtures-only).
- Whole suite `npm run test`: **9 files passed, 103 passed | 1 skipped (104)**.
- `npm run check`: 1156 files, **0 errors, 0 warnings**.
- Quick simulation: moved static-data aside, ran
  `export_static_assets.py --quick --git-sha t --generated-at t` (8.5s, real
  backend), `npm run test` → **103 passed | 1 skipped**; restored:
  EXPORT_RC=0 TEST_RC=0 RESTORE_RC=0, index.json `"quick":false`.

## Limitation (stated plainly)

Cross-platform divergence cannot be reproduced locally (both local sources are
macOS-trained, so their checkpoints agree). The proof here is the pairing
itself plus the per-source spec test asserting each engine reproduces its own
source's checkpoint_id; CI (Linux export + macOS fixtures) exercises the
divergent case.
