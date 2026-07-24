# Batch-1 A1 report — Geometry Lab backend (`llm_geometry/geo/`)

Date: 2026-07-24 · Branch: 002-interactive-model-explorer · Status: COMPLETE, all tests green.

## Files created
- `code/backend/src/llm_geometry/geo/__init__.py` — package docstring/overview.
- `geo/config.py` — frozen arch constants (d_model=3, 4 layers, 1 head, mlp 12, vocab 1003, ctx 50, seed 0), corpus constants + sha256, tuned training hyperparams, gate thresholds.
- `geo/corpus.py` — Alice in Wonderland (Gutenberg #11) acquisition: sha256-verified, atomic download-if-absent, Gutenberg marker stripping.
- `geo/tokenizer.py` — deterministic word-level tokenizer (top-1000 corpus types + specials), unk marking, side-selectable truncation, JSON round-trip.
- `geo/model.py` — GeoTransformer: norm-free pre-residual decoder, learned absolute positional embeddings, unscaled ⟨k,q⟩ attention (matches the paper/contract force formula), per-matrix addressable W_Q/K/V/O, tied unembedding, `forward_trace` returning every contract tensor.
- `geo/weights.py` — 6 presets (both 3×3 and 1003×3), build_weight_set with validation → `InvalidWeightEditError`, content-hash `weights_token`, CacheStore persistence.
- `geo/train.py` — deterministic CPU training (CE + Wang-Isola spherical repulsion, per-step embedding renorm), canonical checkpoint via cache/store.py, gate metrics, `train_canonical(progress_cb)` with "epoch 7/30 · loss 4.12" messages.
- `geo/fields.py` — `next_next` (batched, layer logit-lens, T=0 argmax / top_m prob-weighted) and `force` (per-point W_V·z, antisymmetrize ⇒ exactly tangent, per-position Σ softmax(⟨Kz_j,Qz_i⟩)Vz_j with normal residuals, layer="full" rejected) + Fibonacci-bin metrics.
- `geo/finetune.py` — real SGD fine-tuning from text/file-text/HF streaming datasets (≤500 steps), result caching, never mutates canonical.
- `errors.py` — added `InvalidWeightEditError` (422) only.
- `data/raw/alice-in-wonderland.txt` — committed corpus (public domain), sha256 `a3a27f8e…403754`.
- Tests: `tests/unit/test_geo_tokenizer.py`, `tests/unit/test_geo_weights.py`, `tests/integration/test_geo_{training,fields,finetune}.py` (54 tests, all real — real corpus, real training, real HF streaming of roneneldan/TinyStories).
- Deps: added `datasets>=2.19` to `pyproject.toml`; merged its 13 pins into `requirements.lock` (datasets==5.0.0).

## Design decisions
- **No layer norm** (documented in model.py): at d_model=3, norm would erase the radial signal the tab visualizes; stability comes from small W_O/W_out init + unit-norm embeddings.
- **Learned absolute positional embeddings** (50×3): simplest fully-visualizable choice.
- **Unscaled attention scores** so trace ≡ force-field formula ⟨K z_j, Q z_i⟩ exactly.
- **Punctuation counts as vocab types** (top-1000 by frequency, ties alphabetical): an LM over the corpus is meaningless without it; ~6.4% unk rate on the full corpus.
- **Embedding presets**: identity/toeplitz_fuzzy cycle the diagonal (row i peaks at col i mod 3); every embedding preset unit-normalized; `zero` embedding rejected (can't be unit-norm) — a designed 422.
- **toeplitz σ=0.75**; random presets seeded via `np.random.default_rng(seed)`; random_autocorr = Gaussian-filtered white noise (σ=8 down the vocab axis for embeddings).
- **Training**: Adam lr 2e-2, 30 epochs, batch 64, stride-10 windows, eos between paragraphs; repulsion weight tuned by sweep {0.05, 0.15, 0.3, 0.6} → **0.3** (0.05 fell below the coverage gate at full length). Full canonical train ≈ 25 s CPU (budget 180 s). Determinism: `torch.use_deterministic_algorithms(True)` + seeded Generators; same seed ⇒ bit-identical checkpoint hash (verified).
- **normal_residual** = |⟨f_i, ẑ_i⟩| (contract says "magnitudes"); antisymmetrize applies to per-point field only, aggregate forces always use real W_V.
- **Unusable HF dataset id → `UnsupportedModelError` (422)**: contract fixes the status but not the type, and only one new error class was authorized; flagged for route-layer review.
- finetune with an isolated store resolves `base="learned"` from the shared cache but writes only to the isolated store (tested).

## Gate metrics (canonical checkpoint `be5359a1c66bda29c8c554269e589009`)
| metric | value | threshold |
|-|-|-|
| final_loss (nats) | 4.885 | < 5.91 (ln 1003 − 1) |
| coverage_uniformity | 0.9005 | ≥ 0.80 |
| field_directional_entropy (nats, 64 bins) | 2.812 | ≥ 2.0 (max 4.16) |

## Test output tail
```
......................................................                   [100%]
54 passed in 70.96s (0:01:10)
```
(cold-cache first run 73.8 s incl. canonical training; ruff + black clean on all new files.)

## Unresolved / notes for red-team
- `weights_token` for the *canonical* checkpoint doubles as `checkpoint_id` (same content hash) — routes should treat them interchangeably.
- Error type for unusable HF dataset ids (above) is the one contract-gray area.
- git status also shows another agent's `ModelTooLargeError` in errors.py (expected, batch-parallel).
