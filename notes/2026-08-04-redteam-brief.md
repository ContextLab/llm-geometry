# Red-team brief — the full four-tab demo set

Shared context for every red-team, fix, and verification agent in this campaign.
Read this first; your individual charter names your slice.

## The app

Four tabs, deep-linked by `location.hash`, deployed at https://context-lab.com/llm-geometry/
and runnable locally with `sh scripts/dev.sh` (backend :8000, frontend :5173).

| Tab | Source | Backend |
|-|-|-|
| Architecture Explorer | `code/frontend/src/viz/arch/` | `llm_geometry/arch/`, `api/routes_arch.py` |
| Geometry Lab | `code/frontend/src/viz/geo/` | `llm_geometry/geo/`, `api/routes_geo.py` |
| Lexicon Lab | `code/frontend/src/viz/lex/` | `llm_geometry/lex/`, `api/routes_lex.py` |
| Info | `code/frontend/src/viz/info/` | none (static prose) |

Two data modes, and **both must be attacked**:
- **full stack** — `npm run dev`, real FastAPI backend, real PyTorch.
- **static** — `VITE_DATA_MODE=static`, everything in-browser
  (`src/lib/staticClient/`, `src/lib/geoEngine/`, `src/lib/lexEngine/`), plus
  transformers.js and safetensors HTTP Range reads. This is what the public site runs.

## Project rules that bind you

These come from `CLAUDE.md` and are not negotiable:

1. **No mocks, no simulations, no fallbacks.** If real functionality does not work, the
   code must raise or fail loudly. A plausible-looking number produced by a degraded path
   is the worst possible outcome and is exactly the defect class we are hunting.
2. **Never dismiss a problem as "pre-existing" or "not from our changes."** Every error you
   encounter gets reported, regardless of when it was introduced.
3. **Ground every claim in a verbatim quote** from the file, the console, or the test
   output. Do not paraphrase evidence. If you did not observe it, say "I did not verify this."
4. **Say "I don't know"** rather than guessing.
5. Keep the repo clean: scratch scripts go to `scripts/` or get deleted; screenshots go to
   `docs/screenshots/` or get deleted.

## The defect class we care most about

Not crashes — crashes are easy. We are hunting **plausible wrong answers**: a path that
returns a number, renders a field, or reports a loss that is *wrong* while nothing throws.
Two shipped examples from this feature, both of which passed CI:

- `d6e9d5d` — the Geometry Lab saved weights under the canonical vocabulary after a reload
  for any model with its own word list, and the digests *verified*, because the writer
  computed them over the substituted list.
- `c115ec6` — `q4f16` on WebGPU built a session fine and returned logits carrying no
  information about the input. Three of four curated models were destroyed. Headless
  Chromium has no WebGPU adapter, so e2e never saw it.

When you find something, ask: *would this have thrown?* If not, it is high severity.

## Known gaps — report if you find them worse than stated, not as news

- Issue #4 — the Architecture model list is curated; arbitrary HF ids are not supported.
- Issue #5 — curated ONNX mirrors are pinned to `main`, not a commit sha.
- Issue #6 — Geometry Lab fine-tuning ignores a scratch-trained model's own vocabulary.
- Issue #7 — a11y: hover-only tooltips, misused `tablist` roles, a layout table.
- The `webgpu` Playwright project SKIPS on CI runners (no adapter). It passes locally on
  a machine with a real GPU.
- Static mode **refuses** `nonce − swap` and per-passage deltas under q8 because no error
  bound was measured for them. A refusal with a typed error is correct behaviour, not a bug.
- `swap` is injective only at `p ∈ {0, 1}`; intermediate `p` is refused, citing
  `specs/007-vacancy-transform-field/architecture.md` §5.2a. This is a theorem, not a defect.

## Reporting

Write your findings to `notes/agent-reports/<your-assigned-filename>`. Structure each
finding as:

```
### F<n>. <one-line claim>
**Severity:** critical | high | medium | low
**Where:** file:line (or URL + tab + control)
**Reproduce:** exact steps or exact command
**Observed:** verbatim output/quote
**Expected:** and why
**Would it have thrown?** yes/no
```

Then reply with **under 2000 characters**: the count by severity and the one-line claims
only. List outcomes, not process. The file is the deliverable.
