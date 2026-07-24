# Batch-2 B2 report — Architecture Explorer HTTP routes (`/api/arch/*`)

Status: COMPLETE. 63/63 pass (20 new contract tests + all 43 batch-1 arch regressions); ruff + black clean.

## Files (only the two I own)
- `code/backend/src/llm_geometry/api/routes_arch.py` — the four contract endpoints
- `code/backend/tests/contract/test_api_arch.py` — 20 contract tests, real models, no mocks

## Endpoints
- `GET /api/arch/graph?model_id=` — size gate, then cached `arch.graph.build_graph`
- `GET /api/arch/weights?model_id=&param=&r0=&r1=&c0=&c1=&max_cells=` — `arch.weights.weight_window`; unknown param -> 404 envelope
- `GET /api/arch/trace?model_id=&prompt=&system_prompt=&max_context=` — `arch.trace.trace_forward`
- `POST /api/arch/generate` — pydantic body, `arch.generate.generate`; temp 0 = greedy, seeded sampling reproducible

## Decisions
- **One rounding helper** `_sig6`: recursive numpy->plain-list coercion with every float through `float(f"{x:.6g}")` (unambiguous 6-sig-digit semantics; tests verify with the identical inline formula, independently re-derived from state_dict slices). Bools checked before ints (bool subclasses int) so `chat_template_used`/`downsampled` flags survive.
- **Gate first, on ALL four endpoints** (contract-wide "before any download"): `check_model_size` runs before any `load_model`. Wrapped in `lru_cache(64)` — passes memoized per process (hub-metadata fetch happens once per model), failures always re-raise since lru_cache never caches exceptions.
- **Trace payloads bounded** by the arch layer itself (T <= max_context via left truncation, per-head attention <= 64x64 strided-mean) — the route adds no extra caps, just the 6-sig-digit encoding, which also keeps JSON small.
- **No pydantic bounds on generate**: `max_new_tokens<=128`, `temperature>=0`, non-empty prompt enforced by `arch.generate` so violations return the typed 400 envelope, not FastAPI's 422 validation shape.

## Contract ambiguities resolved
- Arch trace is silent on empty prompts (geo trace says 400): prompt defaults to `""`; `trace_forward` raises `InvalidParamError` -> 400 envelope. Test locks that in.
- "bias C=1": SmolLM2 (Llama-style) has no attention biases, so the literal bias case is tested on Qwen2.5-0.5B-Instruct (`q_proj.bias`, locally cached); everything else reuses SmolLM2-135M.
- Greedy `prob` follows the batch-1/001 one-hot precedent: exactly 1.0 at temperature 0 (asserted).
- Oversized-model 422 verified on graph AND generate via Qwen2.5-7B-Instruct (hub-metadata path only; no-download property covered by the batch-1 gate unit test).

## Test output tail
```
...............................................................          [100%]
63 passed, 1 warning in 9.73s
```
(command: `python -m pytest tests/contract/test_api_arch.py tests/integration/test_arch_{graph,weights,trace,generate}.py tests/unit/test_arch_gate.py -q`; the warning is starlette's pre-existing httpx TestClient deprecation)

## Notes
- No bugs found in `arch/` internals; nothing outside my two files was touched. No git commands run.
