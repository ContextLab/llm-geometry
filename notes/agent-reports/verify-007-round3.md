# Verify 007 — round 4, against the round-3 fixes (agent "VERIFICATION AGENT V4")

Date: 2026-08-04. Branch `main` @ `4c4327a`. No file in the repo was modified by me; every
probe ran from the session scratchpad, and the backend I started ran on **port 8010** with
`LLM_GEOMETRY_CACHE_DIR` pointed at the scratchpad. `npm run test:e2e` was not run.

Charter: refute `notes/agent-reports/fix-007-{geo,arch,lex}-r3.md`. Findings are grouped as
the charter asks: (a) defects surviving, (b) NEW defects the round-3 fixes introduced,
(c) claims I could not verify, (d) what is genuinely fixed, (e) mutation testing of round
3's own new tests.

Verdicts by charter item:

| # | item | verdict |
|-|-|-|
| 1 | the vocabulary-substitution identity, third fix | **PARTIAL** — no fourth path to a self-consistent bad *file*; three live wrong-answer paths remain |
| 2 | "a tampered file is now refused" | **PARTIAL** — refusal confirmed; the conceded hole is not the only one left |
| 3 | the error-bar wording | **REFUTED** |
| 4 | the joiner class | **REFUTED** |
| 5 | `Object.hasOwn` sweep | **PARTIAL** — all four fixes verified; the sweep was not complete |
| 6 | unk bound + seed validation | (a) **VERIFIED** · (b) **PARTIAL** · (c) **PARTIAL** · (d) **VERIFIED still broken, and understated** |
| 7 | teeth | **REFUTED** — 7 fresh mutations, **5 survived** the test that claims to cover them (3 survived the whole suite) |
| — | SC-703 + the TS↔Python differential suite | **VERIFIED undisturbed** |

---

## (a) Defects surviving

### F1. `GET /api/geo/tokenize` answers 200 under the SHIPPED word list for a token it does not have, while `GET /api/geo/trace` answers 404 for the same token
**Severity:** high
**Where:** `code/backend/src/llm_geometry/geo/tokenizer.py:206-211` (`tokenizer_for` returns
`get_tokenizer()` on a store miss); route at `api/routes_geo.py:125-129`. Mirror:
`code/frontend/src/lib/geoEngine/index.ts:285-288` (`tokenizerFor` → `this.vocabs.get(token) ?? this.tokenizer`),
called by `tokenize()` at `:292-295` with no `resolveWeightSet`.
**Reproduce:**
```
curl -s -w "  HTTP %{http_code}\n" "http://127.0.0.1:8010/api/geo/tokenize?text=alice&weights_token=deadbeefdeadbeefdeadbeefdeadbeef"
curl -s -w "  HTTP %{http_code}\n" "http://127.0.0.1:8010/api/geo/trace?prompt=alice&weights_token=deadbeefdeadbeefdeadbeefdeadbeef"
```
**Observed:**
```
{"tokens":[{"id":17,"text":"alice","unk":false}],"n_unk":0,"truncated":false}
  HTTP 200
{"error":{"type":"NotFoundError","message":"weights_token 'deadbeef…' is unknown (never minted here, or evicted); re-submit the edit to mint it again"}}
  HTTP 404
```
and directly, after deleting a scratch model's artifact from the cache dir (Python):
```
  after evicting tb's artifact:
    tokenizer_for(tb).words[:3] = [',', '"', 'the']          # the SHIPPED list
    resolve/load_weight_set(tb): NotFoundError(NotFoundError): weights_token '6a0005…' is unknown
```
(the model's real words were `['nibbling', "needn't", 'muttering']`). Same in the browser
engine — `vite-node` probe: `engine.tokenize("the cat sat", "<unknown>")` returns tokens
resolved against the canonical vocabulary, no throw.
**Expected:** this is the campaign's central corruption — a model's ids read under Alice in
Wonderland's word list — reached by a *fourth* route: not a hash collision, not a dedup, but
a **store miss silently falling back to the canonical tokenizer**. `weights.py:219`'s own
docstring calls the outcome "every label on screen is wrong". The two endpoints disagree
about whether the model exists, so the wrong answer is provably not a "the model is gone"
state. CLAUDE.md: "if real functionality doesn't work … raise an exception or fail."
**Would it have thrown?** **No.** 200 with plausible tokens.
**Reachability:** in the static build `MINTED_SETS_CAP = 8`
(`src/lib/staticClient/geo.ts:67`) LRU-drops persisted sets while `geoWeightsToken` is
persisted under a *separate* sessionStorage key (`src/lib/explorerStores.ts:59-77`), so the
token routinely outlives its payload. `GeometryLab.svelte:161` then calls
`client.geoTokenize(text, token)`, `verifyVocab` compares against the bundled canonical
table, **succeeds**, and the tab reports the vocabulary verified. `healEvictedToken`
(`GeometryLab.svelte:296-306`) only fires later, on the field/trace call.

### F2. Round 3's identity change is a persisted-format change with **no `SCHEMA_VERSION` bump**: a pre-round-3 cache makes the user's own scratch model permanently unsaveable, and can wedge `train_canonical`
**Severity:** high
**Where:** `code/backend/src/llm_geometry/config.py:19` — `SCHEMA_VERSION = 14`, last touched
in `eccd362` (feature 004); the identity change is `0d23123`. `geo/weights.py:274-343`,
`geo/bundle.py:98-107`.
**Reproduce:** write the entry the *previous* build wrote for a scratch model (weights-only
token, own word list in `meta`), then use it — `scratchpad/attack2.py`:
```
  wrote a pre-fix entry, token = be5359a1c66bda29c8c554269e589009  (SCHEMA_VERSION still 14 )
  tokenizer_for(old).words[:3] = ['nibbling', "needn't", 'muttering'] (correct)
    GET /api/geo/model  -> export_bundle(old_token): TYPED InvalidParamError: weights_token
      'be5359a1…' does not match a re-hash of its own weights and vocabulary ('6a000549…') —
      the stored model is inconsistent, so saving it would produce a file that names the wrong model.
    POST /api/geo/weights -> mint_weight_set(base=old_token): OK -> 34c349dcff6dc6eb5ce9bb315bd47206
```
and, with a conflicting claim stored at the canonical key, the app's own bootstrap dies:
```
  File ".../geo/weights.py", line 320, in save_weight_set
    raise InvalidParamError(
llm_geometry.errors.InvalidParamError: weights_token 'be5359a1c66bda29c8c554269e589009' is
already stored with a different vocabulary claim (stored owns_vocab=True, writing
owns_vocab=False) — refusing to overwrite or to reuse it …
```
— raised from `geo/train.py:304` inside `train_canonical()`, i.e. the Geometry Lab cannot
start at all until the cache is deleted by hand.
**Expected:** `SCHEMA_VERSION` exists precisely so that "the format changed -> recompute"
(`cache/store.py:68`). The token *is* the artifact key, so the identity change silently
orphaned every stored `geo-weights-*` entry that carried a vocabulary, and the message
accuses the user's own model of being corrupt with no remedy but "Retrain or reload it".
The browser equivalent is the same change, silent: a pre-fix `:v2` sessionStorage payload
(`ownsVocab: true` + `vocabWords`, weights-only token) fails the new hash check and
`restorePersistedSets` **deletes it** —
`vite-node` probe: `importWeightSet(preFixToken, prefix payload) -> false`. The key is still
`:v2`, so nothing signals the format moved.
**Would it have thrown?** Backend yes (typed 400 / a 500-level bootstrap failure); browser
**no** — the model just vanishes.

### F3. `POST /api/geo/model` — the model-file upload path — returns **untyped HTTP 500** for seven malformed `vocab` blocks
**Severity:** medium
**Where:** `geo/bundle.py:190` (`GeoTokenizer.from_json(vocab_json)  # raises on a malformed
vocabulary`) → `geo/tokenizer.py:182-187`, which does `json.loads`, `data.get`,
`list(data["words"])` with no guards. The comment claims the call is the guard; it is not a
typed one.
**Reproduce:** take a real bundle (`GET /api/geo/model`), substitute `vocab`, recompute
`vocab_sha256`, POST it.
**Observed (verbatim, port 8010):**
```
  not JSON                         HTTP 500  {"error":{"type":"InternalError","message":"Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
  a JSON array                     HTTP 500  {"error":{"type":"InternalError","message":"'list' object has no attribute 'get'","detail":{}}}
  a JSON string                    HTTP 500  {"error":{"type":"InternalError","message":"'str' object has no attribute 'get'","detail":{}}}
  null                             HTTP 500  {"error":{"type":"InternalError","message":"'NoneType' object has no attribute 'get'","detail":{}}}
  words=null                       HTTP 500  {"error":{"type":"InternalError","message":"'NoneType' object is not iterable","detail":{}}}
  words=nested lists               HTTP 500  {"error":{"type":"InternalError","message":"unhashable type: 'list'","detail":{}}}
  `tokens` shape (TS accepts)      HTTP 500  {"error":{"type":"InternalError","message":"'words'","detail":{}}}
```
**Expected:** typed 400s — this is the same class the campaign already fixed for
`weights_token`/`vocab_sha256`; the *vocabulary* half of `import_bundle` was rewritten in
`0d23123` and left leaking.
**Would it have thrown?** Yes, but untyped and as a 500.

### F4. Everything the sibling agents refuted (items 3, 4, 5)
Recorded here in one place; full evidence in the session transcript.

* **Item 3 — the error bar. REFUTED.** The `measured` label is gone from `errorBarTerms`
  and `secondaryLine`, but two *other* shipped strings on the same panel still make the
  claim FR-720a forbids:
  - `src/lib/staticClient/arch.ts:458-469` (`VACANCY_UNKNOWN_FORM_REFUSAL`, rendered at
    `VacancyScorePanel.svelte:270`) — "**Measured on this very configuration: float32 says
    0.273 for gpt2 and q8 says 0.235**, a 14 % error". `architecture.md` §8.3a records
    `0.273 ± 0.041` as the **pre-rewrite** number and says "The q8 arm has not been
    re-measured since the swap rewrite". **critical.**
  - `arch.ts:448-455` (`VACANCY_PER_PASSAGE_REFUSAL`) — "**Only the pooled figure has a
    measured bound.**" The pooled bound *is* `VACANCY_Q8_UNCERTAINTY_NATS`. **high.**
  - The regression guard itself admits the forbidden phrase:
    `node -e 'console.log(/(?<!re-)measured/.test("quantization, retained bound — re-measured on the shipped swap"))'`
    → `false`, i.e. `tests/unit/archVacancyPanel.test.ts:186` passes on the exact claim
    §8.3a bans. **high.**
  - `VacancyScorePanel.svelte:103-105` renders the ±0.2 standing paragraph on an **identity**,
    where no ± is printed at all (FR-720b). **medium.**
  - Two ± terms are printed side by side joined by `·` with no stated combination rule. **low.**
  Verified correct: the backend never attaches a quantization term (24 real gpt2 differences
  at seeds 0/7, p ∈ {0,1}, three passage kinds: `quantizationUncertaintyNats: null`
  throughout), and `errorBarTerms` drops both ± on an identity.

* **Item 4 — the joiner class. REFUTED.** The class is not closed:
  - **`\p{Pc}` (connector punctuation) is missing from both stacks** —
    `vacancy_score.py:151` `WORD_JOINER_CATEGORIES = ("Pd", "Cf")`, `byteSpans.ts:157`.
    `don‿t` (U+203F) scores **HTTP 200** and swaps to `warm‿t`; same for `_`, U+2040,
    U+2054, U+FF3F, U+FE33/34/4D/4E/4F. Character-for-character the reported `don’t` →
    `big’t` defect. **high.**
  - **Unicode-version skew between the stacks: 448 characters** the TS mirror refuses and
    the Python backend accepts (Python 3.10's `unicodedata` is Unicode 13.0; Node ICU is
    15.1). Includes **U+0890/U+0891, which really are `Cf`** — the backend's own declared
    joiner category — and which score 200 (`swap='… warm࢐t brown …'`). **high.**
  - Named-list escapes: U+05F3 Hebrew geresh, U+30FB / U+FF65 katakana middle dot,
    U+1FBF / U+0375 / U+00B4 / U+1FBD / U+2035 / U+A67E. **medium.**
  - A **false refusal**: ASCII `--`, the Gutenberg em-dash convention present in this
    project's own corpus (`['ba--are', 'hea--art', 'Lady--loves', 'legs--upon']`), is
    refused because `J+` accepts a *run* of joiners where `WORD_RE` allows one. The six
    shipped 250-word passages still score; a user-pasted passage or a different
    `count`/`words` split does not. **medium.**

* **Item 5 — `Object.hasOwn`. PARTIAL.** All four claimed fixes **verified by execution**
  (`provenanceFromMetrics` → `unrecorded` for all 8 keys; `decodeScalars` → typed
  `ComputeError` and no `Float32Array`; static `mint` → typed `InvalidParamError` matching
  the backend; bundle weight names reported as extra). Their *"safe"* judgements are
  individually right, but the sweep covered only the `in` operator and therefore missed:
  - `src/lib/lexEngine/vacancy.ts:386-389` — `METER_FEET[foot]` guarded by
    `=== undefined`; `METER_FEET["constructor"]` is `Object`, so `meterScore(...)` returns
    **`0`** instead of throwing (`"bogus"` throws correctly). Latent — `foot` is not a wire
    parameter today — but it is the wrong-number-not-throw form. **medium.**
  - `src/lib/staticClient/safetensors.ts:305` + `staticClient/arch.ts:751-752` — a remote
    header can set the prototype of `header.tensors` via `__proto__`, and
    `resolveTensorName` reads it with a truthiness test. Refused downstream by the new
    `Object.hasOwn` in `readWindow`, so no wrong number reaches the screen. **low.**

### F5. Item 6 — the fix was applied to one route file, not to the class
**Severity:** high
Round 3's `_as_int` rewrite landed in `api/routes_lex.py` only. Verified against a running
backend on :8010:

* **`api/routes_geo.py:248-252` still calls `int(value)`.**
  `POST /api/geo/finetune {"text":"the cat sat on the mat","steps":Infinity}` →
  `(500, '{"error":{"type":"InternalError","message":"cannot convert float infinity to integer","detail":{}}}')`
  — *the exact OverflowError→500 the fix report says it removed*. `steps: 1.5`, `"7"`,
  `true` and `"٧"` (Arabic-Indic seven) all coerce silently. **high.**
* **`api/routes_lex.py:334-338` `_as_float` still accepts `Infinity`/`NaN`,** and the typed
  400 it *should* raise then dies serializing the offending value into `detail`:
  `POST /api/lex/vacancy {"p":Infinity}` →
  `(500, '{"error":{"type":"InternalError","message":"Out of range float values are not JSON compliant"}}')`.
  **high.**
* **Claim (c) is false as stated:** `/api/lex/train` and `/api/lex/vacancy` do enforce
  `MAX_SEED`; **`POST /api/geo/train` does not** — `{"seed":9007199254740993}` → `202`, as do
  `2**63` and `10**40`, echoing back an integer JavaScript cannot read.
  `routes_geo.py:111`, `routes_arch.py:59,78`. **medium.**
* Non-finite `lr` / `weight_decay` are accepted at the wire on `POST /api/lex/train` (202)
  and fail only mid-job (`"training diverged at step 2: the loss is nan"`), where the static
  client throws at the boundary. **medium.**
* **The conceded handoff (d) is understated.** `staticClient/lex.ts:783-812`'s
  `asInt`/`asFloat` (12 call sites) coerce far more than a `null` seed. Against the backend's
  typed 400s: `"7"`→7, `true`→1, `false`→0, `null`→0, `[]`→0, `[7]`→7, `""`→0, `" 7 "`→7, and
  **`"0x10"`→16**. **high.** This is the public site's path.
* No TypeScript test pins the `>=` unk bound (`tests/unit/geoDerivedVocab.test.ts:286` only
  asserts the constant is `0.9`); reverting either TS `>=` to `>` breaks no test. **low.**

**What is genuinely fixed here:** `>=` is present in all four places
(`geo/finetune.py:201`, `geo/jobs.py:131`, `geoEngine/index.ts:525`,
`staticClient/geo.ts:271`) and the Python at-bound refusal / just-below acceptance execute
correctly. The cross-stack floating-point divergence I hypothesised **does not exist** and
was refuted: for (9,10), (90,100), (900,1000), (2700,3000), (63,70), (27,30), (9000,10000)
both languages give exactly `0.90000000000000002220` and compare equal; a near-miss would
need `n_tokens > ~1.8e15`. `routes_lex.py::_as_int` itself answered all 24 hostile values
with typed 400s, including `"٧"`, `"７"`, `2**53` and long integer strings.

---

## (b) NEW defects the round-3 fixes introduced

### F6. The two stacks compute **different** `weights_token` and `vocab_sha256` for the identical word list when a word contains U+007F
**Severity:** medium
**Where:** `code/backend/src/llm_geometry/geo/tokenizer.py:171-180` (`ensure_ascii=True`
escapes everything outside `\x20-\x7e`, DEL included) vs
`code/frontend/src/lib/geoEngine/tokenizer.ts:78-81` — `code < 0x80 ? raw[i] : "\\u"…`,
which leaves U+007F **raw**. The comment above it claims "every non-ASCII character escaped
(Python's `ensure_ascii=True`)"; Python's rule is not "non-ASCII".
**Reproduce:** build a 1000-word list whose last word is `"\x7f"` (the geo token regex
`[^\sa-z0-9]` admits any single non-space symbol, so it really can reach the word list), then:
```
PY vocab sha256      = 927b4da6af32a945bca6d437ca5194d6157fc4d3e3a323f353cae3d4a1bff4a5
TS vocab sha256      = e2ebf0878067f983f5c18868f3a8f9f85963588be6c8cc930450f12bd71d64f7
PY weights_token     = eaa10f1e4cfec1f8c44ba92fbd0b001d
TS weights_token     = adfa9a05351b1c516b1eb423c2f192e7
```
**Expected:** byte-identical, which is the whole point of `canonicalVocabJson`'s docstring.
Round 3 made the vocabulary an input to the *identity*, so a serialization divergence that
used to affect only `vocab_sha256` now also splits the model id: a file saved by the full
stack is refused by the static build as "this model file is corrupt", and vice versa.
**Would it have thrown?** Yes — but with a message that accuses a good file of corruption.

### F7. `GeoEngine.importWeightSet` is the one import path with **no `validateWeightSet`**, and the TS token hashes the *declared* shape rather than the array
**Severity:** medium
**Where:** `src/lib/geoEngine/index.ts:402-444` (no `validateWeightSet(ws)`, unlike
`importBundle` at `:726` and unlike the backend's `weights.validate_weight_set`), plus
`src/lib/geoEngine/weights.ts:340` — `const shape = WEIGHT_SHAPES.get(name)`, where the
backend hashes `repr(arr.shape)`.
**Reproduce (`vite-node`):** persist a set whose `embedding` holds 3 floats and whose
`layers.0.W_K` holds 12, with the token the TS rule gives it:
```
  imported a 3-float 'embedding' as a valid set -> true
  trace threw: GeoEngineError Weight 'embedding' has 3 values, expected shape (1003, 3)
  A: embedding len 3006  W_K len 12   importWeightSet accepts A -> true
```
**Expected:** the same refusal `importBundle` gives. The engine now holds a valid-looking
token for a model that cannot run, and `exportBundle` will write a file declaring
`shape: [1003, 3]` over 12 bytes of data. The hash does not cover the fact it claims to.
**Would it have thrown?** On import, no; on first use, yes.

### F8. `importWeightSet` returns `true` without inspecting the payload whenever the token is already known
**Severity:** low
**Where:** `src/lib/geoEngine/index.ts:403` — `if (this.weightSets.has(token)) return true;`
**Observed:** a `:v2` payload keyed by the canonical checkpoint id, carrying
`ownsVocab: true` and a *reversed* word list, returns `true`:
`importWeightSet(preFixToken, {ownsVocab:true, vocabWords:[...]}) -> true`. The engine is
unaffected (the payload is discarded), but `restorePersistedSets` therefore never deletes
the entry, so a contradictory payload survives every reload.
**Would it have thrown?** No.

---

## (c) Claims I could not verify

1. **The q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS`.** Unchanged from the three previous
   agents' position: it needs a real browser running this build's static scorer, and
   `npm run test:e2e` is out of scope. No q8 number was computed, guessed or extrapolated
   anywhere in this report.
2. **Anything rendered end to end.** No e2e, no browser, no screenshot. The static arm of
   every finding above is verified by running the real functions under `vite-node`/`vitest`,
   not by looking at a page.
3. **Whether the `<pre>`-level UI actually shows F1's wrong words to a user in the full
   stack.** I proved the API returns them and traced the caller
   (`GeometryLab.svelte:161`, `:224`), but I did not observe the DOM.
4. **A true collision in the TS `weightsToken`.** I showed the hash does not cover the real
   per-tensor lengths (F7) but did **not** construct two distinct weight sets with the same
   token; the fixed name+shape literals between tensors appear to prevent the obvious
   construction. I do not know whether one exists.
5. **Models other than gpt2** for the arch findings, and **Linux / Python 3.10 in CI** for
   everything (all runs were local macOS). Note that F4's 448-character skew is a *property*
   of CI's Python 3.10 and would be worse there, not better.

---

## (d) What I confirmed genuinely fixed

* **Backward compatibility of `checkpoint_id`: real.** `weights_token(ws)` and
  `weights_token(ws, None)` both give `be5359a1c66bda29c8c554269e589009`, which is exactly
  the value committed in `code/frontend/public/static-data/geo/checkpoint.json`
  (`metrics.checkpoint_id`), and the browser engine's `canonicalToken` computes the same
  string. `checkpoint.json` was not touched by any round-3 commit. **No pre-existing
  `checkpoint_id` moved.**
* **Two vocabularies are no longer one model.** Identical weights + the shipped word order
  vs. reversed: `be5359a1c66bda29c8c554269e589009` vs `6a000549255f9934f8a3fb7d9cface1d`,
  and `tokenizer_for` returns `[',', '"', 'the']` and `['nibbling', "needn't", 'muttering']`
  respectively. The dedup can no longer hand one model the other's words.
* **The claimed side effect is real (item 2, first half).** A file with genuine weights, a
  substituted word list and a **recomputed `vocab_sha256`** is now refused:
  `InvalidParamError: this model file is corrupt: its weights and vocabulary hash to
  be5359a1… but it declares 6a000549…`.
* **I could not produce a fourth path to a saved file whose weights and word list
  disagree while every digest verifies.** I tried: identical weights with different
  vocabularies; import → derive → re-import → re-derive; the `owns_vocab=True` /
  `vocab_json=None` self-contradictory write; store dedup in both orders; sessionStorage
  `:v2` payloads in all six claim/payload combinations; transposed, extra and missing
  tensors; and eleven malformed vocabulary shapes. Every one is refused or produces a
  correct token. Four full **import → derive → re-import → re-derive** round trips are stable
  and keep the model's own words:
  ```
  scratch       6a000549255f9934f8a3fb7d9cface1d ['nibbling', "needn't"]
  round 0: import=6a000549255f derive=34c349dcff6d reimport=34c349dcff6d same=True words=['nibbling', "needn't"]
  round 3: import=1d42ea547890 derive=ede61681f16b reimport=ede61681f16b same=True words=['nibbling', "needn't"]
  ```
  and 8 concurrent `save_weight_set` calls on the same content converge on one token
  (`concurrent 8x same content -> {'6a000549255f9934f8a3fb7d9cface1d'}`). The weights-payload
  validation in particular is solid:
  ```
    embedding shape transposed to [3,1003]: TYPED InvalidParamError: model file: weight 'embedding' has shape (3, 1003), expected (1003, 3)
    an EXTRA tensor named zzz_extra:        TYPED InvalidParamError: model file is incomplete (… unexpected: ['zzz_extra'])
    a MISSING tensor:                       TYPED InvalidParamError: model file is incomplete (missing: ['layers.0.W_V'] …)
  ```
* **Item 2, second half — what is still accepted between "one digest" and "all of them".**
  The conceded hole (an author who recomputes *both* digests) is the only way to get a
  *self-describing* bad file. But `vocab_sha256` covers the file's **raw** bytes while the
  identity covers the **canonical re-serialization**, so the two need not agree about what
  the file says:
  - a file declaring `"specials":{"<unk>":5}` loads with `<unk>=0` — Python
    `HTTP 200 {"weights_token":"be5359a1…","vocab_size":1003}` — because
    `GeoTokenizer.from_json` ignores `specials` entirely, while the browser engine
    **refuses** it (`vocab.json: special <unk> has id 5, expected 0`);
  - the `tokens`-shaped vocabulary is **accepted by the browser engine** (1000 words) and
    is an untyped **HTTP 500** in Python (F3).
  So the remaining gap is not one hole but three: the conceded one, plus two shapes on
  which the two stacks disagree about whether a file is valid at all.
* **SC-703 and the TS↔Python differential suite are undisturbed.**
  `npx vitest run tests/unit/vacancy.test.ts tests/unit/vacancyGolden.test.ts
  tests/unit/geoDerivedVocab.test.ts tests/unit/geoEngine.test.ts` →
  `Test Files 4 passed (4) · Tests 197 passed | 1 skipped (198)`, and the §10 line prints
  the same numbers the fix report recorded: `stemsTotal=1676`, `tokensVacated(p=1)=8125`,
  `domainTypes 2233/1940`, `bijective=true`, `imageSize=2233` at seeds 0 and 7.

---

## (e) Teeth — 7 fresh mutations against round 3's own new tests, 5 survived

Each mutation is one a careless refactor could plausibly produce, and is *different* from the
one the fix report names. Every mutated file was backed up with `cp`, restored with `cp`, and
re-verified by `shasum` (never `git checkout` — a sibling agent shares this tree).

**Caught cleanly (2):**
* `stores.ts` `beforeunload` firing unconditionally → `an idle app must never interrupt a
  reload: expected true to be false`, `Tests 1 failed | 14 passed (15)`.
* `geoEngine/index.ts:429` agreement check weakened to `if (ownsVocab && words === undefined)`
  → `× refuses a payload whose claim and payload disagree, in either direction`.

**Survived (5):**

### F9. `QUANTIZATION_TERM` can be rewritten into the exact claim §8.3a forbids, and all 596 unit tests pass
**Severity:** high · **Where:** `src/viz/arch/vacancyVerdict.ts:118`, guard at
`tests/unit/archVacancyPanel.test.ts:176`
```
-export const QUANTIZATION_TERM = "quantization, retained bound — not re-measured since the swap rewrite";
+export const QUANTIZATION_TERM = "quantization, retained bound — re-measured on the shipped swap";
```
`npx vitest run tests/unit/archVacancyPanel.test.ts` → `Tests  16 passed (16)`; full suite
`Test Files 29 passed (29) · Tests 596 passed | 1 skipped (597)`. The assertion tests two
*tokens* (`/(?<!re-)measured/`, `/retained/`), not the claim. Only `tests/e2e/static.spec.ts:401`
pins the real string, and e2e is not the unit gate. **This is the same finding two independent
agents reached from opposite directions** (see also F4, item 3).
**Would it have thrown?** no

### F10. The nav-guard panel tests assert on source *text*, so the wiring can be de-reactified undetected
**Severity:** high · **Where:** `tests/unit/navGuard.test.ts:294-310` (a `readFileSync` +
regex); code at `src/viz/lex/VacancyPanel.svelte:523`
```
-  $effect(() => {
+  {
     if (demoBusy) registerWork(WORK_ID, "…"); else releaseWork(WORK_ID);
-  });
+  }
```
→ `Tests  15 passed (15)`, full suite `596 passed | 1 skipped`. The bare block runs once at
init with `demoBusy === false`, so the demo's two real training runs are **never registered** —
exactly the regression round-3 TASK 3 fixed. The test only checks the file *contains* the
statement, which the mutant still does. This is the "asserts on its own fixture" class: the
fixture is the source string.
**Would it have thrown?** no

### F11. `import_bundle`'s key-order-independence is claimed in a comment and tested nowhere
**Severity:** medium · **Where:** `geo/bundle.py:212`
(`owned = own_vocab_json(canonical_vocab)` → `own_vocab_json(vocab_json)`) →
`32 passed, 1 warning in 35.57s`. A behaviour probe on a semantically identical file with
different JSON whitespace/key order (`vocab_sha256` honestly recomputed): mutated →
`REFUSED: InvalidParamError this model file is corrupt: its weights and vocabulary hash to
88f44a3b…`; restored → `LOADED`. The comment at `:191-194` states the property; nothing pins
it, so a valid third-party file would be refused with no test failing.

### F12. The five round-3 vocabulary tests do not cover the shipped-vocab normalization
**Severity:** medium · **Where:** `geo/weights.py:251` — `own_vocab_json` reduced to
`return vocab_json` → `tests/integration/test_geo_derived_vocab.py`: **20 passed**, all five
named tests green, while one model then has two identities (`as checkpoint: 3ae5df00…` vs
`as file with the canonical vocab: a4d2510f…`, `same identity? False`). CI is saved by a
different module (`test_geo_scratch.py::test_model_export_import_over_http` → `1 failed, 531
passed`), not by the module advertised as pinning identity.

### F13. `test_the_singleton_merge_boundary_is_exactly_one_member` is blind to a class-selective over-merge
**Severity:** low · **Where:** `lex/vacancy.py:659` —
`if suffix and len(grouped[suffix]) < 2:` → `… or suffix == "s"` → the boundary test **passes**
(its domain is `bare + ing[:k]`, so an `s`-class rule never fires); two *other* tests fail.
The test advertised as pinning §8.3's exception pins only the `-ing` axis.

**Weakened assertions?** `git log -p 8b3c293^..1b70966 -- tests/**` shows exactly one removed
`expect(` in the round-3 test diffs — the navGuard PANELS loop being parameterized
(`` `busy` `` → `${flag}`), a rewrite, not a weakening. `readmeNumbers.test.ts` is not vacuous:
it derives 1000/1003 from the shipped `public/static-data/geo/vocab.json`.

---

## Repo hygiene

Nothing in the repository was modified by me. All probes
(`attack_geo.py`, `attack2.py`, `ts_probe.ts`, `ts_probe2.ts`) and the two scratch cache
directories lived in the session scratchpad and are deleted. The backend I started on
:8010 was stopped. Ports 8000/5173/4173 were never bound.
