# Feature 007 — UI specification

Normative for the two panels. Read `architecture.md` first; every number named here comes from
`vacancyStats` or a source constant, never from prose.

---

## 1. `VacancyPanel.svelte` (Lexicon Lab)

Lives in `src/viz/lex/`, owned by `LexiconLab.svelte` like every other panel — plain props in,
callback props out, no store. It sits **after `BudgetPanel`/`ModelPanel` and before
`TrainPanel`**, because it changes the corpus the trainer will see.

### 1.1 Controls

| control | type | default | notes |
|-|-|-|-|
| `p` | slider 0→1, step 0.05 | `0` | the headline knob; show the value numerically |
| `seed` | integer input | `0` | changing it re-mints everything, by design |
| condition | radio | `consistent` | `consistent` \| `inconsistent` \| `partial reveal` |
| reveal N | integer, shown only for `partial reveal` | `1` | `revealAfter` |
| prosody | checkbox | on | `matchProsody` |
| mint | radio | `nonce` | `nonce` \| `swap` (§8.3 of the contract) |

Changing any control re-derives the view synchronously. The transform runs in ~ms on 16 000
tokens; do **not** put it behind a spinner or a worker, and do not debounce the slider so hard
that the nesting demonstration stops feeling continuous.

### 1.2 The corpus view — the doc's Figure 5, live

Render the first ~40 token-producing lines of the active corpus with every word carrying one of
three classes, colour-coded to match the source document's figure:

- **closed class, preserved** — full-contrast text
- **open class, not yet vacated** — muted
- **minted** — accent

A legend states exactly that. The classification comes from the real map (`isEligible` +
`u(stem) < p`), never from a hand-annotated list — FR-711.

### 1.3 The nesting ribbon

The single most important thing this panel has to make *visible*, because it is the property
that makes a `p`-sweep interpretable and the source's own implementation gets it wrong.

Pick ~8 eligible stems spanning the `u` range. For each, a row of cells at
`p = 0, 0.25, 0.5, 0.75, 1` showing the surface form at that `p`. The reader must be able to see
at a glance that:

- once a cell turns minted it **never reverts** as `p` grows (nesting), and
- the minted string is **the same string** in every later cell (stability).

Caption states both properties by name. This is FR-711 and it is not satisfied by prose.

### 1.4 Statistics readout

Show, from `vacancyStats` only:

- `corpusTypesVacated` / `corpusTypesEligible` and `tokensVacated` / `tokensTotal`
  — **corpus scope, not domain**: the 22 domain-only Dolch words never appear in the text and
  counting them inflates the rate the reader is being shown (contract §10).
- prosody: `meanSyllablesBefore → After`, `meanAnapestBefore → After`
- **immediately beside them**, the three-way stress split (`stressFromTable` /
  `stressFromMinted` / `stressFromRule`), plus one sentence: the stress table is rule-seeded and
  unverified, it covers ~5 % of this corpus's tokens, so these are indicative and not exact.
  FR-712 / SC-708. No prosody number may appear without it.
- `bijective`, `remintRounds`

### 1.5 The invariance demonstration — FR-714

Two tiers, because the theorem is free to check and the training run is not.

**Instant (always shown).** Compute `tokenStream` under the mapped vocabulary at the current `p`
and at `p = 0`, compare element-for-element, and display the verdict with the count actually
compared — e.g. *"token id streams identical · 19 071 ids compared"*. Recompute on every control
change. In a condition that breaks the theorem (`inconsistent`, `revealAfter > 0`) this must
show the **real** result — the streams differ, with how many positions and the resulting
`<unk>` rate. It is a live check, never a hard-coded ✓.

**On demand (button).** Train at `p = 0` and at the current `p` with the same seed and
hyperparameters, then show both loss curves and `max |Δloss|`. Under `consistent` this is
exactly `0`. Report it as `0`, not "≈0" — and if it is ever not 0, that is a bug and the UI
should say so rather than round it away.

Default the demo to a step count that finishes quickly; the point is the comparison, not the
final loss. Two runs, not three.

### 1.6 Framing — do not let the null read as a bug

The headline result is an exact zero, and a panel that presents it as a flat line looks broken.
Contract §7.4: state plainly that for a word-level model trained from scratch the transform is a
pure relabelling and the model is provably blind to it, that this is *the finding* — all of a
word's meaning is field, none is form — and that the number which is **not** zero lives in the
Architecture Explorer, with a link to it.

---

## 2. `VacancyScorePanel.svelte` (Architecture Explorer)

### 2.1 Controls

Model (from the curated list), passage (default a fixed excerpt of the shipped corpus, editable),
`p`, `seed`, and mint strategy. A single **Score** button — this runs real forward passes and
must not fire on every keystroke.

### 2.2 Output

A three-row table, English / swap / nonce, each with `nllPreserved`, `nllAll`, `bitsPerChar`,
`nTokens`, `nPreservedTokens`; then the two differences that matter, labelled in words:

- `nll(swap) − nll(english)` — **the cost of wrong content**
- `nll(nonce) − nll(swap)` — **the cost of unknown form**

Never display `nll(nonce) − nll(english)` as a headline; it conflates the two (contract §8.3).

Beside it, the tiny arm's exact `0`, labelled as the same measurement on a model with no
locations. That juxtaposition is the whole 2×2 and is the reason this panel exists.

### 2.3 Honesty requirements

- State that the residual — nonce forms fragmenting into more subword tokens — is not separable
  without a tokenizer-level control, so "cost of unknown form" is an upper bound on what
  location was worth.
- State the alignment mechanism (byte-level pieces → UTF-8 byte spans) and that it is verified
  by reconstruction at run time; a mismatch raises rather than mis-attributing (FR-718).
- **Quantization**: the static build runs quantized ONNX whose absolute per-token logprobs
  differ from fp32 by up to several nats. Whatever the measurement of that error's effect on the
  *difference* concludes, the panel states the stated uncertainty in static mode, or refuses and
  names the full stack. Not decided here — it is decided by measurement.

---

## 3. Info tab

A new `<h3 id="vacancy">` after `#lex`, covering: the 2×2 with the vacancy cell marked; the
transform's definition; nesting and stability, and that the source's implementation breaks both;
the invariance theorem, what it proves and what it does not; the swap control and the
decomposition; and the stress table's real status. Update `#real`, `#limits` and `#refs`.

Every number transcribed from a source constant, pinned by `tests/e2e/docs.spec.ts` — feature
005's rule, so changing a constant without changing the sentence fails CI.
