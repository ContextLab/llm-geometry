# Source-paper claim audit — "Why a Tiny Closed-Lexicon Model Is the Right Instrument"

Sources read in full:
- `/Users/jmanning/Desktop/tiny-models/tiny_models.tex` (1067 lines; D. Rockmore, "for J. Manning and M.J. Latifi Jebelli")
- `/Users/jmanning/Desktop/tiny-models/AUDIT.md` (Independent audit, 2026-08-03)
- `/Users/jmanning/Desktop/tiny-models/README.md` and `/Users/jmanning/Desktop/tiny-models/docs/LLM_GEOMETRY_INTEGRATION.md` (read because they bear directly on the UI question)

Every quote below is verbatim from those files with line numbers. **The single most
important fact for the UI: the paper is a proposal. It contains no results.**

---

## 0. The paper's own status disclaimer (put this, or its substance, in the UI)

The paper opens with a boxed disclaimer (tex:30-38):

> **Audit status (3 August 2026): proposal, not results.**
> The supplied bundle contains no generated base corpus, trained checkpoints, or
> model-run logs. Several figures are schematic predictions. Independent review
> also found that the current prosody table is heuristic/incomplete and that key
> tests (class-split vacancy loss, minting trajectories, author offsets, and
> parameter-matched controls) remain to be implemented. See `AUDIT.md`
> before citing any numerical value or "verified" claim in this draft.

And (tex:61-62):

> Everything in Parts I--III below is run on resources that do not yet exist,
> using a small set of instruments that do.

And (tex:70-71):

> Before the design rationale, an inventory of what actually gets made. None of
> these objects exists today

The audit's verdict (AUDIT.md:5-9):

> The core idea is interesting and falsifiable, but the supplied folder is a
> proposal plus partial instrumentation—not a result package. The standard
> library utilities compile and several deterministic properties hold. The
> training, rank, vacancy-loss, author-offset, and masked-decoding claims have not
> been demonstrated.

The repo README is even blunter (README.md:3-7):

> This folder contains an **experimental proposal**, not completed scientific
> results. The supplied paper's rank, learning-curve, voice, and comprehension
> figures are hypotheses/schematics. No trained checkpoints or generated Dolch
> corpus accompanied the original bundle.

---

## 1. Central thesis and research questions (in the paper's own words)

**Thesis.** From the abstract-like frontmatter (tex:45-53):

> Four notions in our discussions --- lexical dark matter, super-words, minting,
> and the tiny/small ladder --- are currently unmeasurable, and all for the same
> reason: in a full-scale model you cannot enumerate what a language fails to
> lexicalize. A closed word budget removes that obstruction and turns each of the
> four into a computable number.

Restated flatly in §"The claim, stated flatly" (tex:562-565):

> Each is unmeasurable in a full-scale model *for the same reason*: you
> cannot enumerate what a language fails to lexicalize, so you have no ground
> truth for the gaps, no denominator for the compression ceiling, and no way to
> see a single new token against a background of $50{,}000$ others.

(tex:568-582):

> **A closed lexicon removes exactly this obstruction.** Fix a
> budget $\V$ of $315$ words. Then:
> 1. The dark matter is *explicitly known*: it is $C_{\mathrm{Eng}} \setminus \V$. […]
> 2. The compression ceiling has a denominator. Bits per concept under $\V$ is
> a finite computable number, and we can move $\V$ by one word and remeasure.
> 3. A minted token is $1/315$ of the vocabulary --- a macroscopic
> perturbation, not a rounding error.
> 4. Nested budgets $\V_0 \subset \V_1 \subset \cdots$ make the ladder an
> experimental object rather than a metaphor.

(tex:584-586):

> The tiny model is therefore not a Seuss pastiche. It is the unique regime in
> which these four notions have ground truth. That is the entire argument for
> building it.

**The four notions being operationalized** (tex:551-559, verbatim):

> - **Lexical dark matter**: the content a language does not lexicalize; the gaps in the chart $\alpha_L : C_L \to \mathbb{R}^d$.
> - **Super-words**: tokens that are fossil records of nested abstraction; in Chaitin's form, the compression ceiling moves only when a new token is added.
> - **Minting**: a token coming into being --- our case had a *location* (hub centroid) and no *field*.
> - **The ladder**: Tarski's metalanguage regress with no top; each rung describes the rung below but not itself.

**The six research questions** are the six tests T1–T6, each phrased as its own
section title (verbatim):

| # | Section title (verbatim) | tex line |
|-|-|-|
| T1 | "Circumlocution cost scales with dark-matter depth" | 679 |
| T2 | "The coverage frontier identifies candidate super-words" | 736 |
| T3 | "The staircase: does the ceiling move, and by how much?" | 765 |
| T4 | "Location without field, field without location" | 822 |
| T5 | "The undefinable kernel is a computable invariant" | 894 |
| T6 | "Trained-tiny is not the same object as masked-small" | 962 |

The paper describes these as (tex:52-53): "Sections~\ref{sec:claim}--\ref{sec:ladder}
state six explicit tests, with predictions and falsifiers."

---

## 2. Every empirical claim, classified

Legend: **(a) demonstrated** = audit confirms; **(b) hypothesis/schematic** = the
paper itself labels it a prediction, or it concerns objects that do not exist;
**(c) contradicted** = the audit disputes it.

### 2A. Claims the paper puts in its own "Checked / Result" table (tex:486-509)

| # | Claim (verbatim from paper) | Status | Evidence |
|-|-|-|-|
| C1 | "Budget sizes, duplicate-free \| $40/92/133/220/315$ as specified" | **(a) demonstrated** | AUDIT:13-14 "Declared budget cardinalities are duplicate-free: 40, 92, 133, 220, 315 (and a separate 50-word mask)." |
| C2 | "Out-of-budget detection \| catches injected violations" | **(a) demonstrated, but narrowed** | AUDIT:15 "Basic out-of-budget detection works **for ordinary single-word entries**." (emphasis added — the audit scopes it) |
| C3 | "Rhyme classifier \| 11/11 on mixed rhyme/near-miss pairs" | **(b) not confirmed by audit; adjacent claim contradicted** | The audit's "What held up" list does not include the rhyme classifier. AUDIT:30-31: "The original generator enforced vocabulary and a heuristic meter threshold, **but not rhyme**." So even if the 11-pair test passes, rhyme was not enforced in generation. |
| C4 | "Vacancy nesting and stability \| verified across $p \in \{0,.35,.70,1\}$" | **(a) demonstrated** | AUDIT:16-17 "Vacancy *type selection* is deterministic and nested in `p`; consistent mode assigns a stable nonce per source stem." |
| C5 | "Prosody preserved under vacancy \| anapest $0.351 \to 0.345$; syllables $1.224 \to 1.211$" | **(b) not confirmed** | Audit does not list these numbers. It reports a different fixture: "anapest score 0.34639" for the shipped demo fingerprint (AUDIT:19). Do not present 0.351→0.345 as verified. |
| C6 | "Lattice mask \| rejects *hamper*, *greengrocer*, *grease*" | **(c) contradicted at the level that matters** | AUDIT:44-47: "The subword trie idea is sound, but the decoder can concatenate unspaced allowed starts after a completed word, and it does not validate prompt state. It therefore **does not yet guarantee that decoded surface words remain in budget**." The unit test may pass; the R1 guarantee does not hold. |
| C7 | "Null baseline (uniform under budget) \| TTR $0.098$, Zipf slope $-0.16$, rhyme rate $0.01$" | **(a) partially — and note a numeric discrepancy** | AUDIT:18-19 confirms a reproducible fingerprint: "400 lines, 3214 tokens, TTR 0.12134, Zipf slope -0.16719, anapest score 0.34639, rhyme rate 0.01." These may be different objects (paper's "null baseline" vs audit's "shipped demo fingerprint"), but **TTR 0.098 ≠ 0.12134**. Do not display either number without saying which artifact it came from. |

### 2B. Claims the paper itself marks "Not yet exercised" (tex:502-507) — all **(b)**

> - "Trainer and probes \| compile only; not run (no GPU to hand)"
> - "Stress table \| seeded by rule; wants roughly an hour of human checking"
> - "Corpus synthesis \| pipeline built; no corpus generated yet"

Audit agrees and extends (AUDIT:23-25): "No generated base corpus, checkpoint, training
log, source-to-nonce map, run manifest, or test suite was supplied."

### 2C. Claims in the prose that the audit specifically contradicts

| # | Paper's claim (verbatim) | Status |
|-|-|-|
| C8 | "**E. The exact prosodic dictionary.** Hand-verified stress patterns and rhyme classes for the Dolch 315." (tex:102-104) | **(c) contradicted.** AUDIT:34-37: "Prosody is not exact. Most of the 315 words use a spelling heuristic, some table entries are incorrect, and nonce stress metadata was not loaded across processes. Documentation must call this **estimated**." The paper contradicts itself too: its own table says the stress table is "seeded by rule" (tex:505). **Never write "exact" or "hand-verified" prosody in the UI. Write "estimated."** |
| C9 | "**R4. Exact prosody.** […] Prosody must therefore be known rather than estimated." (tex:207-213) and "no open-vocabulary verse corpus can claim exact meter" (tex:356-357) | **(c) contradicted** by AUDIT:34-37 as above. The *aspiration* is legitimate; the *achievement* is not. |
| C10 | "**yield is not a nuisance parameter --- it is the primary observable of T2**" (tex:159-160); "The fraction of attempts that survive at budget size $|\V|$ estimates how hard it is to say anything at all under that budget" (tex:337-339) | **(c) contradicted as a measure.** AUDIT:47-49: "Yield is a property of the provider, prompt, lexical compliance, meter/rhyme gates, and deduplication—**not a clean universal measure of linguistic expressibility**." |
| C11 | "three control conditions (inconsistent assignment, no prosody matching, partial reveal)" as valid controls (tex:86-88) | **(c) contradicted for one control.** AUDIT:41-43: "The inconsistent vacancy control changes the number of nonce types and thus vocabulary/model size; it **does not isolate identity by itself**." |
| C12 | "The instrument side is done and verified as far as it can be without a GPU" (tex:1024) | **(c) overstated.** AUDIT:38-40 lists unimplemented pieces: "B1/B2/A4/T6 are not implemented: there is no function-vs-content target loss, contextual nonce span trajectory, saved initialization displacement, author-offset estimator, or per-word dispersion analysis." Note per-word dispersion is the *entire observable of T6*. |
| C13 | Rejection sampling machinery described as built and characterized (tex:330-340) | **(c) qualified.** AUDIT:29-33: "It was unseeded, used process-randomized `hash`, and could retry API failures forever. These mechanics are now corrected, but **no corpus has yet been regenerated with the corrected pipeline**." |

### 2D. All six test predictions — every one is **(b) hypothesis**

Each is inside a `\begin{prediction}` environment and each has a stated falsifier.
Verbatim:

- **T1** (tex:693-695): "$L(c)$ increases monotonically in $\dm(c)$, with $R(c)$ falling off past a threshold depth. The relation is closer to affine in $\dm$ than in $\log(1/\text{freq})$ --- i.e. geometry, not frequency, sets the cost."
- **T2** (tex:752-756): "$\rho$ has a knee, and marginal gain is extremely heavy-tailed: a small number of words (**my guess**: relational and functional --- *like*, *not*, *same*, *make*, *part*) carry disproportionate coverage."
- **T3** (tex:779-783): "The decomposition separates cleanly into two populations. Concrete names are additive (all gain on their own referent). Relational and abstract tokens are superadditive […]"
- **T4** (tex:856-861): "Strong asymmetry, with (iii) $\gg$ (ii): field-without-location converges much faster than location-without-field. If so, an embedding is largely a *summary* of contextual support rather than an independent carrier of content, and ''location'' in our minting memo is doing less work than we assumed."
- **T5** (tex:906-909): "$\B(\V)$ is nonempty for every $\V$ --- the regress has no top […] Its size grows sublinearly in $|\V|$, and its membership converges toward a small, stable set dominated by deixis, quantity, negation, and primitive predicates."
- **T6** (tex:983-988): "Masked-small shows markedly higher per-word dispersion at identical vocabulary size […] Consequently masked-small navigates dark matter far more efficiently (lower $L(c)$ at equal $|\V|$) despite identical lexical resources. Lexical poverty and conceptual poverty are separable, and the separation is measurable."

**These are bets, not findings. The UI must label them as predictions with falsifiers.**
The paper's own T4 caption says it best (tex:889): "Its *sign* is the result; **I have drawn my bet, not a finding.**"

### 2E. Resource/plan claims (aspirational, not empirical)

- "Metered, rhymed verse synthesized under exactly $40 / 92 / 133 / 220 / 315$ words […] Target $\sim$2M tokens each. *Nothing like this exists at any size.*" (tex:76-80) — **(b)**; nothing generated (AUDIT:23-25).
- "$\sim$2M-parameter word-level transformers, one per (budget $\times$ condition $\times$ seed), plus $\sim$30 more for the T3 minting staircase." (tex:91-93) — **(b)**; no checkpoints exist.
- Figure "The model zoo": "**72 training runs** $\approx$ single-GPU hours." (tex:140-141) — **(b)**, a plan.
- "**F. The concept battery.** $\sim$200 targets […] This one does not exist even in draft" (tex:108-110) — the paper says it doesn't exist.
- "T5 is the only test with no instrument yet" (tex:541) — self-declared.

### 2F. Internal inconsistency I found that neither document flags

The nested-budget figure caption (tex:611-612) reads:

> Nested budgets $\V_0 \subset \V_1 \subset \V_2 \subset \V_3$ ($40 / 92 / 133 / 315$ words).

but the budget ladder everywhere else is **five** budgets $\V_0 \dots \V_4$ =
$40/92/133/220/315$ (tex:77-78, 117-118, 241-242, 281). The figure silently drops
**220**. If the UI shows a budget selector, use the five-value ladder
40/92/133/220/315 and do not reproduce the figure's four-value labeling.

---

## 3. The "rank staircase"

### What the paper actually claims

**The phrase "rank staircase" does not appear in the paper.** I grepped: the paper
uses "staircase" only for the *compression* staircase (T3), never for rank:

- tex:93 "plus $\sim$30 more for the T3 minting staircase"
- tex:142 "Green: T3 staircase"
- tex:530 "T3 compression staircase"
- tex:765 "T3. The staircase: does the ceiling move, and by how much?"
- tex:1041 "The compression staircase is an information-theoretic statement with a clean finite denominator"

T3's staircase is over **$\K$, bits per concept**, not rank. Its observable (tex:773-776):

> $\Delta\K = \K(\V) - \K(\V \cup \{\tau\})$ in bits per concept, decomposed into
> the part attributable to $c_\tau$ and the part attributable to everything else.

The paper's *rank/spectrum* material is separate and lives in three places:

1. **R6, Full observability** (tex:222-227): "At $|\V| \approx 320$ the embedding and readout matrices are $320 \times d$, so their entire spectra can be computed at every evaluation step at no cost. This is the reason to work word-level rather than with subword tokenization […]"
2. **Resource C** (tex:93-96): "the *full spectra of $E$ and $U$ are stored at every evaluation step*. The checkpoints are not merely artifacts from which measurements are later extracted; the measurement is in them by construction." And (tex:172-173) "its per-step spectra make it a reasonable public artifact for anyone studying low-rank structure in small language models."
3. **T6's observable** (tex:977-979): "For each budget word: contextual dispersion --- the **effective rank** and spread of its contextual representations across occurrences (the capaciousness measurement, applied per word)."

So "the rank staircase" as a named object comes **from the audit and the integration
doc, not from this paper**. The audit's recommendation (AUDIT:53-55):

> Proceed, but narrow the first paper/demo to the rank staircase with strong nulls
> and fully provenance-tracked artifacts.

and the integration doc (`docs/LLM_GEOMETRY_INTEGRATION.md`:9) titles its MVP
"**MVP: rank staircase explorer**". The audit labels the corresponding experiment
**A1** (AUDIT:29) — a label that appears nowhere in the paper's T1–T6 scheme. If your
UI uses the term "rank staircase," it is adopting the *audit's* framing for a
*proposed* experiment, and there is no paper section you can cite for it.

### What the audit says is mechanically confounded

AUDIT.md:27-29, verbatim and complete:

> 2. Effective rank is bounded by `min(|V|-1, d)`. With fixed `d=128`, apparent
>    saturation as vocabulary grows is expected even for random matrices. A1
>    needs random-init/shuffled-label nulls, several widths, seeds, and uncertainty.

README.md:47-50 restates the bar:

> In particular, effective-rank sweeps must be normalized against
> random matrices and repeated across model width: rank is mechanically bounded
> by `min(|V|-1, d_model)`.

**Read this precisely.** With budgets $|\V| \in \{40, 92, 133, 220, 315\}$ and a fixed
width, rank is capped at $\min(|\V|-1, d)$: 39, 91, 127, 127, 127 for $d=128$. So a
"staircase" that rises with budget and then flattens at $d$ is **exactly what a stack
of random matrices would produce**. Any rise-then-plateau shape is uninformative on
its own. Note also that **`d=128` appears only in the audit and integration doc, not
in the paper**; the paper says only "$320 \times d$" and "$\sim$2M parameters."

### What would be needed to actually demonstrate it

Combining AUDIT:27-29, AUDIT:53-55, README:44-50, and the integration doc's bundle
spec (`LLM_GEOMETRY_INTEGRATION.md`:11-22):

1. Random-initialization nulls and shuffled-label/shuffled-corpus nulls at every budget.
2. Several model widths $d$, so the $\min(|\V|-1,d)$ ceiling can be moved and shown not to be doing the work.
3. Multiple seeds with replicate uncertainty (error bands, not single curves).
4. Trained checkpoints that actually exist — none do.
5. Full provenance per run: "corpus hash, configuration, random seed, environment freeze, source revision, checkpoint/log files, and null baselines" (README:44-46).
6. Exported as immutable static bundles containing "exact singular-value/effective-rank trajectories for embedding and readout" plus "random-init and shuffled-corpus baselines with replicate uncertainty" (integration doc:14-15).

Until all of that exists, **a rank-vs-budget curve in the UI is fabrication.** The
integration doc states the rule outright (`LLM_GEOMETRY_INTEGRATION.md`:27-28):

> **Never make up browser curves when bundles are absent.**

---

## 4. "Vacancy" / "field without location" — the hypothesis

This is **T4**, "Location without field, field without location" (tex:822). The paper's
own framing (tex:824): "This is the one where our own minting experiment sits inside a
$2 \times 2$ rather than alone."

**The 2×2 design** (tex:830-841, verbatim cells):

| | **no field** | **field supplied** |
|-|-|-|
| **no location** | "(i) control: random init, no data" | "(iii) Carroll: nonce form, full syntactic support" |
| **location** | "(ii) our minting: hub centroid, no data" | "(iv) normal word learning" |

The underlying notion from §"The claim" (tex:557-558): "**Minting**: a token coming
into being --- our case had a *location* (hub centroid) and no *field*."

**The hypothesis** (tex:856-861, verbatim):

> Strong asymmetry, with (iii) $\gg$ (ii): field-without-location converges much
> faster than location-without-field. If so, an embedding is largely a
> *summary* of contextual support rather than an independent carrier of
> content, and "location" in our minting memo is doing less work than we
> assumed. Secondarily: $\ell^*(\tau)$ starts late and migrates earlier as the
> field forms, making ignition depth a maturation index for a word.

**Falsifier** (tex:864-866): "Symmetry, or the reverse ordering. The reverse would be
the more interesting result and would mean location is genuinely primitive."

**Observables** (tex:852-855): "Time-to-competence (steps until held-out loss on
$\tau$-containing contexts reaches a threshold); ignition depth $\ell^*(\tau)$ tracked
across checkpoints; KL of $\tau$'s readout from the corpus unigram."

**How the vacancy condition is manufactured** (tex:363-366): "Closed-class words,
inflectional suffixes, syntax, and meter are preserved exactly; open-class stems are
replaced by phonotactically legal nonce forms carrying *the same syllable count and
stress pattern* as the word they replace." Its two design properties are **Nesting**
("A word is vacated iff $u(w) < p$ for a hash-derived $u(w) \in [0,1)$ depending only
on the word and the seed", tex:369-371) and **Stable assignment** ("The nonce for a
word is fixed by the word and seed alone, independent of $p$ and of traversal order",
tex:372-374).

**What you may say in a UI:** the transform itself is real and its determinism is
independently confirmed (AUDIT:16-17). The *asymmetry* is a bet, and the audit puts
T4 in phase two: "Treat the vacancy experiment as phase two after implementing
class-split evaluation and a parameter-matched control" (AUDIT:55-56), because
"there is no function-vs-content target loss, contextual nonce span trajectory,
saved initialization displacement" (AUDIT:38-40) and the inconsistent control is
confounded (AUDIT:41-43).

---

## 5. Figures: which are schematic, which are real

The paper has **nine** figures. Exactly **one** is described as containing real output.

| Fig | Label / location | Kind | Evidence |
|-|-|-|-|
| 1 | `fig:zoo` "The model zoo" (tex:114-149) | **Plan diagram — no data.** Cell entries are counts of runs that have not happened. | "Cell entries are training runs including seed replicates" (tex:147); no checkpoints exist (AUDIT:23-25) |
| 2 | `fig:pipeline` "The pipeline" (tex:264-323) | **Schematic architecture diagram.** | "Ovals are sources and terminals, parallelograms are data, rectangles are processes" (tex:317-318) |
| 3 | `fig:vacancy` (tex:379-424) | **Real output — the only one.** | "**Actual output** of the vacancy transform on one sentence, seed fixed." (tex:415). Caveat: the audit lists "source-to-nonce map" among artifacts **not supplied** (AUDIT:23-24), so the specific strings *swurl*, *wierk*, *sweechenen*, *scaid*, *sauming* were not independently re-derived. The caption's corpus-level numbers ("mean anapest score from $0.351$ to $0.345$", tex:421) are **not** audit-confirmed. |
| 4 | `fig:trie` "The budget as a lattice" (tex:443-481) | **Schematic** illustration of the trie. | Idealized; the audit says the real decoder "does not yet guarantee that decoded surface words remain in budget" (AUDIT:44-47) |
| 5 | Nested budgets ellipses, **no `\label`** (tex:588-617) | **Schematic** conceptual diagram; hand-placed dots. | "Blue points are lexicalized concepts; red points are dark matter" — illustrative only. Also mislabels the ladder (see §2F). |
| 6 | T1 cost-vs-depth plot (tex:706-734) | **Schematic.** | Its own legend entry reads "**schematic data**" (tex:725); curves are closed-form (`4+4.6*x`, `4+18*ln(1+x/2.2)`) |
| 7 | T3 compression-ceiling plot (tex:792-820) | **Predicted.** | "Two hypotheses for how the ceiling responds to minting." (tex:817) — hand-entered coordinates |
| 8 | T4 learning-curve plot (tex:867-890) | **Predicted.** | "T4, **predicted**. […] Its *sign* is the result; **I have drawn my bet, not a finding**." (tex:888-889) — curves are `0.9+4.2*exp(-x/3.4)` etc. |
| 9 | T5 definitional-graph figure (tex:922-960) | **Schematic.** | "T5, **schematic**." (tex:955) |
| 10 | T6 dispersion bar chart (tex:994-1020) | **Predicted.** | "T6, **predicted**. Identical vocabulary, different geometry. **If this gap is real** […]" (tex:1016-1017) — hand-entered bars |

(Counting note: the nested-budget figure has no `\label`, so LaTeX numbering differs
from my row numbers; I've listed them in source order.)

The audit's blanket statement (AUDIT:23-25): "**Paper curves are manually entered
schematics/predictions.**"

**UI rule:** every one of figures 6–10 is a drawn hypothesis. If you reproduce any of
their shapes, they must be visually and textually marked as predictions — ideally
rendered in a distinct "hypothesis" style with the falsifier shown alongside.

---

## 6. Terminology and notation to adopt

### Symbols (LaTeX macros at tex:15-18 plus definitions table tex:623-651)

| Symbol | Meaning (verbatim where quoted) |
|-|-|
| $\mathcal{V}$ (`\V`) | the word budget / closed lexicon |
| $\V_0 \subset \V_1 \subset \V_2 \subset \V_3 \subset \V_4$ | graded budgets at **40 / 92 / 133 / 220 / 315** words |
| $C(\V_k)$ | base corpus synthesized under budget $\V_k$ |
| $C_p$ | the **vacancy family**, at "$p \in \{0, .2, .4, .6, .8, 1\}$" (tex:86-87) |
| $p$ | vacancy rate; "$p$ is the only variable in the design" (tex:377) |
| $\dm(c)$ | **dark-matter depth**: "Distance from $\alpha(c)$ to $\mathrm{span}\,\alpha(\V)$ in a reference model's embedding space"; observable = "Residual norm after projection" |
| $L(c)$ | **circumlocution cost**: "Tokens of budget-only text needed before a held-out judge recovers $c$"; observable = "Token count at first correct recovery" |
| $R(c)$ | recovery rate at a fixed cap (tex:690) |
| $\K(\V)$ (`\K`) | **compression ceiling**: "Cross-entropy of a fixed held-out corpus under a model trained on budget $\V$, in bits per *concept*" |
| $\K(\V; c)$ | description length of concept $c$ under $\V$ |
| $\Delta\K$ | $\K(\V) - \K(\V \cup \{\tau\})$ |
| $\tau$ | a **minted token**; $c_\tau$ is the concept it names |
| $\ell^*(\tau)$ | **ignition depth**: "Least layer at which $\tau$'s tuned-lens probability crosses $\theta$ and stays"; observable = "Layer index" |
| $\B(\V)$ (`\B`) | **undefinable kernel**: "Words of $\V$ not definable in $\V$ without circularity"; observable = "$|\B(\V)|$, and its membership" |
| $\alpha_L : C_L \to \mathbb{R}^d$ | the chart from concepts to embedding space; dark matter = "the gaps in the chart" |
| $\rho(|\V|)$ | coverage curve (tex:748) |
| $G_\V$ | definitional digraph, "with an edge $w \to u$ when $u$ appears in the definition of $w$" (tex:897-899) |
| $E$, $U$ | embedding and readout matrices; "$320 \times d$" (tex:224) |
| $C_{\mathrm{Eng}} \setminus \V$ | the dark matter, explicitly |

### The super-word inequality (tex:657-667) — quote it exactly if you show it

> A minted token $\tau$ naming concept $c_\tau$ is a *super-word* if
> $$\underbrace{\K(\V) - \K(\V \cup \{\tau\})}_{\text{total gain}} \;>\; \underbrace{\K(\V; c_\tau) - \K(\V \cup \{\tau\}; c_\tau)}_{\text{gain on its own referent}}$$
> That is: adding $\tau$ makes it cheaper to say things that are not $\tau$.

Gloss (tex:669-673): "An ordinary new name is additive: it pays for itself and nothing
else. A super-word is superadditive, because the abstraction it packages becomes
available as a component of other descriptions."

### Named terms to use consistently

- **lexical dark matter**, **dark-matter depth**, **super-word**, **minting**, **the ladder** (the four notions)
- **the coverage frontier**, **the compression ceiling**, **the undefinable kernel**, **ignition depth**, **contextual dispersion**
- **trained-tiny** ("the child") vs **masked-small** ("the constrained adult") — from T6 (tex:966-968: "These are the child and the adult under constraint.") and the figure legend (tex:1010, 1013)
- **the vacancy transform** / **jabberwockify** (component name); **field** vs **location**
- **yield** (rejection-sampling survival fraction) — but see C10; do not call it a measure of expressibility
- **the lattice mask** / **trie mask** — "the mask is not a set of permitted token ids but a set of permitted *paths* through the token lattice" (tex:433-435)
- **the concept battery**, **the model zoo**, **the masked-small arm**
- Requirements **R1–R6**; resources **A–F**; tests **T1–T6**
- Component paths: `wordlists/dolch`, `synth/lexicon`, `synth/generate`, `synth/jabberwockify`, `train/tiny_lm`, `eval/fingerprint`, `eval/probe`, `eval/mask_decode` (tex:240-256)
- Provenance: the budget comes from **the Dolch lists (1936; 220 service words + 95 nouns; public domain)** (tex:195-196). **Do not call it a Seuss list.** The paper is explicit (tex:190-193): "The obvious closed budget is the one a publisher handed to Seuss --- but we should not build on his corpus. […] we *re-run the assignment rather than reuse the answer*." And (tex:584): "The tiny model is therefore not a Seuss pastiche."

### Terms to avoid (or restate)

- "exact prosody" / "hand-verified stress" → say **estimated** (AUDIT:34-37 requires this).
- "verified" for anything except budget cardinalities, out-of-budget detection on single words, vacancy nesting/stability, and the demo fingerprint.
- "rank staircase" → not a paper term; it is the audit/integration doc's name for a **proposed** experiment (audit label **A1**), and it is mechanically confounded absent nulls.
- "$d=128$" → appears only in `AUDIT.md`:27 and `LLM_GEOMETRY_INTEGRATION.md`:5, not in the paper.

---

## 7. Bottom line for the UI

**Honest to say:**
- The design, the operational definitions, the six tests with their predictions *and* falsifiers, the super-word inequality, and the four motivating notions — all clearly framed as a **proposal**.
- The budget ladder 40/92/133/220/315 and its Dolch provenance.
- The vacancy transform's behavior, including Figure `fig:vacancy`'s per-sentence output, with nesting and stability.
- The reproducible demo fingerprint (400 lines, 3214 tokens, TTR 0.12134, Zipf slope −0.16719, anapest 0.34639, rhyme rate 0.01) **if attributed to that fixture**.
- The trie/lattice *idea*, flagged that the decoder does not yet guarantee in-budget surface words.

**Fabrication if said:**
- Any trained-model result: rank curves, spectra, learning curves, loss trajectories, dispersion bars, compression ceilings. **No checkpoint exists.**
- Any coverage, circumlocution-cost, or recovery number. **The concept battery does not exist "even in draft."**
- Any corpus statistic for $C(\V_k)$. **No corpus has been generated.**
- "Exact" prosody or meter.
- Yield as a measure of what is sayable.
- Any claim that T1–T6 have outcomes.

**Governing constraint from the integration doc (`LLM_GEOMETRY_INTEGRATION.md`:27-28):**
"Never make up browser curves when bundles are absent." That doc also specifies the
division of labor if this becomes a real explorer: "this repository owns training,
evaluation, provenance, checksums, and `export-web`; llm-geometry owns Svelte/Three.js
presentation and consumes static bundles" (:24-26), and warns that the two models are
**not** checkpoint-compatible (:3-7) and that any token cloud "must be labeled a PCA
projection (the existing llm-geometry sphere is native 3-D and explicitly is not PCA)"
(:21-22).
