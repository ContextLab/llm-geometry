#!/usr/bin/env python
"""Emit the vacancy-transform golden vectors (spec 007, §11 of architecture.md).

Both stacks implement ``specs/007-vacancy-transform-field/architecture.md``:

* Python  — ``code/backend/src/llm_geometry/lex/vacancy.py``
* browser — ``code/frontend/src/lib/lexEngine/vacancy.ts``

This script runs the **real Python side** on the **real committed corpus** (*The Real
Mother Goose*, verified against its recorded digest) and writes what it measured to
``code/frontend/tests/fixtures/vacancy-golden.json``. ``tests/unit/vacancyGolden.test.ts``
then runs the real TypeScript side against that file and asserts agreement. Nothing here
re-derives the maths; every number is read out of the shipped implementation.

Same pattern and the same discipline as ``scripts/export_lex_golden.py`` ->
``tests/fixtures/lex-golden.json`` -> ``tests/unit/lexGolden.test.ts``: the file carries
``format``, ``tolerance``, ``git_sha`` and the generator versions, and the test reads the
tolerance FROM the file so the two cannot drift apart.

Usage (from the backend venv):

    python scripts/export_vacancy_golden.py
    python scripts/export_vacancy_golden.py --out /tmp/vacancy-golden.json

The output is a pure function of the committed corpus plus this file, so regenerating it
twice must produce byte-identical bytes; ``--generated`` is pinned to a date rather than a
timestamp for that reason.

What §11 requires pinned, and where it lives in the document:

1. ``u(stem)`` for 28 stems spanning eligible/ineligible and both budgets, as the EXACT
   float64 — ``stems[].u``. Plain JSON numbers are exact here: ``repr`` and JavaScript's
   number formatting are both shortest-round-trip, so the double survives the trip. That
   exactness is the whole point of departure 2 (``>> 11`` before the divide, §4).
2. The FULL stem -> nonce map at ``seed in {0, 7}`` — ``maps[].mapping``, every pair.
3. The first 400 characters of the vacated corpus at ``p in {0, 0.35, 0.7, 1}``, seed 0 —
   ``cases[].head400``. The corpus is ASCII, so "character" means the same thing in both
   languages. Each case also carries the sha256 of the WHOLE vacated corpus, which pins
   all 86 kB for 64 bytes.
4. ``vacancyStats`` with every §10 field — ``cases[].stats``.
5. Nesting as explicit sets — ``nesting.levels[].stems``, the literal vacated-stem sets at
   ``p in {0, 0.35, 0.7, 1}``, so the test checks containment on data rather than on an
   assertion one language makes about itself.
6. Stability — ``cases[].stemForms``, the surface form of each of the 28 stems at every
   ``p``. A stem's nonce must be byte-identical at every ``p`` where it is vacated.
7. The token-id-stream digest under the mapped vocabulary at each ``p`` —
   ``cases[].idStream``. These are all EQUAL, which is §7.3 (the invariance theorem) pinned
   as DATA rather than as an assertion written twice, once per language.
8. Both control conditions (``consistent = false``, ``revealAfter > 0``) and both
   ``matchProsody`` settings — the ``control-*`` and ``noprosody-*`` cases.
9. The swap control of §8.3 — the ``swap-*`` maps and cases. Four maps (both seeds × both
   ``matchProsody`` settings) pinned by digest and 28 samples, and three cases per seed at
   ``p in {0, 0.7, 1}``. The endpoints carry a real ``idStream``: SC-703 holds for
   ``mint="swap"`` exactly as for ``mint="nonce"`` wherever a swap map can be injective.
   ``swap-*-p0.7`` carries ``idStream: null`` and ``mapVocabWordsRejects: true``, which is
   §5.2a's theorem pinned as data — a map whose images are domain types and which does not
   depend on `p` cannot be injective at intermediate `p` unless it is the identity, so the
   mapped vocabulary does not exist there and BOTH stacks refuse it.

MEASURED, and recorded here because the fixture's shape depends on it: the mapped
vocabulary of §7.2 is defined ONLY for ``consistent = true, revealAfter = 0``; both stacks
raise otherwise. So a control case has ``idStream: null`` and
``mapVocabWordsRejects: true``, and the test asserts the TypeScript side refuses it too —
a silently-accepted control would manufacture a vocabulary matching no corpus.

Also measured: ``consistent = false`` mints a fresh form per OCCURRENCE and registers its
stress pattern on the map's ``minted_stress``, i.e. it MUTATES the map. Every control case
therefore builds its own map, and the test must do the same or its statistics will be
scored against patterns left behind by a previous case.

NO DISAGREEMENTS REMAIN BETWEEN THE STACKS. Every case in this fixture — both seeds, both
``matchProsody`` settings, every `p`, and both control conditions — agrees field for field,
string for string, digest for digest. That was not true when the fixture was first written,
and the history is worth keeping because it is what the mechanism below exists for.

Three defects were pinned here as ``knownDivergence`` blocks and have since been fixed in
the stack the contract said was wrong. Each was fixed in the IMPLEMENTATION; none was fixed
by loosening this file:

1. **``corpusTypesVacated`` under ``revealAfter > 0``** — Python 665, TypeScript 1337.
   Python MEASURED the count from the two texts; TypeScript computed it through the map, so
   a type whose every occurrence fell inside the reveal window was still counted. §10 says
   ``corpusTypes*`` is "what the panel shows a reader", and the reader is looking at the
   text, so the measured reading was the contract's. TypeScript now measures the texts too
   (``domainTypes*`` keeps map membership, which §10 also requires — the 22 Dolch-only words
   have no occurrences to measure).
2. **Prosody of the per-occurrence mint under ``consistent = false``.** §5.8 pins the key as
   ``f"{stem}#{idx}"`` and §5.5's mint read its stress pattern off the string it was handed,
   so Python got ``stress("little#0") == "10"`` instead of ``stress("little") == "100"`` and
   ``Little`` minted as ``Wrerken`` rather than ``Wrerkenle``. §7.1 says the nonce carries
   THE STEM's syllable count and stress, so the key must not reach the prosody lookup.
   Python's ``_mint`` now takes ``stem`` separately from ``key``.
3. **Condition B on the per-occurrence path** (§5.8, added to the contract with this fix).
   It was enforced when building the map and not in the ``consistent = false`` control, so
   at seed 7, `p = 1`, the stem ``tak`` minted the nonce ``tak`` and ``Taking -> Taking``:
   ``corpusTypesVacated`` and ``tokensVacated`` each one short of the consistent path's.
   Both stacks now forbid a per-occurrence nonce from equalling the stem it replaces as well
   as any domain type, and the control vacates 1918 types / 8125 tokens at both seeds,
   exactly as ``consistent = true`` does.

``KNOWN_DIVERGENCES`` is therefore EMPTY, and it is kept — with ``attach_divergence`` and
``DIVERGENCE_STATUS`` — rather than deleted, because it is the honest way to ship a fixture
while a real cross-stack defect is outstanding: record BOTH readings, assert each side
against its own, and refuse to write the fixture at all the moment Python starts agreeing
with the recorded TypeScript value. That guard is what retired all three entries above; it
fired on ``control-inconsistent.vacatedSha256`` and would not let the stale exemption be
regenerated. An entry added here must be MEASURED on both sides, never predicted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "backend" / "src"))

from llm_geometry.errors import ComputeError  # noqa: E402
from llm_geometry.lex.corpus import load_corpus_text  # noqa: E402
from llm_geometry.lex.dolch import dolch_budget  # noqa: E402
from llm_geometry.lex.train import token_stream  # noqa: E402
from llm_geometry.lex.vocab import LexVocab, tokenize  # noqa: E402
from llm_geometry.lex.vacancy import (  # noqa: E402
    VacancyMap,
    VacancyParams,
    build_vacancy_map,
    is_eligible,
    is_vacatable,
    map_vocab_words,
    stem_and_suffix,
    type_counts,
    vacancy_domain,
    vacancy_stats,
    vacancy_u,
    vacate_text,
)

DEFAULT_OUT = (
    REPO_ROOT / "code" / "frontend" / "tests" / "fixtures" / "vacancy-golden.json"
)

FORMAT = "vacancy-golden-v2"

#: Both stacks compute the vacancy transform in float64 and in the same order, so the only
#: floats that can differ at all are the prosody means (a sum over the same tokens in the
#: same sequence). Everything else in this fixture — `u`, every count, every string, every
#: digest — is compared EXACTLY by the test; this bound applies to the means alone, and it
#: is deliberately far tighter than the 1e-5 of `lex-golden.json`, where torch's float32
#: was on the other side.
TOLERANCE = 1e-12

#: The `p` grid. §11 names {0, 0.35, 0.7, 1}; {0.25, 0.5, 0.75} are added because §10
#: quotes measured `corpusTypesVacated` / `domainTypesVacated` at exactly those values, and
#: a number quoted in a contract that no test reads is a number free to rot.
P_GRID: tuple[float, ...] = (0.0, 0.25, 0.35, 0.5, 0.7, 0.75, 1.0)

#: §11's "28 stems spanning eligible/ineligible and both budgets".
#:
#:  * eligible, in the Dolch list AND in the corpus  — the ordinary case
#:  * eligible, Dolch-only                           — one of the 22 domain-only words that
#:                                                     have images but never appear in the
#:                                                     text (§10)
#:  * eligible, corpus-only                          — `crown`, `candlestick`, `diddle`, …
#:  * eligible, in the MAP but in neither budget      — `gum` and `hang`, which reach the map
#:                                                     only as the stems of `gums`/`hanged`
#:                                                     and are the pair §5.7's case-commuting
#:                                                     bug (`GUMS` -> `FLESS`) was found on
#:  * ineligible, closed class                       — `is_eligible` test 1
#:  * ineligible, too short                          — test 3, `len(stem) > 2`
#:  * ineligible, non-alphabetic stem                — test 2; `good-bye` matches no suffix,
#:                                                     so its stem keeps the hyphen
#:  * ineligible, CLOSED CLASS SPLIT OPEN            — `after`, `this`, `does`, `always`:
#:                                                     `stem_and_suffix` splits each into a
#:                                                     stem that is not a function word
#:                                                     (`aft`, `thi`, `doe`, `alway`), so the
#:                                                     stem test alone passed them and all
#:                                                     four were vacated. `is_vacatable`
#:                                                     tests the whole word first (§2.2), and
#:                                                     the `vacatable` field below is what
#:                                                     pins that in both stacks.
PINNED_STEMS: tuple[str, ...] = (
    "little",
    "pretty",
    "run",
    "eat",
    "jump",
    "away",
    "squirrel",
    "funny",
    "today",
    "gum",
    "hang",
    "crown",
    "candlestick",
    "crooked",
    "diddle",
    "moon",
    "pussy",
    "goose",
    "the",
    "and",
    "you",
    "not",
    "ox",
    "good-bye",
    "after",
    "this",
    "does",
    "always",
)

#: §11's excerpt length. The corpus is ASCII (asserted below), so Python code points and
#: JavaScript UTF-16 code units count the same characters.
HEAD_CHARS = 400

#: The nesting sets §11 asks for, as explicit data.
NESTING_SEED = 0
NESTING_PS: tuple[float, ...] = (0.0, 0.35, 0.7, 1.0)

#: Measured cross-stack disagreements, recorded instead of reconciled — see the module
#: docstring. **Currently EMPTY: the two stacks agree on every case in this fixture.** The
#: three entries this dict used to carry are listed there, with what each one was and which
#: stack was fixed; none of them was retired by editing this file.
#:
#: The mechanism stays because it is the honest way to ship a fixture over an outstanding
#: defect. To add an entry, every ``typescript`` value must be MEASURED, by running the real
#: browser engine on the real corpus at that case's parameters:
#:
#:     const vmap = buildVacancyMap(vacancyDomain(new Set(tokenize(CORPUS))), params);
#:     const vacated = vacateText(CORPUS, vmap, params);
#:     const stats = vacancyStats(CORPUS, vacated, vmap, params);
#:
#: Never a prediction, and never derived here. ``attach_divergence`` then asserts that the
#: Python side still disagrees with each recorded value, so the moment the defect is fixed
#: this exporter refuses to write a fixture at all rather than shipping a stale exemption.
#: Shape: ``{case label: {"cause": str, "fields": {field path: typescript value}}}``, where
#: a field path is a case key (``head400``) or ``stats.<name>``.
KNOWN_DIVERGENCES: dict[str, dict[str, Any]] = {}

#: Shared by every entry, so the wording cannot drift between them.
DIVERGENCE_STATUS = (
    "reported, unfixed — neither implementation was modified to build this fixture. Both "
    "readings are pinned exactly, so fixing either stack turns vacancyGolden.test.ts red at "
    "this block and the exemption cannot outlive the defect."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mapping_sha256(mapping: dict[str, str]) -> str:
    """A digest over the whole assignment, so a map can be pinned without shipping it.

    Canonical form: `stem\\tnonce\\n` per pair, stems in ASCII-ascending order — the same
    order §5.2 mints in, and reproducible in one line of TypeScript.
    """
    body = "".join(f"{stem}\t{mapping[stem]}\n" for stem in sorted(mapping))
    return sha256_text(body)


def params_json(params: VacancyParams) -> dict[str, Any]:
    """The parameter block, in the TypeScript field names both stacks' UIs use (§5.8)."""
    return {
        "p": params.p,
        "seed": params.seed,
        "consistent": params.consistent,
        "matchProsody": params.match_prosody,
        "revealAfter": params.reveal_after,
        "keep": sorted(params.keep),
        "mint": params.mint,
    }


def stem_block(
    domain: set[str], corpus_types: set[str], budget: set[str]
) -> list[dict[str, Any]]:
    """`u` for the 24 pinned stems at both seeds, plus why each one is (in)eligible."""
    out: list[dict[str, Any]] = []
    for stem in PINNED_STEMS:
        out.append(
            {
                "stem": stem,
                # BOTH halves of §2.2, separately: `eligible` is the stem-level test and
                # `vacatable` is the whole-word test that must run first. They differ exactly
                # on the closed-class words the splitter breaks open (`after`, `this`, …),
                # which is the pair of readings the fix turned on.
                "eligible": is_eligible(
                    stem_and_suffix(stem)[0], VacancyParams().keep_set
                ),
                "vacatable": is_vacatable(stem, VacancyParams().keep_set),
                "stemOf": stem_and_suffix(stem)[0],
                "suffixOf": stem_and_suffix(stem)[1],
                "inDomain": stem in domain,
                "inCorpus": stem in corpus_types,
                "inDolchFull": stem in budget,
                # EXACT float64 — see the module docstring on round-tripping.
                "u": {"0": vacancy_u(stem, 0), "7": vacancy_u(stem, 7)},
            }
        )
    return out


def vacated_stems(vmap: VacancyMap, seed: int, p: float) -> list[str]:
    """The stems the map vacates at `p`: `{stem : u(stem) < p}`, in canonical order.

    Over ``vmap.stems``, never ``vmap.mapping``: under ``mint="swap"`` the map is keyed by
    TYPE (§8.3) and this is a set of stems under both strategies.
    """
    return sorted(stem for stem in vmap.stems if vacancy_u(stem, seed) < p)


def map_block(label: str, vmap: VacancyMap, full: bool) -> dict[str, Any]:
    """One map's pinned facts. `full` writes every pair; otherwise a digest plus 28 samples.

    ``imagesOutsideDomain`` and ``fixedPoints`` are the swap control's defining property
    (§8.3) pinned as DATA rather than as prose: every image of a swap map must be a domain
    type — a real English word — and none may be the type it replaces. Both must read 0 under
    ``mint="swap"``. Under ``mint="nonce"`` the opposite holds and is equally worth pinning:
    condition B keeps every image OUT of the domain, so ``imagesOutsideDomain`` is the whole
    map.
    """
    images = sorted(set(vmap.mapping.values()))
    return {
        "label": label,
        "seed": vmap.seed,
        "matchProsody": vmap.match_prosody,
        "mint": vmap.mint,
        "injectiveAtEveryP": vmap.injective_at_every_p,
        "remintRounds": vmap.remint_rounds,
        "bijective": vmap.bijective,
        "imageSize": vmap.image_size,
        "domainSize": len(vmap.domain),
        "mappingSize": len(vmap.mapping),
        "stemsSize": len(vmap.stems),
        "imagesOutsideDomain": sum(1 for image in images if image not in vmap.domain),
        "fixedPoints": sorted(k for k, v in vmap.mapping.items() if k == v),
        "mappingSha256": mapping_sha256(dict(vmap.mapping)),
        "mapping": (
            {key: vmap.mapping[key] for key in sorted(vmap.mapping)} if full else None
        ),
        "sampleNonces": (
            None if full else {stem: vmap.mapping.get(stem) for stem in PINNED_STEMS}
        ),
    }


def _mapped_condition(vmap: VacancyMap, params: VacancyParams) -> bool:
    """Is the MAPPED vocabulary of §7.2 defined for this (map, params) pair?

    Three ways it is not, and both stacks raise on each: the inconsistent-assignment control
    and ``reveal_after > 0`` (§7.2 — a source type no longer has a single image), and
    ``mint="swap"`` at intermediate `p` (§5.2a — swap's replacements are domain types, so a
    vacated type can land on an un-vacated one and no `p`-stable swap avoids it).
    """
    if not params.consistent or params.reveal_after:
        return False
    return vmap.injective_at_every_p or params.p in (0.0, 1.0)


def id_stream_block(
    vacated: str, budget_words: list[str], vmap: VacancyMap, params: VacancyParams
) -> dict[str, Any] | None:
    """The §7.3 measurement: the id stream under the MAPPED vocabulary.

    `map_vocab_words` preserves order, so `itos_p = SPECIALS ++ mapped` gives every word the
    id its pre-image had; the stream of ids is then unchanged by vacancy and training is
    bit-identical. Returning the digest at every `p` turns that theorem into data.

    ``None`` for the control conditions, where the mapped vocabulary is undefined and both
    stacks raise — see the module docstring.
    """
    if not _mapped_condition(vmap, params):
        return None
    mapped = map_vocab_words(budget_words, vmap, params)
    vocab_p = LexVocab(tuple(mapped), source="dolch", budget_name="full")
    ids = [int(i) for i in token_stream(vacated, vocab_p)]
    return {
        "digest": sha256_text(",".join(str(i) for i in ids)),
        "length": len(ids),
        "first16": ids[:16],
        "last16": ids[-16:],
        "mappedWordsSha256": sha256_text("\n".join(mapped)),
    }


def apply_or_none(vmap: VacancyMap, word: str, params: VacancyParams) -> str | None:
    """`vmap.apply_word(word, params)`, or ``None`` where the engine REFUSES the word.

    Exactly one case refuses, and it is a property of §8.3 worth pinning rather than an error
    to hide: a swap map is keyed by TYPE, so a word the domain does not contain has no
    assignment at all. `gum` and `hang` are in `PINNED_STEMS` precisely because they reach a
    NONCE map only as the stems of `gums` and `hanged` — they are not types — so the nonce
    strategy transforms them and the swap strategy refuses them. ``None`` records that
    refusal, and the golden test asserts the TypeScript side refuses the same words, the same
    way. Nothing else may raise here: a `ComputeError` on any other word is re-raised.
    """
    try:
        return vmap.apply_word(word, params)
    except ComputeError:
        if params.mint == "swap" and word.lower() not in vmap.domain:
            return None
        raise


def build_case(
    label: str,
    map_label: str,
    corpus: str,
    vmap: VacancyMap,
    params: VacancyParams,
    budget_words: list[str],
) -> dict[str, Any]:
    """One measured condition: the rewritten corpus, its statistics, and the id stream.

    Order matters and the test must repeat it: `vacate_text` runs FIRST, because under
    `consistent = false` it registers the stress pattern of every form it mints on the map,
    and `vacancy_stats` scores the vacated side with exactly those patterns.
    """
    vacated = vacate_text(corpus, vmap, params)
    stats = vacancy_stats(corpus, vacated, vmap, params)
    mapped_condition = _mapped_condition(vmap, params)
    return {
        "label": label,
        "map": map_label,
        "params": params_json(params),
        "head400": vacated[:HEAD_CHARS],
        "vacatedSha256": sha256_text(vacated),
        "vacatedChars": len(vacated),
        "stats": stats,
        # §11's stability assertion: the surface form of each pinned stem at this `p`. It
        # is stated for — and only meaningful in — the MAPPED condition; §7.1 says the two
        # controls have DELIBERATELY no stability property, so there is nothing to pin
        # there and both stacks say so in their own way (TypeScript's single-word
        # `transformWord` refuses an order-dependent condition outright).
        # Stability (§11) is stated for the ORDER-INDEPENDENT conditions, which is a weaker
        # requirement than the mapped vocabulary's: `mint="swap"` at intermediate `p` has no
        # mapped vocabulary (§5.2a) but its map is still built once and still stable, so its
        # surface forms are pinned here exactly as the nonce strategy's are.
        "stemForms": (
            {stem: apply_or_none(vmap, stem, params) for stem in PINNED_STEMS}
            if params.consistent and not params.reveal_after
            else None
        ),
        "idStream": id_stream_block(vacated, budget_words, vmap, params),
        "mapVocabWordsRejects": not mapped_condition,
    }


def _read_field(case: dict[str, Any], field: str) -> Any:
    """Read a `knownDivergence` field path — either a case key or ``stats.<name>``."""
    if field.startswith("stats."):
        return case["stats"][field.split(".", 1)[1]]
    return case[field]


def attach_divergence(case: dict[str, Any]) -> None:
    """Record BOTH readings of a field the two stacks disagree on (see the docstring).

    Nothing is reconciled and nothing is loosened: the Python value is what this run
    measured, the TypeScript value is what a real browser-engine run measured, and the
    golden test asserts each side against its own. The guard below is what stops the
    exemption outliving the defect — the moment Python agrees with the recorded TypeScript
    reading, this exporter refuses to write a fixture at all.
    """
    entry = KNOWN_DIVERGENCES.get(case["label"])
    if entry is None:
        return
    fields = []
    for field, ts_value in entry["fields"].items():
        py_value = _read_field(case, field)
        if py_value == ts_value:
            raise SystemExit(
                f"{case['label']}.{field} now reads the TypeScript value on the Python "
                "side — the divergence this fixture pins has been resolved. Drop it from "
                "KNOWN_DIVERGENCES, drop the exemption in vacancyGolden.test.ts, and say "
                "so in architecture.md."
            )
        fields.append({"field": field, "python": py_value, "typescript": ts_value})
    case["knownDivergence"] = {
        "cause": entry["cause"],
        "status": DIVERGENCE_STATUS,
        "fields": fields,
    }


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as err:  # pragma: no cover - dev only
        raise SystemExit(f"cannot read the git sha for provenance: {err}") from err


def build_document(generated: str) -> dict[str, Any]:
    corpus = load_corpus_text()
    if not corpus.isascii():
        raise SystemExit(
            "the corpus is no longer ASCII, so `head400` no longer means the same 400 "
            "characters in Python and JavaScript — re-derive the excerpt before shipping"
        )
    corpus_types = set(tokenize(corpus))
    counts = type_counts(tokenize(corpus))
    budget_words = list(dolch_budget("full"))
    domain = vacancy_domain(corpus_types)
    domain_set = set(domain)

    maps: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    # --- the two pinned maps, in full (§11) ------------------------------------------
    prosody_maps: dict[int, VacancyMap] = {}
    for seed in (0, 7):
        vmap = build_vacancy_map(domain, VacancyParams(seed=seed))
        prosody_maps[seed] = vmap
        maps.append(map_block(f"seed{seed}", vmap, full=True))
        for p in P_GRID:
            cases.append(
                build_case(
                    f"seed{seed}-p{p}",
                    f"seed{seed}",
                    corpus,
                    vmap,
                    VacancyParams(seed=seed, p=p),
                    budget_words,
                )
            )

    # --- matchProsody = false: a DIFFERENT map, since minting reads the flag ----------
    # Pinned by digest plus the 24 sample nonces rather than in full: §11 asks for the
    # complete map at the two seeds above, and a sha256 over the canonical form is exactly
    # as strong a check for this one at 1/1000th the bytes.
    noprosody = build_vacancy_map(domain, VacancyParams(seed=0, match_prosody=False))
    maps.append(map_block("seed0-noprosody", noprosody, full=False))
    for p in (0.7, 1.0):
        cases.append(
            build_case(
                f"noprosody-p{p}",
                "seed0-noprosody",
                corpus,
                noprosody,
                VacancyParams(seed=0, p=p, match_prosody=False),
                budget_words,
            )
        )

    # --- the swap control (§8.3), on its own maps ------------------------------------
    # `mint="swap"` draws a REAL English word from the domain's open-class types by
    # frequency rank, so it needs the corpus's counts and it produces a different map at
    # every (seed, matchProsody). Pinned by digest plus the 24 sample replacements, like the
    # noprosody map above: a sha256 over the canonical form is exactly as strong a check.
    #
    # The `p` grid here is deliberately {0, 0.7, 1}. §5.2a proves that a map whose images are
    # domain types and which does not depend on `p` CANNOT be injective at intermediate `p`
    # unless it is the identity, so `swap-p0.7` carries `idStream: null` and
    # `mapVocabWordsRejects: true` — measured, and the same refusal both stacks make. The two
    # endpoints carry a real id stream, which is SC-703 holding for swap exactly as it does
    # for nonce wherever a swap map can be injective at all.
    for seed in (0, 7):
        for prosody in (True, False):
            label = f"swap-seed{seed}" + ("" if prosody else "-noprosody")
            base = VacancyParams(seed=seed, mint="swap", match_prosody=prosody)
            swap_map = build_vacancy_map(domain, base, counts)
            maps.append(map_block(label, swap_map, full=False))
            if not prosody:
                continue
            for p in (0.0, 0.7, 1.0):
                cases.append(
                    build_case(
                        f"swap-seed{seed}-p{p}",
                        label,
                        corpus,
                        swap_map,
                        VacancyParams(seed=seed, p=p, mint="swap"),
                        budget_words,
                    )
                )

    # --- the two control conditions (§7.1), each on its OWN map -----------------------
    # `consistent = false` writes to `minted_stress`; sharing a map across cases would let
    # one case's minted patterns score another case's text.
    #
    # `control-inconsistent-seed7` is the seed-7 CONDITION-B regression, pinned as data.
    # It is at `p = 1` and not `0.7` deliberately: §10's identity says every eligible type
    # vacates at full vacancy, so this case must read `corpusTypesVacated == 1918` and
    # `tokensVacated == 8125` — exactly what `consistent = true` reads. Before the fix it
    # read 1921 / 8201, because the stem `tak` minted the nonce `tak` and `Taking` survived
    # the transform. Seed 0 shows nothing here (no stem mints itself), which is why the
    # defect lived in the one control the fixture already had.
    for label, params in (
        ("control-inconsistent", VacancyParams(seed=0, p=0.7, consistent=False)),
        ("control-inconsistent-seed7", VacancyParams(seed=7, p=1.0, consistent=False)),
        ("control-reveal-after-2", VacancyParams(seed=0, p=0.7, reveal_after=2)),
    ):
        fresh = build_vacancy_map(domain, params)
        maps.append(map_block(label, fresh, full=False))
        case = build_case(label, label, corpus, fresh, params, budget_words)
        attach_divergence(case)
        cases.append(case)

    nesting = {
        "seed": NESTING_SEED,
        "map": f"seed{NESTING_SEED}",
        "levels": [
            {
                "p": p,
                "stems": vacated_stems(prosody_maps[NESTING_SEED], NESTING_SEED, p),
            }
            for p in NESTING_PS
        ],
    }

    return {
        "format": FORMAT,
        "generated": generated,
        "git_sha": git_sha(),
        "command": "python scripts/export_vacancy_golden.py",
        "source": (
            "llm_geometry.lex.vacancy run directly on the committed corpus — real code, "
            "real text, no mocks"
        ),
        "contract": "specs/007-vacancy-transform-field/architecture.md",
        "tolerance": TOLERANCE,
        "python_version": platform.python_version(),
        "encoding": (
            "every value is plain JSON; floats are shortest-round-trip in both languages, "
            "so `u` and the digests compare EXACTLY. Only the prosody means "
            "(meanSyllables*, meanAnapest*, stressFrom*) use `tolerance`. Digests are "
            "sha256 hex: `vacatedSha256` over the vacated corpus's UTF-8 bytes, "
            "`mappingSha256` over `stem\\tnonce\\n` in ASCII-ascending stem order, "
            "`idStream.digest` over the ids joined by ','."
        ),
        "corpus": {
            "path": "code/backend/src/llm_geometry/lex/data/real-mother-goose.txt",
            "note": "the Gutenberg body, trimmed exactly as lex/corpus.py trims it",
            "sha256": sha256_text(corpus),
            "chars": len(corpus),
            "tokens": len(tokenize(corpus)),
            "corpusTypes": len(corpus_types),
            "domainSize": len(domain),
            "budget": "dolch/full",
            "budgetSize": len(budget_words),
        },
        "stems": stem_block(domain_set, corpus_types, set(budget_words)),
        "maps": maps,
        "cases": cases,
        "nesting": nesting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    document = build_document(args.generated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(
        f"wrote {args.out} ({size_kb:.0f} KB, {len(document['maps'])} maps, "
        f"{len(document['cases'])} cases)"
    )


if __name__ == "__main__":
    main()
