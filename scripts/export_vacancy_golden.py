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

1. ``u(stem)`` for 24 stems spanning eligible/ineligible and both budgets, as the EXACT
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
6. Stability — ``cases[].stemForms``, the surface form of each of the 24 stems at every
   ``p``. A stem's nonce must be byte-identical at every ``p`` where it is vacated.
7. The token-id-stream digest under the mapped vocabulary at each ``p`` —
   ``cases[].idStream``. These are all EQUAL, which is §7.3 (the invariance theorem) pinned
   as DATA rather than as an assertion written twice, once per language.
8. Both control conditions (``consistent = false``, ``revealAfter > 0``) and both
   ``matchProsody`` settings — the ``control-*`` and ``noprosody-*`` cases.

MEASURED, and recorded here because the fixture's shape depends on it: the mapped
vocabulary of §7.2 is defined ONLY for ``consistent = true, revealAfter = 0``; both stacks
raise otherwise. So a control case has ``idStream: null`` and
``mapVocabWordsRejects: true``, and the test asserts the TypeScript side refuses it too —
a silently-accepted control would manufacture a vocabulary matching no corpus.

Also measured: ``consistent = false`` mints a fresh form per OCCURRENCE and registers its
stress pattern on the map's ``minted_stress``, i.e. it MUTATES the map. Every control case
therefore builds its own map, and the test must do the same or its statistics will be
scored against patterns left behind by a previous case.

TWO MEASURED DISAGREEMENTS BETWEEN THE STACKS, pinned rather than papered over. Each is a
real defect in one implementation, each is reported, and NEITHER implementation was touched
to build this fixture — a golden file that quietly matched the two would destroy the
evidence. Both sit in the CONTROL conditions; every mapped-condition case (both seeds, both
``matchProsody`` settings, every `p`) agrees field for field, string for string, digest for
digest.

**1. ``corpusTypesVacated`` under ``revealAfter > 0``** — Python 665, TypeScript 1337.

* Python (``vacancy.py::vacancy_stats``) MEASURES it: ``len({b.lower() for b, a in
  zip(before, after) if b.lower() != a.lower()})``, i.e. types that actually changed in the
  text.
* TypeScript (``vacancy.ts::countTypes``) computes it THROUGH THE MAP: a type counts if
  ``u(stem) < p`` and its surface form differs, whatever the rewrite did.

They agree everywhere except ``revealAfter > 0``, where a type all of whose occurrences fall
inside the reveal window stays English in the text while the map still assigns it a nonce.
§10 says ``corpusTypes*`` is "what the panel shows a reader", and the reader is looking at
the text — so the MEASURED reading (Python's) is the contract's, and the map-based one
over-reports by 2x in exactly the condition this control exists to measure. ``countTypes``'s
own comment argues condition-independence only for ``consistent = false``, where it is
right; it is silent on ``revealAfter``, where it is wrong.

**2. The minted forms under ``consistent = false``** — the two stacks mint DIFFERENT nonces,
so the vacated text, its digest, its length and four prosody means all differ.

§5.8 pins the per-occurrence KEY as ``f"{stem}#{idx}"`` and §5.5's mint reads its stress
pattern off the string it is handed. Python passes the key and therefore mints to
``stress("little#0") == "10"`` — two syllables, via the spelling rule, because ``little#0``
is not in the hand table. TypeScript computes ``stress("little") == "100"`` — three
syllables, from the table — and passes the key only to the byte stream. So ``Little`` becomes
``Wrerken`` in one stack and ``Wrerkenle`` in the other.

§7.1 defines ``matchProsody`` as "the nonce carries **the stem's** syllable count and
stress", and §10 defines ``stressFromMinted`` as the pattern we registered for a form — both
describe the STEM's prosody, not a synthetic key's. TypeScript's reading is therefore the
contract's; Python's derives prosody from a string containing ``#`` and a digit. The contract
pins the key and says nothing about the pattern, and that silence is the gap. Note the
control still controls: every count, ``tokensVacated`` included, agrees exactly, so SC-705's
measurement is unaffected.

Until each is fixed, both readings are pinned. ``knownDivergence`` on the affected case
carries, per field, the Python value from THIS run and the MEASURED TypeScript value, plus
the explanation above; the golden test asserts each side against its own. Fixing either
stack turns the test red at that block — and ``attach_divergence`` refuses to write a fixture
at all once Python agrees with the recorded TypeScript reading, so the exemption cannot
outlive the defect.
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

from llm_geometry.lex.corpus import load_corpus_text  # noqa: E402
from llm_geometry.lex.dolch import dolch_budget  # noqa: E402
from llm_geometry.lex.train import token_stream  # noqa: E402
from llm_geometry.lex.vocab import LexVocab, tokenize  # noqa: E402
from llm_geometry.lex.vacancy import (  # noqa: E402
    VacancyMap,
    VacancyParams,
    build_vacancy_map,
    is_eligible,
    map_vocab_words,
    stem_and_suffix,
    vacancy_domain,
    vacancy_stats,
    vacancy_u,
    vacate_text,
)

DEFAULT_OUT = (
    REPO_ROOT / "code" / "frontend" / "tests" / "fixtures" / "vacancy-golden.json"
)

FORMAT = "vacancy-golden-v1"

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

#: §11's "24 stems spanning eligible/ineligible and both budgets".
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
)

#: §11's excerpt length. The corpus is ASCII (asserted below), so Python code points and
#: JavaScript UTF-16 code units count the same characters.
HEAD_CHARS = 400

#: The nesting sets §11 asks for, as explicit data.
NESTING_SEED = 0
NESTING_PS: tuple[float, ...] = (0.0, 0.35, 0.7, 1.0)

#: The two measured cross-stack disagreements — see the module docstring for what they are
#: and why they are recorded instead of reconciled. Both live in the CONTROL conditions;
#: every mapped-condition case agrees field for field.
#:
#: Every ``typescript`` value below is MEASURED, by running the real browser engine on the
#: real corpus at that case's parameters:
#:
#:     const vmap = buildVacancyMap(vacancyDomain(new Set(tokenize(CORPUS))), params);
#:     const vacated = vacateText(CORPUS, vmap, params);
#:     const stats = vacancyStats(CORPUS, vacated, vmap, params);
#:
#: None of them is a prediction and none is derived here. ``build_document`` asserts that
#: the Python side still disagrees with each one, so if either stack is fixed this exporter
#: fails loudly rather than shipping a stale exemption.
KNOWN_DIVERGENCES: dict[str, dict[str, Any]] = {
    "control-reveal-after-2": {
        "cause": (
            "corpusTypesVacated: Python MEASURES it from the two texts "
            "(vacancy.py::vacancy_stats), TypeScript computes it through the map "
            "(vacancy.ts::countTypes). They agree everywhere except revealAfter > 0, where "
            "a type all of whose occurrences fall inside the reveal window stays English in "
            "the text while the map still assigns it a nonce. §10 defines corpusTypes* as "
            "what the panel shows a READER — i.e. what the text does — so the measured "
            "reading is the contract's and countTypes over-reports by 2x in exactly the "
            "condition this control exists to measure. countTypes's own comment argues "
            "condition-independence only for consistent = false, where it is right; it is "
            "silent on revealAfter, where it is wrong."
        ),
        "fields": {"stats.corpusTypesVacated": 1337},
    },
    "control-inconsistent": {
        "cause": (
            "the per-occurrence mint derives its stress pattern from different strings. "
            "§5.8 pins the KEY as f'{stem}#{idx}' and §5.5's mint reads its pattern off the "
            "string it is handed, so Python calls _mint('little#0') and gets "
            "stress('little#0') = '10' (2 syllables, via the spelling rule, because "
            "'little#0' is not in the hand table); TypeScript computes stress('little') = "
            "'100' (3 syllables, from the table) and passes the key only to the byte "
            "stream. §7.1 defines matchProsody as 'the nonce carries THE STEM's syllable "
            "count and stress', and §10 defines stressFromMinted as the pattern we "
            "registered for a form — both of which describe the stem's prosody, not a "
            "synthetic key's, so TypeScript's reading is the contract's. The contract pins "
            "the key and is silent on the pattern; that silence is the gap. Only the minted "
            "FORMS differ: every count, including tokensVacated and both vacated-type "
            "counts, agrees exactly, so the control still measures what SC-705 says it does."
        ),
        "fields": {
            "vacatedSha256": (
                "610a0d8375890ec68a788457b957d248e5e8a279644fc5b339aa7849451a68cf"
            ),
            "vacatedChars": 94748,
            "stats.meanSyllablesAfter": 1.28975,
            "stats.meanAnapestAfter": 0.3215726757806852,
            "stats.stressFromMintedAfter": 0.303375,
            "stats.stressFromRuleAfter": 0.67925,
            "head400": (
                "      THE GLAIRN\n    SCIRRDER PLERNT\n\n  _Wroormowyed by_\nBlanche "
                "Thursper Styn\n\n1916\n\n\n\nA HURF OF THE YORCKES\n\nWrerkenle "
                "Bo-Peep\nFlorlkishle Boy Poun\nRain\nThe Sqyft\nWinter\nNietens and "
                "Klisks\nA Gnuntingenle Jai\nSculk Mursh and Her Cat\nThree Thoochousish on "
                "the Ice\nNefs Gask\nThe Soamp Therltle Under a Gloang\nTweedle-Dum and "
                "Tweedle-Dee\nOh Steb!\nSwouck Felkid Rarp\nLainowid Yersking Snof\n"
                "Pat-a-Cake\nSt"
            ),
        },
    },
}

#: Shared by both entries above, so the wording cannot drift between them.
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
                "eligible": is_eligible(stem, VacancyParams().keep_set),
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
    """The stems the map vacates at `p`: `{stem : u(stem) < p}`, in canonical order."""
    return sorted(stem for stem in vmap.mapping if vacancy_u(stem, seed) < p)


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
    if not params.consistent or params.reveal_after:
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
    mapped_condition = params.consistent and not params.reveal_after
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
        "stemForms": (
            {stem: vmap.apply_word(stem, params) for stem in PINNED_STEMS}
            if mapped_condition
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
        maps.append(
            {
                "label": f"seed{seed}",
                "seed": seed,
                "matchProsody": True,
                "remintRounds": vmap.remint_rounds,
                "bijective": vmap.bijective,
                "imageSize": vmap.image_size,
                "domainSize": len(vmap.domain),
                "mappingSize": len(vmap.mapping),
                "mappingSha256": mapping_sha256(dict(vmap.mapping)),
                "mapping": {stem: vmap.mapping[stem] for stem in sorted(vmap.mapping)},
            }
        )
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
    maps.append(
        {
            "label": "seed0-noprosody",
            "seed": 0,
            "matchProsody": False,
            "remintRounds": noprosody.remint_rounds,
            "bijective": noprosody.bijective,
            "imageSize": noprosody.image_size,
            "domainSize": len(noprosody.domain),
            "mappingSize": len(noprosody.mapping),
            "mappingSha256": mapping_sha256(dict(noprosody.mapping)),
            "mapping": None,
            "sampleNonces": {
                stem: noprosody.mapping.get(stem) for stem in PINNED_STEMS
            },
        }
    )
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

    # --- the two control conditions (§7.1), each on its OWN map -----------------------
    # `consistent = false` writes to `minted_stress`; sharing a map across cases would let
    # one case's minted patterns score another case's text.
    for label, params in (
        ("control-inconsistent", VacancyParams(seed=0, p=0.7, consistent=False)),
        ("control-reveal-after-2", VacancyParams(seed=0, p=0.7, reveal_after=2)),
    ):
        fresh = build_vacancy_map(domain, params)
        maps.append(
            {
                "label": label,
                "seed": params.seed,
                "matchProsody": params.match_prosody,
                "remintRounds": fresh.remint_rounds,
                "bijective": fresh.bijective,
                "imageSize": fresh.image_size,
                "domainSize": len(fresh.domain),
                "mappingSize": len(fresh.mapping),
                "mappingSha256": mapping_sha256(dict(fresh.mapping)),
                "mapping": None,
                "sampleNonces": {
                    stem: fresh.mapping.get(stem) for stem in PINNED_STEMS
                },
            }
        )
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
