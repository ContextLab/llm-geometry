"""The vacancy transform, checked against the properties a `p`-sweep depends on.

Everything here runs on the REAL committed corpus (`The Real Mother Goose`) and the real
Dolch budgets. No fixtures, no mocks: the transform's whole claim is about what happens to a
real English text, and a synthetic string would not exercise the cases that matter
(`good-bye`, `don't`, `dog's`, capitalised line openers, hyphenated compounds).

The four properties, in the order the contract states them
(`specs/007-vacancy-transform-field/architecture.md`):

* NESTING (SC-701) — `u` depends only on `(seed, stem)`, so the vacated set at `p` is a
  subset of the vacated set at any larger `p`.
* STABILITY (SC-702) — the map is built once, in canonical order, independently of `p` and of
  document order, so a stem's nonce is the same everywhere it appears.
* INVARIANCE (SC-703) — with `consistent=True, reveal_after=0` and a mapped vocabulary the
  token id stream is unchanged, so a word-level model is *exactly* invariant to `p`.
* INJECTIVITY (SC-704) — verified on the real type set, not assumed; the theorem depends on
  it.
"""

from __future__ import annotations

import random

import pytest

from llm_geometry.errors import ComputeError, InvalidParamError
from llm_geometry.lex.corpus import load_corpus_text
from llm_geometry.lex.dolch import DOLCH_ORDER, dolch_budget
from llm_geometry.lex.vacancy import (
    FUNCTION_WORDS,
    REMINT_SALT_STRIDE,
    SPLIT_EXCEPTIONS,
    STRESS_TABLE,
    SUFFIXES,
    VacancyMap,
    VacancyParams,
    _mint,  # private: the salt semantics of §5.5/§5.8 are a contract detail, so they are pinned
    build_vacancy_map,
    is_eligible,
    map_vocab_words,
    meter_score,
    stem_and_suffix,
    stress,
    stress_source,
    syllables,
    vacancy_domain,
    vacancy_stats,
    vacancy_u,
    vacate_text,
)
from llm_geometry.lex.vocab import WORD_RE, LexVocab, tokenize

P_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 7)


@pytest.fixture(scope="module")
def corpus() -> str:
    return load_corpus_text()


@pytest.fixture(scope="module")
def corpus_types(corpus: str) -> list[str]:
    return sorted(set(tokenize(corpus)))


@pytest.fixture(scope="module")
def budget() -> list[str]:
    """The full Dolch list — the budget whose words must also be in the map's domain."""
    return dolch_budget("full")


@pytest.fixture(scope="module")
def domain(corpus_types: list[str]) -> list[str]:
    """Contract §5.2: the corpus's types UNION the FULL Dolch list, never the active budget."""
    return vacancy_domain(corpus_types)


@pytest.fixture(scope="module")
def maps(domain: list[str]) -> dict[int, VacancyMap]:
    """One map per seed, built once — they are `p`-independent by construction."""
    return {seed: build_vacancy_map(domain, VacancyParams(p=1.0, seed=seed)) for seed in SEEDS}


@pytest.fixture(scope="module")
def vacated(corpus: str, maps: dict[int, VacancyMap]) -> dict[tuple[int, float], str]:
    """The vacated corpus at every (seed, p) on the grid."""
    out = {}
    for seed in SEEDS:
        for p in P_GRID:
            params = VacancyParams(p=p, seed=seed)
            out[(seed, p)] = vacate_text(corpus, maps[seed], params)
    return out


def _changed_types(original: str, vacated_text: str) -> set[str]:
    """The source types whose surface actually changed, measured from the two texts."""
    before = WORD_RE.findall(original)
    after = WORD_RE.findall(vacated_text)
    assert len(before) == len(after)
    return {b.lower() for b, a in zip(before, after) if b != a}


def _images(original: str, vacated_text: str) -> dict[str, set[str]]:
    """source TYPE -> the set of lower-cased surfaces it was rewritten to.

    Keyed on the lower-cased word, which is what the tokenizer sees. That is the strong
    question, and the transform must survive it — see
    :func:`test_transform_commutes_with_lowercasing_over_the_whole_corpus`.
    """
    images: dict[str, set[str]] = {}
    for b, a in zip(WORD_RE.findall(original), WORD_RE.findall(vacated_text)):
        images.setdefault(b.lower(), set()).add(a.lower())
    return images


# --- the verbatim tables ----------------------------------------------------------------


def test_closed_class_is_the_curated_list_only():
    """The source's warning, kept: an earlier version unioned this with the short Dolch
    service words, which silently protected content verbs and understated the vacancy rate."""
    assert len(FUNCTION_WORDS) == 137
    for w in ("run", "eat", "see", "get", "let", "put"):
        assert w not in FUNCTION_WORDS, f"{w} is a content verb, not closed class"
    for w in ("the", "and", "never", "ten"):
        assert w in FUNCTION_WORDS
    assert all(w == w.lower() for w in FUNCTION_WORDS)


def test_suffix_order_is_load_bearing():
    """`ies` must be tried before `es` and `s`, `edly` before `ed`."""
    assert SUFFIXES.index("ies") < SUFFIXES.index("es") < SUFFIXES.index("s")
    assert SUFFIXES.index("edly") < SUFFIXES.index("ed")
    assert stem_and_suffix("berries") == ("berr", "ies")


def test_split_exceptions_come_from_the_audited_copy():
    """Without them `brother -> broth+er` and `morning -> morn+ing`, which the source itself
    flags as a known artifact."""
    assert "brother" in SPLIT_EXCEPTIONS and "morning" in SPLIT_EXCEPTIONS
    assert stem_and_suffix("brother") == ("brother", "")
    assert stem_and_suffix("morning") == ("morning", "")
    # ... and it is a spelling heuristic, still wrong outside the list. Documented, not fixed.
    assert stem_and_suffix("ladder") == ("ladd", "er")


# --- eligibility, contract §2.2 ---------------------------------------------------------


def test_good_bye_is_never_vacated(maps):
    """No suffix matches, so the stem contains a hyphen and fails the ASCII-letters test."""
    assert stem_and_suffix("good-bye") == ("good-bye", "")
    assert not is_eligible("good-bye")
    for seed in SEEDS:
        assert maps[seed].apply_word("good-bye", VacancyParams(p=1.0, seed=seed)) == "good-bye"
        assert "good-bye" not in maps[seed].mapping


def test_dont_is_never_vacated(maps):
    """The MECHANISM, not just the outcome (§2.2). The `n't` suffix does NOT split `don't`,
    because §3's length rule needs `len(word) - len(suffix) >= 3` and `5 - 3 = 2`. The stem is
    therefore the whole `don't`, which contains an apostrophe and fails eligibility test 2 —
    the ASCII-letters test — rather than test 3, the length test."""
    assert stem_and_suffix("don't") == ("don't", "")
    assert not is_eligible("don't")
    assert len("don't") - len("n't") == 2  # < 3, which is why no split happens
    assert len("don't") > 2  # so it is test 2 that rejects it, not test 3
    for seed in SEEDS:
        assert maps[seed].apply_word("don't", VacancyParams(p=1.0, seed=seed)) == "don't"


def test_dogs_apostrophe_s_keeps_its_suffix(maps):
    """`dog's` splits to `dog`, which passes; the output is `<nonce>'s`."""
    assert stem_and_suffix("dog's") == ("dog", "'s")
    assert is_eligible("dog")
    for seed in SEEDS:
        params = VacancyParams(p=1.0, seed=seed)
        out = maps[seed].apply_word("dog's", params)
        assert out.endswith("'s")
        assert out[:-2].lower() == maps[seed].mapping["dog"] or out[:-2].lower() != "dog"
        assert out != "dog's"


def test_short_and_closed_class_stems_are_ineligible():
    assert not is_eligible("cat"[:2])  # len 2
    assert not is_eligible("the")
    assert is_eligible("cat")
    # `keep` extends the closed class.
    assert not is_eligible("cat", FUNCTION_WORDS | {"cat"})


def test_unicode_letters_are_not_ascii_letters():
    """`str.isalpha()` would accept these; JavaScript's `^[A-Za-z]+$` does not."""
    assert not is_eligible("café")
    assert not is_eligible("naïve")


# --- the vacancy decision, contract §4 --------------------------------------------------


def test_u_is_a_53_bit_double_and_depends_only_on_seed_and_stem():
    for stem in ("candle", "mother", "jack", "z"):
        u = vacancy_u(stem, 0)
        assert 0.0 <= u < 1.0
        # The `>> 11` exists so the numerator is exactly representable; if it were not, this
        # product would not be an integer and the two stacks could disagree at the boundary.
        assert (u * 2**53).is_integer()
        assert vacancy_u(stem, 0) == u
        assert vacancy_u(stem.upper(), 0) == u
        assert vacancy_u(stem, 1) != u


def test_nesting_across_p_grid_and_two_seeds(corpus, vacated):
    """SC-701. The vacated sets grow monotonically with `p`, measured from the output."""
    for seed in SEEDS:
        sets = [_changed_types(corpus, vacated[(seed, p)]) for p in P_GRID]
        for smaller, larger in zip(sets, sets[1:]):
            assert smaller <= larger
        assert sets[0] == set()
        assert len(sets[-1]) > len(sets[0])
        # And it is genuinely graded, not a step function.
        assert len(sets[1]) < len(sets[2]) < len(sets[3]) < len(sets[4])


def test_vacancy_rate_tracks_p(corpus, corpus_types, vacated):
    """`p` is the fraction of ELIGIBLE TYPES vacated, so the measured rate should sit close
    to `p` — this is a distributional check on `u`, not a re-implementation of it."""
    eligible = {t for t in corpus_types if is_eligible(stem_and_suffix(t)[0])}
    for seed in SEEDS:
        for p in (0.25, 0.5, 0.75):
            rate = len(_changed_types(corpus, vacated[(seed, p)])) / len(eligible)
            assert abs(rate - p) < 0.05, f"seed={seed} p={p} rate={rate}"


# --- stability, contract §5.6 -----------------------------------------------------------


def test_a_stems_nonce_is_identical_at_every_p(corpus, maps, vacated):
    """SC-702. Nothing about the assignment may depend on `p`."""
    for seed in SEEDS:
        per_p = [_images(corpus, vacated[(seed, p)]) for p in P_GRID]
        for word in per_p[-1]:
            surfaces = {img for images in per_p for img in images.get(word, set()) if img != word}
            assert len(surfaces) <= 1, f"{word} took {surfaces} across the p-sweep"
        # ... and the assignment itself, which is what SC-702 is stated over.
        for stem, nonce in maps[seed].mapping.items():
            for p in P_GRID:
                if vacancy_u(stem, seed) < p:
                    assert maps[seed].nonce_for(stem) == nonce


# --- the case-commuting invariant, contract §5.7 ----------------------------------------


def test_transform_commutes_with_lowercasing_over_the_whole_corpus(corpus, maps):
    """The normative invariant of §5.7: `lower(transform(w)) == transform(lower(w))`.

    The tokenizer lowercases, so a step that branches on case — a seam test against a
    case-preserved suffix, say — gives one TYPE two surface forms and §7.3 is false. This is
    the test that would have caught `gums -> flels` while `GUMS -> FLESS`.
    """
    params = VacancyParams(p=1.0, seed=0)
    vmap = maps[0]
    for word in WORD_RE.findall(corpus):
        assert vmap.apply_word(word, params).lower() == vmap.apply_word(word.lower(), params)
    # The specific pair that broke it, now equal.
    assert vmap.apply_word("GUMS", params) == "FLELS"
    assert vmap.apply_word("gums", params) == "flels"


def test_transform_commutes_over_every_type_in_three_casings(corpus_types, maps):
    """Same invariant, but exercising casings the corpus does not happen to contain."""
    for seed in SEEDS:
        params = VacancyParams(p=1.0, seed=seed)
        vmap = maps[seed]
        for word in corpus_types:
            lowered = vmap.apply_word(word, params)
            assert lowered == lowered.lower()
            for variant in (word.upper(), word.capitalize(), word.lower()):
                assert vmap.apply_word(variant, params).lower() == lowered
            # ... and the case marking itself survives, so the corpus still reads as English.
            assert vmap.apply_word(word.upper(), params).isupper() or len(word) == 1
            assert vmap.apply_word(word.capitalize(), params)[0].isupper()


def test_map_is_unchanged_by_shuffling_the_input_types(domain, corpus_types, maps):
    """The build order is canonical (sorted), never document order."""
    shuffled = list(domain)
    random.Random(1234).shuffle(shuffled)
    rebuilt = build_vacancy_map(shuffled, VacancyParams(p=1.0, seed=0))
    assert rebuilt.mapping == maps[0].mapping
    assert rebuilt.minted_stress == maps[0].minted_stress


def test_map_does_not_depend_on_p(domain, corpus_types, maps):
    for p in (0.0, 0.35, 1.0):
        built = build_vacancy_map(domain, VacancyParams(p=p, seed=0))
        assert built.mapping == maps[0].mapping


def test_seeds_give_different_assignments(maps):
    assert maps[0].mapping != maps[7].mapping
    shared = set(maps[0].mapping) & set(maps[7].mapping)
    differing = sum(1 for k in shared if maps[0].mapping[k] != maps[7].mapping[k])
    assert differing > 0.99 * len(shared)


# --- injectivity, contract §7.3 ---------------------------------------------------------


def test_map_is_injective_on_the_real_corpus(maps, domain):
    """SC-704. Two distinct source types can collide through the stem+suffix construction;
    the build verifies the assembled SURFACE FORMS and re-mints until they do not."""
    for seed in SEEDS:
        vmap = maps[seed]
        assert vmap.bijective is True
        assert vmap.image_size == vmap.type_count == len(domain)
        assert len(set(vmap.mapping.values())) == len(vmap.mapping)
    # Condition B costs exactly one re-mint on this corpus, at seed 7. Measured, not assumed:
    # if a change to the minter makes it zero or two, this number is the first thing to look at.
    assert maps[0].remint_rounds == 0
    assert maps[7].remint_rounds == 1


def test_injectivity_holds_at_every_p_not_just_the_endpoints(corpus, vacated):
    """Trap 1 of §7.3: checking at `p = 1` only is insufficient, because at full vacancy every
    eligible type has moved and a minted form has nothing left to collide with."""
    for seed in SEEDS:
        for p in P_GRID:
            pairs = list(zip(WORD_RE.findall(corpus), WORD_RE.findall(vacated[(seed, p)])))
            image_of: dict[str, str] = {}
            for src, out in pairs:
                image_of.setdefault(src.lower(), out.lower())
            assert len(set(image_of.values())) == len(
                image_of
            ), f"seed={seed} p={p}: two source types share a surface form"


def test_a_remint_is_held_to_the_same_quality_bar_as_an_original_mint(maps):
    """§5.5's thresholds are on the ATTEMPT COUNTER `a`, not on the absolute salt.

    A mint call carries a base salt `S`; the stream is keyed on `S + a` and a re-mint restarts
    `a` at 0. Read the thresholds absolutely instead and a re-mint at `S = 1001` would begin
    with the length and syllable checks already relaxed — and a second round, at `S = 2001`,
    could not run at all, contradicting "raise after 8 rounds".

    The observable: `hang` is monosyllabic, and its RE-MINTED nonce still is.
    """
    assert syllables("hang") == 1
    assert syllables(maps[7].mapping["hang"]) == 1
    assert maps[7].minted_stress[maps[7].mapping["hang"]] == stress("hang")
    assert len(maps[7].mapping["hang"]) >= 3
    # Directly: a mint from a base salt past every threshold still enforces every check.
    for base in (0, 401, 801, 1001, REMINT_SALT_STRIDE + 500):
        nonce, pattern, salt = _mint("hang", 7, True, frozenset(), start_salt=base)
        assert syllables(nonce) == len(pattern) == 1
        assert len(nonce) >= 3
        assert base <= salt < base + 1200
    # ... and the stream is keyed on `S + a`: forbidding the first candidate at `S` gives
    # exactly what a fresh call at `S + 1` produces.
    first, _pattern, first_salt = _mint("hang", 7, True, frozenset(), start_salt=100)
    assert first_salt == 100
    assert _mint("hang", 7, True, frozenset({first}), start_salt=100)[0] == (
        _mint("hang", 7, True, frozenset(), start_salt=101)[0]
    )


def test_hanged_no_longer_surfaces_as_the_english_word_waked(corpus, corpus_types, maps, vacated):
    """Regression for trap 2 of §7.3, the case that motivated condition B.

    At seed 7 the stem `hang` first minted `wak`, so `hanged` assembled to `waked` — a real
    word of this corpus, and one that is NOT vacated at p = 0.25 or p = 0.5, where it therefore
    merged with the vacated `hanged`. A bare-nonce `avoid` check passed (`wak` is not a corpus
    type) and a p = 1 image check passed (there, `waked` had moved too).
    """
    assert "waked" in corpus_types and "hanged" in corpus_types
    vmap = maps[7]
    assert vmap.mapping["hang"] != "wak"
    for p in (0.25, 0.5):
        params = VacancyParams(p=p, seed=7)
        assert vmap.apply_word("hanged", params) != "waked"
        text = vacated[(7, p)]
        pairs = list(zip(WORD_RE.findall(corpus), WORD_RE.findall(text)))
        survivors = {b.lower() for b, a in pairs if b == a}
        moved = {a.lower() for b, a in pairs if b != a}
        assert not (moved & survivors)


def test_no_nonce_is_a_real_corpus_type(maps, corpus_types):
    """FR-706. The source accepts an `avoid` parameter and never passes one, which lets a
    minted form silently merge with an English type."""
    real = set(corpus_types)
    for seed in SEEDS:
        assert not (set(maps[seed].mapping.values()) & real)


def test_domain_is_the_corpus_plus_the_full_dolch_list_never_the_active_budget(
    corpus_types, domain, budget
):
    """§5.2. If the domain tracked the ACTIVE budget, switching budgets in the UI would
    rebuild the map and re-mint the corpus underneath a panel demonstrating that nonces are
    stable."""
    assert set(domain) == {t.lower() for t in corpus_types} | {w.lower() for w in budget}
    assert set(vacancy_domain([])) == {w.lower() for w in dolch_budget("full")}
    assert domain == sorted(domain)  # canonical order, ASCII ascending


def test_map_is_a_pure_function_of_domain_seed_and_prosody(corpus, corpus_types, domain, maps):
    """§5.2: there is no caller-supplied `avoid`, so no call path can build a different map.

    The parameter existed, defaulted to empty, and both stacks passed the type set — so they
    agreed and no parity test could catch it. But the map was then a function of what the
    caller remembered: at seed 0 the same corpus gives `remint_rounds` 0 with the domain
    passed and 1 without, with different nonces either way. Both maps valid; that is the
    problem. Here the same map is built through four different call paths.
    """
    reference = maps[0].mapping
    paths = [
        build_vacancy_map(domain, VacancyParams(p=1.0, seed=0)),
        build_vacancy_map(vacancy_domain(corpus_types), VacancyParams(p=0.0, seed=0)),
        build_vacancy_map(vacancy_domain(tokenize(corpus)), VacancyParams(p=0.5, seed=0)),
        build_vacancy_map(vacancy_domain(set(corpus_types)), VacancyParams(p=1.0, seed=0)),
    ]
    for built in paths:
        assert built.mapping == reference
        assert built.minted_stress == maps[0].minted_stress
        assert built.domain == maps[0].domain
        assert built.remint_rounds == maps[0].remint_rounds
    # ... and the domain is what a nonce is forbidden to equal, with nothing left to a caller.
    assert not set(reference.values()) & maps[0].domain


def test_iterables_of_types_reject_a_bare_text(corpus, domain, budget, maps):
    """`Iterable[str]` accepts a `str` and iterates it character by character, so passing the
    corpus text built a domain of single letters and failed much later, somewhere else."""
    for call in (
        lambda: vacancy_domain(corpus),
        lambda: build_vacancy_map(corpus, VacancyParams(p=1.0, seed=0)),
        lambda: map_vocab_words(corpus, maps[0], VacancyParams(p=1.0, seed=0)),
        lambda: VacancyParams(p=1.0, seed=0, keep="little"),
    ):
        with pytest.raises(TypeError, match="not a text"):
            call()
    # The correct forms still work.
    assert vacancy_domain(tokenize(corpus)) == domain
    assert len(map_vocab_words(budget, maps[0], VacancyParams(p=1.0, seed=0))) == len(budget)
    assert VacancyParams(keep=frozenset({"little"})).keep_set == FUNCTION_WORDS | {"little"}


def test_the_map_does_not_move_when_the_active_budget_changes(corpus_types, maps):
    """The property the panel depends on, and the reason the domain is the FULL Dolch list.

    Once `avoid` became implicit (§5.2), the domain IS the forbidden set — so a *smaller*
    domain forbids fewer words and genuinely mints differently. Measured: building over
    `corpus ∪ dolch_budget(name)` for any name below `full` moves exactly one stem, `jam`,
    because `floor` is a full-list Dolch word that never appears in the corpus and so is
    forbidden in the full domain and free in the smaller ones.

    That is not a defect, it is the argument: if the domain tracked the ACTIVE budget, a
    reader switching budgets would watch the corpus re-mint under a panel demonstrating that
    nonces are stable. `vacancy_domain` unions the full list regardless, so the smaller
    domains are unreachable through the sanctioned API and there is only ever one map.
    """
    for name in DOLCH_ORDER:
        assert vacancy_domain(corpus_types) == vacancy_domain(corpus_types + dolch_budget(name))
        for seed in SEEDS:
            built = build_vacancy_map(
                vacancy_domain(corpus_types + dolch_budget(name)),
                VacancyParams(p=1.0, seed=seed),
            )
            assert built.mapping == maps[seed].mapping, f"{name}/{seed}: the map moved"

    # The counterexample that makes the rule load-bearing, pinned so it cannot drift silently.
    assert "floor" in dolch_budget("full")
    assert "floor" not in corpus_types
    smaller = sorted({t.lower() for t in corpus_types} | set(dolch_budget("pre_primer")))
    off_contract = build_vacancy_map(smaller, VacancyParams(p=1.0, seed=0))
    moved = [s for s, n in off_contract.mapping.items() if maps[0].mapping[s] != n]
    assert moved == ["jam"]
    assert off_contract.mapping["jam"] == "floor" != maps[0].mapping["jam"]


def test_no_surface_form_is_ever_an_english_word_of_the_domain(corpus, domain, maps, vacated):
    """Condition B of §5.2, stated over the whole domain and therefore `p`-independent.

    Deliberately conservative: it forbids a minted form from equalling a word that would
    always have been vacated alongside it. That costs a re-mint and buys a condition that
    holds simultaneously at every `p`, which is what the theorem needs.
    """
    real = set(domain)
    for seed in SEEDS:
        params = VacancyParams(p=1.0, seed=seed)
        for t in domain:
            out = maps[seed].apply_word(t, params)
            if out != t:
                assert out not in real, f"seed={seed}: {t} surfaced as the English word {out}"
        for p in P_GRID:
            pairs = zip(WORD_RE.findall(corpus), WORD_RE.findall(vacated[(seed, p)]))
            assert not {a.lower() for b, a in pairs if b != a} & real


# --- the rewrite, contract §1 -----------------------------------------------------------


def test_every_output_is_one_complete_word_token(corpus, vacated):
    for seed in SEEDS:
        for p in P_GRID:
            for word in WORD_RE.findall(vacated[(seed, p)]):
                m = WORD_RE.fullmatch(word)
                assert m is not None and m.group(0) == word


def test_token_count_and_order_are_preserved(corpus, vacated):
    original = tokenize(corpus)
    for seed in SEEDS:
        for p in P_GRID:
            assert len(tokenize(vacated[(seed, p)])) == len(original)


def test_line_structure_is_preserved(corpus, vacated):
    """Line breaks are untouched, so the `<eos>`-per-line rule fires in the same places."""
    lines = corpus.splitlines()
    token_lines = [i for i, ln in enumerate(lines) if WORD_RE.search(ln)]
    for seed in SEEDS:
        for p in P_GRID:
            out_lines = vacated[(seed, p)].splitlines()
            assert len(out_lines) == len(lines)
            assert [i for i, ln in enumerate(out_lines) if WORD_RE.search(ln)] == token_lines


def test_non_word_characters_pass_through_byte_for_byte(corpus, vacated):
    """Whitespace, punctuation and digits are not the transform's business."""
    stripped = WORD_RE.sub("", corpus)
    for seed in SEEDS:
        for p in P_GRID:
            assert WORD_RE.sub("", vacated[(seed, p)]) == stripped


def test_p_zero_is_the_identity(corpus, vacated):
    for seed in SEEDS:
        assert vacated[(seed, 0.0)] == corpus


def test_p_one_vacates_every_eligible_type(corpus, corpus_types, vacated):
    eligible = {t for t in corpus_types if is_eligible(stem_and_suffix(t)[0])}
    for seed in SEEDS:
        assert _changed_types(corpus, vacated[(seed, 1.0)]) == eligible


def test_capitalisation_is_carried_onto_the_nonce(maps):
    params = VacancyParams(p=1.0, seed=0)
    assert maps[0].apply_word("Jack", params)[0].isupper()
    assert maps[0].apply_word("JACK", params).isupper()
    assert maps[0].apply_word("jack", params).islower()


# --- the invariance theorem, contract §7.3 ----------------------------------------------


def test_mapped_vocabulary_leaves_the_token_id_stream_unchanged(corpus, budget, maps, vacated):
    """SC-703, on the real corpus: the transform is a pure relabelling of the vocabulary, so
    a word-level model sees the identical id stream and trains bit-identically."""
    base = LexVocab(tuple(budget), source="dolch", budget_name="full")
    reference = base.encode(tokenize(corpus))
    for seed in SEEDS:
        for p in P_GRID:
            params = VacancyParams(p=p, seed=seed)
            words = map_vocab_words(budget, maps[seed], params)
            assert len(set(words)) == len(words)
            mapped = LexVocab(tuple(words), source="dolch", budget_name="full")
            assert mapped.rows == base.rows
            assert mapped.encode(tokenize(vacated[(seed, p)])) == reference


def test_map_vocab_words_preserves_order(budget, maps):
    params = VacancyParams(p=1.0, seed=0)
    words = map_vocab_words(budget, maps[0], params)
    assert len(words) == len(budget)
    for src, out in zip(budget, words):
        if is_eligible(stem_and_suffix(src)[0]):
            assert out != src
        else:
            assert out == src


def test_map_vocab_words_refuses_the_conditions_it_is_undefined_for(budget, maps):
    """Under those conditions the budget must be REBUILT from the vacated corpus; the
    coverage collapse is the measurement, and manufacturing a mapped vocabulary instead
    would silently hide it."""
    for params in (
        VacancyParams(p=1.0, seed=0, consistent=False),
        VacancyParams(p=1.0, seed=0, reveal_after=2),
    ):
        with pytest.raises(InvalidParamError):
            map_vocab_words(budget, maps[0], params)


# --- the control conditions, contract §6/§7.1 -------------------------------------------


def test_the_control_conditions_all_differ_from_each_other(corpus, maps, vacated):
    """An invariance is only worth showing against something that breaks it."""
    baseline = vacated[(0, 1.0)]
    inconsistent = vacate_text(corpus, maps[0], VacancyParams(p=1.0, seed=0, consistent=False))
    revealed = vacate_text(corpus, maps[0], VacancyParams(p=1.0, seed=0, reveal_after=2))
    flat = vacate_text(corpus, maps[0], VacancyParams(p=1.0, seed=0, match_prosody=False))
    kept = vacate_text(corpus, maps[0], VacancyParams(p=1.0, seed=0, keep=frozenset({"little"})))
    variants = {
        "baseline": baseline,
        "inconsistent": inconsistent,
        "revealed": revealed,
        "kept": kept,
    }
    for name, text in variants.items():
        assert text != corpus, name
        assert len(tokenize(text)) == len(tokenize(corpus)), name
    assert len(set(variants.values())) == len(variants)

    # `match_prosody=False` needs its own map: the flag changes what is minted, not how it
    # is applied, so re-using a prosody-matched map cannot show a difference.
    assert flat == baseline
    flat_map = build_vacancy_map(
        vacancy_domain(set(tokenize(corpus))),
        VacancyParams(p=1.0, seed=0, match_prosody=False),
    )
    flat = vacate_text(corpus, flat_map, VacancyParams(p=1.0, seed=0, match_prosody=False))
    assert flat != baseline
    assert all(syllables(n) == 1 for n in flat_map.mapping.values())


def test_inconsistent_assignment_destroys_type_identity(corpus, maps):
    """Same vacancy rate, no learnable identity — that is the point of the control."""
    params = VacancyParams(p=1.0, seed=0, consistent=False)
    text = vacate_text(corpus, maps[0], params)
    assert len(_changed_types(corpus, text)) == len(
        _changed_types(corpus, vacate_text(corpus, maps[0], VacancyParams(p=1.0, seed=0)))
    )
    images = _images(corpus, text)
    multiplied = {t for t, surfaces in images.items() if len(surfaces) > 1}
    assert len(multiplied) > 100
    assert len(set(tokenize(text))) > len(set(tokenize(corpus)))


def test_reveal_after_keeps_the_first_n_occurrences(corpus, maps):
    params = VacancyParams(p=1.0, seed=0, reveal_after=3)
    text = vacate_text(corpus, maps[0], params)
    before, after = WORD_RE.findall(corpus), WORD_RE.findall(text)
    seen: dict[str, int] = {}
    for b, a in zip(before, after):
        stem = stem_and_suffix(b)[0]
        if not is_eligible(stem):
            continue
        key = stem.lower()
        seen[key] = seen.get(key, 0) + 1
        assert (a == b) == (seen[key] <= 3)


# --- prosody, contract §6 ---------------------------------------------------------------


def test_syllable_rule_matches_the_contract():
    assert syllables("cat") == 1
    assert syllables("candle") == 2  # trailing `le` keeps its syllable
    assert syllables("make") == 1  # silent `e` dropped
    assert syllables("tree") == 1
    assert syllables("") == 1
    assert syllables("'-") == 1


def test_stress_table_lookup_is_case_sensitive_before_it_is_case_insensitive():
    assert stress("Christmas") == STRESS_TABLE["Christmas"]
    assert stress_source("Christmas") == "table"
    # Only the capitalised key is in the table, so the lower-cased form falls to the rule.
    assert stress_source("christmas") == "rule"
    assert stress_source("little") == "table"
    assert stress_source("candlestick") == "rule"


def test_minted_stress_wins_over_everything(maps):
    vmap = maps[0]
    nonce, pattern = next(iter(vmap.minted_stress.items()))
    assert stress(nonce, vmap.minted_stress) == pattern
    assert stress_source(nonce, vmap.minted_stress) == "minted"


def test_prosody_is_matched_when_asked(maps, corpus_types):
    """`match_prosody` means the nonce carries the stem's syllable count and stress."""
    vmap = maps[0]
    checked = 0
    for stem, nonce in vmap.mapping.items():
        assert vmap.minted_stress[nonce] == stress(stem)
        assert syllables(nonce) == len(stress(stem))
        checked += 1
    assert checked > 1000


def test_meter_score_is_a_fraction_and_rejects_unknown_feet():
    assert meter_score("") == 0.0
    assert 0.0 <= meter_score("Hickory dickory dock") <= 1.0
    assert meter_score("cat", "trochee") == 1.0
    with pytest.raises(InvalidParamError):
        meter_score("cat", "spondee")


# --- statistics, contract §10 -----------------------------------------------------------


def test_stats_have_exactly_the_contract_field_names(corpus, maps, vacated):
    stats = vacancy_stats(corpus, vacated[(0, 0.5)], maps[0], VacancyParams(p=0.5, seed=0))
    assert set(stats) == {
        "domainTypesTotal",
        "domainTypesEligible",
        "domainTypesVacated",
        "corpusTypesTotal",
        "corpusTypesEligible",
        "corpusTypesVacated",
        "stemsTotal",
        "stemsVacated",
        "tokensTotal",
        "tokensVacated",
        "meanSyllablesBefore",
        "meanSyllablesAfter",
        "meanAnapestBefore",
        "meanAnapestAfter",
        "stressFromTableBefore",
        "stressFromTableAfter",
        "stressFromMintedBefore",
        "stressFromMintedAfter",
        "stressFromRuleBefore",
        "stressFromRuleAfter",
        "bijective",
        "imageSize",
        "remintRounds",
    }
    assert not [k for k in stats if k.startswith("types")], "an unprefixed types* is forbidden"


def test_the_three_way_stress_split_sums_to_one_on_each_side(corpus, maps, vacated):
    """§10: token-weighted fractions, unambiguous where a single coverage number was not."""
    for seed in SEEDS:
        for p in P_GRID:
            params = VacancyParams(p=p, seed=seed)
            s = vacancy_stats(corpus, vacated[(seed, p)], maps[seed], params)
            for side in ("Before", "After"):
                total = (
                    s[f"stressFromTable{side}"]
                    + s[f"stressFromMinted{side}"]
                    + s[f"stressFromRule{side}"]
                )
                assert total == pytest.approx(1.0)
            # No English word of the corpus is also a minted form — `avoid` guarantees it.
            assert s["stressFromMintedBefore"] == 0.0
    # The split moves the right way: vacating replaces guessed stress with declared stress.
    at_zero = vacancy_stats(corpus, vacated[(0, 0.0)], maps[0], VacancyParams(p=0.0, seed=0))
    at_one = vacancy_stats(corpus, vacated[(0, 1.0)], maps[0], VacancyParams(p=1.0, seed=0))
    assert at_zero["stressFromMintedAfter"] == 0.0
    assert at_one["stressFromMintedAfter"] > 0.4
    assert at_one["stressFromRuleAfter"] < at_zero["stressFromRuleAfter"]


def test_stats_are_measured_not_asserted(corpus, corpus_types, maps, vacated):
    eligible = {t for t in corpus_types if is_eligible(stem_and_suffix(t)[0])}
    for seed in SEEDS:
        previous = -1
        for p in P_GRID:
            params = VacancyParams(p=p, seed=seed)
            stats = vacancy_stats(corpus, vacated[(seed, p)], maps[seed], params)
            assert stats["corpusTypesTotal"] == len(set(tokenize(corpus)))
            assert stats["corpusTypesEligible"] == len(eligible)
            assert stats["domainTypesTotal"] == len(maps[seed].domain)
            assert stats["stemsTotal"] == len(maps[seed].mapping)
            assert stats["tokensTotal"] == len(tokenize(corpus))
            assert stats["corpusTypesVacated"] <= stats["corpusTypesEligible"]
            assert stats["corpusTypesVacated"] > previous  # strictly graded in p
            previous = stats["corpusTypesVacated"]
            assert stats["bijective"] is True
            assert stats["remintRounds"] == maps[seed].remint_rounds
            assert 0.0 <= stats["stressFromTableAfter"] <= 1.0
        assert stats["corpusTypesVacated"] == len(eligible)
        assert stats["tokensVacated"] > 0


def test_the_two_counting_scopes_and_their_identities(corpus, corpus_types, maps, vacated):
    """§10: the scope is in the name, and the identities are what exposed the confusion.

    The domain has 22 more eligible types than the corpus — Dolch words like `funny`,
    `squirrel` and `today` that are in the budget but never appear in the text. Counting them
    in what the panel shows a reader would inflate the vacancy rate they are being shown;
    leaving them out of the diagnostic would misstate what the map covers. Hence both.
    """
    eligible = {t for t in corpus_types if is_eligible(stem_and_suffix(t)[0])}
    for seed in SEEDS:
        for p in P_GRID:
            params = VacancyParams(p=p, seed=seed)
            s = vacancy_stats(corpus, vacated[(seed, p)], maps[seed], params)
            assert s["domainTypesTotal"] == 2233
            assert s["domainTypesEligible"] == 1944
            assert s["corpusTypesTotal"] == 2211
            assert s["corpusTypesEligible"] == 1922
            assert s["domainTypesEligible"] == s["corpusTypesEligible"] + 22
            assert s["stemsTotal"] == 1680 <= s["domainTypesEligible"]
            assert s["domainTypesVacated"] >= s["corpusTypesVacated"]
            if p == 1.0:
                # `u` lands in [0, 1), so at p = 1 every eligible stem vacates.
                assert s["stemsVacated"] == s["stemsTotal"]
                assert s["domainTypesVacated"] == s["domainTypesEligible"]
                assert s["corpusTypesVacated"] == s["corpusTypesEligible"] == len(eligible)


def test_both_scopes_reproduce_the_numbers_the_typescript_stack_measured(corpus, maps, vacated):
    """Cross-stack parity, pinned as data. These are the sequences the TS side reported; if
    either stack's counting drifts, this is where it shows up."""
    expected = {
        0: {"corpus": [0, 461, 954, 1430, 1922], "domain": [0, 469, 966, 1448, 1944]},
        7: {"corpus": [0, 434, 975, 1440, 1922], "domain": [0, 440, 985, 1455, 1944]},
    }
    for seed in SEEDS:
        stats = [
            vacancy_stats(corpus, vacated[(seed, p)], maps[seed], VacancyParams(p=p, seed=seed))
            for p in P_GRID
        ]
        assert [s["corpusTypesVacated"] for s in stats] == expected[seed]["corpus"]
        assert [s["domainTypesVacated"] for s in stats] == expected[seed]["domain"]
        assert [s["tokensVacated"] for s in stats][-1] == 8202


def test_prosody_survives_full_vacancy(corpus, maps, vacated):
    """The claim is that meter is untouched, so this pins the size of the drift rather than
    asserting equality — and it quotes OUR corpus, never the source's numbers."""
    stats = vacancy_stats(corpus, vacated[(0, 1.0)], maps[0], VacancyParams(p=1.0, seed=0))
    assert abs(stats["meanSyllablesAfter"] - stats["meanSyllablesBefore"]) < 0.02
    assert abs(stats["meanAnapestAfter"] - stats["meanAnapestBefore"]) < 0.02
    # And the honesty number: most of the corpus is not in the 61-entry hand table.
    assert 0.0 < stats["stressFromTableBefore"] < 0.15
    assert stats["stressFromRuleBefore"] > 0.85


def test_stats_reject_texts_that_do_not_align(corpus, maps):
    with pytest.raises(ComputeError):
        vacancy_stats(corpus, corpus + " extra", maps[0], VacancyParams(p=1.0, seed=0))


# --- parameter validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": -0.01},
        {"p": 1.5},
        {"p": float("nan")},
        {"p": "0.5"},
        {"seed": 1.5},
        {"reveal_after": -1},
        {"keep": frozenset({3})},
    ],
)
def test_bad_parameters_raise_invalid_param_error(kwargs):
    with pytest.raises(InvalidParamError):
        VacancyParams(**kwargs)


def test_defaults_are_the_contracts_defaults():
    params = VacancyParams()
    assert (params.p, params.seed, params.consistent) == (0.0, 0, True)
    assert (params.match_prosody, params.reveal_after, params.keep) == (True, 0, frozenset())
    assert params.keep_set == FUNCTION_WORDS
    assert VacancyParams(keep=frozenset({"Little"})).keep_set == FUNCTION_WORDS | {"little"}


def test_a_stem_outside_the_maps_domain_is_a_compute_error(maps):
    """The map's domain must include the budget's words as well as the corpus's types."""
    with pytest.raises(ComputeError):
        maps[0].apply_word("zzzqqqx", VacancyParams(p=1.0, seed=0))
