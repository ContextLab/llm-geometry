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
    MAX_SEED,
    REMINT_SALT_STRIDE,
    SPLIT_EXCEPTIONS,
    STRESS_TABLE,
    SUFFIXES,
    VacancyMap,
    VacancyParams,
    _mint,  # private: the salt semantics of §5.5/§5.8 are a contract detail, so they are pinned
    build_vacancy_map,
    is_eligible,
    is_vacatable,
    map_vocab_words,
    meter_score,
    stem_and_suffix,
    stress,
    stress_source,
    swap_pools,
    syllables,
    type_counts,
    vacancy_domain,
    vacancy_stats,
    vacancy_u,
    vacate_text,
)
from llm_geometry.lex.vocab import WORD_RE, LexVocab, frequency_budget, tokenize

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
    eligible = {t for t in corpus_types if is_vacatable(t)}
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
    eligible = {t for t in corpus_types if is_vacatable(t)}
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
        if is_vacatable(src):
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


def test_the_inconsistent_mint_key_never_reaches_the_prosody_lookup(corpus, maps):
    """§5.8. The key `f"{stem}#{idx}"` feeds the byte stream and the uniqueness check ONLY.

    Let it reach the prosody lookup and the pattern becomes `stress("little#0") == "10"`
    instead of `stress("little") == "100"`, so `Little` mints as a disyllable. §7.1 says the
    nonce carries THE STEM'S syllable count and stress; a mint key is not a word, and the
    spelling rule has no business being asked about one. Caught by the golden fixture, not by
    either stack's tests — hence this one.
    """
    assert stress("little") == "100" != stress("little#0") == "10"
    # At the minter: the key drives the byte stream, `stem` drives the pattern.
    correct = _mint("little#0", 0, True, frozenset(), stem="little")
    as_if_key_were_the_stem = _mint("little#0", 0, True, frozenset())
    assert correct[1] == "100" and syllables(correct[0]) == 3
    assert as_if_key_were_the_stem[1] == "10"
    assert correct[0] != as_if_key_were_the_stem[0]

    # End to end: every minted form still carries its stem's syllable count, occurrence by
    # occurrence. Restricted to un-suffixed words so the seam repair of §5.7 cannot muddy
    # the comparison — it is covered by its own test.
    params = VacancyParams(p=1.0, seed=0, consistent=False)
    text = vacate_text(corpus, maps[0], params)
    checked = 0
    for src, out in zip(WORD_RE.findall(corpus), WORD_RE.findall(text)):
        stem, suffix = stem_and_suffix(src)
        if src == out or suffix:
            continue
        assert maps[0].minted_stress[out.lower()] == stress(stem.lower())
        assert syllables(out.lower()) == syllables(stem.lower())
        checked += 1
    assert checked > 4000


def test_condition_b_applies_to_the_per_occurrence_path_too(corpus, maps):
    """§5.8, and the `tak` case that exposed it.

    Condition B — no minted form may equal a domain type — was enforced when building the
    map and NOT on the `consistent=False` minting path. The gap is observable: at seed 7,
    `p = 1`, the stem `tak` (of `taking`) minted the nonce `tak`, so `Taking -> Taking` and
    one token silently failed to vacate. `corpus_types_vacated` read 1921 against the
    consistent path's 1918 and `tokens_vacated` one short of 8125.

    §7.1 denies this control a STABILITY property — that is about a nonce being reused
    across occurrences — and it does not license a word surviving the transform. A control
    whose vacancy rate is not the stated rate is not a control, so a per-occurrence nonce
    must equal neither a domain type nor the stem it replaces, under the same re-mint loop.
    """
    # `tak` is a stem, not a type, which is exactly why the domain did not already forbid
    # it: the domain is the corpus's TYPES (`taking`, `takes`, …) plus the Dolch list.
    assert stem_and_suffix("taking") == ("tak", "ing")
    assert "tak" not in maps[7].domain and "taking" in maps[7].domain

    for seed in (0, 7):
        params = VacancyParams(p=1.0, seed=seed, consistent=False)
        # A fresh map per condition: `consistent=False` writes to `minted_stress`.
        vmap = build_vacancy_map(vacancy_domain(set(tokenize(corpus))), params)
        text = vacate_text(corpus, vmap, params)
        stats = vacancy_stats(corpus, text, vmap, params)
        # At `p = 1` every eligible type vacates (§10) — in this control exactly as in the
        # mapped condition, which is the whole claim.
        # 1918 / 8125, not 1922 / 8202: §2.2's whole-word test stopped the suffix splitter
        # breaking `after`, `this`, `does` and `always` out of the closed class.
        assert stats["corpusTypesVacated"] == stats["corpusTypesEligible"] == 1918, seed
        assert stats["tokensVacated"] == 8125, seed

        # No token survives the transform, and none survives as itself least of all.
        for src, out in zip(WORD_RE.findall(corpus), WORD_RE.findall(text)):
            if is_vacatable(src):
                assert out.lower() != src.lower(), (seed, src)

    # And the mechanism, directly: forbidding the stem is not implied by forbidding the
    # domain, so `_mint` must be handed it. The losing draw is the FOURTH occurrence of
    # `tak` in document order — `tak#3` at seed 7 mints `tak` on its first attempt.
    assert _mint("tak#3", 7, True, frozenset(), stem="tak")[0] == "tak"
    assert _mint("tak#3", 7, True, frozenset({"tak"}), stem="tak")[0] != "tak"


def test_reveal_after_keeps_the_first_n_occurrences(corpus, maps):
    params = VacancyParams(p=1.0, seed=0, reveal_after=3)
    text = vacate_text(corpus, maps[0], params)
    before, after = WORD_RE.findall(corpus), WORD_RE.findall(text)
    seen: dict[str, int] = {}
    for b, a in zip(before, after):
        stem = stem_and_suffix(b)[0]
        if not is_vacatable(b):
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
    eligible = {t for t in corpus_types if is_vacatable(t)}
    for seed in SEEDS:
        previous = -1
        for p in P_GRID:
            params = VacancyParams(p=p, seed=seed)
            stats = vacancy_stats(corpus, vacated[(seed, p)], maps[seed], params)
            assert stats["corpusTypesTotal"] == len(set(tokenize(corpus)))
            assert stats["corpusTypesEligible"] == len(eligible)
            assert stats["domainTypesTotal"] == len(maps[seed].domain)
            assert stats["stemsTotal"] == len(maps[seed].stems)
            assert stats["tokensTotal"] == len(tokenize(corpus))
            assert stats["corpusTypesVacated"] <= stats["corpusTypesEligible"]
            assert stats["corpusTypesVacated"] > previous  # strictly graded in p
            previous = stats["corpusTypesVacated"]
            assert stats["bijective"] is True
            assert stats["remintRounds"] == maps[seed].remint_rounds
            assert 0.0 <= stats["stressFromTableAfter"] <= 1.0
        assert stats["corpusTypesVacated"] == len(eligible)
        assert stats["tokensVacated"] > 0


def test_corpus_types_vacated_is_measured_from_the_texts_not_from_map_membership(corpus, maps):
    """§10. A type counts as vacated iff at least one of its occurrences ACTUALLY changed.

    Under `reveal_after` the two readings diverge sharply: a type whose every occurrence falls
    inside the reveal window is still in the map and still has `u(stem) < p`, so asking the map
    over-reports roughly 2x. The panel would then tell a reader that 1334 types are vacant in a
    text where 663 of them are printed in plain English.
    """
    params = VacancyParams(p=0.7, seed=0, reveal_after=2)
    text = vacate_text(corpus, maps[0], params)
    stats = vacancy_stats(corpus, text, maps[0], params)

    measured = len(_changed_types(corpus, text))
    by_membership = len(
        {
            t
            for t in set(tokenize(corpus))
            if is_vacatable(t) and vacancy_u(stem_and_suffix(t)[0], params.seed) < params.p
        }
    )
    assert stats["corpusTypesVacated"] == measured == 663
    assert by_membership == 1334  # the map-membership reading, over-reporting 2.01x
    # Every type the two readings disagree about really is still English in the output.
    unchanged = {w for w, surfaces in _images(corpus, text).items() if surfaces == {w}}
    assert len(unchanged & {t.lower() for t in tokenize(corpus)}) >= by_membership - measured
    assert not (unchanged & _changed_types(corpus, text))
    # tokensVacated is measured the same way, and the reveal window is why it drops.
    without_reveal = VacancyParams(p=0.7, seed=0)
    assert (
        stats["tokensVacated"]
        < vacancy_stats(
            corpus, vacate_text(corpus, maps[0], without_reveal), maps[0], without_reveal
        )["tokensVacated"]
    )


def test_the_two_counting_scopes_and_their_identities(corpus, corpus_types, maps, vacated):
    """§10: the scope is in the name, and the identities are what exposed the confusion.

    The domain has 22 more eligible types than the corpus — Dolch words like `funny`,
    `squirrel` and `today` that are in the budget but never appear in the text. Counting them
    in what the panel shows a reader would inflate the vacancy rate they are being shown;
    leaving them out of the diagnostic would misstate what the map covers. Hence both.
    """
    eligible = {t for t in corpus_types if is_vacatable(t)}
    for seed in SEEDS:
        for p in P_GRID:
            params = VacancyParams(p=p, seed=seed)
            s = vacancy_stats(corpus, vacated[(seed, p)], maps[seed], params)
            assert s["domainTypesTotal"] == 2233
            assert s["domainTypesEligible"] == 1940
            assert s["corpusTypesTotal"] == 2211
            assert s["corpusTypesEligible"] == 1918
            assert s["domainTypesEligible"] == s["corpusTypesEligible"] + 22
            assert s["stemsTotal"] == 1676 <= s["domainTypesEligible"]
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
        0: {"corpus": [0, 461, 951, 1427, 1918], "domain": [0, 469, 963, 1445, 1940]},
        7: {"corpus": [0, 433, 972, 1436, 1918], "domain": [0, 439, 982, 1451, 1940]},
    }
    for seed in SEEDS:
        stats = [
            vacancy_stats(corpus, vacated[(seed, p)], maps[seed], VacancyParams(p=p, seed=seed))
            for p in P_GRID
        ]
        assert [s["corpusTypesVacated"] for s in stats] == expected[seed]["corpus"]
        assert [s["domainTypesVacated"] for s in stats] == expected[seed]["domain"]
        assert [s["tokensVacated"] for s in stats][-1] == 8125


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


# --- `forbidden`, contract §5.8 ---------------------------------------------------------


def test_forbidden_is_stored_and_keeps_superseded_remint_nonces(maps, domain):
    """§5.8: `forbidden` is STORED, not rebuilt as `domain | mapping.values()`.

    THE CASE THAT DISTINGUISHES THEM, and the only one the shipped corpus produces: at
    seed 7 the stem `hang` first minted `wak`, whose surface `wak` + `ed` is the real English
    word `waked`; condition B rejected it and the re-mint returned `smeeg`. `wak` is now no
    stem's nonce, so a reconstruction from `mapping.values()` drops it — but it must stay
    forbidden, because it was rejected for cause and the `consistent=False` control draws
    against this very set.
    """
    seed7 = maps[7]
    assert seed7.mapping["hang"] == "smeeg"
    assert seed7.remint_rounds == 1
    assert "wak" in seed7.forbidden
    assert "wak" not in set(seed7.mapping.values())  # exactly what a rebuild would lose
    assert "waked" in seed7.domain  # ... and why it was superseded

    # The rest of the field, so "stored" cannot decay into "stored but wrong".
    for seed, vmap in maps.items():
        assert vmap.domain <= vmap.forbidden, seed
        assert set(vmap.mapping.values()) <= vmap.forbidden, seed
    assert maps[0].forbidden == maps[0].domain | set(maps[0].mapping.values())  # 0 re-mints
    assert maps[7].forbidden == maps[7].domain | set(maps[7].mapping.values()) | {"wak"}


def test_the_inconsistent_control_draws_against_the_stored_forbidden_set(corpus, maps):
    """The per-occurrence path must never hand out a superseded nonce (§5.8)."""
    params = VacancyParams(p=1.0, seed=7, consistent=False)
    text = vacate_text(corpus, maps[7], params)
    assert "wak" not in {w.lower() for w in WORD_RE.findall(text)}


# --- the swap control, contract §8.3 / §5.2a --------------------------------------------


@pytest.fixture(scope="module")
def counts(corpus: str) -> dict[str, int]:
    """The corpus's per-type occurrence counts — the frequency source swap ranks by."""
    return type_counts(tokenize(corpus))


@pytest.fixture(scope="module")
def swap_maps(domain: list[str], counts: dict[str, int]) -> dict[int, VacancyMap]:
    return {
        seed: build_vacancy_map(domain, VacancyParams(seed=seed, mint="swap"), counts)
        for seed in SEEDS
    }


def test_swap_replaces_stems_with_real_corpus_words(swap_maps, corpus_types, budget):
    """The whole point of the control: every replacement is a word English already had."""
    real = {t.lower() for t in corpus_types} | {w.lower() for w in budget}
    for seed, vmap in swap_maps.items():
        assert vmap.mapping, seed
        assert set(vmap.mapping.values()) <= real, seed
        # ... and no stem keeps its own form, which would be a word that failed to vacate.
        assert all(stem != word for stem, word in vmap.mapping.items()), seed


def test_swap_needs_the_frequency_counts_and_says_so(domain):
    """No silent fallback to an alphabetical rank, which would be frequency in name only."""
    with pytest.raises(InvalidParamError):
        build_vacancy_map(domain, VacancyParams(mint="swap"))


def test_swap_refuses_the_inconsistent_control(domain):
    """1676 open-class stems against 8125 vacated tokens — there is no supply (§8.3)."""
    with pytest.raises(InvalidParamError):
        VacancyParams(mint="swap", consistent=False)


def test_an_unknown_mint_strategy_is_rejected():
    with pytest.raises(InvalidParamError):
        VacancyParams(mint="real-words")


def test_counts_do_not_reach_the_nonce_map(domain, counts, maps):
    """The nonce map stays a pure function of `(domain, seed, match_prosody)` (§5.2)."""
    for seed in SEEDS:
        with_counts = build_vacancy_map(domain, VacancyParams(seed=seed), counts)
        assert with_counts.mapping == maps[seed].mapping
        assert with_counts.remint_rounds == maps[seed].remint_rounds


def test_swap_is_stable_in_seed_and_stem(domain, counts, swap_maps, corpus):
    """SC-702 for swap: the map is built once, independently of `p` (§5.6)."""
    for seed in SEEDS:
        again = build_vacancy_map(domain, VacancyParams(p=1.0, seed=seed, mint="swap"), counts)
        assert again.mapping == swap_maps[seed].mapping
    # ... and a stem's surface is byte-identical at every `p` at which it is vacated.
    forms: dict[str, set[str]] = {}
    for p in P_GRID:
        params = VacancyParams(p=p, seed=0, mint="swap")
        for stem in ("little", "moon", "crown"):
            if vacancy_u(stem, 0) < p:
                forms.setdefault(stem, set()).add(swap_maps[0].apply_word(stem, params))
    assert forms and all(len(v) == 1 for v in forms.values())


def test_swap_is_nested_in_p(corpus, swap_maps):
    """SC-701 for swap: `u(stem) < p` is untouched, so the vacated sets are still nested."""
    previous: set[str] = set()
    for p in P_GRID:
        text = vacate_text(corpus, swap_maps[0], VacancyParams(p=p, seed=0, mint="swap"))
        changed = _changed_types(corpus, text)
        assert previous <= changed, p
        previous = changed


def test_swap_is_a_bijection_of_the_domain_at_full_vacancy(swap_maps):
    """A + B₁ of §5.2a, which is what the invariance theorem needs at `p = 1`."""
    for seed, vmap in swap_maps.items():
        assert vmap.bijective, seed
        assert vmap.image_size == vmap.type_count, seed
        assert vmap.remint_rounds == 0, seed
        assert vmap.injective_at_every_p is False, seed


def test_swap_satisfies_the_invariance_theorem_where_it_is_defined(corpus, budget, swap_maps):
    """SC-703 for swap, at the `p` where §5.2a proves a swap map CAN be injective.

    At `p in {0, 1}` the id stream is element-for-element identical to the untransformed
    stream, exactly as it is for `mint="nonce"` — the tiny model is exactly as blind to a
    real-word swap as to an invented form, which is the check that the control is right.
    """
    base = LexVocab(tuple(budget), source="dolch", budget_name="full")
    reference = base.encode(tokenize(corpus))
    for seed in SEEDS:
        for p in (0.0, 1.0):
            params = VacancyParams(p=p, seed=seed, mint="swap")
            text = vacate_text(corpus, swap_maps[seed], params)
            words = map_vocab_words(budget, swap_maps[seed], params)
            assert len(set(words)) == len(words)
            mapped = LexVocab(tuple(words), source="dolch", budget_name="full")
            assert mapped.rows == base.rows
            assert mapped.encode(tokenize(text)) == reference


def test_swap_refuses_the_mapped_vocabulary_at_intermediate_p(budget, swap_maps):
    """§5.2a: no `p`-stable swap into the domain is injective at `0 < p < 1`, so the mapped
    vocabulary does not exist there and is refused rather than silently duplicated."""
    for p in (0.25, 0.5, 0.75):
        with pytest.raises(InvalidParamError):
            map_vocab_words(budget, swap_maps[0], VacancyParams(p=p, seed=0, mint="swap"))


def test_why_swap_cannot_be_injective_at_intermediate_p(corpus, swap_maps):
    """The theorem of §5.2a, measured rather than merely proved.

    A vacated type lands on a real English word; at intermediate `p` that word's own
    occurrences may not have moved, so two source types share one surface. This pins the
    collision count so the refusal above can never be mistaken for over-caution — if a future
    change makes swap injective at `p = 0.5`, this test fails and the contract is wrong.
    """
    params = VacancyParams(p=0.5, seed=0, mint="swap")
    vmap = swap_maps[0]
    images: dict[str, str] = {}
    collisions = 0
    for t in sorted(vmap.domain):
        image = vmap.apply_word(t, params).lower()
        if image in images:
            collisions += 1
        images[image] = t
    assert collisions > 0
    # ... whereas at full vacancy there are none, which is what B₁ buys.
    full = VacancyParams(p=1.0, seed=0, mint="swap")
    assert len({vmap.apply_word(t, full).lower() for t in vmap.domain}) == len(vmap.domain)


def test_swap_honours_match_prosody(domain, counts, swap_maps):
    """`matchProsody` is a real filter under swap, and — unlike minting — it is a filter over
    a FINITE pool, so it cannot always be honoured.

    Measured at seed 0: 1918 of 1940 types get a stress-matched real word, i.e. 98.9 %. The
    other 22 fall through to the deterministic completion of §8.3 stage 2 because their
    pattern is rare inside their own suffix class and the few words carrying it are already
    used — the relaxation of §5.5, applied to a pool that can genuinely run out. Minting has
    no such limit, which is exactly the difference between inventing a form and borrowing
    one, so the bound here is the measurement rather than a claim of exactness.
    """
    matched = swap_maps[0].mapping
    hits = sum(1 for stem, word in matched.items() if stress(word) == stress(stem))
    assert hits / len(matched) > 0.9
    flat = build_vacancy_map(
        domain, VacancyParams(seed=0, mint="swap", match_prosody=False), counts
    )
    assert flat.mapping != matched
    flat_hits = sum(1 for stem, word in flat.mapping.items() if stress(word) == stress(stem))
    assert flat_hits < hits


def test_swap_replacements_are_not_registered_as_minted_stress(swap_maps):
    """They are real English words: their stress comes from the table or the rule, so
    `stressFromMinted` must stay 0 on both sides of a swap (§8.3)."""
    for seed, vmap in swap_maps.items():
        assert vmap.minted_stress == {}, seed


def test_swap_statistics_report_the_same_vacancy_rate_as_nonce(corpus, maps, swap_maps):
    """The control holds the vacancy rate fixed and changes only what replaces the word."""
    for seed in SEEDS:
        nonce = vacancy_stats(
            corpus,
            vacate_text(corpus, maps[seed], VacancyParams(p=1.0, seed=seed)),
            maps[seed],
            VacancyParams(p=1.0, seed=seed),
        )
        params = VacancyParams(p=1.0, seed=seed, mint="swap")
        swap = vacancy_stats(
            corpus, vacate_text(corpus, swap_maps[seed], params), swap_maps[seed], params
        )
        for field in ("corpusTypesVacated", "tokensVacated", "stemsVacated", "stemsTotal"):
            assert swap[field] == nonce[field], (seed, field)


def test_sc703_over_the_full_grid_for_both_mint_strategies(corpus, domain, counts):
    """SC-703 as the spec states it, for BOTH minting strategies.

    The grid the spec names: all five Dolch budgets plus a frequency budget,
    ``p in {0, 0.25, 0.5, 0.75, 1}``, ``seed in {0, 7}``, both ``match_prosody`` settings —
    120 cases.

    ``mint="nonce"`` passes all 120: condition B keeps every image out of the domain, so the
    map is injective at every `p` and the id stream is element-for-element unchanged.

    ``mint="swap"`` passes 48 — every case at `p in {0, 1}` — and REFUSES the other 72. That
    is not a weaker test of the same claim; it is the claim §5.2a proves. A swap map's images
    are domain types, so at intermediate `p` a vacated type can land on one that has not
    moved, and no `p`-stable map avoids it short of the identity. Where a swap map can be
    injective at all it is exactly as invisible to the model as an invented form, which is
    what makes the control trustworthy. The counts are asserted, so neither number can drift
    without this failing.
    """
    budgets = {name: list(dolch_budget(name)) for name in DOLCH_ORDER}
    budgets["frequency-top300"] = frequency_budget(corpus, 300)
    assert len(budgets) == 6

    passed = {"nonce": 0, "swap": 0}
    refused = {"nonce": 0, "swap": 0}
    for mint in ("nonce", "swap"):
        for seed in SEEDS:
            for match_prosody in (True, False):
                vmap = build_vacancy_map(
                    domain,
                    VacancyParams(seed=seed, mint=mint, match_prosody=match_prosody),
                    counts,
                )
                for p in P_GRID:
                    params = VacancyParams(p=p, seed=seed, mint=mint, match_prosody=match_prosody)
                    text = vacate_text(corpus, vmap, params)
                    for words in budgets.values():
                        base = LexVocab(tuple(words), source="dolch", budget_name="full")
                        try:
                            mapped_words = map_vocab_words(words, vmap, params)
                        except InvalidParamError:
                            refused[mint] += 1
                            continue
                        assert len(set(mapped_words)) == len(mapped_words), (mint, seed, p)
                        mapped = LexVocab(tuple(mapped_words), source="dolch", budget_name="full")
                        assert mapped.rows == base.rows
                        assert mapped.encode(tokenize(text)) == base.encode(tokenize(corpus))
                        passed[mint] += 1

    assert (passed["nonce"], refused["nonce"]) == (120, 0)
    assert (passed["swap"], refused["swap"]) == (48, 72)


# --- §2.2 test 1: the closed class is never split open ----------------------------------


def test_the_suffix_splitter_breaks_function_words_open_and_test_1_stops_it():
    """The defect `is_vacatable` exists for, at the level of the two predicates.

    `stem_and_suffix` is a spelling heuristic, so it splits seven function words into stems
    that are not function words. The stem-level test therefore passes them, and before the
    whole-word test was applied they were vacated: at seed 0 `after` came out as `kitser`.
    """
    split_open = sorted(w for w in FUNCTION_WORDS if is_eligible(stem_and_suffix(w)[0]))
    assert split_open == ["after", "always", "does", "during", "having", "this", "unless"]
    for word in split_open:
        assert is_eligible(stem_and_suffix(word)[0]) is True, word  # test 2 alone passes it
        assert is_vacatable(word) is False, word  # test 1 stops it
    assert stem_and_suffix("after") == ("aft", "er")
    assert stem_and_suffix("this") == ("thi", "s")


def test_the_closed_class_survives_the_transform_under_both_mints(corpus, domain, counts):
    """§0's claim, checked on the text: every function-word TOKEN is character-identical.

    Both strategies, at full vacancy, where nothing eligible is left standing — so any
    function word that moved, moved because the transform was allowed to touch it.
    """
    before = WORD_RE.findall(corpus)
    for mint in ("nonce", "swap"):
        params = VacancyParams(p=1.0, seed=0, mint=mint)
        vmap = build_vacancy_map(domain, params, counts)
        after = WORD_RE.findall(vacate_text(corpus, vmap, params))
        assert len(before) == len(after)
        moved = sorted({b for b, a in zip(before, after) if b.lower() in FUNCTION_WORDS and a != b})
        assert moved == [], (mint, moved[:10])


# --- §8.3: every swap form is a REAL word -----------------------------------------------


def test_every_swap_form_is_a_real_word_of_the_domain(corpus, domain, counts):
    """The control's defining property, on the real text: *every form known*.

    The first implementation drew a replacement for the STEM and re-attached the SOURCE
    word's suffix, so the pool's own inflected types produced forms that are not words:
    `jumped -> wented`, `leaping -> thying`, `huffed -> sacksed`, `after -> kitser`,
    `November -> huffeder`. Measured over the six shipped Architecture passages at
    `p = 1, seed = 0`, 165 of 776 vacated words (21.3 %) were not words of their own domain.
    The map now runs type -> type inside one suffix class, so the image IS a domain type.
    """
    for seed in SEEDS:
        vmap = build_vacancy_map(domain, VacancyParams(p=1.0, seed=seed, mint="swap"), counts)
        for p in (0.5, 1.0):
            params = VacancyParams(p=p, seed=seed, mint="swap")
            before = WORD_RE.findall(corpus)
            after = WORD_RE.findall(vacate_text(corpus, vmap, params))
            unreal = sorted(
                {
                    a.lower()
                    for b, a in zip(before, after)
                    if a != b and a.lower() not in vmap.domain
                }
            )
            assert unreal == [], (seed, p, unreal[:10])


def test_a_swap_replacement_keeps_the_inflection_of_the_word_it_replaces(domain, counts):
    """§8.3: the image is drawn from the SAME suffix class, so `-ed` replaces `-ed`.

    That is what keeps the swap arm's morphology as parseable as the nonce arm's — the two
    variants must differ in the lexicon, not in how much syntax survives.
    """
    params = VacancyParams(p=1.0, seed=0, mint="swap")
    vmap = build_vacancy_map(domain, params, counts)
    wrong_class = [
        (t, image)
        for t, image in vmap.mapping.items()
        if stem_and_suffix(t)[1] != stem_and_suffix(image)[1]
    ]
    assert wrong_class == []
    # The words the red team quoted, all of them now real and all still inflected.
    for word in ("jumped", "leaping", "huffed", "november"):
        image = vmap.apply_word(word, params).lower()
        assert image in vmap.domain, (word, image)
        assert stem_and_suffix(image)[1] == stem_and_suffix(word)[1], (word, image)


def test_a_swap_class_that_cannot_be_permuted_is_refused():
    """A class with one member has no non-identity image, and that raises rather than
    leaving a word where it was. Every non-bare class on a domain this small is a singleton,
    so they are merged into the bare class (§8.3); the bare class has nothing to merge with.
    """
    keep = VacancyParams().keep_set
    assert set(swap_pools(["cat", "dog"], {"cat": 2, "dog": 1}, keep)) == {""}
    with pytest.raises(ComputeError, match="cannot be permuted"):
        build_vacancy_map(["cat"], VacancyParams(mint="swap"), {"cat": 1})


def test_a_singleton_suffix_class_is_merged_into_the_bare_class():
    """The passage-sized case, on real text: most of the six shipped Architecture passages
    have a suffix class with exactly one member, so §8.3's merge rule is the common path and
    not a corner. What it must never do is emit a form that is not a domain word.
    """
    from llm_geometry.arch.vacancy_score import default_passages

    keep = VacancyParams().keep_set
    merged = 0
    for passage in default_passages():
        tokens = tokenize(passage)
        dom = vacancy_domain(tokens)
        counts = type_counts(tokens)
        classes = {stem_and_suffix(t)[1] for t in dom if is_vacatable(t, keep)}
        pools = swap_pools(dom, counts, keep)
        if classes - set(pools):
            merged += 1
        params = VacancyParams(p=1.0, seed=0, mint="swap")
        vmap = build_vacancy_map(dom, params, counts)
        for t, image in vmap.mapping.items():
            assert image in vmap.domain, (t, image)
    assert merged >= 4


# --- §4: the seed's domain ---------------------------------------------------------------


def test_seed_is_bounded_to_exactly_representable_integers():
    """Beyond 2**53 Python stringifies the exact integer and JavaScript the rounded double,
    so the two stacks hash different strings and build different maps — with nothing raised
    in either. The bound is enforced, not clamped: a clamp would use a seed nobody asked for.
    """
    assert MAX_SEED == 2**53 - 1
    assert str(MAX_SEED) == "9007199254740991"
    for seed in (0, 7, -7, MAX_SEED, -MAX_SEED):
        assert 0.0 <= vacancy_u("little", seed) < 1.0
        assert VacancyParams(seed=seed).seed == seed
    for seed in (MAX_SEED + 1, -(MAX_SEED + 1), 2**53 + 1, 12345678901234567890):
        with pytest.raises(InvalidParamError, match="seed must"):
            VacancyParams(seed=seed)
        with pytest.raises(InvalidParamError, match="seed must"):
            vacancy_u("little", seed)


# --- §5.2a: the collision count, with its counting definition ----------------------------


def test_swap_collisions_at_intermediate_p_are_measured_under_one_definition(swap_maps):
    """§5.2a's measured claim, with the definition and the seed stated (both were missing).

    LOST IMAGE SLOTS: `|domain| - |{T_p(t) : t in domain}|`, i.e. how many of the domain's
    2 233 slots the type map at `p` fails to reach. That is the number the panel measures
    live, and it is the one the contract now quotes.
    """
    expected = {
        0: {0.0: 0, 0.25: 349, 0.5: 484, 0.75: 364, 1.0: 0},
        7: {0.0: 0, 0.25: 336, 0.5: 475, 0.75: 372, 1.0: 0},
    }
    for seed, vmap in swap_maps.items():
        assert len(vmap.domain) == 2233
        for p, lost in expected[seed].items():
            params = VacancyParams(p=p, seed=seed, mint="swap")
            images = {vmap.apply_word(t, params).lower() for t in vmap.domain}
            assert len(vmap.domain) - len(images) == lost, (seed, p)


# --- §10: one rule for `domainTypesVacated` ----------------------------------------------


def test_domain_types_vacated_is_map_membership_under_both_mints(corpus, domain, counts):
    """§10: one rule, in both stacks — vacatable, and `u(stem) < p`.

    TypeScript additionally required the image to DIFFER from the type. The two readings
    coincide, because conditions B and B₁ both forbid an image equal to its own source — and
    that coincidence is asserted here rather than assumed, so a regression in B shows up as a
    failure of the DEFINITION rather than as two stacks quietly reporting two numbers.
    """
    keep = VacancyParams().keep_set
    for mint in ("nonce", "swap"):
        vmap = build_vacancy_map(domain, VacancyParams(p=1.0, seed=0, mint=mint), counts)
        for p in P_GRID:
            params = VacancyParams(p=p, seed=0, mint=mint)
            stats = vacancy_stats(corpus, vacate_text(corpus, vmap, params), vmap, params)
            by_membership = sum(
                1
                for t in vmap.domain
                if is_vacatable(t, keep) and vacancy_u(stem_and_suffix(t)[0], 0) < p
            )
            by_image = sum(
                1
                for t in vmap.domain
                if is_vacatable(t, keep)
                and vacancy_u(stem_and_suffix(t)[0], 0) < p
                and vmap.apply_word(t, params).lower() != t
            )
            assert stats["domainTypesVacated"] == by_membership == by_image, (mint, p)
