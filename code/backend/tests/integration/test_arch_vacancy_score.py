"""The pretrained arm, end to end on a REAL model (contract §8, FR-717/719/719a).

Runtime cost: `gpt2` (124M, float32, CPU) over ONE ~250-word passage is three forward
passes of ~350 tokens — about 1 s once the weights are cached, ~10 s on a cold cache
including the download. The measurement of §8.3a pools six passages; one is enough to
assert every structural property, and the six-passage run is what the panel does.
"""

from __future__ import annotations

import math

import pytest

from llm_geometry.api.encoding import jsonable_6sig
from llm_geometry.arch.vacancy_score import (
    VARIANTS,
    default_passages,
    vacancy_score,
)
from llm_geometry.errors import InvalidParamError

MODEL = "gpt2"  # the smallest curated model, and one of the two §8.3a measured


@pytest.fixture(scope="module")
def scored() -> dict:
    return vacancy_score(MODEL, default_passages(count=1), p=1.0, seed=0)


def test_reports_every_field_of_section_8_1(scored: dict) -> None:
    assert [v["id"] for v in scored["variants"]] == list(VARIANTS)
    for variant in scored["variants"]:
        stats = variant["pooled"]
        assert set(stats) == {
            "nllPreserved",
            "nllAll",
            "bitsPerChar",
            "nTokens",
            "nPreservedTokens",
            "nChars",
        }
        assert stats["nTokens"] > 0
        assert stats["nPreservedTokens"] > 0
        assert stats["nllPreserved"] > 0
        # bitsPerChar is the passage's total surprisal per character, so it must agree
        # with nllAll·nTokens/(ln2·nChars) — the definition, not an independent number.
        assert stats["bitsPerChar"] == pytest.approx(
            stats["nllAll"] * stats["nTokens"] / (math.log(2) * stats["nChars"]), rel=1e-4
        )
    assert scored["dtype"] == "float32"
    assert scored["stack"] == "backend"
    assert scored["alignment"]["verified"] is True
    assert scored["alignment"]["unit"] == "utf8_bytes"


def test_the_same_scaffolding_is_scored_in_every_variant(scored: dict) -> None:
    """The preserved token sets correspond one-for-one, which is what makes the
    differences paired and the comparison meaningful at all."""
    counts = {v["id"]: v["pooled"]["nPreservedTokens"] for v in scored["variants"]}
    assert len(set(counts.values())) == 1, counts
    # …while the variants as a whole tokenize differently: nonce forms fragment, which
    # is the residual the "unknown form" difference is an upper bound because of.
    tokens = {v["id"]: v["pooled"]["nTokens"] for v in scored["variants"]}
    assert tokens["nonce"] > tokens["swap"], tokens


def test_the_decomposition_is_labelled_and_adds_up(scored: dict) -> None:
    by_id = {d["id"]: d for d in scored["differences"]}
    assert set(by_id) == {"wrong_content", "unknown_form", "total"}

    wrong = by_id["wrong_content"]
    form = by_id["unknown_form"]
    total = by_id["total"]

    assert wrong["expr"] == "nll(swap) − nll(english)"
    assert form["expr"] == "nll(nonce) − nll(swap)"
    assert total["expr"] == "nll(nonce) − nll(english)"
    # FR-719a: the conflated difference is never a headline.
    assert wrong["headline"] and form["headline"]
    assert total["headline"] is False
    assert form["upperBound"] is True and "UPPER BOUND" in form["note"]

    # Paired means over the same tokens are exactly additive — a real invariant, not a
    # tolerance: if it ever fails, the three differences were not computed over one
    # aligned token set and none of them means what its label says.
    assert total["nats"] == pytest.approx(wrong["nats"] + form["nats"], abs=1e-9)
    assert wrong["nPairs"] == form["nPairs"] == total["nPairs"]
    for d in (wrong, form, total):
        assert d["se"] > 0


def test_both_costs_are_real_and_wrong_content_dominates(scored: dict) -> None:
    """SC-707/707b: the measured result, asserted as a sign and an ordering.

    Not asserted as a fixed value — it is a real model on real text and would be a
    brittle golden — but the ORDER is the finding: most of the damage is saying the
    wrong thing, and only a minority of it is the form being unknown.
    """
    by_id = {d["id"]: d for d in scored["differences"]}
    wrong = by_id["wrong_content"]["nats"]
    form = by_id["unknown_form"]["nats"]
    total = by_id["total"]["nats"]
    assert wrong > 0, "swapping in real but wrong words must cost something"
    assert form > 0, "nonce forms must cost more than known wrong words"
    assert wrong > form, (wrong, form)
    assert 0.4 < total < 2.0, total


def test_the_tiny_arms_exact_zero_travels_with_the_number(scored: dict) -> None:
    """FR-719: the pretrained delta is only interpretable beside the exact 0."""
    assert scored["tiny_arm"]["delta_nats"] == 0.0
    assert scored["tiny_arm"]["exact"] is True
    assert "exactly 0" in scored["tiny_arm"]["note"]
    assert "higher entropy" in scored["confound"]


def test_per_passage_rows_are_reported_by_the_full_stack(scored: dict) -> None:
    rows = scored["passages"]
    assert len(rows) == 1
    assert set(rows[0]["variants"]) == set(VARIANTS)
    assert rows[0]["nPreservedWords"] > 0
    # The English text is returned as scored, so the panel shows the reader the passage
    # the number came from rather than describing it.
    used = scored["passages_used"]
    assert len(used) == 1 and used[0].strip()
    assert used[0].startswith(scored["variants"][0]["preview"][:40])


def test_pooling_is_token_weighted_across_passages() -> None:
    """Two passages pooled must equal the token-weighted combination of their rows."""
    result = vacancy_score(MODEL, default_passages(count=2), p=1.0, seed=0)
    for variant in result["variants"]:
        rows = [p["variants"][variant["id"]] for p in result["passages"]]
        expected = sum(r["nllAll"] * r["nTokens"] for r in rows) / sum(r["nTokens"] for r in rows)
        assert variant["pooled"]["nllAll"] == pytest.approx(expected, rel=1e-6)


def test_p_zero_is_the_identity_and_costs_nothing() -> None:
    """u ∈ [0,1), so p = 0 vacates nothing and every variant is the same text.

    Red team F7: the zero is real, but it is an IDENTITY, not a measurement, and the
    payload has to say which one it is — the panel was rendering "0.000 ± 0.000 nats
    (sampling, 20 paired tokens)" for "the cost of wrong content" with nothing on screen
    saying the three variants were character-identical.
    """
    result = vacancy_score(MODEL, default_passages(count=1), p=0.0, seed=0)
    previews = {v["preview"] for v in result["variants"]}
    assert len(previews) == 1
    for d in result["differences"]:
        assert d["nats"] == pytest.approx(0.0, abs=1e-9)
        assert d["identity"] is True
        assert "exactly 0 by construction" in d["identityNote"]
    # …and a real run is flagged as NOT an identity, so the flag distinguishes something.
    real = vacancy_score(MODEL, default_passages(count=1), p=1.0, seed=0)
    assert all(d["identity"] is False for d in real["differences"])
    assert all("identityNote" not in d for d in real["differences"])


@pytest.mark.parametrize(
    "passage",
    [
        "I like cats and dogs.",  # 1 paired preserved token -> se = NaN
        "the the",  # 1 paired preserved token
        "the dog",  # the only preserved word is token 0, which has no prediction
        "The dog barked.",
        "Hello world",  # no word survives the transform at all
    ],
)
def test_short_passages_get_a_typed_400_naming_the_cause(passage: str) -> None:
    """Red team F2. Every one of these returned HTTP 500 to a first-try user input.

    Two of them as `InternalError: non-finite value in response payload` — the designed
    "standard error undefined at n = 1" path met `jsonable_6sig`, which refuses non-finite
    numbers, so the design was unreachable through the API and the panel's red box showed
    the reader that string. The rest as untyped `ComputeError`s about averaging no tokens.
    """
    with pytest.raises(InvalidParamError) as exc:
        vacancy_score(MODEL, [passage], p=1.0, seed=0)
    assert exc.value.http_status == 400
    message = str(exc.value)
    assert "non-finite" not in message
    # The message has to name the real cause, in terms of the reader's own input.
    assert any(
        phrase in message
        for phrase in ("paired preserved token", "no scored token", "no word that survives")
    ), message


def test_a_result_that_is_returned_always_encodes(scored: dict) -> None:
    """The endpoint's own encoder is the thing that turned NaN into a 500, so the
    invariant is stated where it belongs: anything `vacancy_score` returns must survive
    `jsonable_6sig`. There is no path that computes a number the API cannot express."""
    jsonable_6sig(scored)
    jsonable_6sig(vacancy_score(MODEL, default_passages(count=1), p=0.0, seed=0))


def test_intermediate_p_is_refused_end_to_end() -> None:
    """Red team F7, through the endpoint: 0 < p < 1 is where swap is non-injective."""
    with pytest.raises(InvalidParamError, match="§5.2a"):
        vacancy_score(MODEL, default_passages(count=1), p=0.5, seed=4)


def test_a_diacritic_passage_is_refused_before_any_number_is_computed() -> None:
    """Red team F6, through the endpoint. `café` used to score and return `washé`."""
    with pytest.raises(InvalidParamError, match="ASCII letters only"):
        vacancy_score(
            MODEL,
            ["The dog and the cat sat on the mat with a café and a résumé on the table."],
            p=1.0,
            seed=0,
        )


def test_bad_parameters_raise_typed_errors() -> None:
    with pytest.raises(InvalidParamError):
        vacancy_score(MODEL, [], p=1.0)
    with pytest.raises(InvalidParamError):
        vacancy_score(MODEL, ["   "], p=1.0)
    with pytest.raises(InvalidParamError):
        vacancy_score(MODEL, default_passages(count=1), p=1.5)
    with pytest.raises(InvalidParamError, match="context"):
        # gpt2 holds 1024 positions; a passage past it is refused rather than truncated,
        # because a truncated variant is not the same text as the one it is compared to.
        vacancy_score(MODEL, ["the cow jumped over the moon. " * 400], p=1.0)
