"""Derived weight sets, honest training reports, and bundle completeness (red-team 007).

Every test here reproduces a defect the red team observed on the running product
(`notes/agent-reports/redteam-007-geo.md`, findings F1–F4 and F6). Real training on a
real (generated, but genuinely structureless) corpus, real cache round-trips, real
routes — no mocks anywhere.

The invented corpus is deliberately made of words the shipped Alice-in-Wonderland
vocabulary does not contain. That is the whole point: it makes "did this derived model
keep ITS OWN word list?" and "was this text tokenized with the right vocabulary?"
observable rather than a coincidence of overlapping English.
"""

from __future__ import annotations

import json
import math
import random

import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.cache.store import CacheStore
from llm_geometry.errors import InvalidParamError
from llm_geometry.geo.bundle import export_bundle, import_bundle
from llm_geometry.geo.config import VOCAB_SIZE, VOCAB_WORDS
from llm_geometry.geo.finetune import finetune
from llm_geometry.geo.jobs import mint_weight_set
from llm_geometry.geo.scratch import SCRATCH_LEARNED_MARGIN, train_scratch
from llm_geometry.geo.tokenizer import get_tokenizer, tokenizer_for
from llm_geometry.geo.weights import load_weight_set, save_weight_set

client = TestClient(app)

_SYLLABLES = (
    "ba",
    "de",
    "fi",
    "go",
    "hu",
    "ka",
    "le",
    "mo",
    "nu",
    "pa",
    "ri",
    "so",
    "tu",
    "va",
    "ze",
    "bo",
    "da",
    "fe",
    "gi",
    "ho",
)


def _invented_words(n: int) -> list[str]:
    """`n` distinct pronounceable nonsense words, none of them English."""
    words: list[str] = []
    seen: set[str] = set()
    for a in _SYLLABLES:
        for b in _SYLLABLES:
            for c in _SYLLABLES:
                w = a + b + c
                if w not in seen:
                    seen.add(w)
                    words.append(w)
                if len(words) >= n:
                    return words
    raise AssertionError("not enough syllable combinations")


@pytest.fixture(scope="module")
def invented_corpus() -> str:
    """A corpus with no English in it at all: 1200 word types, i.i.d., 13,200 tokens.

    Structureless on purpose — F3 is about a run that CANNOT learn being reported as
    though it did.
    """
    words = _invented_words(1200)
    rng = random.Random(1865)
    return " ".join(rng.choice(words) for _ in range(13_200))


@pytest.fixture(scope="module")
def scratch_model(invented_corpus: str, tmp_path_factory) -> dict:
    """One real from-scratch run, shared by the tests below (training is not cheap)."""
    store = CacheStore(tmp_path_factory.mktemp("scratch-store"))
    result = train_scratch(text=invented_corpus, epochs=1, store=store)
    return {"store": store, "result": result, "text": invented_corpus}


# -- F1: a derived set must inherit the vocabulary of the base it came from -------------


def test_finetuning_a_scratch_model_keeps_its_vocabulary(scratch_model: dict) -> None:
    """RED TEAM F1: `save_weight_set(..., source="finetuned")` passed no `vocab_json`,
    so the fine-tune silently reverted to the canonical Alice vocabulary — and the
    saved file's digests verified, because the writer hashed the substituted list."""
    store: CacheStore = scratch_model["store"]
    base = scratch_model["result"]["weights_token"]
    base_words = tokenizer_for(base, store=store).words
    assert base_words != get_tokenizer().words, "fixture is not exercising a new vocabulary"

    ft = finetune(base=base, text=scratch_model["text"][:4000], steps=2, store=store)
    ft_words = tokenizer_for(ft["weights_token"], store=store).words
    assert ft_words == base_words, "the fine-tune reverted to the canonical vocabulary"

    # And the file you would save carries that same word list, not a substituted one.
    bundle = export_bundle(ft["weights_token"], store=store)
    assert json.loads(bundle["vocab"])["words"] == base_words


def test_editing_weights_of_a_scratch_model_keeps_its_vocabulary(scratch_model: dict) -> None:
    """RED TEAM F1, second path: one preset edit through POST /api/geo/weights."""
    store: CacheStore = scratch_model["store"]
    base = scratch_model["result"]["weights_token"]
    base_words = tokenizer_for(base, store=store).words

    minted = mint_weight_set(
        base, [{"layer": 0, "matrix": "W_Q", "preset": "identity"}], store=store
    )
    edited_words = tokenizer_for(minted["weights_token"], store=store).words
    assert edited_words == base_words, "the weight edit reverted to the canonical vocabulary"

    bundle = export_bundle(minted["weights_token"], store=store)
    assert json.loads(bundle["vocab"])["words"] == base_words


def test_chained_derivation_still_carries_the_vocabulary(scratch_model: dict) -> None:
    """scratch → edit → fine-tune → edit: inheritance must survive every hop."""
    store: CacheStore = scratch_model["store"]
    base = scratch_model["result"]["weights_token"]
    base_words = tokenizer_for(base, store=store).words

    step1 = mint_weight_set(
        base, [{"layer": 1, "matrix": "W_K", "preset": "identity"}], store=store
    )
    step2 = finetune(
        base=step1["weights_token"], text=scratch_model["text"][:3000], steps=1, store=store
    )
    step3 = mint_weight_set(
        step2["weights_token"], [{"layer": 2, "matrix": "W_O", "preset": "identity"}], store=store
    )
    assert tokenizer_for(step3["weights_token"], store=store).words == base_words


def test_export_refuses_when_an_owned_vocabulary_is_missing(scratch_model: dict, tmp_path) -> None:
    """Where inheritance is impossible, the writer must REFUSE, never substitute.

    Mirrors the guard `GeoEngine.exportBundle` already had for a scratch set with no
    vocabulary: a file pairing these weights with Alice's words would be internally
    consistent (the writer computes the digest over the substituted list), so no
    reader could ever detect it.
    """
    store: CacheStore = scratch_model["store"]
    ws = load_weight_set(scratch_model["result"]["weights_token"], store=store)
    broken = CacheStore(tmp_path / "broken")
    # A set that DECLARES it owns a word list but has none — the stale-cache shape.
    token = save_weight_set(ws, source="scratch", store=broken, owns_vocab=True)
    with pytest.raises(InvalidParamError) as exc:
        export_bundle(token, store=broken)
    assert "vocabulary" in exc.value.message


# -- F2: fine-tuning must tokenize with the base model's own vocabulary ------------------


def test_finetune_tokenizes_with_the_base_models_vocabulary(scratch_model: dict) -> None:
    """RED TEAM F2 (issue #6): `get_tokenizer().encode_stream(text)` ignored `base`,
    so fine-tuning a scratch model produced `n_tokens 40 n_unk 38` and still reported
    a full nat of "improvement on your text"."""
    store: CacheStore = scratch_model["store"]
    base = scratch_model["result"]["weights_token"]
    text = scratch_model["text"][:4000]

    # Under the CANONICAL tokenizer this text is essentially all <unk> ...
    canonical_enc = get_tokenizer().encode(text, truncate=False)
    assert canonical_enc.n_unk / len(canonical_enc.ids) > 0.9

    # ... so a fine-tune that reported a clean loss drop was training on <unk>.
    # Under the model's OWN vocabulary the only unknowns left are the 200 word types
    # a 1000-word vocabulary cannot hold out of this corpus's 1200 — real, small, and
    # reported, rather than 100 % silently.
    ft = finetune(base=base, text=text, steps=2, store=store)
    assert ft["n_tokens"] > 0
    assert ft["unk_rate"] < 0.2, "fine-tuning still used the canonical tokenizer"
    assert ft["n_unk"] == pytest.approx(ft["unk_rate"] * ft["n_tokens"], abs=1)


def test_finetune_refuses_an_all_unk_stream(scratch_model: dict) -> None:
    """An almost-entirely-<unk> fine-tuning stream must fail loudly, never as a
    clean loss drop (RED TEAM F2: "loss 6.58 → 5.58 on your text")."""
    from llm_geometry.geo.finetune import FINETUNE_MAX_UNK_RATE

    text = scratch_model["text"][:4000]
    with pytest.raises(InvalidParamError) as exc:
        finetune(base="learned", text=text, steps=2)
    assert "unk" in exc.value.message.lower()
    assert f"{FINETUNE_MAX_UNK_RATE:.0%}" in exc.value.message or "%" in exc.value.message


# -- F3: a run that ended at the uniform baseline must say so ----------------------------


def test_structureless_training_is_reported_as_not_learned(scratch_model: dict) -> None:
    """RED TEAM F3: final_loss 6.78 vs ln(1003)=6.91 was activated and announced as
    "trained a new model · final loss 6.89" with nothing saying it never learned."""
    result = scratch_model["result"]
    baseline = math.log(VOCAB_SIZE)
    assert result["uniform_baseline"] == pytest.approx(baseline, abs=1e-9)
    assert result["final_loss"] > baseline - SCRATCH_LEARNED_MARGIN
    assert result["learned"] is False


def test_real_corpus_training_is_reported_as_learned(tmp_path) -> None:
    """The positive control: real text with real structure clears the baseline."""
    from llm_geometry.geo.corpus import load_corpus_text

    store = CacheStore(tmp_path)
    result = train_scratch(text=load_corpus_text(), epochs=4, store=store)
    assert result["learned"] is True
    assert result["final_loss"] < result["uniform_baseline"] - SCRATCH_LEARNED_MARGIN


def test_the_two_existing_gates_do_not_catch_a_baseline_run(scratch_model: dict) -> None:
    """Pins the red team's measurement that the collapse gates score the degenerate
    model BETTER than the canonical one, so nobody later mistakes them for coverage."""
    from llm_geometry.geo.train import compute_gate_metrics, load_canonical_weight_set

    store: CacheStore = scratch_model["store"]
    degenerate = compute_gate_metrics(
        load_weight_set(scratch_model["result"]["weights_token"], store=store)
    )
    canonical = compute_gate_metrics(load_canonical_weight_set())
    assert degenerate["field_directional_entropy"] > canonical["field_directional_entropy"]
    assert degenerate["coverage_uniformity"] > canonical["coverage_uniformity"]


# -- F4: an incomplete model file must be refused, and never surface a bare KeyError -----


def test_incomplete_bundle_is_refused_on_import(scratch_model: dict, tmp_path) -> None:
    """RED TEAM F4: a bundle carrying only `embedding`, with every digest recomputed
    over what was left, imported with a 200 and then 500'd with the string
    `'layers.0.W_V'`."""
    from llm_geometry.geo.weights import weights_token

    store: CacheStore = scratch_model["store"]
    bundle = export_bundle(scratch_model["result"]["weights_token"], store=store)
    subset = {k: v for k, v in bundle["weights"].items() if k == "embedding"}
    crafted = {**bundle, "weights": subset}
    # Recompute the declared token over what is left, exactly as the red team did.
    from llm_geometry.geo.bundle import _unb64

    partial = {name: _unb64(p["data"], [int(d) for d in p["shape"]]) for name, p in subset.items()}
    crafted["weights_token"] = weights_token(partial)

    with pytest.raises(InvalidParamError) as exc:
        import_bundle(crafted, store=CacheStore(tmp_path / "incomplete"))
    assert "missing" in exc.value.message.lower()
    assert "layers.0.W_Q" in exc.value.message


def test_weights_route_never_leaks_a_bare_key_error() -> None:
    """An incomplete set already in the cache (written by an older build) must produce
    a typed error, not `500 {"message": "'layers.0.W_V'"}`."""
    from llm_geometry.geo.train import load_canonical_weight_set

    ws = load_canonical_weight_set()
    partial = {"embedding": ws["embedding"]}
    token = save_weight_set(partial, source="imported")

    resp = client.get(
        "/api/geo/weights", params={"matrix": "W_V", "layer": 0, "weights_token": token}
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()["error"]
    assert body["type"] == "InvalidParamError"
    assert body["message"] != "'layers.0.W_V'"
    assert "layers.0.W_V" in body["message"]
    assert "incomplete" in body["message"].lower()


# -- F6: one canonical vocabulary serialization in both stacks ---------------------------


def test_vocabulary_json_is_the_pinned_canonical_serialization() -> None:
    """RED TEAM F6: Python wrote `{"format": "geo-tokenizer-v1", "specials": {...` and
    the browser wrote `{"format":"geo-tokenizer-v1","specials":{"<unk>":0,...`, so the
    same model had two `vocab_sha256` values depending on which build saved it."""
    payload = get_tokenizer().to_json()
    assert payload.startswith(
        '{"format":"geo-tokenizer-v1","specials":{"<eos>":1,"<pad>":2,"<unk>":0},"words":['
    )
    assert ", " not in payload[: payload.index('"words":')]  # compact separators
    assert payload.isascii()  # ensure_ascii, so byte-identical across encodings
    # And it still round-trips.
    from llm_geometry.geo.tokenizer import GeoTokenizer

    assert GeoTokenizer.from_json(payload).words == get_tokenizer().words


def test_non_ascii_vocabulary_words_are_escaped_identically() -> None:
    """The token regex admits any non-space symbol, so a corpus with typographic or
    accented characters puts non-ASCII into the word list. Both stacks must escape it
    the same way or the digests diverge again."""
    from llm_geometry.geo.tokenizer import GeoTokenizer

    words = _invented_words(VOCAB_WORDS)
    words[0] = "é"
    words[1] = "—"
    payload = GeoTokenizer(words).to_json()
    assert payload.isascii()
    assert "\\u00e9" in payload
    assert GeoTokenizer.from_json(payload).words[:2] == ["é", "—"]
