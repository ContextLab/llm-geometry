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

import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.cache.store import CacheStore
from llm_geometry.errors import InvalidParamError
from llm_geometry.geo.bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    _b64,
    _EXPECTED_CONFIG,
    export_bundle,
    import_bundle,
    vocab_digest,
)
from llm_geometry.geo.config import VOCAB_SIZE, VOCAB_WORDS
from llm_geometry.geo.finetune import finetune
from llm_geometry.geo.jobs import mint_weight_set
from llm_geometry.geo.scratch import SCRATCH_LEARNED_MARGIN, train_scratch
from llm_geometry.geo.tokenizer import GeoTokenizer, get_tokenizer, tokenizer_for
from llm_geometry.geo.weights import (
    load_weight_set,
    own_vocab_json,
    save_weight_set,
    weight_set_owns_vocab,
    weights_token,
)


def _bundle_for(ws: dict, vocab_json: str) -> dict:
    """A real, self-consistent model file for (weights, word list) — what a writer emits."""
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "weights_token": weights_token(ws, own_vocab_json(vocab_json)),
        "config": dict(_EXPECTED_CONFIG),
        "vocab": vocab_json,
        "vocab_sha256": vocab_digest(vocab_json),
        "weights": {
            name: {"shape": list(np.asarray(arr).shape), "data": _b64(arr)}
            for name, arr in sorted(ws.items())
        },
    }


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
def canonical_ready() -> None:
    """The canonical checkpoint, in the shared cache this run points at.

    `tests/conftest.py` gives every run a throwaway `LLM_GEOMETRY_CACHE_DIR`, so a case
    that resolves `base="learned"` (or calls `load_canonical_weight_set`) only passed
    when some earlier module in the same session happened to have trained it. Three cases
    below did exactly that and failed when this file was run alone — the first thing a
    bisect or a `-k` run hits. Train it explicitly instead: the same real training the app
    does on first open, cached for the rest of the session.
    """
    from llm_geometry.geo.train import train_canonical

    train_canonical()


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


def test_two_models_with_identical_weights_keep_their_own_word_lists(
    scratch_model: dict, tmp_path
) -> None:
    """RED TEAM F1, third path: the cache's content-hash DEDUP, not the derivation chain.

    ``save_weight_set`` wrote its metadata only when the key was new, so for a hash
    already present the incoming vocabulary was discarded — first-write-wins. Two models
    whose weights coincide (a pre-fix file loaded, then the same model trained from
    scratch) therefore shared one word list, with every digest recomputed over it and
    verifying. The fix is to the IDENTITY, not the caching policy: the token covers the
    vocabulary, so two word lists are two models.
    """
    store = CacheStore(tmp_path / "collide")
    ws = load_weight_set(scratch_model["result"]["weights_token"], store=scratch_model["store"])
    words_a = _invented_words(VOCAB_WORDS)
    words_b = [f"zz{w}" for w in words_a]
    vocab_a = GeoTokenizer(words_a).to_json()
    vocab_b = GeoTokenizer(words_b).to_json()

    token_a = save_weight_set(ws, source="scratch", store=store, vocab_json=vocab_a)
    token_b = save_weight_set(ws, source="scratch", store=store, vocab_json=vocab_b)
    assert token_a != token_b, "identical weights + different words must not share a token"

    assert tokenizer_for(token_a, store=store).words == words_a
    assert tokenizer_for(token_b, store=store).words == words_b
    # And the FILE each one saves carries its own list, which is what the user sees.
    assert json.loads(export_bundle(token_a, store=store)["vocab"])["words"] == words_a
    assert json.loads(export_bundle(token_b, store=store)["vocab"])["words"] == words_b


def test_loading_a_file_then_training_the_same_model_does_not_swap_the_word_list(
    scratch_model: dict, tmp_path
) -> None:
    """The realistic trigger the red team reproduced end to end: load a pre-fix model
    file, then train the same weights from scratch on your own text. The scratch run's
    own 1,000 words were discarded and its saved file read the loaded file's list."""
    store = CacheStore(tmp_path / "load-then-train")
    ws = load_weight_set(scratch_model["result"]["weights_token"], store=scratch_model["store"])
    other_words = [f"qq{w}" for w in _invented_words(VOCAB_WORDS)]

    # 1) A real model file carrying somebody else's word list, loaded first.
    loaded = import_bundle(_bundle_for(ws, GeoTokenizer(other_words).to_json()), store=store)
    assert tokenizer_for(loaded["weights_token"], store=store).words == other_words

    # 2) Now "train from scratch" and land on the same weights, with your own words.
    mine = _invented_words(VOCAB_WORDS)
    mine_token = save_weight_set(
        ws, source="scratch", store=store, vocab_json=GeoTokenizer(mine).to_json()
    )
    assert mine_token != loaded["weights_token"]
    assert tokenizer_for(mine_token, store=store).words == mine
    assert json.loads(export_bundle(mine_token, store=store)["vocab"])["words"] == mine


def test_a_file_whose_word_list_was_swapped_after_the_fact_is_refused(
    scratch_model: dict, tmp_path
) -> None:
    """A tampered file — genuine weights, a substituted word list, and `vocab_sha256`
    recomputed over the substitute — used to load with a 200 and mislabel every token.
    The two digests could not catch it; the identity hash can, because the declared
    `weights_token` names a model with the ORIGINAL words."""
    store: CacheStore = scratch_model["store"]
    bundle = export_bundle(scratch_model["result"]["weights_token"], store=store)
    fake_words = [f"xx{w}" for w in _invented_words(VOCAB_WORDS)]
    fake_vocab = GeoTokenizer(fake_words).to_json()
    tampered = {**bundle, "vocab": fake_vocab, "vocab_sha256": vocab_digest(fake_vocab)}

    with pytest.raises(InvalidParamError) as exc:
        import_bundle(tampered, store=CacheStore(tmp_path / "tampered"))
    assert "corrupt" in exc.value.message
    assert "vocabulary" in exc.value.message


def test_save_weight_set_refuses_a_conflicting_vocabulary_claim(
    scratch_model: dict, tmp_path
) -> None:
    """The dedup no longer keeps whichever entry arrived first and discards the other's
    word list: a second write that disagrees about ownership is refused, loudly."""
    store = CacheStore(tmp_path / "conflict")
    ws = load_weight_set(scratch_model["result"]["weights_token"], store=scratch_model["store"])
    token = save_weight_set(ws, source="scratch", store=store, owns_vocab=True)
    # The same bytes written as a canonical-vocabulary model: same hash, different claim.
    with pytest.raises(InvalidParamError) as exc:
        save_weight_set(ws, source="learned", store=store, owns_vocab=False)
    assert token in exc.value.message
    assert "vocabulary" in exc.value.message


def test_the_token_covers_the_vocabulary_byte_for_byte() -> None:
    """A cross-language golden for the identity hash.

    The same two constants are pinned in `tests/unit/geoDerivedVocab.test.ts` against the
    TypeScript `weightsToken`. They are what makes "a model saved by the browser and the
    same model saved by the Python backend are the same file" checkable rather than
    hoped for — the two stacks used to resolve a token collision in OPPOSITE directions
    (Python kept the first word list, the browser the last).
    """
    from llm_geometry.geo.weights import WEIGHT_SHAPES, weights_token

    ws = {
        name: (np.arange(int(np.prod(shape)), dtype=np.float32) * np.float32(0.001))
        .reshape(shape)
        .astype(np.float32)
        for name, shape in WEIGHT_SHAPES.items()
    }
    words = [f"w{i}" for i in range(VOCAB_WORDS)]
    # Unchanged from before the vocabulary joined the hash: a model that reads under the
    # shipped word list keeps the token it always had, so `checkpoint_id` never moved.
    assert weights_token(ws) == "38cb99338fb6c40f022641b579a7e827"
    assert weights_token(ws, GeoTokenizer(words).to_json()) == "50246246e336794517fcc299b505659a"


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


def test_finetune_refuses_an_all_unk_stream(scratch_model: dict, canonical_ready: None) -> None:
    """An almost-entirely-<unk> fine-tuning stream must fail loudly, never as a
    clean loss drop (RED TEAM F2: "loss 6.58 → 5.58 on your text")."""
    from llm_geometry.geo.finetune import FINETUNE_MAX_UNK_RATE

    text = scratch_model["text"][:4000]
    with pytest.raises(InvalidParamError) as exc:
        finetune(base="learned", text=text, steps=2)
    assert "unk" in exc.value.message.lower()
    assert f"{FINETUNE_MAX_UNK_RATE:.0%}" in exc.value.message or "%" in exc.value.message


def _stream_with_unk_rate(n_unk: int, n_known: int) -> str:
    """A text of exactly ``n_unk + n_known`` canonical tokens, ``n_unk`` of them unknown."""
    known = [w for w in get_tokenizer().words if w.isalpha()][:n_known]
    assert len(known) == n_known
    unknown = _invented_words(n_unk)
    pieces: list[str] = []
    for i in range(max(n_unk, n_known)):
        if i < n_unk:
            pieces.append(unknown[i])
        if i < n_known:
            pieces.append(known[i])
    return " ".join(pieces)


def test_the_unk_bound_refuses_a_stream_that_is_exactly_at_it(canonical_ready: None) -> None:
    """The comparison was `>`, so a stream that is EXACTLY 90 % <unk> was accepted and
    reported as a clean loss drop, while 901 of 1000 was refused with a message that
    rounds to the same "(90%)". One token apart, indistinguishable on screen."""
    from llm_geometry.geo.finetune import FINETUNE_MAX_UNK_RATE

    text = _stream_with_unk_rate(900, 100)
    enc = get_tokenizer().encode(text, truncate=False)
    assert (enc.n_unk, len(enc.ids)) == (900, 1000)
    assert enc.n_unk / len(enc.ids) == FINETUNE_MAX_UNK_RATE  # exactly at the bound

    with pytest.raises(InvalidParamError) as exc:
        finetune(base="learned", text=text, steps=1)
    assert "outside the active model's vocabulary" in exc.value.message
    assert f"the limit is {FINETUNE_MAX_UNK_RATE:.0%}" in exc.value.message


def test_the_unk_bound_still_accepts_a_stream_just_below_it(
    canonical_ready: None, tmp_path
) -> None:
    """The control: the bound is one-sided, not a blanket refusal. 89.9 % still trains —
    fine-tuning the shipped model on modern prose legitimately unks a large share."""
    text = _stream_with_unk_rate(899, 101)
    enc = get_tokenizer().encode(text, truncate=False)
    assert enc.n_unk / len(enc.ids) == 0.899

    result = finetune(base="learned", text=text, steps=1, store=CacheStore(tmp_path / "below"))
    assert result["n_unk"] == 899
    assert result["unk_rate"] == pytest.approx(0.899)


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


def test_the_two_existing_gates_do_not_catch_a_baseline_run(
    scratch_model: dict, canonical_ready: None
) -> None:
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


def test_weights_route_never_leaks_a_bare_key_error(canonical_ready: None) -> None:
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


# -- round 7: the two properties `own_vocab_json` exists for --------------------------
#
# Both were stated in a docstring and pinned by nothing. `own_vocab_json` reduced to
# `return vocab_json` left every test in this module green — the module whose whole
# subject is which model a set of weights IS — while one model acquired two identities;
# and `import_bundle`'s `own_vocab_json(canonical_vocab)` could be swapped for
# `own_vocab_json(vocab_json)` with the same result. Every case below is a behaviour
# probe: build a real file, load it, and compare the identity that comes back.


def test_own_vocab_json_treats_the_shipped_word_list_as_nothing_to_own() -> None:
    """The function's contract, as three cases rather than a paragraph.

    A model whose words ARE the shipped words has nothing of its own to substitute, so it
    must hash as the checkpoint does; a model with its own list hashes with it. Reduced to
    `return vocab_json`, the first case is what breaks, and it breaks silently.
    """
    shipped = get_tokenizer().to_json()
    assert own_vocab_json(shipped) is None
    assert own_vocab_json(None) is None
    mine = GeoTokenizer(_invented_words(VOCAB_WORDS)).to_json()
    assert own_vocab_json(mine) == mine


def test_a_file_spelling_out_the_shipped_word_list_is_the_same_model_as_the_checkpoint(
    scratch_model: dict, tmp_path
) -> None:
    """One model, one identity, whichever door it came through.

    A checkpoint-descended set stores no vocabulary; a file always spells one out. If the
    file's copy of the SHIPPED list counted as a vocabulary of its own, the same weights
    would be two models — `3ae5df00…` as a checkpoint and `a4d2510f…` as a file — the
    store would hold both, and the two entries would disagree about which words the ids
    mean while every digest in each of them verified.
    """
    store = CacheStore(tmp_path / "one-identity")
    ws = load_weight_set(scratch_model["result"]["weights_token"], store=scratch_model["store"])

    as_checkpoint = save_weight_set(ws, source="learned", store=store)
    as_file = import_bundle(_bundle_for(ws, get_tokenizer().to_json()), store=store)

    assert as_file["weights_token"] == as_checkpoint
    assert weight_set_owns_vocab(as_file["weights_token"], store=store) is False
    assert tokenizer_for(as_file["weights_token"], store=store).words == get_tokenizer().words


def test_a_writers_key_order_does_not_change_which_model_a_file_is(
    scratch_model: dict, tmp_path
) -> None:
    """A file's identity is its word LIST, not the bytes a writer chose to spell it with.

    `import_bundle` hashes `own_vocab_json(canonical_vocab)` — the serialization this
    build would have emitted — precisely so that a third-party writer using a different
    key order or indentation produces the same model. Hashing the file's own bytes instead
    refuses it: `this model file is corrupt: its weights and vocabulary hash to 88f44a3b…`,
    on a file that is entirely honest.

    `vocab_sha256` still covers the bytes as written (it is a transport checksum), so the
    variant recomputes it — that is what an honest writer does.
    """
    store = CacheStore(tmp_path / "key-order")
    ws = load_weight_set(scratch_model["result"]["weights_token"], store=scratch_model["store"])
    words = _invented_words(VOCAB_WORDS)
    canonical = GeoTokenizer(words).to_json()
    as_written_here = _bundle_for(ws, canonical)

    data = json.loads(canonical)
    # Same object, emitted by a writer that does not sort keys and indents its output.
    reordered = json.dumps(
        {"words": data["words"], "specials": data["specials"], "format": data["format"]},
        indent=2,
    )
    assert reordered != canonical, "the variant must differ in bytes to test anything"
    assert json.loads(reordered) == data, "…and must not differ in meaning"
    as_written_elsewhere = {
        **as_written_here,
        "vocab": reordered,
        "vocab_sha256": vocab_digest(reordered),
    }

    here = import_bundle(as_written_here, store=store)
    elsewhere = import_bundle(as_written_elsewhere, store=store)

    assert elsewhere["weights_token"] == here["weights_token"]
    assert tokenizer_for(elsewhere["weights_token"], store=store).words == words
    # And what the store kept is the canonical spelling, so re-saving it is byte-stable.
    assert export_bundle(elsewhere["weights_token"], store=store)["vocab"] == canonical


# -- round 5: three wrong-answer paths around the identity fix ---------------------------
#
# The version the identity change (0d23123) shipped under. Entries written by that build
# are keyed by a hash of the WEIGHTS ALONE while carrying a word list of their own, so
# this build cannot check one against the other — which is the whole reason it must not
# read them. Written literally, not as `SCHEMA_VERSION - 1`: the point of these cases is
# what happens to the caches that exist in the wild.
_PRE_IDENTITY_SCHEMA_VERSION = 14


def _write_pre_identity_entry(store: CacheStore, ws: dict, vocab_json: str) -> str:
    """The entry the previous build wrote for a scratch model, byte-for-byte in shape.

    Weights-only token, its own word list in `meta`, tagged with the schema version that
    build used.
    """
    from llm_geometry.config import SCHEMA_VERSION
    from llm_geometry.geo.weights import _artifact_key

    token = weights_token(ws)  # the OLD identity: weights, no vocabulary
    key = _artifact_key(token)
    store.put(
        key,
        {"schema_version": SCHEMA_VERSION, "artifact_type": "geo-weights", "weights_token": token},
        {
            "weights_token": token,
            "source": "scratch",
            "names": sorted(ws),
            "owns_vocab": True,
            "vocab": vocab_json,
        },
        {name: np.asarray(a, np.float32) for name, a in ws.items()},
    )
    path = store.cache_dir / f"{key}.json"
    sidecar = json.loads(path.read_text())
    sidecar["schema_version"] = _PRE_IDENTITY_SCHEMA_VERSION
    path.write_text(json.dumps(sidecar))
    return token


def test_an_unknown_token_is_unknown_to_tokenize_too(canonical_ready: None) -> None:
    """ROUND 5 F1: `/tokenize` answered 200 under the SHIPPED word list for a token the
    store does not have, while `/trace` answered 404 for the identical request.

    That is the campaign's central corruption — a model's ids read under Alice in
    Wonderland's words — reached through a store miss rather than through a hash
    collision. Nothing threw, and the token strip's own verification probe compares
    against exactly that vocabulary, so the tab reported the word list verified.
    """
    ghost = "deadbeefdeadbeefdeadbeefdeadbeef"
    tok = client.get(f"/api/geo/tokenize?text=alice&weights_token={ghost}")
    assert tok.status_code == 404, tok.text
    assert tok.json()["error"]["type"] == "NotFoundError"
    tr = client.get(f"/api/geo/trace?prompt=alice&weights_token={ghost}")
    assert tr.status_code == 404
    # The defect was the DISAGREEMENT: one route said the model exists, the other said it
    # does not, and the one that said yes answered with another model's words.
    assert tok.json()["error"]["type"] == tr.json()["error"]["type"]


def test_tokenizer_for_refuses_an_evicted_model_instead_of_relabelling_it(
    scratch_model: dict, tmp_path
) -> None:
    """The same hole below the routes: an LRU eviction is not "this model reads under the
    shipped vocabulary", and answering as though it were relabels every token."""
    from llm_geometry.geo.weights import _artifact_key

    source: CacheStore = scratch_model["store"]
    origin = scratch_model["result"]["weights_token"]
    ws = load_weight_set(origin, store=source)
    vocab_json = tokenizer_for(origin, store=source).to_json()

    # A store of this test's own, so evicting from it cannot disturb the shared fixture.
    store = CacheStore(tmp_path / "evictable")
    token = save_weight_set(
        ws, source="scratch", store=store, vocab_json=vocab_json, owns_vocab=True
    )
    own_words = tokenizer_for(token, store=store).words
    assert own_words != get_tokenizer().words

    key = _artifact_key(token)
    (store.cache_dir / f"{key}.json").unlink()
    (store.cache_dir / f"{key}.npz").unlink()
    with pytest.raises(Exception) as exc:  # NotFoundError
        tokenizer_for(token, store=store)
    assert "unknown" in str(exc.value)
    assert type(exc.value).__name__ == "NotFoundError"


def test_a_cache_from_before_the_identity_change_says_what_happened(
    scratch_model: dict, tmp_path
) -> None:
    """ROUND 5 F2: the identity change moved a persisted format without bumping
    SCHEMA_VERSION, so pre-change entries stayed readable — and being readable is what
    made the user's own model unsaveable ("does not match a re-hash of its own weights and
    vocabulary … Retrain or reload it", about a model that was never corrupt).

    They are cache-missed now, and the miss is EXPLAINED: "evicted, re-submit the edit"
    describes neither what happened nor what to do.
    """
    ws = load_weight_set(scratch_model["result"]["weights_token"], store=scratch_model["store"])
    vocab_json = tokenizer_for(
        scratch_model["result"]["weights_token"], store=scratch_model["store"]
    ).to_json()
    store = CacheStore(tmp_path / "old-cache")
    token = _write_pre_identity_entry(store, ws, vocab_json)

    with pytest.raises(Exception) as exc:
        load_weight_set(token, store=store)
    message = str(exc.value)
    assert type(exc.value).__name__ == "NotFoundError"
    assert f"v{_PRE_IDENTITY_SCHEMA_VERSION}" in message
    assert "earlier build" in message
    # It must not accuse the user's model of being corrupt, and it must say that the file
    # they saved is unaffected — the model is recoverable, and the message is the only
    # place that can say so.
    assert "SAVED MODEL FILE" in message.upper()
    assert "corrupt" not in message


def test_a_cache_from_before_the_identity_change_does_not_wedge_the_lab(tmp_path) -> None:
    """ROUND 5 F2, the other half: a pre-change entry sitting at the canonical
    checkpoint's key made `train_canonical` raise on startup —

        InvalidParamError: weights_token '…' is already stored with a different vocabulary
        claim (stored owns_vocab=True, writing owns_vocab=False)

    — so the Geometry Lab could not open at all until the cache was deleted by hand.
    """
    rng = np.random.default_rng(11)
    ws = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in __import__(
            "llm_geometry.geo.weights", fromlist=["WEIGHT_SHAPES"]
        ).WEIGHT_SHAPES.items()
    }
    other_words = [f"zz{i:04d}" for i in range(VOCAB_WORDS)]
    store = CacheStore(tmp_path / "wedge-cache")
    token = _write_pre_identity_entry(store, ws, GeoTokenizer(other_words).to_json())

    # The same weights arriving as the canonical-vocabulary model this build knows: the
    # identical key, the opposite ownership claim.
    written = save_weight_set(ws, source="learned", store=store)
    assert written == token
    assert load_weight_set(written, store=store).keys() == ws.keys()
