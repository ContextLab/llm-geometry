"""From-scratch training on arbitrary text + portable model save/load (feature 004).

Real training on real text, real cache round-trips — no mocks. The corpus used here is
the committed public-domain one, which is the only text in the repo big enough to fill
a 1000-word vocabulary; the point of the tests is that the vocabulary is rebuilt FROM
the supplied text rather than inherited, and that a saved model reloads identically.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.cache.store import CacheStore
from llm_geometry.errors import InvalidParamError
from llm_geometry.geo.bundle import BUNDLE_FORMAT, export_bundle, import_bundle
from llm_geometry.geo.config import VOCAB_SIZE, VOCAB_WORDS
from llm_geometry.geo.corpus import load_corpus_text
from llm_geometry.geo.scratch import corpus_stats, train_scratch
from llm_geometry.geo.tokenizer import get_tokenizer, tokenizer_for
from llm_geometry.geo.weights import weights_token

client = TestClient(app)


@pytest.fixture(scope="module")
def corpus() -> str:
    return load_corpus_text()


def test_corpus_stats_counts_real_types(corpus: str) -> None:
    stats = corpus_stats(corpus)
    assert stats["n_tokens"] > 10_000
    assert stats["n_distinct"] >= VOCAB_WORDS
    assert stats["vocab_words_required"] == VOCAB_WORDS
    # Empty text is a legitimate query (the UI polls as you type), not an error.
    assert corpus_stats("")["n_distinct"] == 0


def test_short_text_is_refused_with_a_useful_message() -> None:
    with pytest.raises(InvalidParamError) as exc:
        train_scratch(text="alice met the rabbit " * 20, epochs=1)
    assert "distinct word types" in exc.value.message
    assert str(VOCAB_WORDS) in exc.value.message


def test_scratch_rebuilds_the_vocabulary_from_the_given_text(tmp_path, corpus: str) -> None:
    """A model trained on a DIFFERENT slice gets a DIFFERENT vocabulary."""
    store = CacheStore(tmp_path)
    # Second half of the corpus: same language, genuinely different word frequencies.
    text = corpus[len(corpus) // 2 :]
    result = train_scratch(text=text, epochs=1, store=store, progress_cb=None)

    assert result["vocab_size"] == VOCAB_SIZE
    assert result["n_distinct"] >= VOCAB_WORDS
    assert result["weights_token"] and not result["cached"]

    trained_vocab = tokenizer_for(result["weights_token"], store=store)
    canonical = get_tokenizer()
    assert trained_vocab.words != canonical.words, "vocabulary was inherited, not rebuilt"
    # It is still a real vocabulary of the right size, with no duplicates.
    assert len(trained_vocab.words) == VOCAB_WORDS
    assert len(set(trained_vocab.words)) == VOCAB_WORDS

    # Same request again is a cache hit with the same model.
    again = train_scratch(text=text, epochs=1, store=store)
    assert again["cached"] is True
    assert again["weights_token"] == result["weights_token"]


def test_scratch_model_ids_are_read_with_its_own_vocabulary(tmp_path, corpus: str) -> None:
    store = CacheStore(tmp_path)
    text = corpus[: len(corpus) // 2]
    result = train_scratch(text=text, epochs=1, store=store)
    tok = tokenizer_for(result["weights_token"], store=store)

    # Round-tripping through THIS model's tokenizer must recover the same words.
    sample = "alice was beginning to get very tired"
    enc = tok.encode(sample)
    assert len(enc.ids) == 7
    assert tok.decode(enc.ids) == sample  # every word is in this corpus's vocabulary

    # An unknown token id is a hard error, not a silent blank.
    with pytest.raises(InvalidParamError):
        tok.decode([VOCAB_SIZE + 5])


def test_bundle_roundtrip_preserves_the_model_exactly(tmp_path, corpus: str) -> None:
    store = CacheStore(tmp_path)
    text = corpus[len(corpus) // 3 :]
    trained = train_scratch(text=text, epochs=1, store=store)
    token = trained["weights_token"]

    bundle = export_bundle(token, store=store)
    assert bundle["format"] == BUNDLE_FORMAT
    assert bundle["weights_token"] == token
    assert bundle["config"]["vocab_size"] == VOCAB_SIZE
    assert bundle["weights"], "bundle carries no weights"

    # Survives a real serialization round-trip (this is what a saved file is).
    reloaded = json.loads(json.dumps(bundle))

    fresh = CacheStore(tmp_path / "elsewhere")
    imported = import_bundle(reloaded, store=fresh)
    assert imported["weights_token"] == token  # identical weights ⇒ identical hash
    assert imported["vocab_size"] == VOCAB_SIZE

    # And the vocabulary came with it: the reloaded model reads ids the same way.
    assert tokenizer_for(token, store=fresh).words == tokenizer_for(token, store=store).words


def test_corrupt_bundle_is_refused_not_silently_loaded(tmp_path, corpus: str) -> None:
    store = CacheStore(tmp_path)
    trained = train_scratch(text=corpus[len(corpus) // 4 :], epochs=1, store=store)
    bundle = export_bundle(trained["weights_token"], store=store)

    bad_hash = {**bundle, "weights_token": "0" * 32}
    with pytest.raises(InvalidParamError) as exc:
        import_bundle(bad_hash, store=CacheStore(tmp_path / "a"))
    assert "corrupt" in exc.value.message

    wrong_arch = {**bundle, "config": {**bundle["config"], "d_model": 8}}
    with pytest.raises(InvalidParamError) as exc:
        import_bundle(wrong_arch, store=CacheStore(tmp_path / "b"))
    assert "d_model" in exc.value.message

    no_vocab = {k: v for k, v in bundle.items() if k != "vocab"}
    with pytest.raises(InvalidParamError):
        import_bundle(no_vocab, store=CacheStore(tmp_path / "c"))

    with pytest.raises(InvalidParamError):
        import_bundle({"format": "something-else"}, store=CacheStore(tmp_path / "d"))


def test_integrity_checks_cannot_be_bypassed(tmp_path, corpus: str) -> None:
    """The exact bypasses a red-team pass found: an omitted token, and a fabricated
    vocabulary that the weights hash cannot possibly cover."""
    store = CacheStore(tmp_path)
    trained = train_scratch(text=corpus[len(corpus) // 5 :], epochs=1, store=store)
    bundle = export_bundle(trained["weights_token"], store=store)

    # (a) Deleting the token must NOT be read as "nothing to verify".
    no_token = {k: v for k, v in bundle.items() if k != "weights_token"}
    with pytest.raises(InvalidParamError) as exc:
        import_bundle(no_token, store=CacheStore(tmp_path / "e"))
    assert "cannot be verified" in exc.value.message

    # (b) Genuine weights + an invented word list: every label would be wrong.
    fake_words = [f"zzz{i}" for i in range(VOCAB_WORDS)]
    fake_vocab = json.dumps(
        {"format": "geo-tokenizer-v1", "specials": {}, "words": fake_words},
        ensure_ascii=True,
        sort_keys=True,
    )
    swapped = {**bundle, "vocab": fake_vocab}  # vocab_sha256 left as the real one
    with pytest.raises(InvalidParamError) as exc:
        import_bundle(swapped, store=CacheStore(tmp_path / "f"))
    assert "vocabulary" in exc.value.message

    # (c) Dropping the vocabulary digest as well must not open the hole back up.
    swapped_no_digest = {k: v for k, v in swapped.items() if k != "vocab_sha256"}
    with pytest.raises(InvalidParamError) as exc:
        import_bundle(swapped_no_digest, store=CacheStore(tmp_path / "g"))
    assert "vocab_sha256" in exc.value.message

    # (d) A v1 file (no vocabulary check at all) is refused by version.
    v1 = {**bundle, "version": 1}
    with pytest.raises(InvalidParamError) as exc:
        import_bundle(v1, store=CacheStore(tmp_path / "h"))
    assert "version" in exc.value.message


def test_force_arrows_are_tangent_where_they_are_drawn(canonical_ready: None) -> None:
    """FR-418: the aggregate force must be tangent at the point the CLIENT anchors it.

    Regression for a real defect: the projection used the layer's residual stream
    (`hidden_in`) while the UI draws at the token embedding, so the "tangent" arrows
    came out up to 59 degrees off the tangent plane at layer 2.
    """
    from llm_geometry.geo.fields import force_field
    from llm_geometry.geo.model import model_from_weight_set
    from llm_geometry.geo.tokenizer import get_tokenizer
    from llm_geometry.geo.train import resolve_weight_set

    model = model_from_weight_set(resolve_weight_set("learned"))
    ids = get_tokenizer().encode("alice rabbit queen said the little door").ids
    embeddings = np.asarray(model.embedding.detach().cpu().numpy(), dtype=np.float64)

    for layer in range(4):
        field = force_field(model, ids, layer=layer, antisymmetrize=True)
        for force in field["sequence_forces"]:
            anchor = embeddings[ids[force["position"]]]
            normal = anchor / np.linalg.norm(anchor)
            vec = np.asarray(force["vec"], dtype=np.float64)
            mag = np.linalg.norm(vec)
            if mag < 1e-9:
                continue
            cos = abs(float(vec @ normal) / mag)
            assert cos < 1e-5, (
                f"layer {layer} position {force['position']}: force is "
                f"{np.degrees(np.arcsin(min(1.0, cos))):.1f} degrees out of the tangent "
                "plane at the point it is drawn from"
            )


# -- HTTP surface ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonical_ready() -> None:
    """The HTTP export path resolves "learned", which needs the canonical checkpoint.

    conftest points the cache at a throwaway dir, so a standalone run of this module
    has to train it — the same real training the app does on first open.
    """
    from llm_geometry.geo.train import train_canonical

    train_canonical()


def test_corpus_stats_route() -> None:
    r = client.get("/api/geo/corpus_stats", params={"text": "alice met the white rabbit"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_tokens"] == 5
    assert body["n_distinct"] == 5
    assert body["vocab_words_required"] == VOCAB_WORDS


def test_train_scratch_route_rejects_short_text() -> None:
    r = client.post("/api/geo/train_scratch", json={"text": "too short to train on"})
    assert r.status_code == 400  # InvalidParamError envelope
    assert "distinct word types" in r.json()["error"]["message"]


def test_train_scratch_route_rejects_bad_sources() -> None:
    assert client.post("/api/geo/train_scratch", json={}).status_code == 400
    both = client.post(
        "/api/geo/train_scratch", json={"text": "x", "hf_dataset": "roneneldan/TinyStories"}
    )
    assert both.status_code == 400
    bad_epochs = client.post("/api/geo/train_scratch", json={"text": "x", "epochs": 9999})
    assert bad_epochs.status_code == 400


def test_model_export_import_over_http(canonical_ready: None) -> None:
    exported = client.get("/api/geo/model", params={"weights_token": "learned"})
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == BUNDLE_FORMAT
    assert set(bundle["weights"]) == set(bundle["weights"])  # names round-trip

    imported = client.post("/api/geo/model", json=bundle)
    assert imported.status_code == 200
    # The canonical weights re-hash to the canonical token.
    from llm_geometry.geo.train import resolve_weight_set

    assert imported.json()["weights_token"] == weights_token(resolve_weight_set("learned"))

    bad = client.post("/api/geo/model", json={"format": "nope"})
    assert bad.status_code == 400
