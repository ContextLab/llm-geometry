"""The pretrained arm of the vacancy instrument (contract §8).

What this measures, in one sentence: **does a model that knows English still predict the
closed-class scaffolding when the content words have been vacated?** The scaffolding is
character-identical across the variants, so the same words — very often the same token
ids, at the same word positions — are scored in each, and the difference is attributable
to what happened around them.

Three variants of one passage are scored (§8.3):

* ``english`` — the passage as written;
* ``swap``    — every vacated stem replaced by a REAL, frequency-rank-matched English
  word. Equally nonsensical, ordinarily tokenized, every form known;
* ``nonce``   — every vacated stem replaced by a phonotactically legal invention.

and the two differences that are worth anything are the *labelled* ones:

* ``nll(swap) − nll(english)`` — the cost of **wrong content**;
* ``nll(nonce) − nll(swap)``   — the cost of **unknown form**.

``nll(nonce) − nll(english)`` is never a headline: it is the sum of the two and conflates
them. The second difference is also only an UPPER BOUND on what a word's *location* was
worth, because nonce forms fragment into more subword tokens than real words do and that
residual is not separable without a tokenizer-level control (§8.3, ``UNKNOWN_FORM_NOTE``).

ALIGNMENT (§8.2, FR-718). Tokens are attributed to words by **UTF-8 byte spans** derived
from the tokenizer's byte-level pieces, never by "characters":

* transformers.js exposes no offsets at all, so HF's ``return_offsets_mapping`` is not a
  mechanism the two stacks can share;
* per-token decoding emits U+FFFD and destroys the text on any split multi-byte character,
  in both stacks;
* Python indexes code points where JavaScript indexes UTF-16 units, so the two disagree on
  the same string (31 vs 32 for one probe text) — bytes are the only safe contract unit;
* HF's own offsets OVERLAP on multi-byte characters, so per-token quantities summed over a
  word would be double-counted; the byte spans are a true partition.

Every step is verified rather than trusted: the concatenated token bytes must equal
``utf8(text)`` exactly, and a mismatch RAISES (never mis-attributes). The input is
NFC-normalized once up front because Qwen's tokenizer carries an NFC normalizer while
gpt2's and SmolLM2's do not — without that, a decomposed character silently shifts every
span after it, and the check above would fire on a passage that is perfectly fine.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from ..errors import ComputeError, InvalidParamError
from ..lex.vacancy import (
    VacancyParams,
    build_vacancy_map,
    type_counts,
    vacancy_domain,
    vacate_text,
)
from ..lex.vocab import WORD_RE, tokenize
from ..models.loader import LoadedModel, load_model
from .tracing import _TRACE_LOCK

#: The three variants, in the order the UI reads them. ``english`` is the reference;
#: ``swap`` sits between the other two by construction, which is what makes the
#: decomposition of §8.3 a decomposition rather than two unrelated numbers.
VARIANTS: tuple[str, ...] = ("english", "swap", "nonce")

#: `mint` values for the two vacated variants (§8.3). ``english`` is not minted at all.
VARIANT_MINT: dict[str, str] = {"swap": "swap", "nonce": "nonce"}

#: Ceiling on the passages accepted in one request. Each one costs three real forward
#: passes; the measurement of §8.3a used six.
MAX_PASSAGES = 12

#: Words per default passage, and how many of them — the shape of the run the numbers in
#: §8.3a come from, so the panel's default reproduces the measured configuration.
DEFAULT_PASSAGE_WORDS = 250
DEFAULT_PASSAGE_COUNT = 6

#: Fraction of the shipped corpus's 250-word blocks that are front matter (title page +
#: the alphabetical index of first lines). Measured: the index ends in block 11 of 63.
FRONT_MATTER_FRACTION = 0.2

#: Verbatim in the response, and rendered by the panel. The residual is stated, not hidden.
UNKNOWN_FORM_NOTE = (
    "Nonce forms fragment into more subword tokens than real words do, so this difference "
    "is the cost of an unknown form TOGETHER WITH the cost of a stranger, longer context. "
    "The two are not separable without a tokenizer-level control, so treat this as an "
    "UPPER BOUND on what a word's location was worth — never as pure location."
)

#: The entropy confound (§8.4), stated wherever a delta is.
CONFOUND_NOTE = (
    "A vacated passage genuinely has higher entropy, so every prediction inside it gets "
    "worse — the scaffolding included. A positive difference is therefore expected, not a "
    "surprise: its MAGNITUDE is the result, and it is only interpretable against the tiny "
    "arm's exact zero."
)

#: The tiny arm's side of the 2×2 (§7.3, §8.3a). Not a measurement made here — it is the
#: Lexicon Lab's, restated so the panel can put the two numbers side by side, which is the
#: entire reason this panel exists.
TINY_ARM = {
    "delta_nats": 0.0,
    "exact": True,
    "label": "the same measurement on a model with no locations",
    "note": (
        "For the from-scratch word-level GeoTransformer of the Lexicon Lab, the vacancy "
        "transform is a pure relabelling of the vocabulary: with consistent=true and "
        "revealAfter=0 the token id stream is element-for-element identical, so the "
        "training loss is bit-identical and a word's FORM is worth exactly 0. That is not "
        "a rounding — it is an identity, and it is asserted in that tab, not assumed."
    ),
}


# --- byte-level alignment (§8.2) ----------------------------------------------------------


@lru_cache(maxsize=1)
def byte_decoder() -> dict[str, int]:
    """GPT-2's unicode→byte table: the inverse of ``bytes_to_unicode``.

    Byte-level BPE renders every byte as a printable character so the vocabulary is
    text; this undoes that, which is what turns a piece into a byte count and therefore
    into a span. All four curated models use a byte-level BPE, and a piece carrying a
    character outside this table is a tokenizer this code has not been verified against —
    so it raises rather than guessing a width.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


def token_byte_spans(pieces: list[str], text: str) -> list[tuple[int, int]]:
    """``[start, end)`` UTF-8 byte range owned by each token, verified by reconstruction.

    The spans tile ``utf8(text)`` exactly once — contiguous, covering, non-overlapping —
    so a per-token quantity may be summed over a word without double-counting. A token
    that is a bare continuation byte gets a degenerate empty span, which is the honest
    answer for it; it still resolves to the right word because its start byte lies strictly
    inside the multi-byte character it continues.

    Raises ``ComputeError`` if the reconstruction is not byte-identical to the input
    (FR-718). There is no fallback: a wrong span silently attributes a token to the wrong
    word, which would corrupt the very number this module exists to report.
    """
    table = byte_decoder()
    spans: list[tuple[int, int]] = []
    rebuilt = bytearray()
    cursor = 0
    for i, piece in enumerate(pieces):
        try:
            raw = bytes(table[c] for c in piece)
        except KeyError as exc:
            raise ComputeError(
                f"token {i} ({piece!r}) contains a character outside the byte-level BPE "
                "table, so its byte width cannot be determined",
                {"index": i, "piece": piece},
            ) from exc
        rebuilt += raw
        spans.append((cursor, cursor + len(raw)))
        cursor += len(raw)

    expected = text.encode("utf-8")
    if bytes(rebuilt) != expected:
        raise ComputeError(
            "token→text alignment failed: the concatenated byte-level pieces do not "
            f"reproduce the passage ({len(rebuilt)} bytes rebuilt vs {len(expected)} "
            "expected). Refusing to attribute tokens to words rather than mis-attribute "
            "them.",
            {"rebuilt_bytes": len(rebuilt), "text_bytes": len(expected)},
        )
    return spans


@dataclass(frozen=True)
class WordSpan:
    """One ``WORD_RE`` match of a passage, in UTF-8 byte coordinates."""

    index: int
    word: str
    start: int
    end: int


def word_spans(text: str) -> list[WordSpan]:
    """Every word of `text`, in byte coordinates, using the TOKENIZER'S OWN regex.

    The same ``WORD_RE`` the vacancy transform rewrites with, so a word here is exactly a
    word there — otherwise the two would disagree about ``good-bye`` and the preserved set
    would not be the set that was actually preserved.
    """
    out: list[WordSpan] = []
    for i, m in enumerate(WORD_RE.finditer(text)):
        start = len(text[: m.start()].encode("utf-8"))
        out.append(WordSpan(i, m.group(0), start, start + len(m.group(0).encode("utf-8"))))
    return out


def preserved_token_indices(
    spans: list[tuple[int, int]],
    words: list[WordSpan],
    preserved: frozenset[int],
) -> list[int]:
    """Indices of the tokens belonging to a PRESERVED word.

    A token belongs to a word when their byte ranges **overlap** — not "starts inside".
    Byte-level BPE folds a word's leading space into the word's own token, so the token
    that *is* the function word starts one byte before the word does; the start rule would
    drop nearly every one of them.

    A token overlapping words with different preserved-ness cannot be attributed, and that
    RAISES. It cannot happen with the byte-level pretokenizers of the curated models
    (each token lies within one word plus its leading whitespace), so if it ever does, the
    assumption behind this whole attribution has changed and the number must not be
    reported (§8.2).
    """
    out: list[int] = []
    for i, (a, b) in enumerate(spans):
        hits = [w for w in words if a < w.end and b > w.start]
        if not hits:
            continue  # punctuation, whitespace, line breaks — scored, attributed to no word
        flags = {w.index in preserved for w in hits}
        if len(flags) > 1:
            raise ComputeError(
                f"token {i} spans both a preserved and a vacated word "
                f"({', '.join(repr(w.word) for w in hits)}); refusing to attribute it",
                {"index": i, "words": [w.word for w in hits]},
            )
        if flags == {True}:
            out.append(i)
    return out


# --- the three variants (§8.3) ------------------------------------------------------------


def _params(p: float, seed: int, match_prosody: bool, keep: frozenset[str], mint: str) -> Any:
    """`VacancyParams` for one variant, with the swap control wired in when it exists.

    `mint` is the newest knob of §7.1 and lands with the transform itself. Rather than
    silently scoring the nonce variant twice if it is not there yet — which would report a
    'cost of wrong content' of ~0 and look like a finding — this raises and names the
    control that is missing.
    """
    if mint not in VARIANT_MINT.values():
        raise InvalidParamError(f"mint must be one of {sorted(set(VARIANT_MINT.values()))}")
    fields = {f.name for f in VacancyParams.__dataclass_fields__.values()}
    if "mint" not in fields:
        raise ComputeError(
            "this build's vacancy transform has no `mint` parameter, so the swap control "
            "of contract §8.3 does not exist and the decomposition cannot be computed",
            {"available": sorted(fields)},
        )
    return VacancyParams(
        p=float(p),
        seed=int(seed),
        consistent=True,
        match_prosody=bool(match_prosody),
        reveal_after=0,
        keep=frozenset(keep),
        mint=mint,
    )


def variant_texts(
    passage: str,
    *,
    p: float,
    seed: int,
    match_prosody: bool,
    keep: frozenset[str],
) -> dict[str, str]:
    """The passage and its two vacated twins, keyed by variant name.

    The map is built over ``vacancy_domain(passage types)`` — the passage's own types plus
    the full Dolch list — exactly as everywhere else, so the nonce a stem gets here is the
    nonce it gets in the Lexicon Lab for the same seed.

    `consistent=True` and `reveal_after=0` are not options here: they are the condition the
    invariance theorem is stated for, and the whole point of this panel is to put the tiny
    arm's exact zero (which holds only there) beside the pretrained number.
    """
    tokens = tokenize(passage)
    domain = vacancy_domain(tokens)
    counts = type_counts(tokens)
    texts = {"english": passage}
    for variant, mint in VARIANT_MINT.items():
        params = _params(p, seed, match_prosody, keep, mint)
        # `counts` is the passage's own type frequencies: the swap control ranks the
        # replacement pool by them (§8.3), and the nonce strategy ignores them entirely.
        vmap = build_vacancy_map(domain, params, counts)
        texts[variant] = vacate_text(passage, vmap, params)
    return texts


def preserved_word_indices(texts: dict[str, str]) -> tuple[list[WordSpan], frozenset[int]]:
    """Word spans of the English passage, and the indices PRESERVED in every variant.

    "Preserved" is character identity across all three variants, which is stronger than
    "closed class": it also covers eligible stems the `u(stem) < p` decision happened to
    spare, and it is exactly the property §8.1 relies on ("character-identical in both
    passages"). Restricting all three NLLs to the SAME word set is what makes them
    comparable at all.

    The transform guarantees each rewritten word is itself a single complete ``WORD_RE``
    match, so the variants have the same word count in the same order; a mismatch means
    that guarantee broke and it raises rather than aligning by luck.
    """
    per_variant = {name: word_spans(text) for name, text in texts.items()}
    counts = {name: len(spans) for name, spans in per_variant.items()}
    if len(set(counts.values())) != 1:
        raise ComputeError(
            "the variants do not have the same number of words, so preserved words "
            f"cannot be aligned: {counts}",
            {"word_counts": counts},
        )
    english = per_variant["english"]
    preserved = frozenset(
        w.index for w in english if all(per_variant[name][w.index].word == w.word for name in texts)
    )
    return english, preserved


# --- scoring ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredText:
    """One real forward pass over one variant of one passage."""

    text: str
    n_tokens: int  # tokens in the passage (position 0 has no prediction)
    nll: list[float]  # nats, index i = cost of predicting token i given tokens < i
    preserved: list[int]  # indices into `nll` (all > 0) of preserved-word tokens

    @property
    def scored(self) -> list[float]:
        return self.nll[1:]


def _max_positions(lm: LoadedModel) -> int | None:
    config = getattr(lm.model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def score_text(
    lm: LoadedModel, text: str, words: list[WordSpan], preserved: frozenset[int]
) -> ScoredText:
    """Per-token NLL from ONE teacher-forced forward pass, plus the preserved indices.

    No special tokens and no chat template: this scores the passage as written, so that
    the three variants differ by the transform and by nothing else. Position 0 has no
    prediction and is excluded everywhere — it is not a zero, it is absent.
    """
    tokenizer = lm.tokenizer
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    n = int(ids.shape[1])
    if n < 2:
        raise InvalidParamError(
            f"a passage must tokenize to at least 2 tokens to be scored, got {n}"
        )
    limit = _max_positions(lm)
    if limit is not None and n > limit:
        raise InvalidParamError(
            f"the passage tokenizes to {n} tokens but {lm.model_id} has a context of "
            f"{limit}. Shorten the passage (or split it across several) — this "
            "measurement is one forward pass over the whole thing, never a truncation, "
            "because a truncated variant would be scoring different text.",
            {"n_tokens": n, "max_context": limit},
        )

    pieces = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    spans = token_byte_spans(list(pieces), text)
    preserved_idx = [i for i in preserved_token_indices(spans, words, preserved) if i > 0]

    with _TRACE_LOCK, torch.no_grad():
        out = lm.model(input_ids=ids, attention_mask=enc["attention_mask"])
    logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
    targets = ids[0, 1:]
    per_token = -logprobs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    nll = [math.nan] + [float(v) for v in per_token]
    return ScoredText(text=text, n_tokens=n, nll=nll, preserved=preserved_idx)


def _mean(values: list[float]) -> float:
    if not values:
        raise ComputeError("no tokens to average — the passage has no scored positions")
    return sum(values) / len(values)


def _stats(scored: ScoredText) -> dict[str, Any]:
    """The fields of §8.1 for one scored variant.

    ``nTokens`` is the number of SCORED positions (every token but the first), because it
    is the count ``nllAll`` averages over — which is what makes
    ``bitsPerChar = nllAll · nTokens / (ln 2 · nChars)`` the passage's total surprisal
    per character rather than an off-by-one approximation of it.
    """
    all_nll = scored.scored
    n_chars = len(scored.text)
    nll_all = _mean(all_nll)
    return {
        "nllPreserved": _mean([scored.nll[i] for i in scored.preserved]),
        "nllAll": nll_all,
        "bitsPerChar": nll_all * len(all_nll) / (math.log(2.0) * n_chars) if n_chars else 0.0,
        "nTokens": len(all_nll),
        "nPreservedTokens": len(scored.preserved),
        "nChars": n_chars,
    }


def _pooled(scores: list[ScoredText]) -> dict[str, Any]:
    """§8.1's fields over several passages, pooled at the TOKEN level.

    Token-weighted, never a mean of means: a passage with twice the tokens carries twice
    the weight, which is the only pooling for which the pooled `nllPreserved` is the mean
    surprisal of a preserved token and the measured quantization bound applies.
    """
    all_nll = [v for s in scores for v in s.scored]
    preserved = [s.nll[i] for s in scores for i in s.preserved]
    n_chars = sum(len(s.text) for s in scores)
    nll_all = _mean(all_nll)
    return {
        "nllPreserved": _mean(preserved),
        "nllAll": nll_all,
        "bitsPerChar": nll_all * len(all_nll) / (math.log(2.0) * n_chars) if n_chars else 0.0,
        "nTokens": len(all_nll),
        "nPreservedTokens": len(preserved),
        "nChars": n_chars,
    }


def _paired_difference(a: list[ScoredText], b: list[ScoredText], label: str) -> dict[str, Any]:
    """``mean(nll_b − nll_a)`` over preserved tokens, PAIRED, with its standard error.

    The pairing is exact and is not an assumption: preserved words are character-identical
    across variants, and the curated models' pretokenizers never merge across a word
    boundary, so each preserved word yields the same pieces in every variant and the
    preserved token lists correspond one-for-one. That is checked here; if it ever fails,
    the difference is refused rather than computed over mismatched tokens.

    Pairing also removes the between-token variance — the same function word is compared
    with itself in the other condition — which is what makes a standard error on an effect
    of ~0.1 nats worth printing at all.
    """
    diffs: list[float] = []
    for sa, sb in zip(a, b):
        if len(sa.preserved) != len(sb.preserved):
            raise ComputeError(
                f"{label}: the variants have {len(sa.preserved)} and {len(sb.preserved)} "
                "preserved tokens, so they cannot be paired",
                {"a": len(sa.preserved), "b": len(sb.preserved)},
            )
        for ia, ib in zip(sa.preserved, sb.preserved):
            diffs.append(sb.nll[ib] - sa.nll[ia])
    n = len(diffs)
    mean = _mean(diffs)
    if n > 1:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = math.nan
    return {"nats": mean, "se": se, "nPairs": n}


# --- default passages ---------------------------------------------------------------------


def default_passages(
    count: int = DEFAULT_PASSAGE_COUNT, words: int = DEFAULT_PASSAGE_WORDS
) -> list[str]:
    """Evenly spaced excerpts of the shipped corpus — the measured configuration (§8.3a).

    Contiguous whole lines, so verse structure survives, and the front matter is skipped:
    *The Real Mother Goose* opens with a title page and an alphabetical index of first
    lines, and a "passage" cut from that is a column of titles rather than English. It
    would raise every condition's NLL together and dilute the contrast with text that is
    not the thing being measured. MEASURED on the shipped corpus: the index runs to block
    11 of 63, so :data:`FRONT_MATTER_FRACTION` of the blocks are dropped. The constant is
    corpus-specific on purpose — this function only ever reads the shipped corpus, and a
    detector for "is this an index?" that worked on one book and silently mis-fired on the
    next would be worse than a measured number with its measurement written down.

    Deterministic: the same corpus gives the same passages, which is what lets the panel's
    default reproduce the run the reference numbers came from, and what the cross-stack
    digest fixture pins.
    """
    from ..lex.corpus import load_corpus_text

    text = unicodedata.normalize("NFC", load_corpus_text())
    blocks: list[str] = []
    current: list[str] = []
    n = 0
    for line in text.split("\n"):
        current.append(line)
        n += len(WORD_RE.findall(line))
        if n >= words:
            blocks.append("\n".join(current).strip("\n"))
            current, n = [], 0
    if not blocks:
        raise ComputeError("the shipped corpus produced no passage of the requested size")
    start = max(1, round(len(blocks) * FRONT_MATTER_FRACTION))
    step = max(1, (len(blocks) - start) // max(1, count))
    out = [blocks[min(start + i * step, len(blocks) - 1)] for i in range(count)]
    return out


# --- the endpoint's computation -----------------------------------------------------------


def vacancy_score(
    model_id: str,
    passages: list[str] | None = None,
    *,
    p: float = 1.0,
    seed: int = 0,
    match_prosody: bool = True,
    keep: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Score every variant of every passage and return the payload of §8.1 and §8.3.

    One real forward pass per (passage, variant) — 3n passes in total — on the real
    pretrained weights at float32. Nothing here is cached across requests: the passages
    are user text and the whole computation is a few seconds of CPU on the curated
    models.
    """
    if passages is None:
        passages = default_passages()
    if not isinstance(passages, list) or not passages:
        raise InvalidParamError("passages must be a non-empty list of strings")
    if len(passages) > MAX_PASSAGES:
        raise InvalidParamError(
            f"at most {MAX_PASSAGES} passages per request, got {len(passages)}; each one "
            "costs three real forward passes"
        )
    for passage in passages:
        if not isinstance(passage, str) or not passage.strip():
            raise InvalidParamError("every passage must be a non-empty string")
    if not 0.0 <= float(p) <= 1.0:
        raise InvalidParamError(f"p must lie in [0, 1], got {p!r}")

    # NFC once, up front, and everything downstream indexes THIS string (§8.2). Qwen's
    # tokenizer normalizes internally and the others do not; without this the byte-span
    # check would fire on a passage that is perfectly well formed.
    normalized = [unicodedata.normalize("NFC", passage) for passage in passages]

    lm = load_model(model_id)
    scored: dict[str, list[ScoredText]] = {name: [] for name in VARIANTS}
    per_passage: list[dict[str, Any]] = []
    for index, passage in enumerate(normalized):
        texts = variant_texts(passage, p=p, seed=seed, match_prosody=match_prosody, keep=keep)
        words, preserved = preserved_word_indices(texts)
        if not preserved:
            raise ComputeError(
                f"passage {index} has no word that survives the transform, so there is no "
                "scaffolding to score. Lower p, or use a passage with closed-class words.",
                {"passage": index},
            )
        row: dict[str, Any] = {
            "index": index,
            "nWords": len(words),
            "nPreservedWords": len(preserved),
            "variants": {},
        }
        for name in VARIANTS:
            # The preserved WORD indices are the English passage's; each variant's own
            # word spans are recomputed because a nonce is a different number of bytes.
            variant_words = word_spans(texts[name])
            result = score_text(lm, texts[name], variant_words, preserved)
            scored[name].append(result)
            row["variants"][name] = _stats(result)
        per_passage.append(row)

    differences = [
        {
            "id": "wrong_content",
            "label": "the cost of wrong content",
            "expr": "nll(swap) − nll(english)",
            "headline": True,
            **_paired_difference(scored["english"], scored["swap"], "swap − english"),
        },
        {
            "id": "unknown_form",
            "label": "the cost of unknown form",
            "expr": "nll(nonce) − nll(swap)",
            "headline": True,
            "upperBound": True,
            "note": UNKNOWN_FORM_NOTE,
            **_paired_difference(scored["swap"], scored["nonce"], "nonce − swap"),
        },
        {
            # Reported, never headlined (§8.3): it is the SUM of the two above, so
            # showing it as "what location was worth" would credit the cost of nonsense
            # to the cost of an unknown word. It is here because it is the one contrast
            # the quantized static build has a measured bound for.
            "id": "total",
            "label": "both costs together",
            "expr": "nll(nonce) − nll(english)",
            "headline": False,
            "note": (
                "The sum of the two differences above. It conflates wrong content with "
                "unknown form and is never the headline."
            ),
            **_paired_difference(scored["english"], scored["nonce"], "nonce − english"),
        },
    ]

    return {
        "model_id": lm.model_id,
        "revision": lm.revision,
        # What actually ran, so a reader never has to guess which error bounds apply.
        "stack": "backend",
        "dtype": "float32",
        "p": float(p),
        "seed": int(seed),
        "match_prosody": bool(match_prosody),
        "keep": sorted(keep),
        "alignment": {
            "mechanism": "byte-level pieces → UTF-8 byte spans",
            "unit": "utf8_bytes",
            "verified": True,
            "note": (
                "Token→word attribution is verified at run time by reconstructing the "
                "passage from the token byte spans; a mismatch raises rather than "
                "mis-attributing."
            ),
        },
        "variants": [
            {
                "id": name,
                "pooled": _pooled(scored[name]),
                "preview": scored[name][0].text[:400],
            }
            for name in VARIANTS
        ],
        # The English passages exactly as scored (NFC-normalized), so the panel can show
        # the reader the text the number came from and let them edit it. Without this the
        # default set would be a black box the UI could only describe.
        "passages_used": normalized,
        "differences": differences,
        "passages": per_passage,
        "tiny_arm": TINY_ARM,
        "confound": CONFOUND_NOTE,
    }
