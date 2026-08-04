"""Generation from the Lexicon Lab's model — in-budget by construction.

The vocabulary **is** the budget, so there is no trie, no post-filter, and no way for an
out-of-budget word to appear: the only rows that could print something that is not a
budget word are the specials, and `GENERATION_BANNED_IDS` masks `<unk>`, `<bos>` and
`<pad>` to `-inf` before sampling. `<eos>` stays sampleable — it is how the model ends a
line. This is what makes the source's `mask_decode.py` defect (a trie that re-opened its
root mid-word and emitted `" hameat"`) structurally impossible here.

`temperature = 0` is greedy. The source clamped it to `1e-6` and always sampled, which
makes "show me this model's single most likely continuation" unavailable.

`<bos>` seeds an empty prompt, so that row is a real input the model has seen rather than
the source's dead embedding row (its `bos` variable actually held `<eos>`).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from ..errors import ComputeError, InvalidParamError
from .config import (
    BOS_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    EOS_ID,
    GENERATION_BANNED_IDS,
    MAX_NEW_TOKENS,
    SPECIAL_TOKENS,
)
from .model import LexModel
from .vocab import LexVocab, tokenize


def _banned_mask(vocab_rows: int, banned_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    """Additive `-inf` mask over the banned rows, built once per call."""
    mask = torch.zeros(vocab_rows, device=device)
    for token_id in banned_ids:
        if not 0 <= int(token_id) < vocab_rows:
            raise InvalidParamError(f"banned id {token_id} is outside 0..{vocab_rows - 1}")
        mask[int(token_id)] = float("-inf")
    if not torch.isfinite(mask).any():
        raise InvalidParamError(
            "every vocabulary row is banned — there is nothing left to generate"
        )
    return mask


@torch.no_grad()
def generate_ids(
    model: LexModel,
    prompt_ids: Sequence[int] | None = None,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    banned_ids: Sequence[int] = GENERATION_BANNED_IDS,
    stop_at_eos: bool = False,
) -> list[int]:
    """Continue `prompt_ids`; return the whole sequence (prompt included).

    `<eos>` ends a *line*, not the sample, so by default generation runs the full
    `max_new_tokens` and the caller renders line breaks. Pass `stop_at_eos=True` to get a
    single line — the model's own idea of where the line stops.

    Sampling is seeded, so a saved model reloaded from a bundle regenerates exactly the
    same text (SC-607). At `temperature == 0` the sampler is greedy and the seed is inert.
    """
    max_new_tokens = int(max_new_tokens)
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise InvalidParamError(f"max_new_tokens must be in 1..{MAX_NEW_TOKENS}")
    temperature = float(temperature)
    if temperature < 0:
        raise InvalidParamError(f"temperature must be non-negative, got {temperature}")

    cfg = model.cfg
    ids = list(int(i) for i in (prompt_ids or []))
    if not ids:
        ids = [BOS_ID]  # a real input row, not a dead one
    for token_id in ids:
        if not 0 <= token_id < cfg.vocab_rows:
            raise InvalidParamError(f"prompt id {token_id} is outside 0..{cfg.vocab_rows - 1}")

    device = next(model.parameters()).device
    mask = _banned_mask(cfg.vocab_rows, banned_ids, device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    was_training = model.training
    model.eval()  # dropout must not perturb generation
    try:
        for _ in range(max_new_tokens):
            window = torch.tensor([ids[-cfg.ctx :]], dtype=torch.long, device=device)
            logits = model(window)[0, -1].float() + mask
            if not torch.isfinite(logits).any():
                raise ComputeError("all logits are -inf; generation cannot continue")
            if temperature == 0.0:
                next_id = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                if not torch.isfinite(probs).all():
                    raise ComputeError(
                        "the model produced non-finite probabilities; refusing to sample"
                    )
                next_id = int(torch.multinomial(probs.cpu(), 1, generator=gen))
            ids.append(next_id)
            if stop_at_eos and next_id == EOS_ID:
                break
    finally:
        model.train(was_training)
    return ids


def generate_text(
    model: LexModel,
    vocab: LexVocab,
    prompt: str = "",
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    banned_ids: Sequence[int] = GENERATION_BANNED_IDS,
    stop_at_eos: bool = False,
) -> str:
    """Generate a continuation of `prompt` as words. Every word is in `vocab` by design.

    `<eos>` renders as a line break, which is what it means in a corpus of nursery rhymes.
    Out-of-budget prompt words become `<unk>` on the way IN — that is legitimate input, and
    the reported `<unk>` rate is exactly the measurable form of what a budget cannot say —
    but no special token can ever come back OUT.
    """
    if vocab.rows != model.cfg.vocab_rows:
        raise InvalidParamError(
            f"this vocabulary has {vocab.rows} rows but the model has "
            f"{model.cfg.vocab_rows}; they are not the same model"
        )
    prompt_ids = vocab.encode(tokenize(prompt))
    n_prompt = len(prompt_ids)
    ids = generate_ids(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        banned_ids=banned_ids,
        stop_at_eos=stop_at_eos,
    )
    specials = set(SPECIAL_TOKENS)
    lines: list[list[str]] = [[]]
    for word in vocab.decode(ids[max(n_prompt, 1) :]):
        if word == "<eos>":
            lines.append([])
        elif word not in specials:
            lines[-1].append(word)
    return "\n".join(" ".join(line) for line in lines if line)
