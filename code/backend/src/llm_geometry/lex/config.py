"""Constants for the Lexicon Lab (feature 006).

Every number the UI quotes about this tab is imported from here or measured at runtime.
`tests/e2e/docs.spec.ts` pins the ones cheapest to let rot, as feature 005 established.
"""

from __future__ import annotations

from pathlib import Path

# --- corpus --------------------------------------------------------------------------

#: Project Gutenberg #10607, "The Real Mother Goose" (1916). Public domain in the USA.
#: Committed whole, header and licence footer intact, exactly as the Geometry Lab ships
#: Alice — that satisfies the PG licence with no stripping machinery.
CORPUS_PATH = Path(__file__).parent / "data" / "real-mother-goose.txt"
CORPUS_TITLE = "The Real Mother Goose"
CORPUS_YEAR = 1916
CORPUS_GUTENBERG_ID = 10607
#: Verified by download on 2026-08-03; `test_corpus.py` re-checks the committed bytes.
CORPUS_SHA256 = "d514f0fd2cd40967eb6cf35b140a6cddc11200126e07d76603fae3f88bf1e0ab"
CORPUS_BYTES = 110_445
#: `ebooks/<id>.txt.utf-8` and `files/<id>/<id>-0.txt` serve DIFFERENT bytes for this
#: book; only the first matches CORPUS_SHA256. Order matters.
CORPUS_URLS = (
    f"https://www.gutenberg.org/ebooks/{CORPUS_GUTENBERG_ID}.txt.utf-8",
    f"https://www.gutenberg.org/cache/epub/{CORPUS_GUTENBERG_ID}/pg{CORPUS_GUTENBERG_ID}.txt",
)

GUTENBERG_START_MARKER = "*** START OF THE PROJECT GUTENBERG"
GUTENBERG_END_MARKER = "*** END OF THE PROJECT GUTENBERG"

# --- vocabulary ----------------------------------------------------------------------

#: Reserved rows, always present whatever the budget. `<unk>` is index 0 so that a
#: missing entry can never silently alias a real word.
SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>", "<pad>")
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

#: Never sampled during generation: `<unk>` would print a hole, `<bos>`/`<pad>` are
#: structural. `<eos>` IS sampleable — it is how the model ends a line.
GENERATION_BANNED_IDS = (UNK_ID, BOS_ID, PAD_ID)

BUDGET_SOURCES = ("dolch", "frequency")
DEFAULT_BUDGET_SOURCE = "dolch"
DEFAULT_BUDGET = "full"

# --- model ---------------------------------------------------------------------------

D_MODEL_CHOICES = (16, 32, 64, 128)
N_LAYER_CHOICES = (1, 2, 3, 4)
N_HEAD_CHOICES = (1, 2, 4)
CTX_CHOICES = (32, 64, 128)

DEFAULT_D_MODEL = 64
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 2
#: 32, not 64. Two independent reasons, both measured on the committed corpus at the
#: other defaults (400 steps, 318 vocab rows, `d_model` 64, 2 layers):
#:
#:   * QUALITY. 19,050 tokens is a small corpus and a 64-token window spans several
#:     unrelated rhymes. ctx 64 reaches a lower TRAIN loss (2.247 vs 2.406 at batch 32)
#:     and a WORSE held-out loss (2.294 vs 2.197) — it is memorizing, not generalizing.
#:   * SPEED. Per-step cost is roughly `batch x ctx x params` for the projections and
#:     `batch x ctx^2` for attention, so halving ctx is the cheapest real lever there is.
#:
#: `CTX_CHOICES` still offers 64 and 128 for anyone who wants the longer window.
DEFAULT_CTX = 32
DEFAULT_TIED = True
#: The source hard-codes 0.1 and does not expose it. A live demo wants determinism.
DEFAULT_DROPOUT = 0.0

LAYER_NORM_EPS = 1e-5
MLP_RATIO = 4

# --- training ------------------------------------------------------------------------

DEFAULT_STEPS = 400
MAX_STEPS = 3000
DEFAULT_LR = 3e-3  # peak of the one-cycle schedule
#: 16, not 32 — the other half of making the default run feel live rather than batched.
#: With DEFAULT_CTX = 32 this puts a default 400-step run at ~47 s of browser compute
#: (measured in Node on the same engine), against ~193 s for the old ctx 64 / batch 32.
#: The cost is a slightly noisier gradient: held-out loss after 400 steps is 2.333 rather
#: than 2.294. Raising `steps` to 500 recovers it (2.258) at ~56 s, which is the trade
#: the UI's sliders exist to let a user make; the DEFAULT keeps the headroom.
DEFAULT_BATCH = 16
DEFAULT_WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
#: One-cycle: warm up over this fraction of the run, then anneal.
ONECYCLE_PCT_START = 0.3
ONECYCLE_DIV_FACTOR = 25.0
ONECYCLE_FINAL_DIV_FACTOR = 1e4
VAL_FRACTION = 0.05
DEFAULT_SEED = 0
#: Emit a sample every N steps while training so the user watches text improve.
DEFAULT_SAMPLE_EVERY = 50

DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_NEW_TOKENS = 40
MAX_NEW_TOKENS = 200

# --- geometry ------------------------------------------------------------------------

#: Components kept for the PCA token cloud. This is a PROJECTION and is labelled as one:
#: unlike the Geometry Lab's sphere, these coordinates are not the representation itself.
PCA_COMPONENTS = 3
#: Bars drawn in the spectrum plot (the leading singular values).
SPECTRUM_DISPLAY_K = 48


def param_count(vocab_rows: int, d_model: int, n_layers: int, ctx: int, tied: bool) -> int:
    """Exact parameter count.

    Verified against the source implementation on 7 configurations
    (`notes/agent-reports/006-source-model-arch.md`):

        N = (2 if untied else 1)·V·d + ctx·d + L·(12d² + 13d) + 2d

    The `12d² + 13d` per block is: QKV packed projection (3d² + 3d), attention output
    projection (d² + d), MLP up (4d² + 4d), MLP down (4d² + d), and two LayerNorms (4d).
    """
    embed = (1 if tied else 2) * vocab_rows * d_model
    return embed + ctx * d_model + n_layers * (12 * d_model**2 + 13 * d_model) + 2 * d_model
