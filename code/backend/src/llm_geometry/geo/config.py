"""Geometry Lab configuration: the fixed GeoTransformer architecture + training knobs.

These values are FROZEN by ``specs/002-interactive-model-explorer/contracts/api.md``
(`/api/geo/spec`): d_model=3, n_layers=4, n_heads=1, mlp_hidden=12, vocab_size=1003
(1000 corpus word types + <unk>/<eos>/<pad>), context_window=50, tied unembedding,
unit-norm embeddings on S². Do not change them without editing the contract first.

This module is geo-local on purpose — the top-level ``llm_geometry.config`` is owned
by the 001 machinery (and by another workstream this batch); the Geometry Lab only
*reads* shared values (CACHE_DIR, SCHEMA_VERSION) from it.
"""

from __future__ import annotations

from ..config import REPO_ROOT

# --- Fixed architecture (frozen contract) -------------------------------------------
D_MODEL = 3
N_LAYERS = 4
N_HEADS = 1
MLP_HIDDEN = 12
VOCAB_WORDS = 1000  # word/punctuation types drawn from the corpus by frequency
VOCAB_SIZE = 1003  # VOCAB_WORDS + the three specials below
CONTEXT_WINDOW = 50
SEED = 0

# Special token ids (frozen contract: "special_tokens": {"unk": 0, "eos": 1, "pad": 2}).
UNK_ID = 0
EOS_ID = 1
PAD_ID = 2
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS: dict[str, int] = {"unk": UNK_ID, "eos": EOS_ID, "pad": PAD_ID}

# --- Corpus (real public-domain text; FR-109 forbids mock text) ---------------------
# Alice's Adventures in Wonderland, Project Gutenberg ebook #11. The raw file is
# committed to the repo (public domain) so CI never re-downloads it.
CORPUS_ID = "gutenberg-11-alice-in-wonderland"
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "alice-in-wonderland.txt"
CORPUS_URLS = (
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/ebooks/11.txt.utf-8",
)
# sha256 of the raw downloaded file (recorded at first download, 2026-07-24).
CORPUS_SHA256 = "a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754"
GUTENBERG_START_MARKER = "*** START OF THE PROJECT GUTENBERG"
GUTENBERG_END_MARKER = "*** END OF THE PROJECT GUTENBERG"

# --- Canonical training hyperparameters (must finish < ~3 min on CPU) ---------------
TRAIN_EPOCHS = 30
TRAIN_LR = 2e-2  # Adam
TRAIN_BATCH_SIZE = 64
TRAIN_WINDOW_STRIDE = 10  # tokens between successive training windows
# Spherical-uniformity auxiliary loss (pairwise Gaussian repulsion on sampled rows).
# Tuned empirically (2026-07-24 sweep over {0.05, 0.15, 0.3, 0.6}): 0.3 keeps the
# embedding well spread (coverage ≈ 0.90, directional entropy ≈ 2.8 nats) at a
# negligible CE cost (4.89 vs 4.76 nats at 0.05, which fell below the coverage gate).
REPULSION_WEIGHT = 0.3
REPULSION_SAMPLE = 256  # embedding rows sampled per step
REPULSION_T = 2.0  # Gaussian-potential temperature (Wang & Isola uniformity loss)

# --- Fine-tuning limits (frozen contract: steps <= 500, default 100, lr 1e-2) -------
FINETUNE_MAX_STEPS = 500
FINETUNE_DEFAULT_STEPS = 100
FINETUNE_DEFAULT_LR = 1e-2
FINETUNE_DEFAULT_MAX_SAMPLES = 200

# --- Non-degeneracy gates (SC-103) --------------------------------------------------
# Both metrics bin S² into SPHERE_BINS cells around a Fibonacci-lattice direction set.
SPHERE_BINS = 64
# coverage_uniformity: normalized entropy (0..1) of embedding-row occupancy over the
# bins. Uniformly scattered points score ~0.95+; a single cluster scores near 0.
MIN_COVERAGE_UNIFORMITY = 0.80
# field_directional_entropy: Shannon entropy (nats) of next_next arrow directions over
# the bins; max is ln(64) ≈ 4.16. A degenerate "always predict the same token" field
# scores near 0. Threshold chosen with margin below the trained model's actual value.
MIN_FIELD_DIRECTIONAL_ENTROPY = 2.0
