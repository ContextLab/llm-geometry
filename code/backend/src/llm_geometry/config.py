"""Central configuration: paths, the cache schema version, and the model menu.

Everything downstream modules must agree on (cache location, the curated model menu,
the cache schema version) lives here so there is a single source of truth
(Constitution II).

Feature 004 removed the three embedding-geometry views; the reduction defaults,
reference-set sizing, and per-view performance budgets that existed only for them
went with the code they configured.
"""

from __future__ import annotations

import os
from pathlib import Path

# Bump whenever any cached artifact's on-disk format changes. Reads of artifacts
# tagged with a different schema version are treated as cache misses (FR-007).
SCHEMA_VERSION = 14

# Repo root resolved from this file:
# .../code/backend/src/llm_geometry/config.py -> parents[4] == repo root
REPO_ROOT = Path(__file__).resolve().parents[4]


def _default_cache_dir() -> Path:
    override = os.environ.get("LLM_GEOMETRY_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / ".cache" / "llm-geometry"


# Derived cache of precomputed artifacts (git-ignored, regenerable — Constitution II/III).
CACHE_DIR = _default_cache_dir()

# The Architecture Explorer's curated menu (id -> display name).
#
# Curated ONLY (feature 004, FR-412): the app no longer offers an "any open-weights
# HuggingFace id" affordance, because the static build's live path needs a community
# ONNX export that most repos do not have — promising arbitrary ids was a claim the
# deployed demo could not keep. Growing this list (or filtering the Hub for genuinely
# loadable repos) is tracked in GitHub issue #4.
#
# Entries are instruct-tuned unless marked otherwise; a base model has no chat
# template and completes text rather than answering, which the UI states outright.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CURATED_MODELS: dict[str, str] = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B Instruct (default)",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2 360M Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2 135M Instruct (smallest)",
    "gpt2": "GPT-2 124M (base — completes text)",
}

# --- Architecture Explorer (feature 002, /api/arch/*) ------------------------------
# Hard parameter ceiling: larger models are rejected BEFORE any download (FR-107).
ARCH_MAX_PARAMS = 1_500_000_000
# Version of the traced-graph artifact format; part of the graph cache key.
ARCH_GRAPH_SCHEMA_VERSION = 1
# Trace prompts truncate LEFT to this many tokens by default (contracts/api.md).
ARCH_DEFAULT_MAX_CONTEXT = 64
# Attention matrices are downsampled to at most this many rows/cols per head.
ARCH_ATTENTION_MAX_SIDE = 64
# Default exact-window cell budget for /api/arch/weights before downsampling kicks in.
ARCH_WEIGHTS_MAX_CELLS = 4096
# Hard cap on tokens generated per /api/arch/generate call.
ARCH_MAX_NEW_TOKENS = 128

# --- Decoding defaults (feature 004, FR-405) ---------------------------------------
# Unfiltered full-vocab sampling (top_k=0, top_p=1.0) was the primary cause of poor
# replies: at T>0 the long tail gets drawn constantly, which small models cannot
# survive. These are the standard constraints, and the transformers.js runtime in
# code/frontend/src/lib/staticClient/transformersRuntime.ts MUST mirror them so the
# static build and the full stack decode the same way.
ARCH_TOP_P = 0.9
ARCH_TOP_K = 50
ARCH_REPETITION_PENALTY = 1.1
