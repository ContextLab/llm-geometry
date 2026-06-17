"""Central configuration: paths, defaults, seeds, and performance budgets.

Everything that downstream modules need to agree on (cache location, default model,
the cache schema version, reproducibility seeds, and the feature's measurable
performance budgets) lives here so there is a single source of truth (Constitution II).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Bump whenever any cached artifact's on-disk format changes. Reads of artifacts
# tagged with a different schema version are treated as cache misses (FR-007).
SCHEMA_VERSION = 11

# Repo root resolved from this file:
# .../code/backend/src/llm_geometry/config.py -> parents[4] == repo root
REPO_ROOT = Path(__file__).resolve().parents[4]


def _default_cache_dir() -> Path:
    override = os.environ.get("LLM_GEOMETRY_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "data" / "processed" / "cache"


# Derived cache of precomputed artifacts (git-ignored, regenerable — Constitution II/III).
CACHE_DIR = _default_cache_dir()

# Default app model + curated menu (id -> display name). Arbitrary open-weights
# HuggingFace ids are also accepted at runtime (FR-001); the menu is just a
# convenience starting point.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CURATED_MODELS: dict[str, str] = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B Instruct (default)",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2 1.7B Instruct",
    "openai-community/gpt2-large": "GPT-2 Large (774M)",
    "gpt2": "GPT-2 (124M)",
    "distilgpt2": "DistilGPT-2 (82M, fast)",
}

# Reproducibility: a single fixed default seed keeps reductions deterministic (FR-013).
DEFAULT_SEED = 0

# Dimensionality-reduction defaults.
DEFAULT_GRID_N = 25
DEFAULT_2D_METHOD = "pca"
# pca3 is the default 3D method: fast and deterministic (reproducible cache, SC-002).
# Metric MDS is available as an opt-in alternative.
DEFAULT_3D_METHOD = "pca3"

# Manifold RBF cap width on the UNIT sphere (smaller = tighter, more localized domes toward
# likely tokens). Exposed in the UI; this is the default a fresh session starts from.
DEFAULT_RBF_WIDTH = 0.18

# Embedding "reference set": the tokens we compute embeddings for. The full vocab is
# free for the static (input-embedding) source but costs one forward pass per token
# for contextual layers, so the default caps the set for tractability and snappy
# demos (spec Assumption: "vocabulary or a documented, configurable subset"). Set to
# None to use the full vocabulary.
DEFAULT_REFERENCE_SET_SIZE: int | None = 2000

# Batch size for the contextual-embedding forward passes.
EMBED_BATCH_SIZE = 64


@dataclass(frozen=True)
class PerfBudget:
    """The feature's measurable performance budgets (spec Success Criteria)."""

    cache_hit_ms: float = 100.0       # SC-001: backend cache-hit response
    full_roundtrip_ms: float = 1000.0  # SC-001: control change -> preview updated
    target_fps: float = 60.0          # SC-004: shell animations
    first_precompute_s: float = 180.0  # SC-003: first default-model precompute
    progress_min_hz: float = 1.0      # SC-003: progress updates per second


PERF = PerfBudget()
