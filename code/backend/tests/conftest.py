"""Shared test setup.

Points the cache at a throwaway dir BEFORE any llm_geometry import (config reads the
env at import time) so tests never touch the repo cache. Tests use real HuggingFace
models — ``sshleifer/tiny-gpt2`` for speed and ``distilgpt2`` for real-output
assertions — never mocks (Constitution I).
"""

import os
import tempfile

os.environ.setdefault("LLM_GEOMETRY_CACHE_DIR", tempfile.mkdtemp(prefix="llmgeo_test_cache_"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TINY = "sshleifer/tiny-gpt2"
SMALL = "distilgpt2"
