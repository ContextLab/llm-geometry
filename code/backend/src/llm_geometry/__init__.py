"""llm_geometry — core machinery for geometric LLM visualizations.

Subpackages:
  models   — open-weights model loading + capability detection
  compute  — real model-derived quantities (next-token distributions, embeddings)
  reduce   — dimensionality reduction (2D PCA/UMAP, grid, 3D spherical)
  cache    — deterministic keys + integrity-checked artifact store
  jobs     — precompute job registry (single-flight + progress)
  api      — FastAPI app serving cached artifacts + SSE progress
"""

__version__ = "0.1.0"
