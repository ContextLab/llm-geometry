"""llm_geometry — the backend for the two-tab LLM geometry explorer.

Subpackages:
  models   — open-weights model loading + capability detection
  arch     — Architecture Explorer: traced forward passes, weight windows, generation
  geo      — Geometry Lab: the from-scratch d_model=3 GeoTransformer and its fields
  cache    — deterministic keys + integrity-checked artifact store
  jobs     — job registry (single-flight + progress)
  api      — FastAPI app serving /api/arch/*, /api/geo/*, and shared job/SSE routes
"""

__version__ = "0.2.0"
