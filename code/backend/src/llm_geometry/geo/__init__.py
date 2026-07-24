"""Geometry Lab: a tiny, fully-transparent decoder-only transformer (GeoTransformer).

This package is the backend for the Geometry tab (feature 002): a from-scratch
``d_model=3`` transformer trained on a real public-domain corpus whose 3-D token
embeddings live on the unit sphere. Everything here is real — real corpus, real
training, real SGD fine-tuning — per FR-109 (no mocks, no simulated data).

Modules:
  config     — fixed architecture + training hyperparameters + gate thresholds
  corpus     — Alice's Adventures in Wonderland (Project Gutenberg #11) acquisition
  tokenizer  — deterministic word-level tokenizer (top-1000 corpus types + specials)
  model      — the GeoTransformer itself (PyTorch, per-matrix addressable weights)
  weights    — presets, weight-set editing, content-hash tokens, persistence
  train      — seeded deterministic training + canonical checkpoint + gate metrics
  fields     — the two vector-field modes (next_next, force) per the frozen contract
  finetune   — real SGD fine-tuning on text / files / HF datasets (never mutates
               the canonical checkpoint)
"""
