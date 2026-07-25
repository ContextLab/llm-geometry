#!/usr/bin/env python
"""Generate REAL golden fixtures for the TypeScript geoEngine port (feature 003).

Everything here is produced by the real Python backend — HTTP responses come from a
locally started uvicorn serving llm_geometry.api.app (the same routes the frozen 002
contract freezes), and full-precision artifacts (checkpoint state_dict, seeded preset
matrices, fine-tune losses) come from the backend's own functions. Nothing is
hand-written (FR-203: no fabricated data).

Outputs (all git-committed test fixtures):
  tests/fixtures/geo/checkpoint.json   full-precision canonical state_dict + meta
  tests/fixtures/geo/vocab.json        the canonical tokenizer vocabulary (to_json())
  tests/fixtures/geo/golden.json       route request/response pairs for golden tests
  src/lib/geoEngine/presetFixtures.json
      base64(float32-LE) matrices for the seeded numpy presets random /
      random_autocorr (seeds 0..2) — numpy's PCG64+ziggurat stream is not portable
      to TS, so the static build ships these real backend-computed matrices.

Usage: backend venv python, from anywhere:
  code/backend/.venv/bin/python code/frontend/tests/fixtures/geo/generate.py
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BACKEND = REPO / "code" / "backend"
FRONTEND = REPO / "code" / "frontend"
PORT = 8077
BASE = f"http://127.0.0.1:{PORT}"

sys.path.insert(0, str(BACKEND / "src"))

from llm_geometry.geo.finetune import finetune  # noqa: E402
from llm_geometry.geo.tokenizer import get_tokenizer  # noqa: E402
from llm_geometry.geo.train import train_canonical  # noqa: E402
from llm_geometry.geo.weights import preset_matrix  # noqa: E402
from llm_geometry.geo.config import (  # noqa: E402
    CONTEXT_WINDOW,
    CORPUS_ID,
    D_MODEL,
    MLP_HIDDEN,
    N_HEADS,
    N_LAYERS,
    SEED,
    VOCAB_SIZE,
)

import numpy as np  # noqa: E402


def http_get(path: str, params: dict | None = None):
    url = BASE + path + (("?" + urllib.parse.urlencode(params)) if params else "")
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(path: str, body: dict):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    # 1. Make sure the canonical checkpoint exists (idempotent; cache hit is instant).
    meta = train_canonical()
    print("canonical checkpoint:", meta["checkpoint_id"])

    # 2. Full-precision checkpoint export (state_dict as nested lists; floats are
    #    exact float64 images of the float32 weights, so JSON round-trips bit-exactly).
    from llm_geometry.geo.train import load_canonical_weight_set

    ws = load_canonical_weight_set()
    checkpoint = {
        "format": "geo-checkpoint-v1",
        "config": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "mlp_hidden": MLP_HIDDEN,
            "vocab_size": VOCAB_SIZE,
            "context_window": CONTEXT_WINDOW,
            "tied_unembedding": True,
            "corpus": CORPUS_ID,
            "seed": SEED,
        },
        "meta": {k: meta[k] for k in (
            "checkpoint_id", "final_loss", "coverage_uniformity",
            "field_directional_entropy", "seed",
        )},
        "state_dict": {name: np.asarray(arr, np.float32).tolist() for name, arr in ws.items()},
    }
    (HERE / "checkpoint.json").write_text(json.dumps(checkpoint))
    print("checkpoint.json written")

    # 3. Vocab (the tokenizer's own serialization).
    (HERE / "vocab.json").write_text(get_tokenizer().to_json())
    print("vocab.json written")

    # 4. Seeded preset matrices (numpy PCG64 stream — not portable; ship real values).
    def b64(arr: np.ndarray) -> str:
        a = np.ascontiguousarray(np.asarray(arr, np.float32))
        assert a.dtype == np.dtype("<f4"), a.dtype  # little-endian float32
        return base64.b64encode(a.tobytes()).decode("ascii")

    # 0..2 for the WeightLab UI; 7 because the static-site golden export
    # (static-data/geo/golden.json) exercises random presets with seed 7.
    seeds = [0, 1, 2, 7]
    fixtures = {
        "format": "geo-preset-fixtures-v1",
        "note": (
            "Real backend-computed matrices for the seeded numpy presets. "
            "square = any of W_Q/W_K/W_V/W_O (the backend RNG stream depends only on "
            "shape, not the matrix name); embedding rows are already unit-normalized "
            "exactly as preset_matrix() returns them. base64 of float32 little-endian, "
            "row-major."
        ),
        "seeds": seeds,
        "square": {
            p: {str(s): b64(preset_matrix(p, "W_Q", seed=s)) for s in seeds}
            for p in ("random", "random_autocorr")
        },
        "embedding": {
            p: {str(s): b64(preset_matrix(p, "embedding", seed=s)) for s in seeds}
            for p in ("random", "random_autocorr")
        },
    }
    (FRONTEND / "src" / "lib" / "geoEngine" / "presetFixtures.json").write_text(
        json.dumps(fixtures)
    )
    print("presetFixtures.json written")

    # 5. Fine-tune goldens via the real backend function (synchronous; the HTTP route
    #    wraps this exact function in a job).
    ft_text = (
        "alice was beginning to get very tired of sitting by her sister on the bank , "
        "and of having nothing to do : once or twice she had peeped into the book her "
        "sister was reading , but it had no pictures or conversations in it , and what "
        "is the use of a book , thought alice , without pictures or conversations ? "
        "so she was considering in her own mind , as well as she could , for the hot "
        "day made her feel very sleepy and stupid , whether the pleasure of making a "
        "daisy - chain would be worth the trouble of getting up and picking the "
        "daisies , when suddenly a white rabbit with pink eyes ran close by her ."
    )
    ft = finetune(base="learned", text=ft_text, steps=60, lr=1e-2, seed=SEED)
    finetune_golden = [{
        "body": {"text": ft_text, "steps": 60, "lr": 1e-2, "base": "learned", "seed": SEED},
        "result": {k: ft[k] for k in ("weights_token", "loss_before", "loss_after", "base_token")},
    }]
    print("finetune golden:", finetune_golden[0]["result"])

    # 6. Route goldens over real HTTP.
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "llm_geometry.api.app:app", "--port", str(PORT)],
        cwd=str(BACKEND),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            try:
                http_get("/api/geo/spec")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("uvicorn did not come up on port %d" % PORT)

        spec = http_get("/api/geo/spec")
        assert spec["checkpoint"]["status"] == "ready", spec

        tokenize_texts = [
            "Alice was beginning to get very tired of sitting by her sister",
            "The Queen of Hearts, she made some tarts!",
            "supercalifragilistic quantum chromodynamics xylophone",  # unks
            "“Well!” thought Alice — ‘down the rabbit-hole’",
            "don't won't alice's it's 1865 12 o'clock",
            " ".join(["very"] * 60),  # truncation past CONTEXT_WINDOW
            "",
        ]
        tokenize = [
            {"text": t, "response": http_get("/api/geo/tokenize", {"text": t})}
            for t in tokenize_texts
        ]

        trace_prompts = [
            "alice was beginning to get very tired",
            "the queen of hearts said off with her head",
            "down , down , down . would the fall never come to an end ?",
        ]
        trace = [
            {"prompt": p, "response": http_get("/api/geo/trace", {"prompt": p})}
            for p in trace_prompts
        ]

        vf_cases = [
            {"mode": "next_next", "layer": "full", "prompt": "alice was beginning",
             "temperature": 0.0, "top_m": 1},
            {"mode": "next_next", "layer": 1, "prompt": "the white rabbit",
             "temperature": 0.7, "top_m": 3},
            {"mode": "next_next", "layer": 0, "prompt": "", "temperature": 0.0, "top_m": 1},
            {"mode": "force", "layer": 0, "prompt": "alice was beginning to get very tired",
             "antisymmetrize": False},
            {"mode": "force", "layer": 2, "prompt": "the queen of hearts",
             "antisymmetrize": True},
        ]
        vector_field = [
            {"params": c, "response": http_get("/api/geo/vector_field", c)} for c in vf_cases
        ]

        # Weight minting: exact-token goldens (content hash over the full weight set).
        wp_bodies = [
            {"base": "learned", "edits": [
                {"layer": 0, "matrix": "W_Q", "preset": "identity", "seed": 0}]},
            {"base": "learned", "edits": [
                {"layer": None, "matrix": "embedding", "preset": "toeplitz_fuzzy", "seed": 0}]},
            {"base": "learned", "edits": [
                {"layer": 2, "matrix": "W_V", "preset": "zero", "seed": 0},
                {"layer": 1, "matrix": "W_O", "preset": "learned", "seed": 0}]},
            {"base": "learned", "edits": [
                {"layer": 3, "matrix": "W_K", "preset": "random", "seed": 0},
                {"layer": None, "matrix": "embedding", "preset": "random_autocorr", "seed": 1}]},
            {"base": "learned", "edits": [
                {"layer": 1, "matrix": "W_Q", "values":
                 [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.925]]}]},
            {"base": "learned", "edits": [
                {"layer": 0, "matrix": "W_V", "preset": "random", "seed": 2},
                {"layer": 0, "matrix": "W_Q", "preset": "toeplitz_fuzzy", "seed": 0}]},
        ]
        weights_post = [{"body": b, "response": http_post("/api/geo/weights", b)}
                        for b in wp_bodies]
        # A chained mint: base = the token minted by case 0 (tests base resolution +
        # per-matrix source inheritance).
        chain_body = {
            "base": weights_post[0]["response"]["weights_token"],
            "edits": [{"layer": 0, "matrix": "W_K", "preset": "random_autocorr", "seed": 0}],
        }
        weights_post.append(
            {"body": chain_body, "response": http_post("/api/geo/weights", chain_body)}
        )

        weights_get = [
            {"params": {"matrix": "W_Q", "layer": 0},
             "response": http_get("/api/geo/weights", {"matrix": "W_Q", "layer": 0})},
            {"params": {"matrix": "embedding"},
             "response": http_get("/api/geo/weights", {"matrix": "embedding"})},
            {"params": {"matrix": "W_Q", "layer": 0,
                        "weights_token": weights_post[0]["response"]["weights_token"]},
             "response": http_get("/api/geo/weights", {
                 "matrix": "W_Q", "layer": 0,
                 "weights_token": weights_post[0]["response"]["weights_token"]})},
            {"params": {"matrix": "W_V", "layer": 2,
                        "weights_token": weights_post[2]["response"]["weights_token"]},
             "response": http_get("/api/geo/weights", {
                 "matrix": "W_V", "layer": 2,
                 "weights_token": weights_post[2]["response"]["weights_token"]})},
        ]

        golden = {
            "format": "geo-golden-v1",
            "generated": date.today().isoformat(),
            "source": "real backend routes via uvicorn on :%d + llm_geometry.geo functions"
                      % PORT,
            "spec": spec,
            "tokenize": tokenize,
            "trace": trace,
            "vector_field": vector_field,
            "weights_get": weights_get,
            "weights_post": weights_post,
            "finetune": finetune_golden,
        }
        (HERE / "golden.json").write_text(json.dumps(golden))
        print("golden.json written (%.1f KB)" % ((HERE / "golden.json").stat().st_size / 1024))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
