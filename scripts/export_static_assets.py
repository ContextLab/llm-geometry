#!/usr/bin/env python
"""Export precomputed static assets for the GitHub Pages build (feature 003).

Runs the REAL backend (FastAPI TestClient against ``llm_geometry.api.app:app``) and
writes contract-exact JSON/bin artifacts into ``code/frontend/public/static-data/``
so the static site (``VITE_DATA_MODE=static``) can serve them with no Python backend.
Every exported response body is literally what the live route returned — the frozen
contract in specs/002-interactive-model-explorer/contracts/api.md governs the shapes.

Usage (from the backend venv):

    python scripts/export_static_assets.py \
        --git-sha "$(cat .git/refs/heads/...)" --generated-at 2026-07-24T00:00:00Z

    python scripts/export_static_assets.py --quick --out /tmp/static-data ...

``--quick`` (integration-test mode): geo assets in full; arch graph + tiles + 2
traces for gpt2 only; one small preset per 001 view (gpt2). Everything still comes
from the real backend — quick only shrinks model/param coverage, never fabricates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "code" / "frontend" / "public" / "static-data"

SCHEMA_VERSION = 1

# --- curated coverage -------------------------------------------------------------------

ARCH_MODELS_FULL = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
ARCH_MODELS_QUICK = ["gpt2"]

PRESET_MODEL_FULL = "Qwen/Qwen2.5-0.5B-Instruct"
PRESET_MODEL_QUICK = "gpt2"

# The frontend's default shared state (code/frontend/src/lib/stores.ts) + per-view
# constants (viz/VectorField.svelte GRID_N/REF/SEED, viz/Manifold.svelte MARKERS,
# viz/Sankey.svelte SEED). Preset 1 of each view IS the default state so the tab
# renders instantly in static mode.
DEFAULT_PREFIX = "The capital of France is"
VECTOR_GRID_N = 24
VECTOR_REF = 400
VECTOR_ANIM_REF = 576
MANIFOLD_MARKERS = 2000
SEED = 0

# 6 example prompts for the precomputed Architecture-Explorer trace dropdown
# (spec US-2: includes the France one and a system-prompt one).
TRACE_PROMPTS: list[dict[str, Any]] = [
    {"label": "Capital of France", "prompt": "The capital of France is"},
    {
        "label": "Greeting (with system prompt)",
        "prompt": "Hello! How are you today?",
        "system_prompt": "You are a concise, friendly assistant.",
    },
    {"label": "Story opener", "prompt": "Once upon a time"},
    {"label": "Simple arithmetic", "prompt": "2 + 2 ="},
    {"label": "Translation", "prompt": "The French word for cat is"},
    {"label": "Quick brown fox", "prompt": "The quick brown fox jumps over the lazy"},
]

# --- geo golden vectors -----------------------------------------------------------------

GOLDEN_PROMPT_PLAIN = (
    "alice was beginning to get very tired of sitting by her sister on the bank"
)
GOLDEN_PROMPT_UNK = (
    "alice saw a quantum xylophone and a blockchain beside the pool of tears"
)
GOLDEN_PROMPT_LONG = (
    "So she was considering in her own mind, as well as she could, for the hot day made "
    "her feel very sleepy and stupid, whether the pleasure of making a daisy-chain would "
    "be worth the trouble of getting up and picking the daisies, when suddenly a White "
    "Rabbit with pink eyes ran close by her."
)
GOLDEN_PROMPTS = [GOLDEN_PROMPT_PLAIN, GOLDEN_PROMPT_UNK, GOLDEN_PROMPT_LONG]

GOLDEN_FIELD_CONFIGS: list[dict[str, Any]] = [
    {
        "mode": "next_next",
        "layer": "full",
        "temperature": 0.0,
        "top_m": 1,
        "antisymmetrize": False,
    },
    {
        "mode": "next_next",
        "layer": 1,
        "temperature": 1.0,
        "top_m": 3,
        "antisymmetrize": False,
    },
    {
        "mode": "force",
        "layer": 0,
        "temperature": 0.0,
        "top_m": 1,
        "antisymmetrize": False,
    },
    {
        "mode": "force",
        "layer": 0,
        "temperature": 0.0,
        "top_m": 1,
        "antisymmetrize": True,
    },
]

GOLDEN_EDIT_BODIES: list[dict[str, Any]] = [
    {"base": "learned", "edits": [{"layer": 0, "matrix": "W_Q", "preset": "identity"}]},
    {
        "base": "learned",
        "edits": [{"layer": 1, "matrix": "W_V", "preset": "toeplitz_fuzzy"}],
    },
    {
        "base": "learned",
        "edits": [{"layer": 2, "matrix": "W_K", "preset": "random", "seed": 7}],
    },
    {
        "base": "learned",
        "edits": [
            {
                "layer": 3,
                "matrix": "W_O",
                "values": [[0.25, -0.5, 0.1], [0.0, 1.0, -0.25], [0.75, 0.3, -1.5]],
            }
        ],
    },
    {
        "base": "learned",
        "edits": [{"layer": 0, "matrix": "embedding", "preset": "random", "seed": 7}],
    },
]

GOLDEN_FINETUNE_TEXT = (
    "alice took up the fan and gloves, and, as the hall was very hot, she kept fanning "
    "herself all the time she went on talking: dear, dear! how queer everything is today!"
)
GOLDEN_FINETUNE_STEPS = 50


# --- small helpers ----------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def write_json(path: Path, obj: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, separators=(",", ":"))
    path.write_text(data, encoding="utf-8")
    return len(data.encode("utf-8"))


def full_precision(arr: np.ndarray) -> list:
    """float32 array -> nested Python lists whose JSON repr round-trips exactly.

    ``tolist()`` yields the float64 value of each float32 entry; ``json.dumps`` emits
    the shortest string that round-trips that float64, so parsing in JS (f64) and
    narrowing to Float32Array reproduces the backend's float32 bits exactly.
    """
    return np.asarray(arr, dtype=np.float32).tolist()


def _clean(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def get_json(
    client: Any, path: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    r = client.get(path, params=_clean(params or {}))
    if r.status_code != 200:
        raise SystemExit(f"GET {path} {params} -> HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def post_json(
    client: Any, path: str, body: dict[str, Any], ok: tuple[int, ...] = (200,)
) -> tuple[int, dict[str, Any]]:
    r = client.post(path, json=body)
    if r.status_code not in ok:
        raise SystemExit(f"POST {path} -> HTTP {r.status_code}: {r.text[:500]}")
    return r.status_code, r.json()


def wait_job(client: Any, job_id: str, label: str, timeout_s: float = 3600.0) -> None:
    t0 = time.time()
    last = ""
    while True:
        job = get_json(client, f"/api/jobs/{job_id}")
        msg = f"{job['status']} {job['progress']:.2f} {job['message']}"
        if msg != last:
            log(f"  {label}: {msg}")
            last = msg
        if job["status"] == "done":
            return
        if job["status"] == "error":
            raise SystemExit(f"{label} job failed: {job['error']}")
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"{label} job timed out after {timeout_s}s")
        time.sleep(0.5)


# --- geo --------------------------------------------------------------------------------


def export_geo(client: Any, out: Path) -> dict[str, Any]:
    from llm_geometry.geo.config import (
        CONTEXT_WINDOW,
        CORPUS_ID,
        D_MODEL,
        FINETUNE_DEFAULT_LR,
        MLP_HIDDEN,
        N_HEADS,
        N_LAYERS,
        SPECIAL_TOKENS,
        VOCAB_SIZE,
        SEED as GEO_SEED,
    )
    from llm_geometry.geo.model import GeoTransformer
    from llm_geometry.geo.tokenizer import get_tokenizer
    from llm_geometry.geo.train import load_canonical_weight_set
    from llm_geometry.geo.weights import load_weight_set

    log("geo: ensuring the canonical checkpoint is trained")
    _, train = post_json(client, "/api/geo/train", {}, ok=(200, 202))
    if not train.get("ready"):
        wait_job(client, train["job_id"], "geo train")

    spec = get_json(client, "/api/geo/spec")
    if spec["checkpoint"]["status"] != "ready":
        raise SystemExit(
            f"geo checkpoint not ready after training: {spec['checkpoint']}"
        )
    write_json(out / "geo" / "spec.json", spec)

    # checkpoint.json — the complete state_dict (nothing the forward pass needs missing).
    ws = load_canonical_weight_set()
    model_names = set(GeoTransformer().state_dict().keys())
    if set(ws.keys()) != model_names:
        raise SystemExit(
            f"canonical weight set does not cover state_dict: missing="
            f"{sorted(model_names - set(ws))} extra={sorted(set(ws) - model_names)}"
        )
    checkpoint = {
        "format": "geo-checkpoint-v1",
        "config": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "mlp_hidden": MLP_HIDDEN,
            "vocab_size": VOCAB_SIZE,
            "context_window": CONTEXT_WINDOW,
            "seed": GEO_SEED,
            "tied_unembedding": True,
            "corpus": CORPUS_ID,
            "special_tokens": dict(SPECIAL_TOKENS),
        },
        "metrics": dict(spec["checkpoint"]),
        "weight_names": sorted(ws.keys()),
        "shapes": {name: list(np.asarray(a).shape) for name, a in ws.items()},
        "weights": {name: full_precision(a) for name, a in ws.items()},
    }
    write_json(out / "geo" / "checkpoint.json", checkpoint)

    tok = get_tokenizer()
    vocab = {
        "format": "geo-vocab-v1",
        "vocab_size": VOCAB_SIZE,
        "specials": dict(SPECIAL_TOKENS),
        "tokens": [tok.id_to_text[i] for i in range(VOCAB_SIZE)],
    }
    write_json(out / "geo" / "vocab.json", vocab)

    # golden.json — real route responses the TS engine must reproduce (<= 1e-5).
    log("geo: golden vectors (trace + fields per prompt)")
    tk_unk = get_json(client, "/api/geo/tokenize", {"text": GOLDEN_PROMPT_UNK})
    if tk_unk["n_unk"] < 1:
        raise SystemExit("golden unk prompt produced no <unk> tokens")
    tk_long = get_json(client, "/api/geo/tokenize", {"text": GOLDEN_PROMPT_LONG})
    if not tk_long["truncated"]:
        raise SystemExit(
            "golden long prompt did not exceed the 50-token context window"
        )

    cases = []
    for prompt in GOLDEN_PROMPTS:
        trace = get_json(client, "/api/geo/trace", {"prompt": prompt})
        fields = []
        for cfg in GOLDEN_FIELD_CONFIGS:
            resp = get_json(client, "/api/geo/vector_field", {"prompt": prompt, **cfg})
            fields.append({"params": dict(cfg), "response": resp})
        cases.append({"prompt": prompt, "trace": trace, "fields": fields})

    log("geo: golden weight edits")
    weight_edits = []
    for body in GOLDEN_EDIT_BODIES:
        _, posted = post_json(client, "/api/geo/weights", body)
        resolved_ws = load_weight_set(posted["weights_token"])
        weight_edits.append(
            {
                "body": body,
                "weights_token": posted["weights_token"],
                "resolved": {
                    name: full_precision(a) for name, a in resolved_ws.items()
                },
            }
        )

    log("geo: golden fine-tune (50 steps)")
    ft_body = {"text": GOLDEN_FINETUNE_TEXT, "steps": GOLDEN_FINETUNE_STEPS}
    status, ft = post_json(client, "/api/geo/finetune", ft_body, ok=(200, 202))
    if status == 202:
        wait_job(client, ft["job_id"], "geo finetune")
        _, ft = post_json(client, "/api/geo/finetune", ft_body, ok=(200,))
    golden = {
        "cases": cases,
        "weight_edits": weight_edits,
        "finetune": {
            "text": GOLDEN_FINETUNE_TEXT,
            "steps": GOLDEN_FINETUNE_STEPS,
            "lr": FINETUNE_DEFAULT_LR,
            "seed": GEO_SEED,
            "loss_before": ft["loss_before"],
            "loss_after": ft["loss_after"],
            "weights_token": ft["weights_token"],
        },
    }
    write_json(out / "geo" / "golden.json", golden)
    return {"checkpoint_id": spec["checkpoint"]["checkpoint_id"]}


# --- arch -------------------------------------------------------------------------------


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _safetensors_meta(model_id: str, revision: str) -> dict[str, Any]:
    """Verify + record how the raw weights are served on the HF CDN (pinned revision)."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(model_id, revision=revision)
    base = f"https://huggingface.co/{model_id}/resolve/{revision}"
    meta: dict[str, Any] = {"model_id": model_id, "revision": revision}
    if "model.safetensors" in files:
        meta["safetensors_url"] = f"{base}/model.safetensors"
    elif "model.safetensors.index.json" in files:
        shards = sorted(f for f in files if f.endswith(".safetensors"))
        meta["safetensors_index_url"] = f"{base}/model.safetensors.index.json"
        meta["shard_urls"] = [f"{base}/{f}" for f in shards]
    else:
        raise SystemExit(
            f"{model_id}@{revision} has no model.safetensors (files: {files[:20]})"
        )
    return meta


def export_arch_model(
    client: Any, out: Path, model_id: str, n_traces: int
) -> dict[str, Any]:
    from llm_geometry.models.loader import load_model, resolve_model

    slug = _slug(model_id)
    mdir = out / "arch" / slug
    log(f"arch[{model_id}]: graph")
    graph = get_json(client, "/api/arch/graph", {"model_id": model_id})
    write_json(mdir / "graph.json", graph)

    ref = resolve_model(model_id)
    revision = ref["revision"]
    write_json(mdir / "meta.json", _safetensors_meta(model_id, revision))

    # Weight-overview tiles: the exact grid GET /api/arch/weights serves for the FULL
    # window of every state_dict param, quantized to uint8 with per-tile min/max.
    log(f"arch[{model_id}]: tiles for every param")
    lm = load_model(model_id)
    param_names = list(lm.model.state_dict().keys())
    tiles: list[dict[str, Any]] = []
    blob = bytearray()
    for pname in param_names:
        w = get_json(
            client, "/api/arch/weights", {"model_id": model_id, "param": pname}
        )
        vals = np.asarray(w["values"], dtype=np.float64)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax > vmin:
            q = np.clip(np.round((vals - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(
                np.uint8
            )
        else:
            q = np.zeros(vals.shape, dtype=np.uint8)
        tiles.append(
            {
                "param": pname,
                "shape": w["shape"],
                "grid_shape": w["grid_shape"],
                "downsampled": w["downsampled"],
                "method": w["method"],
                "stats": w["stats"],
                "offset": len(blob),
                "nbytes": int(q.size),
                "vmin": vmin,
                "vmax": vmax,
            }
        )
        blob.extend(q.tobytes(order="C"))
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "tiles.bin").write_bytes(bytes(blob))
    write_json(
        mdir / "tiles.json",
        {
            "model_id": model_id,
            "revision": revision,
            "dtype": "uint8",
            "bin": "tiles.bin",
            "encoding": "value = vmin + (u8 / 255) * (vmax - vmin), row-major grid_shape",
            "tiles": tiles,
        },
    )

    prompts = TRACE_PROMPTS[:n_traces]
    index = []
    for n, spec in enumerate(prompts, start=1):
        log(f"arch[{model_id}]: trace {n}/{len(prompts)} ({spec['label']})")
        params = {"model_id": model_id, "prompt": spec["prompt"]}
        if spec.get("system_prompt"):
            params["system_prompt"] = spec["system_prompt"]
        trace = get_json(client, "/api/arch/trace", params)
        write_json(mdir / "traces" / f"{n}.json", trace)
        entry = {
            "n": n,
            "label": spec["label"],
            "prompt": spec["prompt"],
            "file": f"{n}.json",
        }
        if spec.get("system_prompt"):
            entry["system_prompt"] = spec["system_prompt"]
        index.append(entry)
    write_json(mdir / "traces" / "index.json", {"model_id": model_id, "traces": index})
    return {
        "model_id": model_id,
        "slug": slug,
        "revision": revision,
        "n_params": len(param_names),
    }


# --- 001 presets ------------------------------------------------------------------------


def _req(client: Any, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "params": params,
        "response": get_json(client, endpoint, params),
    }


def _vector_static(
    client: Any,
    mid: str,
    prefix: str,
    temp: float,
    layer: int,
    fanout: int,
    grid_n: int,
    ref: int,
) -> dict[str, Any]:
    return _req(
        client,
        "/api/vector_field",
        {
            "model_id": mid,
            "prefix_text": prefix,
            "response_text": "",
            "response_step": 0,
            "temperature": temp,
            "layer_from": layer,
            "layer_to": layer,
            "grid_n": grid_n,
            "fanout": fanout,
            "reference_set_size": ref,
            "seed": SEED,
        },
    )


def preset_specs(model_id: str, quick: bool) -> dict[str, list[dict[str, Any]]]:
    """The labeled preset configs per 001 view (preset 1 = the view's default state)."""
    if quick:
        # Quick mode trims COUNTS (1 preset per view, fewer particles/steps — those are
        # state-carried, so the UI requests them once the preset state is applied) but
        # MUST keep the request constants the views hard-code (VectorField GRID_N/REF,
        # Manifold MARKERS): a preset recorded with different grid_n/reference_set_size
        # can never be matched by a real view request (staticClient serves only exact
        # param-dict hits), which would make quick exports invisible to the UI and to
        # the static e2e suite (003-D).
        return {
            "vector": [
                {
                    "label": "Default (quick)",
                    "state": {
                        "prefix_text": DEFAULT_PREFIX,
                        "temperature": 1.0,
                        "layer_from": 0,
                        "layer_to": 0,
                        "fanout": 2,
                        "response_text": "",
                    },
                    "kind": "static",
                },
            ],
            "sankey": [
                {
                    "label": "Default (quick)",
                    "state": {
                        "prefix_text": DEFAULT_PREFIX,
                        "temperature": 1.0,
                        "n_particles": 40,
                        "n_steps": 4,
                        "response_text": "",
                    },
                },
            ],
            "manifold": [
                {
                    "label": "Default (quick)",
                    "state": {
                        "prefix_text": DEFAULT_PREFIX,
                        "temperature": 1.0,
                        "width": 0.18,
                        "response_text": "",
                    },
                },
            ],
        }
    return {
        "vector": [
            {
                "label": "Default · layer 0",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.0,
                    "layer_from": 0,
                    "layer_to": 0,
                    "fanout": 2,
                    "response_text": "",
                },
                "kind": "static",
            },
            {
                "label": "Final layer, deterministic (T=0)",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 0.0,
                    "layer_from": 23,
                    "layer_to": 23,
                    # fanout 1: the backend clamps fan-out at temperature 0, so a
                    # recorded 2 only creates a control-vs-caption contradiction
                    # (red-team static F3).
                    "fanout": 1,
                    "response_text": "",
                },
                "kind": "static",
            },
            {
                "label": "Response animation · “Once upon a time”",
                "state": {
                    "prefix_text": "Once upon a time",
                    "temperature": 1.0,
                    "layer_from": 0,
                    "layer_to": 0,
                    "fanout": 2,
                    "response_text": " there was a princess",
                },
                "kind": "animation",
            },
            {
                "label": "Hot fan-out (T=1.5, 4 arrows)",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.5,
                    "layer_from": 0,
                    "layer_to": 0,
                    "fanout": 4,
                    "response_text": "",
                },
                "kind": "static",
            },
        ],
        "sankey": [
            {
                "label": "Default swarm",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.0,
                    "n_particles": 1000,
                    "n_steps": 10,
                    "response_text": "",
                },
            },
            {
                "label": "Cool swarm (T=0.3)",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 0.3,
                    "n_particles": 1000,
                    "n_steps": 10,
                    "response_text": "",
                },
            },
            {
                "label": "“Once upon a time”",
                "state": {
                    "prefix_text": "Once upon a time",
                    "temperature": 1.0,
                    "n_particles": 1000,
                    "n_steps": 10,
                    "response_text": "",
                },
            },
            {
                "label": "Response highlight · “ Paris”",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.0,
                    "n_particles": 1000,
                    "n_steps": 10,
                    "response_text": " Paris",
                },
            },
        ],
        "manifold": [
            {
                "label": "Default manifold",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.0,
                    "width": 0.18,
                    "response_text": "",
                },
            },
            {
                "label": "Tight caps (width 0.10)",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.0,
                    "width": 0.10,
                    "response_text": "",
                },
            },
            {
                "label": "Wide caps (width 0.30)",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.0,
                    "width": 0.30,
                    "response_text": "",
                },
            },
            {
                "label": "Response animation · “ Paris”",
                "state": {
                    "prefix_text": DEFAULT_PREFIX,
                    "temperature": 1.0,
                    "width": 0.18,
                    "response_text": " Paris",
                },
            },
        ],
    }


def export_presets(
    client: Any, out: Path, model_id: str, quick: bool
) -> dict[str, Any]:
    specs = preset_specs(model_id, quick)
    labels: dict[str, list[dict[str, Any]]] = {}

    def tokenize_req(text: str) -> dict[str, Any]:
        # ResponseAnimator calls /api/tokenize for the response text in these flows.
        return _req(client, "/api/tokenize", {"model_id": model_id, "text": text})

    for n, spec in enumerate(specs["vector"], start=1):
        log(f"presets/vector/{n}: {spec['label']}")
        st = spec["state"]
        grid_n = spec.get("grid_n", VECTOR_GRID_N)
        requests: list[dict[str, Any]] = []
        if spec.get("kind") == "animation":
            requests.append(
                _req(
                    client,
                    "/api/vector_field_animation",
                    {
                        "model_id": model_id,
                        "prefix_text": st["prefix_text"],
                        "response_text": st["response_text"],
                        "temperature": st["temperature"],
                        "layer_to": st["layer_to"],
                        "reference_set_size": VECTOR_ANIM_REF,
                        "grid_n": grid_n,
                        "seed": SEED,
                    },
                )
            )
            requests.append(tokenize_req(st["response_text"]))
        else:
            requests.append(
                _vector_static(
                    client,
                    model_id,
                    st["prefix_text"],
                    st["temperature"],
                    st["layer_from"],
                    st["fanout"],
                    grid_n,
                    spec.get("ref", VECTOR_REF),
                )
            )
        size = write_json(
            out / "presets" / "vector" / f"{n}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "view": "vector",
                "n": n,
                "label": spec["label"],
                "model_id": model_id,
                "state": st,
                "requests": requests,
            },
        )
        labels.setdefault("vector", []).append(
            {"n": n, "label": spec["label"], "file": f"{n}.json", "bytes": size}
        )

    for n, spec in enumerate(specs["sankey"], start=1):
        log(f"presets/sankey/{n}: {spec['label']}")
        st = spec["state"]
        requests = [
            _req(
                client,
                "/api/sankey",
                {
                    "model_id": model_id,
                    "prefix_text": st["prefix_text"],
                    "temperature": st["temperature"],
                    "n_particles": st["n_particles"],
                    "n_steps": st["n_steps"],
                    "seed": SEED,
                },
            )
        ]
        if st["response_text"]:
            requests.append(
                _req(
                    client,
                    "/api/sankey_highlight",
                    {
                        "model_id": model_id,
                        "prefix_text": st["prefix_text"],
                        "response_text": st["response_text"],
                        "temperature": st["temperature"],
                        "n_steps": st["n_steps"],
                    },
                )
            )
            requests.append(tokenize_req(st["response_text"]))
        size = write_json(
            out / "presets" / "sankey" / f"{n}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "view": "sankey",
                "n": n,
                "label": spec["label"],
                "model_id": model_id,
                "state": st,
                "requests": requests,
            },
        )
        labels.setdefault("sankey", []).append(
            {"n": n, "label": spec["label"], "file": f"{n}.json", "bytes": size}
        )

    for n, spec in enumerate(specs["manifold"], start=1):
        log(f"presets/manifold/{n}: {spec['label']}")
        st = spec["state"]
        markers = spec.get("markers", MANIFOLD_MARKERS)
        base_params = {
            "model_id": model_id,
            "prefix_text": st["prefix_text"],
            "temperature": st["temperature"],
            "seed": SEED,
            "reference_set_size": markers,
            "width": st["width"],
        }
        if st["response_text"]:
            requests = [
                _req(
                    client,
                    "/api/manifold_animation",
                    {**base_params, "response_text": st["response_text"]},
                ),
                tokenize_req(st["response_text"]),
            ]
        else:
            requests = [_req(client, "/api/manifold", base_params)]
        size = write_json(
            out / "presets" / "manifold" / f"{n}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "view": "manifold",
                "n": n,
                "label": spec["label"],
                "model_id": model_id,
                "state": st,
                "requests": requests,
            },
        )
        labels.setdefault("manifold", []).append(
            {"n": n, "label": spec["label"], "file": f"{n}.json", "bytes": size}
        )

    # The shared full-vocab token cloud at the frontend's default (seed=0, spread_mu=0.65).
    log("presets: token_cloud (default seed/spread)")
    cloud = _req(
        client,
        "/api/token_cloud",
        {"model_id": model_id, "seed": SEED, "spread_mu": 0.65},
    )
    write_json(out / "presets" / "token_cloud.json", cloud)
    return labels


# --- manifest ---------------------------------------------------------------------------


def build_index(
    out: Path,
    args: argparse.Namespace,
    arch_meta: list[dict[str, Any]],
    preset_labels: dict[str, Any],
    geo_meta: dict[str, Any],
) -> None:
    files: dict[str, int] = {}
    for p in sorted(out.rglob("*")):
        # Exclude only the TOP-LEVEL manifest itself; per-model traces/index.json
        # files are real assets and must be listed.
        if p.is_file() and str(p.relative_to(out)) != "index.json":
            files[str(p.relative_to(out))] = p.stat().st_size

    # A partial `--only` run must MERGE the untouched sections' metadata from the
    # existing manifest — clobbering them empties the static model catalog while
    # the section files are still on disk (broke the static build once).
    prior: dict[str, Any] = {}
    prior_path = out / "index.json"
    if prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text())
        except Exception:
            prior = {}
    if not geo_meta and prior.get("geo"):
        geo_meta = prior["geo"]
    if not arch_meta and prior.get("arch_models"):
        arch_meta = prior["arch_models"]
    if not preset_labels and prior.get("presets"):
        preset_labels = prior["presets"]

    write_json(
        out / "index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": args.generated_at,
            "git_sha": args.git_sha,
            "quick": bool(args.quick),
            "geo": geo_meta,
            "arch_models": arch_meta,
            "preset_model": args.preset_model,
            "presets": preset_labels,
            "files": files,
            "total_bytes": sum(files.values()),
        },
    )
    log(f"manifest: {len(files)} files, {sum(files.values()) / 1e6:.2f} MB total")


# --- main -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="output directory"
    )
    parser.add_argument(
        "--quick", action="store_true", help="reduced coverage for tests"
    )
    parser.add_argument(
        "--git-sha", default="unknown", help="git commit sha for the manifest"
    )
    parser.add_argument(
        "--generated-at",
        default="unknown",
        help="ISO-8601 build timestamp for the manifest",
    )
    parser.add_argument(
        "--only",
        default="geo,arch,presets",
        help="comma list of sections to export (geo,arch,presets)",
    )
    parser.add_argument(
        "--preset-model", default=None, help="override the 001 preset model id"
    )
    args = parser.parse_args(argv)

    sections = {s.strip() for s in args.only.split(",") if s.strip()}
    if args.preset_model is None:
        args.preset_model = PRESET_MODEL_QUICK if args.quick else PRESET_MODEL_FULL
    arch_models = ARCH_MODELS_QUICK if args.quick else ARCH_MODELS_FULL
    n_traces = 2 if args.quick else len(TRACE_PROMPTS)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from fastapi.testclient import TestClient

    from llm_geometry.api.app import app

    geo_meta: dict[str, Any] = {}
    arch_meta: list[dict[str, Any]] = []
    preset_labels: dict[str, Any] = {}
    with TestClient(app) as client:
        if "geo" in sections:
            geo_meta = export_geo(client, out)
        if "arch" in sections:
            for mid in arch_models:
                arch_meta.append(export_arch_model(client, out, mid, n_traces))
        if "presets" in sections:
            preset_labels = export_presets(client, out, args.preset_model, args.quick)

    build_index(out, args, arch_meta, preset_labels, geo_meta)
    log(f"done in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
