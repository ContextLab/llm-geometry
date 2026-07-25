"""Integration test for scripts/export_static_assets.py (feature 003, agent 003-A).

Runs the REAL exporter in ``--quick`` mode (geo assets in full; arch graph + tiles +
2 traces for gpt2; one small preset per 001 view) into a tmp dir, then validates the
emitted artifacts against the live backend (FastAPI TestClient, no mocks):

- every emitted .json parses;
- golden.json carries the coordinated schema (cases/weight_edits/finetune) with the
  required prompt coverage (an <unk> prompt, a >50-token truncated prompt);
- checkpoint.json enumerates the GeoTransformer state_dict COMPLETELY and matches the
  canonical learned weights bit-for-bit (float32 round-trip);
- tiles.bin dequantizes back to the live /api/arch/weights grids within the uint8
  quantization bound (max abs err <= (max-min)/254);
- exported responses equal live route responses (contract field names + values).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.geo.model import GeoTransformer
from llm_geometry.geo.train import load_canonical_weight_set

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "export_static_assets.py"

GEO_WEIGHT_NAMES = 2 + 4 * 8  # embedding + pos_embedding + 4 layers x 8 params


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("static-data")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--out",
            str(out),
            "--git-sha",
            "test-sha",
            "--generated-at",
            "1970-01-01T00:00:00Z",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert (
        result.returncode == 0
    ), f"exporter failed:\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
    return out


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_json_parses(export_dir: Path) -> None:
    paths = sorted(export_dir.rglob("*.json"))
    assert len(paths) >= 12, f"expected a full quick export, found only {paths}"
    for p in paths:
        _load(p)  # raises on malformed JSON


def test_golden_schema(export_dir: Path) -> None:
    golden = _load(export_dir / "geo" / "golden.json")
    assert set(golden.keys()) == {"cases", "weight_edits", "finetune"}

    cases = golden["cases"]
    assert len(cases) >= 3
    for case in cases:
        assert set(case.keys()) == {"prompt", "trace", "fields"}
        trace = case["trace"]
        assert set(trace.keys()) == {
            "tokens",
            "embeddings",
            "layers",
            "probs",
            "logits_topk",
            "next_token",
        }
        assert len(trace["layers"]) == 4
        assert len(trace["probs"]) == 1003
        for field in case["fields"]:
            assert set(field.keys()) == {"params", "response"}
            assert set(field["params"].keys()) == {
                "mode",
                "layer",
                "temperature",
                "top_m",
                "antisymmetrize",
            }
            resp = field["response"]
            assert set(resp.keys()) == {
                "mode",
                "layer",
                "points",
                "token_ids",
                "arrows",
                "sequence_forces",
                "tangent_exact",
            }
            assert len(resp["points"]) == 1003
    # Required prompt coverage: one prompt with <unk> tokens, one truncated at 50.
    assert any(any(t["unk"] for t in c["trace"]["tokens"]) for c in cases)
    assert any(len(c["trace"]["tokens"]) == 50 for c in cases)
    # Field coverage: both modes, both antisymmetrize values, a T>0 top_m>1 config.
    all_params = [f["params"] for c in cases for f in c["fields"]]
    assert any(p["mode"] == "next_next" and p["layer"] == "full" for p in all_params)
    assert any(
        p["mode"] == "next_next" and p["temperature"] > 0 and p["top_m"] > 1
        for p in all_params
    )
    assert any(p["mode"] == "force" and p["antisymmetrize"] for p in all_params)
    assert any(p["mode"] == "force" and not p["antisymmetrize"] for p in all_params)

    edits = golden["weight_edits"]
    assert len(edits) >= 4
    for entry in edits:
        assert set(entry.keys()) == {"body", "weights_token", "resolved"}
        assert len(entry["resolved"]) == GEO_WEIGHT_NAMES
        assert isinstance(entry["weights_token"], str) and entry["weights_token"]
    presets_used = {
        e["body"]["edits"][0].get("preset")
        for e in edits
        if e["body"]["edits"][0].get("preset") is not None
    }
    assert {"identity", "toeplitz_fuzzy", "random"} <= presets_used
    assert any(e["body"]["edits"][0].get("values") is not None for e in edits)

    ft = golden["finetune"]
    assert set(ft.keys()) == {
        "text",
        "steps",
        "lr",
        "seed",
        "loss_before",
        "loss_after",
        "weights_token",
    }
    assert ft["steps"] == 50
    assert np.isfinite(ft["loss_before"]) and np.isfinite(ft["loss_after"])


def test_checkpoint_complete_and_exact(export_dir: Path) -> None:
    ckpt = _load(export_dir / "geo" / "checkpoint.json")
    state_names = set(GeoTransformer().state_dict().keys())
    assert (
        set(ckpt["weights"].keys()) == state_names
    ), "checkpoint must cover state_dict fully"
    assert set(ckpt["shapes"].keys()) == state_names
    canonical = load_canonical_weight_set()
    for name, arr in canonical.items():
        exported = np.asarray(ckpt["weights"][name], dtype=np.float32)
        assert list(exported.shape) == ckpt["shapes"][name]
        assert np.array_equal(
            exported, np.asarray(arr, dtype=np.float32)
        ), f"checkpoint weight {name} does not round-trip exactly"
    cfg = ckpt["config"]
    assert cfg["d_model"] == 3 and cfg["n_layers"] == 4 and cfg["vocab_size"] == 1003
    assert cfg["special_tokens"] == {"unk": 0, "eos": 1, "pad": 2}

    vocab = _load(export_dir / "geo" / "vocab.json")
    assert len(vocab["tokens"]) == 1003
    assert vocab["tokens"][0] == "<unk>" and vocab["specials"] == {
        "unk": 0,
        "eos": 1,
        "pad": 2,
    }


def test_geo_spec_matches_live(export_dir: Path, client: TestClient) -> None:
    exported = _load(export_dir / "geo" / "spec.json")
    live = client.get("/api/geo/spec").json()
    assert exported == live
    assert exported["checkpoint"]["status"] == "ready"


def test_tiles_dequantize_within_bound(export_dir: Path, client: TestClient) -> None:
    mdir = export_dir / "arch" / "gpt2"
    manifest = _load(mdir / "tiles.json")
    blob = (mdir / "tiles.bin").read_bytes()
    tiles = manifest["tiles"]
    assert len(blob) == sum(t["nbytes"] for t in tiles)
    assert len(blob) < 2_000_000, "tiles must stay under 2 MB per model"
    param_names = {t["param"] for t in tiles}
    assert "transformer.wte.weight" in param_names

    sample = tiles[:3] + [t for t in tiles if t["param"] == "transformer.wte.weight"]
    for tile in sample:
        q = np.frombuffer(
            blob, dtype=np.uint8, count=tile["nbytes"], offset=tile["offset"]
        ).reshape(tile["grid_shape"])
        vmin, vmax = tile["vmin"], tile["vmax"]
        deq = vmin + (q.astype(np.float64) / 255.0) * (vmax - vmin)
        live = client.get(
            "/api/arch/weights", params={"model_id": "gpt2", "param": tile["param"]}
        ).json()
        assert live["grid_shape"] == tile["grid_shape"]
        assert live["shape"] == tile["shape"]
        err = np.max(np.abs(deq - np.asarray(live["values"], dtype=np.float64)))
        assert (
            err <= (vmax - vmin) / 254.0 + 1e-12
        ), f"{tile['param']}: dequantization error {err} exceeds uint8 bound"


def test_arch_graph_meta_and_traces(export_dir: Path, client: TestClient) -> None:
    mdir = export_dir / "arch" / "gpt2"
    graph = _load(mdir / "graph.json")
    assert set(graph.keys()) == {"model_id", "schema_version", "meta", "nodes", "edges"}
    live = client.get("/api/arch/graph", params={"model_id": "gpt2"}).json()
    assert graph == live

    meta = _load(mdir / "meta.json")
    assert meta["model_id"] == "gpt2"
    assert meta["revision"]
    assert "safetensors_url" in meta or "safetensors_index_url" in meta
    if "safetensors_url" in meta:
        assert meta["revision"] in meta["safetensors_url"]

    index = _load(mdir / "traces" / "index.json")
    assert len(index["traces"]) == 2
    for entry in index["traces"]:
        trace = _load(mdir / "traces" / entry["file"])
        assert set(trace.keys()) == {
            "tokens",
            "chat_template_used",
            "truncated",
            "layers",
            "logits_topk",
            "node_activations",
        }
        assert len(trace["layers"]) == 12  # gpt2
        assert trace["node_activations"], "trace must carry per-node activations"


def test_presets_match_live_routes(export_dir: Path, client: TestClient) -> None:
    for view in ("vector", "sankey", "manifold"):
        preset = _load(export_dir / "presets" / view / "1.json")
        assert preset["view"] == view and preset["label"] and preset["state"]
        assert preset["requests"], f"{view} preset has no recorded requests"
        for req in preset["requests"]:
            assert set(req.keys()) == {"endpoint", "params", "response"}
            live = client.get(req["endpoint"], params=req["params"])
            assert (
                live.status_code == 200
            ), f"{req['endpoint']} {req['params']}: {live.text[:300]}"
            assert (
                live.json() == req["response"]
            ), f"{view} preset response drifted from the live route {req['endpoint']}"
    cloud = _load(export_dir / "presets" / "token_cloud.json")
    assert set(cloud.keys()) == {"endpoint", "params", "response"}
    assert {"coords", "token_ids", "token_strs"} <= set(cloud["response"].keys())


def test_index_manifest(export_dir: Path) -> None:
    index = _load(export_dir / "index.json")
    assert index["git_sha"] == "test-sha"
    assert index["generated_at"] == "1970-01-01T00:00:00Z"
    assert index["quick"] is True
    assert index["files"], "manifest must list emitted files"
    actual = {
        str(p.relative_to(export_dir))
        for p in export_dir.rglob("*")
        if p.is_file() and str(p.relative_to(export_dir)) != "index.json"
    }
    assert set(index["files"]) == actual, (
        f"manifest file list drifted: unlisted={sorted(actual - set(index['files']))} "
        f"missing={sorted(set(index['files']) - actual)}"
    )
    for rel, size in index["files"].items():
        p = export_dir / rel
        assert p.is_file(), f"manifest lists missing file {rel}"
        assert p.stat().st_size == size, f"manifest size drifted for {rel}"
    assert index["total_bytes"] == sum(index["files"].values())
    assert [m["model_id"] for m in index["arch_models"]] == ["gpt2"]
    assert set(index["presets"].keys()) == {"vector", "sankey", "manifold"}
