"""FR-102 — weight views serve real values: exact under zoom, honest when downsampled."""

import numpy as np
import pytest

from llm_geometry.arch.weights import weight_window
from llm_geometry.errors import InvalidParamError, NotFoundError
from llm_geometry.models.loader import load_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Llama-style but WITH attention biases (1-D case)
PARAM = "model.layers.0.self_attn.q_proj.weight"


@pytest.fixture(scope="module")
def lm():
    return load_model(MODEL)


def test_exact_window_equals_state_dict_slice(lm):
    res = weight_window(MODEL, PARAM, r0=3, r1=11, c0=5, c1=13)
    expected = lm.model.state_dict()[PARAM][3:11, 5:13].float().numpy()
    assert res["method"] == "exact" and res["downsampled"] is False
    assert res["grid_shape"] == [8, 8]
    assert res["values"] == expected.astype(float).tolist()  # elementwise, no rounding
    assert res["stats"]["min"] == pytest.approx(float(expected.min()))
    assert res["stats"]["max"] == pytest.approx(float(expected.max()))


def test_large_window_downsamples_with_true_window_stats(lm):
    res = weight_window(MODEL, PARAM)  # full 896x896 >> max_cells
    full = lm.model.state_dict()[PARAM].float().numpy()
    assert res["method"] == "strided_mean" and res["downsampled"] is True
    assert res["shape"] == [896, 896]
    gr, gc = res["grid_shape"]
    assert gr <= 64 and gc <= 64
    assert len(res["values"]) == gr and len(res["values"][0]) == gc
    # stats are over the REQUESTED window, not the downsampled grid
    assert res["stats"]["mean"] == pytest.approx(float(full.mean()), abs=1e-6)
    assert res["stats"]["min"] == pytest.approx(float(full.min()), abs=1e-6)
    assert res["stats"]["std"] == pytest.approx(float(full.std()), abs=1e-6)
    # grid values are real bin means, inside the window's range
    grid = np.array(res["values"])
    assert grid.min() >= full.min() and grid.max() <= full.max()


def test_1d_param_served_as_single_column(lm):
    bias_path = "model.layers.0.self_attn.q_proj.bias"
    res = weight_window(MODEL, bias_path, r0=0, r1=8)
    expected = lm.model.state_dict()[bias_path][0:8].float().numpy().reshape(-1, 1)
    assert res["shape"][1] == 1  # C=1 by contract
    assert res["grid_shape"] == [8, 1]
    assert res["values"] == expected.astype(float).tolist()


def test_unknown_param_raises_not_found():
    with pytest.raises(NotFoundError):
        weight_window(MODEL, "model.layers.0.no_such_param.weight")


def test_out_of_range_window_rejected():
    with pytest.raises(InvalidParamError):
        weight_window(MODEL, PARAM, r0=10, r1=5)
    with pytest.raises(InvalidParamError):
        weight_window(MODEL, PARAM, r0=0, r1=10_000_000)
