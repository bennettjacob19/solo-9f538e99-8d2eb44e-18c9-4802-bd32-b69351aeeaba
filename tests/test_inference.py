"""Correctness tests: tiled inference must equal whole-image inference
pixel-for-pixel, across odd/even sizes, tile boundaries and image borders.
"""

from __future__ import annotations

import io
import json
import os
import tempfile

import numpy as np
import pytest

from app.inference import (
    ModelParamError,
    parse_model_params,
    reference_full_inference,
    tiled_segment,
)


def _make_layers(seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return parse_model_params(
        json.dumps(
            {
                "layers": [
                    {"kernel": rng.normal(scale=2.0, size=(3, 3)).tolist(),
                     "bias": float(rng.normal(scale=0.5))},
                    {"kernel": rng.normal(scale=2.0, size=(3, 3)).tolist(),
                     "bias": float(rng.normal(scale=0.5))},
                ]
            }
        )
    )


def _run_tiled(image: np.ndarray, layers: list, tile: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        img_path = os.path.join(d, "in.npy")
        feat_path = os.path.join(d, "feat.npy")
        out_path = os.path.join(d, "out.npy")
        np.save(img_path, image)
        h2, w2 = tiled_segment(img_path, layers, feat_path, out_path, tile=tile)
        result = np.load(out_path, mmap_mode="r")
        assert result.shape == (h2, w2)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1})
        return np.array(result)


@pytest.mark.parametrize(
    "shape,tile",
    [
        ((1, 1), 1),
        ((2, 2), 1),
        ((3, 3), 1),
        ((5, 7), 2),       # both odd, tiles smaller than the image
        ((7, 5), 3),
        ((11, 13), 4),
        ((16, 16), 4),     # tile grid divides evenly
        ((17, 19), 4),     # odd dimensions + ragged last tile
        ((33, 31), 8),
        ((64, 48), 16),
        ((100, 77), 32),
        ((257, 255), 64),  # larger, many seams, odd sizes
    ],
)
def test_tiled_matches_reference(shape, tile):
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    layers = _make_layers(seed=shape[0] * 31 + shape[1])
    expected = reference_full_inference(image, layers)
    got = _run_tiled(image, layers, tile)
    assert got.shape == expected.shape == (
        (shape[0] + 1) // 2,
        (shape[1] + 1) // 2,
    )
    assert np.array_equal(got, expected)


def test_different_tile_sizes_agree():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(53, 41), dtype=np.uint8)
    layers = _make_layers(seed=99)
    ref = reference_full_inference(image, layers)
    for tile in (1, 3, 7, 16, 64, 512):
        assert np.array_equal(_run_tiled(image, layers, tile), ref)


@pytest.mark.parametrize("tile", [1, 3, 8, 512])
def test_checkpoint_per_tile_keeps_result_identical(tmp_path, tile):
    """A hand-over hook at every tile boundary must not change any pixel."""
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(37, 45), dtype=np.uint8)
    layers = _make_layers(seed=4)
    img_path = tmp_path / "in.npy"
    np.save(img_path, image)
    calls = []

    def hook():
        calls.append(1)

    tiled_segment(
        str(img_path), layers,
        str(tmp_path / "f.npy"), str(tmp_path / "o.npy"),
        tile=tile, checkpoint=hook,
    )
    got = np.load(tmp_path / "o.npy")
    assert np.array_equal(got, reference_full_inference(image, layers))
    h1, w1 = (image.shape[0] + 1) // 2, (image.shape[1] + 1) // 2
    # Checks after every tile of layer 1, plus every layer-2 tile except the
    # last one (the mask is complete there, so there is nothing to resume).
    expected_calls = (
        len(range(0, h1, tile)) * len(range(0, w1, tile))
        + len(range(0, h1, tile)) * len(range(0, w1, tile)) - 1
    )
    assert len(calls) == expected_calls


def test_checkpoint_exception_aborts_and_is_not_swallowed(tmp_path):
    image = np.zeros((8, 8), dtype=np.uint8)
    img_path = tmp_path / "in.npy"
    np.save(img_path, image)

    def hook():
        raise RuntimeError("budget exhausted")

    with pytest.raises(RuntimeError):
        tiled_segment(
            str(img_path), _make_layers(),
            str(tmp_path / "f.npy"), str(tmp_path / "o.npy"),
            tile=2, checkpoint=hook,
        )


def test_borders_see_zero_padding_only():
    # All-zero image with a bias pattern: layer1 bias 1 -> all ReLU outputs 1;
    # layer2 sums the 3x3 ones-kernel over padded feat -> interior 9, border 6,
    # corner 4; bias -5 keeps corners (4-5 < 0) at 0, everything else at 1.
    params = {
        "layers": [
            {"kernel": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "bias": 1.0},
            {"kernel": [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "bias": -5.0},
        ]
    }
    layers = parse_model_params(json.dumps(params))
    image = np.zeros((9, 7), dtype=np.uint8)
    got = _run_tiled(image, layers, tile=2)
    expected = reference_full_inference(image, layers)
    assert np.array_equal(got, expected)
    # explicit border/corner expectations for a 5x4 output
    assert got[0, 0] == 0 and got[0, -1] == 0 and got[-1, 0] == 0 and got[-1, -1] == 0
    assert got[0, 1] == 1 and got[1, 0] == 1 and got[-1, 1] == 1


def test_threshold_is_strictly_greater_than_zero():
    # Score forced to exactly 0 everywhere: must be labelled 0, not 1.
    params = {
        "layers": [
            {"kernel": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "bias": 0.0},
            {"kernel": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "bias": 0.0},
        ]
    }
    layers = parse_model_params(json.dumps(params))
    image = np.full((5, 5), 200, dtype=np.uint8)
    got = _run_tiled(image, layers, tile=3)
    assert np.array_equal(got, np.zeros((3, 3), dtype=np.uint8))


def test_integer_inputs_not_scaled():
    # A single hot pixel: hand-checkable receptive-field behaviour.
    params = {
        "layers": [
            {"kernel": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "bias": 0.0},
            {"kernel": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "bias": -1.0},
        ]
    }
    layers = parse_model_params(json.dumps(params))
    image = np.zeros((7, 7), dtype=np.uint8)
    image[2, 2] = 42
    got = _run_tiled(image, layers, tile=2)
    expected = reference_full_inference(image, layers)
    assert np.array_equal(got, expected)
    # hot pixel lands on feat[1,1] = 42; layer2 at (1,1) scores 42-1 > 0
    assert got[1, 1] == 1
    assert got.sum() == 1


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def _valid_params_json() -> bytes:
    return json.dumps(
        {
            "layers": [
                {"kernel": np.zeros((3, 3)).tolist(), "bias": 0.0},
                {"kernel": np.zeros((3, 3)).tolist(), "bias": 1.0},
            ]
        }
    ).encode()


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        b"{}",
        json.dumps({"layers": []}).encode(),
        json.dumps({"layers": [{}, {}]}).encode(),
        json.dumps(
            {"layers": [
                {"kernel": [[0] * 3] * 3, "bias": 0},
                {"kernel": [[0] * 3] * 3, "bias": 0},
                {"kernel": [[0] * 3] * 3, "bias": 0},
            ]}
        ).encode(),
        json.dumps(
            {"layers": [
                {"kernel": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "bias": 0},
                {"kernel": [[0, 0, 0], [0, 0, 0], [0, 0]], "bias": 0},
            ]}
        ).encode(),
        json.dumps(
            {"layers": [
                {"kernel": [[0] * 3] * 3, "bias": "x"},
                {"kernel": [[0] * 3] * 3, "bias": 0},
            ]}
        ).encode(),
        json.dumps(
            {"layers": [
                {"kernel": [["x", 0, 0], [0, 0, 0], [0, 0, 0]], "bias": 0},
                {"kernel": [[0] * 3] * 3, "bias": 0},
            ]}
        ).encode(),
        b'{"layers":[{"kernel":[[0,0,0],[0,0,0],[0,0,0]],"bias":NaN},'
        b'{"kernel":[[0,0,0],[0,0,0],[0,0,0]],"bias":0}]}',
    ],
)
def test_bad_params_rejected(payload):
    with pytest.raises(ModelParamError):
        parse_model_params(payload)


def test_bool_entries_rejected():
    payload = json.dumps(
        {"layers": [
            {"kernel": [[True, 0, 0], [0, 0, 0], [0, 0, 0]], "bias": 0},
            {"kernel": [[0] * 3] * 3, "bias": 0},
        ]}
    )
    with pytest.raises(ModelParamError):
        parse_model_params(payload)


def test_valid_params_parsed():
    layers = parse_model_params(_valid_params_json())
    assert len(layers) == 2
    assert layers[0].kernel.shape == (3, 3)
    assert layers[0].kernel.dtype == np.float32
    assert isinstance(layers[1].bias, np.float32)


# ---------------------------------------------------------------------------
# Input validation on disk artefacts
# ---------------------------------------------------------------------------


def test_rejects_wrong_dtype():
    with tempfile.TemporaryDirectory() as d:
        img = np.zeros((4, 4), dtype=np.float32)
        p = os.path.join(d, "in.npy")
        np.save(p, img)
        with pytest.raises(ValueError):
            tiled_segment(
                p, _make_layers(),
                os.path.join(d, "f.npy"), os.path.join(d, "o.npy"),
            )


def test_rejects_three_dimensional():
    with tempfile.TemporaryDirectory() as d:
        img = np.zeros((4, 4, 1), dtype=np.uint8)
        p = os.path.join(d, "in.npy")
        np.save(p, img)
        with pytest.raises(ValueError):
            tiled_segment(
                p, _make_layers(),
                os.path.join(d, "f.npy"), os.path.join(d, "o.npy"),
            )


def test_rejects_garbage_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.npy")
        with open(p, "wb") as fh:
            fh.write(b"definitely not a npy file")
        with pytest.raises(ValueError):
            tiled_segment(
                p, _make_layers(),
                os.path.join(d, "f.npy"), os.path.join(d, "o.npy"),
            )
