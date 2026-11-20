"""End-to-end HTTP tests against the FastAPI app."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inference import reference_full_inference, parse_model_params
from app.main import app


client = TestClient(app)


def _params_bytes(seed: int = 5) -> bytes:
    rng = np.random.default_rng(seed)
    return json.dumps(
        {
            "layers": [
                {"kernel": rng.normal(size=(3, 3)).tolist(), "bias": 0.25},
                {"kernel": rng.normal(size=(3, 3)).tolist(), "bias": -0.25},
            ]
        }
    ).encode()


def _post(image: np.ndarray, params: bytes | None = None, filename="img.npy"):
    buf = io.BytesIO()
    np.save(buf, image)
    buf.seek(0)
    return client.post(
        "/segment",
        files={
            "image": (filename, buf, "application/octet-stream"),
            "params": ("params.json", params or _params_bytes(), "application/json"),
        },
    )


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_segment_endpoint_matches_reference():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(37, 29), dtype=np.uint8)
    resp = _post(image)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert "mask.npy" in resp.headers["content-disposition"]
    mask = np.load(io.BytesIO(resp.content))
    assert mask.dtype == np.uint8
    layers = parse_model_params(_params_bytes())
    assert np.array_equal(mask, reference_full_inference(image, layers))
    assert mask.shape == (19, 15)  # ceil(37/2), ceil(29/2)


def test_odd_one_pixel_image():
    resp = _post(np.array([[7]], dtype=np.uint8))
    assert resp.status_code == 200
    mask = np.load(io.BytesIO(resp.content))
    assert mask.shape == (1, 1)


def test_rejects_bad_dtype():
    resp = _post(np.zeros((4, 4), dtype=np.uint16))
    assert resp.status_code == 400
    assert "uint8" in resp.json()["detail"]


def test_rejects_3d_image():
    resp = _post(np.zeros((4, 4, 3), dtype=np.uint8))
    assert resp.status_code == 400


def test_rejects_garbage_npy():
    buf = io.BytesIO(b"not a npy file at all")
    resp = client.post(
        "/segment",
        files={
            "image": ("x.npy", buf, "application/octet-stream"),
            "params": ("p.json", _params_bytes(), "application/json"),
        },
    )
    assert resp.status_code == 400


def test_rejects_bad_params_json():
    resp = _post(np.zeros((4, 4), dtype=np.uint8), params=b"{broken")
    assert resp.status_code == 400


def test_rejects_params_wrong_schema():
    bad = json.dumps({"layers": [{"kernel": [[0] * 3] * 3, "bias": 0}]}).encode()
    resp = _post(np.zeros((4, 4), dtype=np.uint8), params=bad)
    assert resp.status_code == 400


def test_docs_available():
    resp = client.get("/docs")
    assert resp.status_code == 200
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
