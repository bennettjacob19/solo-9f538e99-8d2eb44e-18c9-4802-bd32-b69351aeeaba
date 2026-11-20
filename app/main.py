"""FastAPI service: upload a single-channel uint8 .npy image plus model
parameter JSON, get the binary defect mask back as a .npy file.

Run locally::

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Interactive docs (upload form included) are served at ``/docs``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from .inference import (
    DEFAULT_TILE,
    ModelParamError,
    open_input_image,
    parse_model_params,
    tiled_segment,
)

app = FastAPI(
    title="Steel Plate Defect Segmentation",
    description=(
        "Tiled two-layer 3x3 conv inference over single-channel uint8 .npy "
        "images. Processes 16384x16384 inputs inside a 256 MiB memory budget "
        "using disk-backed tiled I/O (pread/pwrite), never dense arrays."
    ),
    version="1.0.0",
)

# Spool every upload to its own temp directory (never into RAM).
_TMP_ROOT = os.environ.get("TMPDIR", "/tmp")
_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(2 * 1024 ** 3)))
_CHUNK = 8 * 1024 * 1024

# Serialise inference so concurrent uploads cannot stack their working
# buffers and break the 256 MiB container memory ceiling. Uploads still
# spool concurrently; only the compute phase is gated.
_inference_gate = threading.Semaphore(1)


def _spool_upload(upload: UploadFile, dst_path: str) -> None:
    """Stream an UploadFile to disk, bounding the accepted request size."""
    total = 0
    with open(dst_path, "wb") as dst:
        while True:
            chunk = upload.file.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"uploaded file exceeds {_MAX_UPLOAD_BYTES} bytes",
                )
            dst.write(chunk)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "tile_size": DEFAULT_TILE}


@app.post(
    "/segment",
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Binary defect mask, uint8 .npy, values in {0, 1}.",
        }
    },
    response_class=FileResponse,
    summary="Segment defects in an .npy image",
)
def segment(
    image: UploadFile = File(..., description="Single-channel uint8 .npy image (2-D)."),
    params: UploadFile = File(
        ...,
        description=(
            "JSON model parameters: "
            '{"layers": [{"kernel": 3x3 numbers, "bias": number}, ...]} '
            "exactly two layers."
        ),
    ),
) -> FileResponse:
    workdir = tempfile.mkdtemp(prefix="seg-", dir=_TMP_ROOT)
    try:
        image_path = os.path.join(workdir, "input.npy")
        params_path = os.path.join(workdir, "params.json")
        feat_path = os.path.join(workdir, "feat.npy")
        mask_path = os.path.join(workdir, "mask.npy")

        try:
            _spool_upload(image, image_path)
            _spool_upload(params, params_path)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - transport failures
            raise HTTPException(status_code=400, detail=f"failed to read upload: {exc}")

        try:
            with open(params_path, "rb") as fh:
                layers = parse_model_params(fh.read())
        except ModelParamError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Validate the .npy header through pread only (no mmap, no data
        # pages touched); open_input_image raises ValueError on bad inputs.
        try:
            probe = open_input_image(image_path)
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid .npy image: {exc}"
            )
        probe.close()
        try:
            # Only one inference runs at a time: keeps peak resident memory
            # bounded regardless of how many clients upload concurrently.
            with _inference_gate:
                tiled_segment(image_path, layers, feat_path, mask_path)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"inference failed: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected failure
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}")

        if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            raise HTTPException(status_code=500, detail="inference produced no output")

        # The whole image must be processed before the response is sent;
        # cleanup happens only after the file has been streamed out.
        return FileResponse(
            mask_path,
            media_type="application/octet-stream",
            filename="mask.npy",
            background=BackgroundTask(_cleanup, workdir),
        )
    except HTTPException:
        _cleanup(workdir)
        raise
    except Exception as exc:  # pragma: no cover - defensive
        _cleanup(workdir)
        return JSONResponse(status_code=500, content={"detail": str(exc)})


def _cleanup(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
