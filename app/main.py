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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from .inference import (
    DEFAULT_TILE,
    ModelParamError,
    open_input_image,
    parse_model_params,
    tiled_segment,
)
from .scheduler import ComputeScheduler

app = FastAPI(
    title="Steel Plate Defect Segmentation",
    description=(
        "Tiled two-layer 3x3 conv inference over single-channel uint8 .npy "
        "images. Processes 16384x16384 inputs inside a 256 MiB memory budget "
        "using disk-backed tiled I/O (pread/pwrite), never dense arrays. "
        "A validated urgent job (urgent=true, at most 1024x1024) preempts a "
        "running normal job once, at a tile boundary; the normal job then "
        "resumes from the tiles already written."
    ),
    version="1.1.0",
)

# Spool every upload to its own temp directory (never into RAM).
_TMP_ROOT = os.environ.get("TMPDIR", "/tmp")
_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(2 * 1024 ** 3)))
_CHUNK = 8 * 1024 * 1024

# Urgent (re-shot) images are small: width and height each at most 1024 px.
URGENT_MAX_PIXELS = 1024

# Serialise inference so concurrent uploads cannot stack their working
# buffers and break the 256 MiB container memory ceiling. Uploads still
# spool concurrently; only the compute phase is gated. The scheduler also
# implements the urgent hand-over policy (see app/scheduler.py).
_scheduler = ComputeScheduler()


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
    return {
        "status": "ok",
        "tile_size": DEFAULT_TILE,
        "urgent_max_pixels": URGENT_MAX_PIXELS,
    }


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
    image: UploadFile = File(..., description="Single-channel uint8 .npy image."),
    params: UploadFile = File(
        ...,
        description=(
            "JSON model parameters: "
            '{"layers": [{"kernel": 3x3 numbers, "bias": number}, ...]} '
            "exactly two layers."
        ),
    ),
    urgent: bool = Form(
        False,
        description=(
            "Urgent re-shot job. Width and height must each be at most "
            f"{URGENT_MAX_PIXELS} pixels (HTTP 400 otherwise). A validated "
            "urgent job preempts the running normal job once, after its "
            "current tile is written, and is served in acceptance order. "
            "Default false keeps the original behaviour."
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
        shape = probe.shape
        probe.close()

        # Urgent jobs are validated *before* being accepted, so a rejected
        # request can never preempt (or briefly hold up) a running job.
        if urgent:
            h, w = shape
            if h > URGENT_MAX_PIXELS or w > URGENT_MAX_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "urgent image must be at most "
                        f"{URGENT_MAX_PIXELS}x{URGENT_MAX_PIXELS} pixels, "
                        f"got {h}x{w}"
                    ),
                )

        # Only one inference runs at a time: keeps peak resident memory
        # bounded regardless of how many clients upload concurrently.
        ticket = _scheduler.ticket(urgent=urgent)
        _scheduler.acquire(ticket)
        try:
            tiled_segment(
                image_path,
                layers,
                feat_path,
                mask_path,
                checkpoint=lambda: _scheduler.checkpoint(ticket),
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"inference failed: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected failure
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}")
        finally:
            # Also on failure: a yielded normal job resumes and an urgent
            # job frees the slot for the next caller.
            _scheduler.release(ticket)

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
