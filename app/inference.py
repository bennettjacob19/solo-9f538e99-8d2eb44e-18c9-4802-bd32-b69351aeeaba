"""Tiled two-layer 3x3 conv defect segmentation.

The network is intentionally tiny:

    layer 1: 3x3 conv, stride 2, global zero padding of one ring, then ReLU
    layer 2: 3x3 conv, stride 1, global zero padding of one ring

Both layers operate on float32 and the accumulation loop order is identical
to :func:`reference_full_inference`, so tiled inference is bit-for-bit equal
to running the whole image at once (same summation order, same dtypes).

Memory strategy for the 256 MiB container limit
------------------------------------------------
None of the large arrays (input, layer-1 feature map, mask) are ever
mmap-ed. Each is a standard ``.npy`` file on disk accessed through
:class:`DiskArray`, which pulls and pushes row bands with
:func:`os.preadv`/:func:`os.pwritev` into small reusable buffers. Only
tile-sized NumPy arrays are resident at any moment, so peak RSS stays far
below 256 MiB even for 16384x16384 inputs, and the virtual address space
remains small as well.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass

import numpy as np

# One tile of the *output* grid per layer. 512 output pixels -> layer-1
# local buffer of at most (512*2+3) x (512*2+3) float32 ~= 4.2 MiB.
DEFAULT_TILE = int(os.environ.get("TILE_SIZE", "512"))


class ModelParamError(ValueError):
    """Raised when the JSON model parameters do not match the schema."""


@dataclass(frozen=True)
class ConvLayer:
    kernel: np.ndarray  # shape (3, 3), float32
    bias: np.float32

    def to_dict(self) -> dict:
        return {
            "kernel": self.kernel.tolist(),
            "bias": float(self.bias),
        }


def parse_model_params(raw: bytes | str) -> list[ConvLayer]:
    """Validate and parse the two-layer model parameter JSON.

    Expected schema::

        {
          "layers": [
            {"kernel": [[...3x3 floats...]], "bias": <float>},
            {"kernel": [[...3x3 floats...]], "bias": <float>}
          ]
        }
    """
    try:
        if isinstance(raw, (bytes, bytearray)):
            payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        else:
            payload = json.loads(raw, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelParamError(f"model params are not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ModelParamError("model params must be a JSON object")
    layers_raw = payload.get("layers")
    if not isinstance(layers_raw, list) or len(layers_raw) != 2:
        raise ModelParamError('"layers" must be a list of exactly 2 layers')

    layers: list[ConvLayer] = []
    for idx, layer in enumerate(layers_raw):
        if not isinstance(layer, dict):
            raise ModelParamError(f"layer {idx} must be an object")
        kernel_raw = layer.get("kernel")
        bias_raw = layer.get("bias")

        if not isinstance(kernel_raw, list) or len(kernel_raw) != 3:
            raise ModelParamError(f"layer {idx}: kernel must be a 3x3 array")
        kernel = np.empty((3, 3), dtype=np.float32)
        for r, row in enumerate(kernel_raw):
            if not isinstance(row, list) or len(row) != 3:
                raise ModelParamError(f"layer {idx}: kernel must be a 3x3 array")
            for c, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ModelParamError(
                        f"layer {idx}: kernel entries must be finite numbers"
                    )
                kernel[r, c] = np.float32(value)
        if not np.isfinite(kernel).all():
            raise ModelParamError(f"layer {idx}: kernel contains NaN or Infinity")

        if isinstance(bias_raw, bool) or not isinstance(bias_raw, (int, float)):
            raise ModelParamError(f"layer {idx}: bias must be a finite number")
        bias = np.float32(bias_raw)
        if not np.isfinite(bias):
            raise ModelParamError(f"layer {idx}: bias is NaN or Infinity")

        layers.append(ConvLayer(kernel=kernel, bias=bias))
    return layers


def _reject_constant(value: str) -> None:
    raise ModelParamError(f"invalid JSON constant {value!r} (NaN/Infinity not allowed)")


def _conv_out_size(size: int, stride: int) -> int:
    # padding of exactly one ring on the full feature map: pad = 1
    return (size + 2 * 1 - 3) // stride + 1


class DiskArray:
    """A 2-D ``.npy`` file on disk with bounded-memory row access.

    The file is a fully valid NumPy ``.npy`` (header written up front) but
    data is exchanged only through :func:`read_rows` / :func:`write_rows`,
    each using a single pread/pwrite of a small band. No mmap is involved,
    so neither RSS nor virtual address space grows with the array size.
    """

    def __init__(self, path: str, dtype: np.dtype, shape: tuple[int, int],
                 create: bool):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.shape = (int(shape[0]), int(shape[1]))
        self.itemsize = self.dtype.itemsize
        self.row_bytes = self.shape[1] * self.itemsize
        flags = os.O_RDWR | os.O_CREAT if create else os.O_RDONLY
        self._fd = os.open(path, flags, 0o644)
        if create:
            # Write a real .npy header and reserve the data region.
            header = self._build_header()
            os.ftruncate(self._fd, len(header) + self.shape[0] * self.row_bytes)
            os.pwrite(self._fd, header, 0)
            self.header_len = len(header)
        else:
            dtype, shape, _fortran, header_len = _read_npy_header(self._fd, path)
            if dtype != self.dtype or tuple(shape) != self.shape:
                raise ValueError(
                    f"{path}: expected {self.dtype} {self.shape}, "
                    f"got {dtype} {shape}"
                )
            self.header_len = header_len

    # -- .npy header handling -------------------------------------------------

    def _build_header(self) -> bytes:
        # Delegate to NumPy's own v1.0 header writer: present in every NumPy
        # release and sufficient here (dtypes are fixed-width, arrays < 4 GiB),
        # so the on-disk file is a standards-compliant .npy readable by np.load.
        from io import BytesIO

        buf = BytesIO()
        np.lib.format.write_array_header_1_0(
            buf,
            {
                "descr": np.lib.format.dtype_to_descr(self.dtype),
                "fortran_order": False,
                "shape": self.shape,
            },
        )
        return buf.getvalue()

    # -- band I/O -------------------------------------------------------------

    def read_rows(self, r0: int, r1_exclusive: int) -> np.ndarray:
        n = r1_exclusive - r0
        raw = bytearray(n * self.row_bytes)
        view = memoryview(raw)
        offset = self.header_len + r0 * self.row_bytes
        while view:
            got = os.preadv(self._fd, [view], offset)
            if got == 0:
                raise OSError("unexpected EOF reading .npy data")
            view = view[got:]
            offset += got
        return np.frombuffer(raw, dtype=self.dtype).reshape(n, self.shape[1])

    def write_rows(self, r0: int, band: np.ndarray) -> None:
        if band.dtype != self.dtype or band.shape[1] != self.shape[1]:
            raise ValueError("write_rows: dtype/width mismatch")
        payload = np.ascontiguousarray(band).tobytes()
        offset = self.header_len + r0 * self.row_bytes
        view = memoryview(payload)
        while view:
            written = os.pwritev(self._fd, [view], offset)
            if written == 0:
                raise OSError("short write to .npy data")
            view = view[written:]
            offset += written

    def write_tile(self, r0: int, c0: int, tile: np.ndarray) -> None:
        """Write a tile at row/col offset without allocating a full-width band.

        Each destination row sits at its own file offset (because of the
        columns before and after the tile), so one ``pwritev`` per tile row is
        issued; all rows share one contiguous payload, with no extra copy.
        Full-width tiles (c0 == 0) are contiguous and use a single pwritev.
        """
        if tile.dtype != self.dtype or tile.shape[1] > self.shape[1]:
            raise ValueError("write_tile: dtype/width mismatch")
        n_rows, n_cols = tile.shape
        width_bytes = n_cols * self.itemsize
        payload = np.ascontiguousarray(tile).tobytes()
        col_byte = c0 * self.itemsize

        if c0 == 0 and n_cols == self.shape[1]:
            # Contiguous block: a single syscall for the whole tile.
            offset = self.header_len + r0 * self.row_bytes
            view = memoryview(payload)
            while view:
                written = os.pwritev(self._fd, [view], offset)
                if written == 0:
                    raise OSError("short write to .npy data")
                view = view[written:]
                offset += written
            return

        for r in range(n_rows):
            offset = self.header_len + (r0 + r) * self.row_bytes + col_byte
            view = memoryview(payload)[r * width_bytes : (r + 1) * width_bytes]
            while view:
                written = os.pwritev(self._fd, [view], offset)
                if written == 0:
                    raise OSError("short write to .npy data")
                view = view[written:]
                offset += written

    def flush(self) -> None:
        os.fsync(self._fd)

    def close(self) -> None:
        fd = getattr(self, "_fd", -1)
        if fd >= 0:
            os.close(fd)
            self._fd = -1

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _read_npy_header(fd: int, path: str) -> tuple[np.dtype, tuple[int, ...], bool, int]:
    """Parse a .npy header through pread only (no mmap, no data pages).

    Returns ``(dtype, shape, fortran_order, header_length)``.
    """
    magic = os.pread(fd, 6, 0)
    if magic != b"\x93NUMPY":
        raise ValueError(f"{path}: not a valid .npy file (bad magic)")
    version = os.pread(fd, 2, 6)
    if version == b"\x01\x00":
        (hlen,) = np.frombuffer(os.pread(fd, 2, 8), dtype="<u2")
        header_start = 10
    elif version == b"\x02\x00":
        (hlen,) = np.frombuffer(os.pread(fd, 4, 8), dtype="<u4")
        header_start = 12
    else:
        raise ValueError(f"{path}: unsupported .npy version {version!r}")
    header_bytes = os.pread(fd, int(hlen), header_start)
    try:
        # The .npy header is a literal dict (str/bool/int/tuple); literal_eval
        # evaluates it without executing any code.
        meta = ast.literal_eval(header_bytes.decode("latin1"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}: malformed .npy header: {exc}") from exc
    if not isinstance(meta, dict) or "descr" not in meta or "shape" not in meta:
        raise ValueError(f"{path}: malformed .npy header metadata")
    dtype = np.dtype(meta["descr"])
    shape = tuple(int(s) for s in meta["shape"])
    fortran = bool(meta.get("fortran_order", False))
    return dtype, shape, fortran, header_start + int(hlen)


def open_input_image(path: str) -> DiskArray:
    """Open and validate the uploaded uint8 single-channel .npy image."""
    # Parse only the small header through pread; the data region is never
    # mapped, so opening a 256 MiB file costs essentially no resident memory.
    fd = os.open(path, os.O_RDONLY)
    try:
        dtype, shape, fortran, _ = _read_npy_header(fd, path)
    finally:
        os.close(fd)
    if fortran:
        raise ValueError(
            "Fortran-ordered .npy images are not supported; save in C order"
        )
    if dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {dtype}")
    if len(shape) != 2:
        raise ValueError(
            f"image must be a 2-D single-channel array, got ndim={len(shape)}"
        )
    if 0 in shape:
        raise ValueError("image must be non-empty")
    return DiskArray(path, np.uint8, shape, create=False)


def tiled_segment(
    image_path: str,
    layers: list[ConvLayer],
    intermediate_path: str,
    output_path: str,
    tile: int = DEFAULT_TILE,
) -> tuple[int, int]:
    """Run tiled inference and write the mask as a uint8 ``.npy`` file.

    Returns ``(out_h, out_w)``. The mask has shape
    ``(ceil(H/2), ceil(W/2))`` and dtype uint8 with values in {0, 1}.
    """
    if tile < 1:
        raise ValueError("tile size must be >= 1")

    image = open_input_image(image_path)
    try:
        h, w = image.shape
        layer1, layer2 = layers
        h1, w1 = _conv_out_size(h, 2), _conv_out_size(w, 2)  # ceil(H/2), ceil(W/2)
        h2, w2 = _conv_out_size(h1, 1), _conv_out_size(w1, 1)  # same as (h1, w1)

        feat = DiskArray(intermediate_path, np.float32, (h1, w1), create=True)
        try:
            _run_layer1(image, feat, layer1, tile)
            feat.flush()

            mask = DiskArray(output_path, np.uint8, (h2, w2), create=True)
            try:
                _run_layer2(feat, mask, layer2, tile)
                mask.flush()
            finally:
                mask.close()
        finally:
            feat.close()
    finally:
        image.close()

    return h2, w2


def _run_layer1(
    image: DiskArray, feat: DiskArray, layer: ConvLayer, tile: int
) -> None:
    """Stride-2 3x3 conv + ReLU, tiled over the layer-1 output grid.

    Each output tile materialises only the small padded input region it
    needs (zeros outside the image), reproducing global one-ring padding.
    Input rows arrive via pread, so the source image never resides in RAM.
    """
    h, w = image.shape
    h1, w1 = feat.shape
    k, b = layer.kernel, layer.bias

    for oy0 in range(0, h1, tile):
        oy1 = min(oy0 + tile, h1)
        for ox0 in range(0, w1, tile):
            ox1 = min(ox0 + tile, w1)

            # Output pixel (oy, ox) reads input rows [2oy-1, 2oy+1].
            r_lo = 2 * oy0 - 1
            r_hi = 2 * (oy1 - 1) + 1  # inclusive
            c_lo = 2 * ox0 - 1
            c_hi = 2 * (ox1 - 1) + 1
            local = np.zeros((r_hi - r_lo + 1, c_hi - c_lo + 1), dtype=np.float32)

            src_r0 = max(r_lo, 0)
            src_r1 = min(r_hi, h - 1)
            src_c0 = max(c_lo, 0)
            src_c1 = min(c_hi, w - 1)
            if src_r0 <= src_r1 and src_c0 <= src_c1:
                block = image.read_rows(src_r0, src_r1 + 1)
                local[
                    src_r0 - r_lo : src_r1 - r_lo + 1,
                    src_c0 - c_lo : src_c1 - c_lo + 1,
                ] = block[:, src_c0:src_c1 + 1]
                del block

            out = np.zeros((oy1 - oy0, ox1 - ox0), dtype=np.float32)
            # Exact same (kh, kw) accumulation order as the full-image path.
            for kh in range(3):
                for kw in range(3):
                    out += k[kh, kw] * local[kh : kh + 2 * (oy1 - oy0) : 2,
                                             kw : kw + 2 * (ox1 - ox0) : 2]
            out += b
            np.maximum(out, 0.0, out=out)  # ReLU
            # Persist this tile immediately; only one tile is ever resident.
            feat.write_tile(oy0, ox0, out)
            del local, out


def _run_layer2(
    feat: DiskArray, mask: DiskArray, layer: ConvLayer, tile: int
) -> None:
    """Stride-1 3x3 conv on the layer-1 feature map, thresholded to {0, 1}.

    A one-output-pixel halo is loaded around every tile so that the same
    feature values are read as in a whole-image convolution; seams between
    tiles are therefore identical. Only the current row band plus a one-row
    halo is held in RAM.
    """
    h1, w1 = feat.shape
    k, b = layer.kernel, layer.bias

    for oy0 in range(0, h1, tile):
        oy1 = min(oy0 + tile, h1)
        for ox0 in range(0, w1, tile):
            ox1 = min(ox0 + tile, w1)

            # Padded local window: output (oy,ox) reads feat[oy-1:oy+2,
            # ox-1:ox+2]; anything outside the feature map stays zero, which
            # reproduces the single global zero ring. Only this window is
            # resident, so tile seams read the exact same values as a
            # whole-image convolution.
            r_lo, r_hi = oy0 - 1, oy1 + 1  # exclusive upper bound
            c_lo, c_hi = ox0 - 1, ox1 + 1
            local = np.zeros((r_hi - r_lo, c_hi - c_lo), dtype=np.float32)
            src_r0 = max(r_lo, 0)
            src_r1 = min(r_hi - 1, h1 - 1)
            src_c0 = max(c_lo, 0)
            src_c1 = min(c_hi - 1, w1 - 1)
            if src_r0 <= src_r1 and src_c0 <= src_c1:
                block = feat.read_rows(src_r0, src_r1 + 1)
                local[
                    src_r0 - r_lo : src_r1 - r_lo + 1,
                    src_c0 - c_lo : src_c1 - c_lo + 1,
                ] = block[:, src_c0 : src_c1 + 1]
                del block

            out = np.zeros((oy1 - oy0, ox1 - ox0), dtype=np.float32)
            th, tw = oy1 - oy0, ox1 - ox0
            for kh in range(3):
                for kw in range(3):
                    out += k[kh, kw] * local[kh : kh + th, kw : kw + tw]
            out += b
            mask.write_tile(oy0, ox0, (out > 0).astype(np.uint8))
            del local, out


# ---------------------------------------------------------------------------
# Reference: whole-image inference. Used by the test-suite to prove that the
# tiled path is pixel-identical (including seams, borders and odd sizes).
# ---------------------------------------------------------------------------


def reference_full_inference(image: np.ndarray, layers: list[ConvLayer]) -> np.ndarray:
    """Naive whole-image implementation of the same two-layer network."""
    x = image.astype(np.float32, copy=True)

    # layer 1: pad one ring on the whole feature map, stride 2, ReLU
    layer1, layer2 = layers
    padded = np.zeros((x.shape[0] + 2, x.shape[1] + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = x
    h1 = _conv_out_size(x.shape[0], 2)
    w1 = _conv_out_size(x.shape[1], 2)
    feat = np.zeros((h1, w1), dtype=np.float32)
    for kh in range(3):
        for kw in range(3):
            feat += layer1.kernel[kh, kw] * padded[
                kh : kh + 2 * h1 : 2, kw : kw + 2 * w1 : 2
            ]
    feat += layer1.bias
    np.maximum(feat, 0.0, out=feat)

    # layer 2: pad one ring on the whole feature map, stride 1, threshold
    padded2 = np.zeros((h1 + 2, w1 + 2), dtype=np.float32)
    padded2[1:-1, 1:-1] = feat
    scores = np.zeros((h1, w1), dtype=np.float32)
    for kh in range(3):
        for kw in range(3):
            scores += layer2.kernel[kh, kw] * padded2[kh : kh + h1, kw : kw + w1]
    scores += layer2.bias
    return (scores > 0).astype(np.uint8)
