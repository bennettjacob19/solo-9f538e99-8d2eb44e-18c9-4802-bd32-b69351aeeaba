"""Tests for the bounded-memory DiskArray and .npy header handling."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from app.inference import DiskArray, _read_npy_header, open_input_image


def test_diskarray_roundtrip_and_npy_compat():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.npy")
        da = DiskArray(p, np.float32, (7, 5), create=True)
        da.write_rows(0, np.arange(10, dtype=np.float32).reshape(2, 5))
        da.write_tile(2, 1, np.full((3, 2), 7.5, np.float32))
        da.close()

        loaded = np.load(p)
        assert loaded.dtype == np.float32 and loaded.shape == (7, 5)
        assert np.array_equal(loaded[:2], np.arange(10).reshape(2, 5))
        assert loaded[2:5, 1:3].tolist() == [[7.5, 7.5]] * 3
        # untouched region is zeros from ftruncate
        assert loaded[5:].sum() == 0
        # column 0 below the write_rows band was never written
        assert loaded[2:, 0].sum() == 0
        # columns 3..4 below the tile's column span were never written
        assert loaded[2:, 3:].sum() == 0

        # reopen for read and compare band access
        ro = DiskArray(p, np.float32, (7, 5), create=False)
        band = ro.read_rows(2, 5)
        assert np.array_equal(band, loaded[2:5])
        ro.close()


def test_diskarray_uint8_tiles_at_all_offsets():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.npy")
        h, w, tile = 11, 13, 4
        da = DiskArray(p, np.uint8, (h, w), create=True)
        truth = np.zeros((h, w), np.uint8)
        rng = np.random.default_rng(0)
        for r0 in range(0, h, tile):
            for c0 in range(0, w, tile):
                t = rng.integers(0, 2, (min(tile, h - r0), min(tile, w - c0)), np.uint8)
                da.write_tile(r0, c0, t)
                truth[r0 : r0 + t.shape[0], c0 : c0 + t.shape[1]] = t
        da.close()
        assert np.array_equal(np.load(p), truth)


def test_read_header_v1_and_v2():
    with tempfile.TemporaryDirectory() as d:
        for ver, kw in ((1, {"version": (1, 0)}), (2, {"version": (2, 0)})):
            p = os.path.join(d, f"v{ver}.npy")
            arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
            np.save(p, arr) if ver == 1 else np.lib.format.write_array(
                open(p, "wb"), arr, version=(2, 0)
            )
            fd = os.open(p, os.O_RDONLY)
            try:
                dtype, shape, fortran, hlen = _read_npy_header(fd, p)
            finally:
                os.close(fd)
            assert dtype == np.uint8 and shape == (2, 3) and fortran is False
            img = open_input_image(p)
            assert img.shape == (2, 3)
            assert np.array_equal(img.read_rows(0, 2), arr)
            img.close()


def test_open_input_rejects_bad_files():
    with tempfile.TemporaryDirectory() as d:
        # not npy
        p = os.path.join(d, "bad.npy")
        with open(p, "wb") as fh:
            fh.write(b"hello world")
        with pytest.raises(ValueError):
            open_input_image(p)

        # wrong dtype
        p2 = os.path.join(d, "f.npy")
        np.save(p2, np.zeros((3, 3), np.float32))
        with pytest.raises(ValueError):
            open_input_image(p2)

        # 3-D
        p3 = os.path.join(d, "c.npy")
        np.save(p3, np.zeros((2, 2, 2), np.uint8))
        with pytest.raises(ValueError):
            open_input_image(p3)

        # empty
        p4 = os.path.join(d, "e.npy")
        np.save(p4, np.zeros((0, 4), np.uint8))
        with pytest.raises(ValueError):
            open_input_image(p4)


def test_fortran_order_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fortran.npy")
        arr = np.asfortranarray(np.zeros((4, 5), np.uint8))
        np.save(p, arr)
        with pytest.raises(ValueError):
            open_input_image(p)
