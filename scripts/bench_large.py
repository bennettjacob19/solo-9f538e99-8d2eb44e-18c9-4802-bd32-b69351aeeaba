"""Standalone benchmark: full 16384x16384 tiled inference with RSS tracking.

Run with isolated mode; all large artefacts live in a data directory passed
as the first argument::

    python -I scripts/bench_large.py /path/to/data-dir
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

# Under `python -I` neither PYTHONPATH nor the user site is auto-added; allow
# an explicit site directory via env so isolated runs still find NumPy.
_site = os.environ.get("BENCH_SITE_PACKAGES")
if _site and _site not in sys.path:
    sys.path.insert(0, _site)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.inference import (  # noqa: E402
    parse_model_params,
    reference_full_inference,
    tiled_segment,
)

SIZE = int(os.environ.get("BENCH_SIZE", "16384"))
TILE = int(os.environ.get("TILE_SIZE", "512"))
# Cross-check against the whole-image reference only for smaller sizes
# (the reference is deliberately dense and would blow the memory budget).
REFERENCE_LIMIT = 1100


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    img_path = os.path.join(data_dir, f"image_{SIZE}.npy")
    feat_path = os.path.join(data_dir, f"feat_{SIZE}.npy")
    mask_path = os.path.join(data_dir, f"mask_{SIZE}.npy")

    rng = np.random.default_rng(2026)
    params = {
        "layers": [
            {"kernel": rng.normal(scale=0.1, size=(3, 3)).tolist(),
             "bias": 0.05},
            {"kernel": rng.normal(scale=0.1, size=(3, 3)).tolist(),
             "bias": -0.02},
        ]
    }
    layers = parse_model_params(json.dumps(params))

    if not os.path.exists(img_path):
        print(f"generating {SIZE}x{SIZE} uint8 image ...", flush=True)
        t0 = time.time()
        img = np.lib.format.open_memmap(
            img_path, mode="w+", dtype=np.uint8, shape=(SIZE, SIZE)
        )
        # Fill in row blocks so no large temporary exists.
        block = 1024
        rng2 = np.random.default_rng(11)
        for r0 in range(0, SIZE, block):
            img[r0 : r0 + block] = rng2.integers(
                0, 256, size=(min(block, SIZE - r0), SIZE), dtype=np.uint8
            )
        img.flush()
        del img
        print(f"  image ready in {time.time() - t0:.1f}s", flush=True)

    print(f"peak RSS before inference: {rss_mib():.1f} MiB", flush=True)
    t0 = time.time()
    h2, w2 = tiled_segment(img_path, layers, feat_path, mask_path, tile=TILE)
    elapsed = time.time() - t0
    print(f"tiled inference done in {elapsed:.1f}s", flush=True)
    print(f"peak RSS after inference:  {rss_mib():.1f} MiB", flush=True)

    mask = np.lib.format.open_memmap(mask_path, mode="r")
    print(f"mask shape={mask.shape} dtype={mask.dtype} "
          f"ones={int(np.count_nonzero(mask))} total={mask.size}", flush=True)
    expected_shape = ((SIZE + 1) // 2, (SIZE + 1) // 2)
    assert (h2, w2) == mask.shape == expected_shape
    assert mask.dtype == np.uint8
    del mask

    if SIZE <= REFERENCE_LIMIT:
        image = np.load(img_path)
        ref = reference_full_inference(image, layers)
        got = np.load(mask_path)
        assert np.array_equal(got, ref), "tiled result differs from reference!"
        print("reference cross-check: EXACT MATCH", flush=True)
    else:
        print("size above reference limit; correctness covered by test-suite",
              flush=True)

    for p in (feat_path, mask_path):
        print(f"{os.path.basename(p)} on-disk size: "
              f"{os.path.getsize(p) / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/seg-bench")
