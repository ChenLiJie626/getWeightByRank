#!/usr/bin/env python3
import os
import sys
from typing import Tuple

import numpy as np


ROWS = 256


def check(name: str, actual_path: str, golden_path: str, shape: Tuple[int, ...],
          atol: float, rtol: float) -> bool:
    actual = np.fromfile(actual_path, dtype=np.float32).reshape(shape)
    golden = np.fromfile(golden_path, dtype=np.float32).reshape(shape)

    diff = np.abs(actual - golden)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / np.maximum(np.abs(golden), 1.0)))
    ok = np.allclose(actual, golden, atol=atol, rtol=rtol)
    print(f"{name}: {'PASS' if ok else 'FAIL'}, max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")
    if not ok:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        print(f"  worst index={idx}, actual={actual[idx]}, golden={golden[idx]}")
    return ok


def main() -> int:
    elem_count = os.path.getsize("output/golden_weightout_r.bin") // np.dtype(np.float32).itemsize
    if elem_count % ROWS != 0:
        print(f"Invalid golden element count: {elem_count}")
        return 1
    shape = (elem_count // ROWS, ROWS)
    weightout_r_ok = check(
        "weightout_r",
        "output/output_weightout_r.bin",
        "output/golden_weightout_r.bin",
        shape,
        atol=0.0,
        rtol=0.0,
    )
    weightout_i_ok = check(
        "weightout_i",
        "output/output_weightout_i.bin",
        "output/golden_weightout_i.bin",
        shape,
        atol=0.0,
        rtol=0.0,
    )
    return 0 if weightout_r_ok and weightout_i_ok else 1


if __name__ == "__main__":
    sys.exit(main())
