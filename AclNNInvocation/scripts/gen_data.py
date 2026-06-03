#!/usr/bin/env python3
import os

import numpy as np


USER_COUNT = 4
INDEX_COUNT = 136
ROWS = 384
RANKS = 8
GET_IDXS = np.array([0, 2, 16], dtype=np.int32)
LENS = np.array([2, 3, 2], dtype=np.int32)
GET_USER_IDS = np.array([0, 2, 1, 3, 0, 2, 1], dtype=np.int32)
GET_USER_RANKS = np.array([0, 3, 7, 2, 5, 4, 1], dtype=np.int32)
OUT_COLS = 8


def build_golden(weight: np.ndarray) -> np.ndarray:
    out = np.zeros((len(GET_IDXS) * RANKS, ROWS, OUT_COLS), dtype=np.float32)
    user_offset = 0
    for i, idx_group in enumerate(GET_IDXS):
        cur_len = int(LENS[i])
        user_ids = GET_USER_IDS[user_offset:user_offset + cur_len]
        ranks = GET_USER_RANKS[user_offset:user_offset + cur_len]
        for local_index in range(RANKS):
            src_index = int(idx_group) * RANKS + local_index
            dst_index = i * RANKS + local_index
            for col_offset, (user_id, rank) in enumerate(zip(user_ids, ranks)):
                out[dst_index, :, user_offset + col_offset] = weight[int(user_id), src_index, :, int(rank)]
        user_offset += cur_len
    return out


def main() -> None:
    rng = np.random.default_rng(20260603)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    shape = (USER_COUNT, INDEX_COUNT, ROWS, RANKS)
    weight_r = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    weight_i = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    weight_r.tofile("input/input_weight_r.bin")
    weight_i.tofile("input/input_weight_i.bin")
    GET_IDXS.tofile("input/input_get_idxs.bin")
    LENS.tofile("input/input_lens.bin")
    GET_USER_IDS.tofile("input/input_user_ids.bin")
    GET_USER_RANKS.tofile("input/input_user_ranks.bin")

    build_golden(weight_r).tofile("output/golden_weightout_r.bin")
    build_golden(weight_i).tofile("output/golden_weightout_i.bin")
    print(
        "Generate input and golden data success. "
        f"userCount={USER_COUNT}, idxCount={len(GET_IDXS)}, validCols={int(np.sum(LENS))}, paddedCols={OUT_COLS}"
    )


if __name__ == "__main__":
    main()
