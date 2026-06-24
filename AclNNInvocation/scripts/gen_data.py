#!/usr/bin/env python3
import os

import numpy as np


INDEX_COUNT = 136
ROWS = 256
RANKS = 8
SCENARIO_SCALE_USER30_IDX17_LENS2 = "scale_user30_idx17_lens2"
SCENARIO_SCALE_USER30_IDX17_LENS4_RANK2 = "scale_user30_idx17_lens4_rank2"
SCENARIO_PREFIX_OUTCOL_REGRESSION = "prefix_outcol_regression"
SCENARIO = os.getenv("GET_WEIGHT_SCENARIO", SCENARIO_SCALE_USER30_IDX17_LENS4_RANK2)


def build_scenario(name: str):
    if name == SCENARIO_SCALE_USER30_IDX17_LENS2:
        user_count = 30
        get_idxs = np.arange(17, dtype=np.int32)
        lens = np.full(len(get_idxs), 2, dtype=np.int32)
        total_user_entries = int(np.sum(lens))
        user_ids = (np.arange(total_user_entries, dtype=np.int32) % user_count).astype(np.int32)
        user_ranks = np.ones(total_user_entries, dtype=np.int32)
        return user_count, get_idxs, lens, user_ids, user_ranks
    if name == SCENARIO_SCALE_USER30_IDX17_LENS4_RANK2:
        user_count = 30
        get_idxs = np.arange(17, dtype=np.int32)
        lens = np.array([6, 5] + [4] * 13 + [3, 2], dtype=np.int32)
        total_user_entries = int(np.sum(lens))
        user_ids = (np.arange(total_user_entries, dtype=np.int32) % user_count).astype(np.int32)
        user_ranks = np.full(total_user_entries, 2, dtype=np.int32)
        return user_count, get_idxs, lens, user_ids, user_ranks
    if name == SCENARIO_PREFIX_OUTCOL_REGRESSION:
        user_count = 4
        get_idxs = np.array([0, 2, 16], dtype=np.int32)
        lens = np.array([3, 4, 3], dtype=np.int32)
        user_ids = np.array([0, 2, 1, 1, 3, 0, 2, 2, 1, 3], dtype=np.int32)
        user_ranks = np.array([1, 2, 1, 2, 1, 2, 1, 3, 1, 2], dtype=np.int32)
        return user_count, get_idxs, lens, user_ids, user_ranks
    raise ValueError(f"Unknown GET_WEIGHT_SCENARIO: {name}")


USER_COUNT, GET_IDXS, LENS, GET_USER_IDS, GET_USER_RANKS = build_scenario(SCENARIO)


def get_group_valid_rows() -> list[int]:
    group_valid_rows = []
    user_offset = 0
    for cur_len in LENS:
        next_offset = user_offset + int(cur_len)
        group_valid_rows.append(int(np.sum(GET_USER_RANKS[user_offset:next_offset])))
        user_offset = next_offset
    return group_valid_rows


GROUP_VALID_ROWS = get_group_valid_rows()
TOTAL_OUTPUT_ROWS = int(np.sum(GROUP_VALID_ROWS) * RANKS)


def build_golden(weight: np.ndarray) -> np.ndarray:
    out = np.zeros((TOTAL_OUTPUT_ROWS, ROWS), dtype=np.float32)
    user_offset = 0
    output_row_base = 0
    for i, idx_group in enumerate(GET_IDXS):
        cur_len = int(LENS[i])
        user_ids = GET_USER_IDS[user_offset:user_offset + cur_len]
        ranks = GET_USER_RANKS[user_offset:user_offset + cur_len]
        group_rows = GROUP_VALID_ROWS[i]
        for local_index in range(RANKS):
            src_index = int(idx_group) * RANKS + local_index
            dst_row = output_row_base + local_index * group_rows
            dst_col = 0
            for user_id, rank_count in zip(user_ids, ranks):
                copy_rows = max(0, min(int(rank_count), group_rows - dst_col))
                scale = np.sqrt(1.0 / float(min(max(int(rank_count), 1), RANKS)))
                for rank_offset in range(copy_rows):
                    src_row = int(user_id) * INDEX_COUNT + src_index
                    out[dst_row + dst_col + rank_offset, :] = weight[src_row, rank_offset, :] * scale
                dst_col += int(rank_count)
        output_row_base += group_rows * RANKS
        user_offset += cur_len
    return out


def main() -> None:
    rng = np.random.default_rng(20260603)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    shape = (USER_COUNT * INDEX_COUNT, RANKS, ROWS)
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
        f"scenario={SCENARIO}, userCount={USER_COUNT}, idxCount={len(GET_IDXS)}, "
        f"lensAvg={float(np.mean(LENS)):.2f}, "
        f"validRowsPerGroup={GROUP_VALID_ROWS}, "
        f"totalOutputRows={TOTAL_OUTPUT_ROWS}"
    )


if __name__ == "__main__":
    main()
