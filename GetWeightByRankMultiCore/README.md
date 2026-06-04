# GetWeightByRankMultiCore

This directory is the AscendC custom-operator project for `GetWeightByRank`.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[userCount, 136, 384, 8]`.
- `getIdxs`: `int32`, shape `[idxCount]`. Each value maps to eight source indexes: `idx * 8` through `idx * 8 + 7`.
- `lens`: `int32`, shape `[idxCount]`.
- `getuserIds`, `getuserIdRank`: `int32`, shape `[sum(lens)]`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[idxCount * 8, 384, 8]`.

For idx group `i`, user entry `k`, and local index `t` in `[0, 7]`, `getuserIdRank[user_offset + k]` is the number of leading rank columns to copy for that user. These columns are concatenated from column `0` inside each idx group:

```text
user_offset = sum(lens[:i])
rank_prefix = sum(getuserIdRank[user_offset:user_offset + k])
for r in range(getuserIdRank[user_offset + k]):
    weightout_[i * 8 + t, row, rank_prefix + r] =
        weight_[getuserIds[user_offset + k], getIdxs[i] * 8 + t, row, r]
```

`user_offset` is the sum of prior `lens` entries and only selects the current group's user/rank slice. The output column prefix is local to the current idx group; the rank sum for each group is guaranteed to be no more than `8`, and columns beyond that local sum are zero-filled. Each rank count is greater than zero.

Implementation notes:

- The host tiling launches up to `DEFAULT_BLOCK_DIM` vector blocks, capped by `idxCount * 8`.
- Each block owns complete `dstIndex` slices, so blocks never write the same output region.
- For every owned `dstIndex`, the kernel builds the complete `[384, 8]` real and imaginary outputs in UB. It uses aligned contiguous `DataCopy` operations to load each source `[384, 8]` matrix from GM and to store each complete output matrix back to GM. The rank-column concatenation is performed inside UB with `LocalTensor::GetValue` / `SetValue`, avoiding per-element GM reads and writes.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before running `../AclNNInvocation/run.sh`.
