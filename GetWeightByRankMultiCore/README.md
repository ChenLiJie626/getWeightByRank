# GetWeightByRankMultiCore

This directory is the AscendC custom-operator project for `GetWeightByRank`.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[userCount, 136, 8, 256]`.
- `getIdxs`: `int32`, shape `[idxCount]`. Each value maps to eight source indexes: `idx * 8` through `idx * 8 + 7`.
- `lens`: `int32`, shape `[idxCount]`.
- `getuserIds`, `getuserIdRank`: `int32`, shape `[sum(lens)]`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[idxCount * 8, outputRows, 256]`, where `outputRows` is the actual concatenated rank-row count for the run instead of a fixed padded `8`.

For idx group `i`, user entry `k`, and local index `t` in `[0, 7]`, `getuserIdRank[user_offset + k]` is the number of leading rank columns to copy for that user. These columns are concatenated from column `0` inside each idx group:

```text
user_offset = sum(lens[:i])
rank_prefix = sum(getuserIdRank[user_offset:user_offset + k])
for r in range(getuserIdRank[user_offset + k]):
    weightout_[i * 8 + t, rank_prefix + r, row] =
        weight_[getuserIds[user_offset + k], getIdxs[i] * 8 + t, r, row]
```

`user_offset` is the sum of prior `lens` entries and only selects the current group's user/rank slice. The output row prefix is local to the current idx group; the rank sum for each group is guaranteed to be no more than `8`. Each rank count is greater than zero.

Implementation notes:

- The host tiling launches up to `DEFAULT_BLOCK_DIM` vector blocks, capped by `idxCount * 8`.
- Each block owns complete `dstIndex` slices, so blocks never write the same output region.
- For every owned `dstIndex`, the kernel builds the complete `[outputRows, 256]` real and imaginary outputs in UB. Because each rank block is contiguous, it copies each user's selected rank block with one aligned `DataCopy` per real/imag tensor and stores each complete output matrix back to GM with one contiguous `DataCopy` per real/imag tensor.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before running `../AclNNInvocation/run.sh`.
