# GetWeightByRankMultiCore

This directory is the AscendC custom-operator project for `GetWeightByRank`.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[userCount * 136, 8, 256]`.
- `getIdxs`: `uint32`, shape `[idxCount]`. Each value maps to eight source indexes: `idx * 8` through `idx * 8 + 7`.
- `lens`: `uint32`, shape `[idxCount]`.
- `getuserIds`: `uint32`, shape `[sum(lens)]`.
- `getuserIdRank`: `uint32`, shape `[sum(lens)]`.
- `totalRows`: `uint32`, shape `[totalOutputRows]`. Its first-dimension length is the exact first dimension of the two outputs; its values are not read by the operator.
- `idxCount`: `uint32`, shape `[1]`. The kernel reads `idxCount[0]` as the number of valid entries to consume from `getIdxs` and `lens`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[totalOutputRows, 256]`, where `totalOutputRows = 8 * sum(group_rows for every idx group)`.

Shape/type inference:

- Output dtypes follow `weight_r` and `weight_i` and are inferred as `float`.
- The op infers output shape as `[shape(totalRows)[0], 256]`.
- `lens`, `getuserIdRank`, `totalRows`, and `idxCount` are normal runtime tensor inputs; no input is declared value-dependent.
- `totalOutputRows = 8 * sum(sum(clamp(getuserIdRank[sum(lens[:i]):sum(lens[:i + 1])], 0, 8)) for i in range(idxCount))`.

For idx group `i`, user entry `k`, and local index `t` in `[0, 7]`, `getuserIdRank[user_offset + k]` is the number of leading rank columns to copy for that user. These columns are concatenated from column `0` inside each idx group:

```text
user_offset = sum(lens[:i])
output_row_base = 8 * sum(sum(getuserIdRank[sum(lens[:j]):sum(lens[:j + 1])]) for j in range(i))
group_rows = sum(getuserIdRank[user_offset:user_offset + lens[i]])
rank_prefix = sum(getuserIdRank[user_offset:user_offset + k])
scale = sqrt(1.0 / getuserIdRank[user_offset + k])
for r in range(getuserIdRank[user_offset + k]):
    weightout_[output_row_base + t * group_rows + rank_prefix + r, row] =
        weight_[getuserIds[user_offset + k] * 136 + getIdxs[i] * 8 + t, r, row] * scale
```

`user_offset` is the sum of prior `lens` entries and only selects the current group's user/rank slice. The output row prefix is local to the current idx group. Each copied user block is scaled by `sqrt(1.0 / rank_count)`. Each rank count is greater than zero and no greater than the source rank dimension.

Implementation notes:

- The host tiling launches `DEFAULT_BLOCK_DIM` vector blocks; the kernel reads `idxCount` from GM and loops over `idxCount * 8` destination slices.
- `idxCount` and `totalUserEntries` are not stored in tiling data. The kernel reads `idxCount` from GM and computes user/rank offsets by summing `lens`.
- Each block owns complete `dstIndex` slices, so blocks never write the same output region.
- For every owned `dstIndex`, the kernel writes the current local index's actual concatenated rank rows into the flat `[totalOutputRows, 256]` output. Because each rank block is contiguous, it copies each user's selected rank block with one aligned `DataCopy` per real/imag tensor.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before running `../AclNNInvocation/run.sh`.
