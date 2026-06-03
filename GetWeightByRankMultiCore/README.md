# GetWeightByRankMultiCore

This directory is the AscendC custom-operator project for `GetWeightByRank`.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[userCount, 136, 384, 8]`.
- `getIdxs`: `int32`, shape `[idxCount]`. Each value maps to eight source indexes: `idx * 8` through `idx * 8 + 7`.
- `lens`: `int32`, shape `[idxCount]`.
- `getuserIds`, `getuserIdRank`: `int32`, shape `[sum(lens)]`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[idxCount * 8, 384, 8]`.

For idx group `i`, user entry `k`, and local index `t` in `[0, 7]`, `getuserIdRank[prefix + k]` is the number of leading rank columns to copy for that user. These columns are concatenated into the output:

```text
rank_prefix = sum(getuserIdRank[:prefix + k])
for r in range(getuserIdRank[prefix + k]):
    weightout_[i * 8 + t, row, rank_prefix + r] =
        weight_[getuserIds[prefix + k], getIdxs[i] * 8 + t, row, r]
```

`prefix` is the sum of prior `lens` entries. The valid output column count is `sum(getuserIdRank)`, which is guaranteed to be no more than `8`; columns beyond that are zero-filled. Each rank count is greater than zero.

The kernel runs with one block for this fixed-size sample and writes each concatenated output column by copying the matching `[384]` rank slice from global memory.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before running `../AclNNInvocation/run.sh`.
