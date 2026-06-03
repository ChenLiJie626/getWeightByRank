# GetWeightByRankMultiCore

This directory is the AscendC custom-operator project for `GetWeightByRank`.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[userCount, 136, 384, 8]`.
- `getIdxs`: `int32`, shape `[idxCount]`. Each value maps to eight source indexes: `idx * 8` through `idx * 8 + 7`.
- `lens`: `int32`, shape `[idxCount]`.
- `getuserIds`, `getuserIdRank`: `int32`, shape `[sum(lens)]`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[idxCount * 8, 384, 8]`.

For idx group `i`, user column `k`, and local index `t` in `[0, 7]`, the operator writes:

```text
weightout_[i * 8 + t, row, prefix + k] =
    weight_[getuserIds[prefix + k], getIdxs[i] * 8 + t, row, getuserIdRank[prefix + k]]
```

`prefix` is the sum of prior `lens` entries. The valid output column count is `sum(lens)`, which is guaranteed to be no more than `8`; columns beyond `sum(lens)` are zero-filled.

The vector kernel uses `DataCopyPad` with GM byte strides to gather one rank column from the `[384, 8]` source slice and scatter it into the output column.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before running `../AclNNInvocation/run.sh`.
