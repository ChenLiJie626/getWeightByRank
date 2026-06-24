# GetWeightByRank AscendC Operator

This workspace contains an AscendC vector-kernel custom operator named `GetWeightByRank`.

Directory layout:

- `GetWeightByRankMultiCore/`: AscendC custom operator source, host registration, tiling, and build scripts.
- `AclNNInvocation/`: deterministic ACLNN invocation sample and numpy verification.
- `GetWeightByRank.json`: op metadata for the six-input, two-output operator.

Logical behavior:

```text
user_offset = 0
output_row_base = 0
for i in range(idxCount):
    idx = getIdxs[i]
    group_rows = sum(getuserIdRank[user_offset:user_offset + lens[i]])
    for t in range(8):
        src_index = idx * 8 + t
        dst_row = output_row_base + t * group_rows
        dst_col = 0
        for k in range(lens[i]):
            user = getuserIds[user_offset + k]
            rank_count = getuserIdRank[user_offset + k]
            scale = sqrt(1.0 / rank_count)
            for r in range(rank_count):
                weightout_[dst_row + dst_col + r, :] = (
                    weight_[user * 136 + src_index, r, :] * scale
                )
            dst_col += rank_count
    output_row_base += 8 * group_rows
    user_offset += lens[i]
```

`weight_r` and `weight_i` have shape `[userCount * 136, 8, 256]`. `weightout_r` and `weightout_i` have shape `[totalOutputRows, 256]`, where `totalOutputRows = 8 * sum(group_rows for every idx group)`. `getuserIdRank` stores how many leading rank blocks to copy for each user, and each copied user block is scaled by `sqrt(1.0 / rank_count)`. Valid output rank blocks are concatenated independently for each `idxCount` group, and each group's row count may be greater than `8` when multiple users are concatenated.

Build the operator:

```bash
cd GetWeightByRankMultiCore
./build.sh
```

After installing the generated package, run the sample:

```bash
cd ../AclNNInvocation
./run.sh
```
