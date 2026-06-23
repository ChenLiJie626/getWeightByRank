# GetWeightByRank AscendC Operator

This workspace contains an AscendC vector-kernel custom operator named `GetWeightByRank`.

Directory layout:

- `GetWeightByRankMultiCore/`: AscendC custom operator source, host registration, tiling, and build scripts.
- `AclNNInvocation/`: deterministic ACLNN invocation sample and numpy verification.
- `GetWeightByRank.json`: op metadata for the six-input, two-output operator.

Logical behavior:

```text
user_offset = 0
for i in range(idxCount):
    idx = getIdxs[i]
    for t in range(8):
        src_index = idx * 8 + t
        dst_index = i * 8 + t
        dst_col = 0
        for k in range(lens[i]):
            user = getuserIds[user_offset + k]
            rank_count = getuserIdRank[user_offset + k]
            for r in range(rank_count):
                weightout_[dst_index, dst_col + r, :] = weight_[user, src_index, r, :]
            dst_col += rank_count
    user_offset += lens[i]
```

`weight_r` and `weight_i` have shape `[userCount, 136, 8, 256]`. `weightout_r` and `weightout_i` have shape `[idxCount * 8, outputRows, 256]`, where `outputRows` is the actual concatenated rank-row count for the run instead of a fixed padded `8`. `getuserIdRank` stores how many leading rank blocks to copy for each user. Valid output rank blocks are concatenated independently for each `idxCount` group, and each group has no more than `8` rows.

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
