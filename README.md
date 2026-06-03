# GetWeightByRank AscendC Operator

This workspace contains an AscendC vector-kernel custom operator named `GetWeightByRank`.

Directory layout:

- `GetWeightByRankMultiCore/`: AscendC custom operator source, host registration, tiling, and build scripts.
- `AclNNInvocation/`: deterministic ACLNN invocation sample and numpy verification.
- `GetWeightByRank.json`: op metadata for the six-input, two-output operator.

Logical behavior:

```text
prefix = 0
out_col = 0
for i in range(idxCount):
    idx = getIdxs[i]
    for t in range(8):
        src_index = idx * 8 + t
        dst_index = i * 8 + t
        dst_col = out_col
        for k in range(lens[i]):
            user = getuserIds[prefix + k]
            rank_count = getuserIdRank[prefix + k]
            for r in range(rank_count):
                weightout_[dst_index, :, dst_col + r] = weight_[user, src_index, :, r]
            dst_col += rank_count
    out_col += sum(getuserIdRank[prefix:prefix + lens[i]])
    prefix += lens[i]
```

`weight_r` and `weight_i` have shape `[userCount, 136, 384, 8]`. `weightout_r` and `weightout_i` have fixed shape `[idxCount * 8, 384, 8]`. `getuserIdRank` stores how many leading rank columns to copy for each user. The valid output columns are concatenated in global `getuserIds` order; `sum(getuserIdRank) <= 8`, and remaining columns are zero-filled.

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
