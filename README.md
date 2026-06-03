# GetWeightByRank AscendC Operator

This workspace contains an AscendC vector-kernel custom operator named `GetWeightByRank`.

Directory layout:

- `GetWeightByRankMultiCore/`: AscendC custom operator source, host registration, tiling, and build scripts.
- `AclNNInvocation/`: deterministic ACLNN invocation sample and numpy verification.
- `GetWeightByRank.json`: op metadata for the six-input, two-output operator.

Logical behavior:

```text
prefix = 0
for i in range(idxCount):
    idx = getIdxs[i]
    for t in range(8):
        src_index = idx * 8 + t
        dst_index = i * 8 + t
        for k in range(lens[i]):
            user = getuserIds[prefix + k]
            rank = getuserIdRank[prefix + k]
            weightout_[dst_index, :, prefix + k] = weight_[user, src_index, :, rank]
    prefix += lens[i]
```

`weight_r` and `weight_i` have shape `[userCount, 136, 384, 8]`. `weightout_r` and `weightout_i` have fixed shape `[idxCount * 8, 384, 8]`. The valid output columns are `sum(lens)` in global `getuserIds` order; `sum(lens) <= 8`, and remaining columns are zero-filled.

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
