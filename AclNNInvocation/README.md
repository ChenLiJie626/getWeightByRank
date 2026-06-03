# AclNNInvocation

This directory calls the installed `GetWeightByRank` custom operator through the generated two-stage aclnn API:

```cpp
aclnnGetWeightByRankGetWorkspaceSize(..., &workspaceSize, &executor);
aclnnGetWeightByRank(workspace, workspaceSize, executor, stream);
```

Run it after building and installing the operator package from `../GetWeightByRankMultiCore`:

```bash
./run.sh
```

The sample generates a small deterministic input set, runs the operator, and compares `weightout_r` / `weightout_i` with numpy golden output.
