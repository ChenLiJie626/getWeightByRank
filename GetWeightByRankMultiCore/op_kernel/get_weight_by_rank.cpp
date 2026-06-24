#include "kernel_operator.h"

using namespace AscendC;

namespace GetWeightByRankKernel {
constexpr uint32_t INDEX_COUNT = 136;
constexpr uint32_t ROWS = 256;
constexpr uint32_t RANKS_PER_INDEX = 8;
constexpr uint32_t INDEX_GROUP_WIDTH = 8;
constexpr uint32_t INDEX_GROUP_COUNT = INDEX_COUNT / INDEX_GROUP_WIDTH;
constexpr uint32_t OUTPUT_COLS = 8;

constexpr uint32_t FLOAT_BYTES = sizeof(float);
constexpr uint32_t ZERO_CHUNK_ELEMS = 1024;
constexpr uint32_t COPY_BUFFER_ELEMS = RANKS_PER_INDEX * ROWS;
constexpr uint32_t COPY_BUFFER_BYTES = COPY_BUFFER_ELEMS * FLOAT_BYTES;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum());
}

__aicore__ inline void ClearLocalTensor(LocalTensor<float> &dst, uint32_t elemCount)
{
    for (uint32_t offset = 0; offset < elemCount; offset += ZERO_CHUNK_ELEMS) {
        const uint32_t count = static_cast<uint32_t>(MinU64(ZERO_CHUNK_ELEMS, elemCount - offset));
        Duplicate(dst[offset], 0.0f, count);
    }
}

__aicore__ inline bool IsValidUserId(int32_t userId, uint32_t userCount)
{
    return userId >= 0 && static_cast<uint32_t>(userId) < userCount;
}

__aicore__ inline uint32_t GetRankCount(int32_t rankCount)
{
    if (rankCount <= 0) {
        return 0;
    }
    const uint32_t count = static_cast<uint32_t>(rankCount);
    return count < RANKS_PER_INDEX ? count : RANKS_PER_INDEX;
}

__aicore__ inline float GetRankScale(uint32_t rankCount)
{
    switch (rankCount) {
        case 2:
            return 0.7071067811865475f;
        case 3:
            return 0.5773502691896258f;
        case 4:
            return 0.5f;
        case 5:
            return 0.4472135954999579f;
        case 6:
            return 0.4082482904638630f;
        case 7:
            return 0.3779644730092272f;
        case 8:
            return 0.3535533905932738f;
        default:
            return 1.0f;
    }
}

__aicore__ inline void AdvanceToGroup(__gm__ int32_t *lensGm, __gm__ int32_t *ranksGm,
                                      uint32_t groupCount, uint32_t totalUserEntries,
                                      uint32_t &userOffset, uint32_t &groupRowOffset)
{
    userOffset = 0;
    groupRowOffset = 0;
    for (uint32_t i = 0; i < groupCount; ++i) {
        const int32_t rawLen = lensGm[i];
        const uint32_t currentLen = rawLen > 0 ? static_cast<uint32_t>(rawLen) : 0;
        const uint32_t remaining = userOffset < totalUserEntries ? totalUserEntries - userOffset : 0;
        const uint32_t availableLen = currentLen < remaining ? currentLen : remaining;
        for (uint32_t k = 0; k < availableLen; ++k) {
            groupRowOffset += GetRankCount(ranksGm[userOffset + k]);
        }
        if (currentLen >= remaining) {
            userOffset = totalUserEntries;
            return;
        }
        userOffset += currentLen;
    }
}

__aicore__ inline uint32_t GetCurrentGroupRows(__gm__ int32_t *ranksGm, uint32_t userOffset,
                                               uint32_t availableLen)
{
    uint32_t groupRows = 0;
    for (uint32_t k = 0; k < availableLen; ++k) {
        groupRows += GetRankCount(ranksGm[userOffset + k]);
    }
    return groupRows;
}

__aicore__ inline void WriteZeros(GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                  LocalTensor<float> &tmpR, LocalTensor<float> &tmpI,
                                  uint64_t baseOffset, uint32_t elemCount)
{
    uint32_t offset = 0;
    while (offset < elemCount) {
        const uint32_t copyElems = static_cast<uint32_t>(MinU64(COPY_BUFFER_ELEMS, elemCount - offset));
        ClearLocalTensor(tmpR, copyElems);
        ClearLocalTensor(tmpI, copyElems);
        PipeBarrier<PIPE_ALL>();
        DataCopy(outR[baseOffset + offset], tmpR, copyElems);
        DataCopy(outI[baseOffset + offset], tmpI, copyElems);
        PipeBarrier<PIPE_ALL>();
        offset += copyElems;
    }
}

__aicore__ inline void ProcessDstIndex(GlobalTensor<float> &weightR, GlobalTensor<float> &weightI,
                                       GM_ADDR getIdxs, GM_ADDR lens,
                                       GM_ADDR getUserIds, GM_ADDR getUserIdRank,
                                       GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                       TBuf<> &tmpRBuf, TBuf<> &tmpIBuf,
                                       uint32_t userCount, uint32_t totalUserEntries,
                                       uint32_t dstIndex)
{
    auto idxsGm = reinterpret_cast<__gm__ int32_t *>(getIdxs);
    auto lensGm = reinterpret_cast<__gm__ int32_t *>(lens);
    auto userIdsGm = reinterpret_cast<__gm__ int32_t *>(getUserIds);
    auto ranksGm = reinterpret_cast<__gm__ int32_t *>(getUserIdRank);

    const uint32_t groupIndex = dstIndex / INDEX_GROUP_WIDTH;
    const uint32_t localIndex = dstIndex % INDEX_GROUP_WIDTH;

    LocalTensor<float> tmpRLocal = tmpRBuf.Get<float>();
    LocalTensor<float> tmpILocal = tmpIBuf.Get<float>();

    uint32_t userOffset = 0;
    uint32_t groupRowOffset = 0;
    AdvanceToGroup(lensGm, ranksGm, groupIndex, totalUserEntries, userOffset, groupRowOffset);
    uint32_t outputColOffset = 0;

    const int32_t rawLen = lensGm[groupIndex];
    const uint32_t currentLen = rawLen > 0 ? static_cast<uint32_t>(rawLen) : 0;
    const uint32_t remaining = userOffset < totalUserEntries ? totalUserEntries - userOffset : 0;
    const uint32_t availableLen = currentLen < remaining ? currentLen : remaining;
    const uint32_t currentGroupRows = GetCurrentGroupRows(ranksGm, userOffset, availableLen);
    const uint64_t baseDstRow =
        (static_cast<uint64_t>(groupRowOffset) * INDEX_GROUP_WIDTH) +
        (static_cast<uint64_t>(localIndex) * currentGroupRows);
    const int32_t idxGroup = idxsGm[groupIndex];
    if (idxGroup < 0 || static_cast<uint32_t>(idxGroup) >= INDEX_GROUP_COUNT) {
        WriteZeros(outR, outI, tmpRLocal, tmpILocal, baseDstRow * ROWS, currentGroupRows * ROWS);
        return;
    }
    const uint32_t srcIndex = static_cast<uint32_t>(idxGroup) * INDEX_GROUP_WIDTH + localIndex;

    for (uint32_t k = 0; k < availableLen; ++k) {
        const int32_t userId = userIdsGm[userOffset + k];
        const uint32_t rankCount = GetRankCount(ranksGm[userOffset + k]);
        const uint32_t remainingCols =
            outputColOffset < currentGroupRows ? currentGroupRows - outputColOffset : 0;
        const uint32_t copyCols = rankCount < remainingCols ? rankCount : remainingCols;
        if (copyCols > 0) {
            const uint64_t baseDstOffset = (baseDstRow + outputColOffset) * ROWS;
            const uint32_t copyElems = copyCols * ROWS;
            if (IsValidUserId(userId, userCount)) {
                const uint64_t baseSrcOffset =
                    ((static_cast<uint64_t>(userId) * INDEX_COUNT + srcIndex) * RANKS_PER_INDEX) * ROWS;
                DataCopy(tmpRLocal, weightR[baseSrcOffset], copyElems);
                DataCopy(tmpILocal, weightI[baseSrcOffset], copyElems);
                PipeBarrier<PIPE_ALL>();
                const float rankScale = GetRankScale(rankCount);
                if (rankScale != 1.0f) {
                    Muls(tmpRLocal, tmpRLocal, rankScale, copyElems);
                    Muls(tmpILocal, tmpILocal, rankScale, copyElems);
                    PipeBarrier<PIPE_ALL>();
                }
                DataCopy(outR[baseDstOffset], tmpRLocal, copyElems);
                DataCopy(outI[baseDstOffset], tmpILocal, copyElems);
                PipeBarrier<PIPE_ALL>();
            } else {
                WriteZeros(outR, outI, tmpRLocal, tmpILocal, baseDstOffset, copyElems);
            }
        }
        outputColOffset += rankCount;
    }
}
} // namespace GetWeightByRankKernel

extern "C" __global__ __aicore__ void get_weight_by_rank(GM_ADDR weight_r, GM_ADDR weight_i,
                                                          GM_ADDR getIdxs, GM_ADDR lens,
                                                          GM_ADDR getuserIds, GM_ADDR getuserIdRank,
                                                          GM_ADDR weightout_r, GM_ADDR weightout_i,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__DAV_CUBE__)
    (void)weight_r;
    (void)weight_i;
    (void)getIdxs;
    (void)lens;
    (void)getuserIds;
    (void)getuserIdRank;
    (void)weightout_r;
    (void)weightout_i;
    (void)workspace;
    (void)tiling;
    return;
#else
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    using namespace GetWeightByRankKernel;

    const uint32_t userCount = tilingData.userCount;
    const uint32_t idxCount = tilingData.idxCount;
    const uint32_t totalUserEntries = tilingData.totalUserEntries;
    const uint32_t totalOutputRows = tilingData.totalOutputRows;
    if (userCount == 0 || idxCount == 0) {
        return;
    }

    GlobalTensor<float> outRGm;
    GlobalTensor<float> outIGm;
    GlobalTensor<float> weightRGm;
    GlobalTensor<float> weightIGm;
    const uint64_t outElems = static_cast<uint64_t>(totalOutputRows) * ROWS;
    const uint64_t weightElems = static_cast<uint64_t>(userCount) * INDEX_COUNT * ROWS * RANKS_PER_INDEX;
    outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_r), outElems);
    outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_i), outElems);
    weightRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_r), weightElems);
    weightIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_i), weightElems);

    TPipe pipe;
    TBuf<> tmpRBuf;
    TBuf<> tmpIBuf;
    pipe.InitBuffer(tmpRBuf, COPY_BUFFER_BYTES);
    pipe.InitBuffer(tmpIBuf, COPY_BUFFER_BYTES);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetAivBlockNum());
    const uint32_t dstCount = idxCount * INDEX_GROUP_WIDTH;
    for (uint32_t dstIndex = blockIdx; dstIndex < dstCount; dstIndex += blockNum) {
        ProcessDstIndex(weightRGm, weightIGm, getIdxs, lens, getuserIds, getuserIdRank,
                        outRGm, outIGm, tmpRBuf, tmpIBuf,
                        userCount, totalUserEntries, dstIndex);
    }
#endif
}
